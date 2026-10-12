"""
Ground-truth test: how close is the tool's output to real survey data it never saw?

The tool is built only on the General Social Survey (GSS). This script tests it against a separate real survey of
398 working adults (Prolific), using 153 items from 34 organizational scales. The real survey is NEVER used as an
anchor: it enters only here, as the thing the output is compared with.

WHAT IT DOES
  1. GENERATE (real API calls, resumable): every one of the 153 items is generated, in batches of at most
     30 items that keep each scale together. The whole set is repeated 3 times with different random groupings of
     scales. Respondent-level generated data AND the tool's own coverage/tier report are saved to
     validation/results/real_data_test/, so nothing has to be inferred later and nothing is lost.
     Sampling is mode="custom", filters=["employed"] (working adults), N = 398, 7-point scale.
  2. ANALYZE (free, no API calls, can be re-run any time):
       - mean, SD, shape (skewness, kurtosis), tails (share answering 1-2 and 6-7), full-distribution distance
       - item-level correlations (all / different scales / same scale / by anchor status)
       - scale-level correlations
       - the "divide by 1.1" correction from the app note
       - breakdowns by anchor status, tier, wording direction and response format
       - a "noise floor": how big the gaps would be if the generated data were simply a second independent
         survey of 398 people, so every number has a yardstick
       - 95% confidence intervals (bootstrap over items for marginals, over batches for correlations)
  3. Writes summary_for_paper.csv plus per-item and per-pair CSVs.

HOW TO RUN (from the repository root)
    python validation/12_validate_against_real_data.py             # generate what is missing, then analyze
    python validation/12_validate_against_real_data.py --analyze   # analysis only, from saved files (free)
    python validation/12_validate_against_real_data.py --design    # show every batch and the call count, call nothing

COST: 3 partitions = 19 batches x about 50 API calls each. The batch design is fixed in
validation/results/real_data_test/design.json.

NEEDS (in validation/reference_data/): items_metadata_FINAL_34.json, prolific_item_value_proportions.json,
                  prolific_item_correlation_matrix.csv, prolific_scale_correlation_matrix.csv
"""

import importlib.util
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # this file lives in validation/
DATA_DIR = os.path.join(REPO_ROOT, "validation", "reference_data")   # the real survey used as the test set
OUT_DIR = os.path.join(REPO_ROOT, "validation", "results", "real_data_test")

# ------------------------------------------------------------------------------------------ settings
N_PARTITIONS = 3          # how many partitions to RUN (1-3). The design below has 3; use 1 for a cheaper first pass.
DESIGN_PARTITIONS = 3     # size of the saved design (validation/results/real_data_test/design.json); do not change once it exists
MAX_BATCH_ITEMS = 30      # items per generation call (your validated range is up to 50)
N_RESPONDENTS = 398       # matches the real Prolific sample size
N_BOOT = 1000             # bootstrap repetitions for confidence intervals
N_SIM_NOISE = 200         # simulations per item for the noise-floor benchmark
OUT_POINTS = 7            # every item is generated on a 7-point scale
DEFAULT_POINTS = 7
CORRECTION_DIVISOR = 1.1  # the divisor in the app's correlation note

# Answer options in your real survey for items that are NOT 7-point (stretched onto 1-7 for comparison).
SCALE_POINTS_BY_ITEM = {
    "religiosity1": 6, "religiosity2": 6,
    "religiosity3": 5, "religiosity4": 5, "religiosity5": 5,
    "volunteeringncharity1": 5, "volunteeringncharity2": 5, "volunteeringncharity3": 5,
    "volunteeringncharity4": 5, "volunteeringncharity5": 5, "volunteeringncharity6": 5,
}
# "auto": a nested JSON ({"n":..,"proportions":..}) holds reverse-worded items as asked; an old flat JSON holds them
# reverse-scored. Override with "asked" or "reverse_scored" if needed.
REVERSE_STORAGE = "auto"

GRID = np.arange(1, OUT_POINTS + 1)
EPS = 1e-9


# ------------------------------------------------------------------------------------------ shape metrics
def positions(k):
    """Where the k answer options sit on the 1-7 scale (equally spaced, first -> 1, last -> 7)."""
    return np.array([1.0 + j * (OUT_POINTS - 1) / (k - 1) for j in range(k)])


def shape_metrics(shares, pos):
    """Mean, SD, skewness, excess kurtosis, tail shares and cumulative curve; works on a batch of distributions."""
    s = np.atleast_2d(np.asarray(shares, float))
    m = s @ pos
    sd = np.sqrt(np.maximum(s @ pos ** 2 - m ** 2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (pos[None, :] - m[:, None]) / sd[:, None]
        skew = (s * z ** 3).sum(1)
        kurt = (s * z ** 4).sum(1) - 3.0
    skew[sd < 1e-9] = np.nan
    kurt[sd < 1e-9] = np.nan
    cdf = np.stack([s[:, pos <= x + EPS].sum(1) for x in GRID], axis=1)
    return {"mean": m, "sd": sd, "skew": skew, "kurt": kurt,
            "bottom2": s[:, pos <= 2 + EPS].sum(1), "top2": s[:, pos >= 6 - EPS].sum(1), "cdf": cdf}


# ------------------------------------------------------------------------------------------ real data
def fix_codes(cats):
    """Items exported with shifted codes: 36-42 -> subtract 35, 8-14 -> subtract 7 (both become 1-7)."""
    lo, hi = min(cats), max(cats)
    if lo >= 36 and hi <= 42:
        return {k - 35: v for k, v in cats.items()}, "subtracted 35"
    if lo >= 8 and hi <= 14:
        return {k - 7: v for k, v in cats.items()}, "subtracted 7"
    return cats, ""


def load_meta():
    meta = json.load(open(os.path.join(DATA_DIR, "items_metadata_FINAL_34.json")))
    return [m for m in meta if m["scale"] != "value_exercise"]


def load_reference():
    meta = load_meta()
    info = {m["prolific_column"].lower(): {"scale": m["scale"], "rev": bool(m["reverse"]), "text": m["text"]} for m in meta}
    raw = json.load(open(os.path.join(DATA_DIR, "prolific_item_value_proportions.json")))
    nested = any(isinstance(v, dict) and "proportions" in v for v in raw.values())
    mirror = (not nested) if REVERSE_STORAGE == "auto" else (REVERSE_STORAGE == "reverse_scored")
    rawl = {k.lower(): v for k, v in raw.items()}
    ref, skipped = {}, []
    for item, inf in info.items():
        entry = rawl.get(item)
        props = None if entry is None else (entry["proportions"] if isinstance(entry, dict) and "proportions" in entry else entry)
        cats = {float(k): float(v) for k, v in (props or {}).items() if k != "__other__"}
        if not cats:
            skipped.append(item)
            continue
        cats, note = fix_codes(cats)
        k = SCALE_POINTS_BY_ITEM.get(item, DEFAULT_POINTS)
        shares, outside = np.zeros(k), False
        for key, v in cats.items():
            i = int(round(key))
            if 1 <= i <= k:
                shares[i - 1] += v
            else:
                outside = True
        if outside or shares.sum() <= 0:
            skipped.append(f"{item} (answers outside 1-{k})")
            continue
        shares /= shares.sum()
        if mirror and inf["rev"]:
            shares = shares[::-1]
        ref[item] = {**inf, "k": k, "shares": shares, "pos": positions(k), "note": note}
    return info, ref, nested, mirror, skipped


def load_real_corr():
    ic = pd.read_csv(os.path.join(DATA_DIR, "prolific_item_correlation_matrix.csv"), index_col=0)
    ic.index, ic.columns = ic.index.str.lower(), ic.columns.str.lower()
    sc = pd.read_csv(os.path.join(DATA_DIR, "prolific_scale_correlation_matrix.csv"), index_col=0)
    return ic, sc


# ------------------------------------------------------------------------------------------ generation
def make_design():
    """The batch design is saved in validation/results/real_data_test/design.json and reused as is, so every run (and every
    computer) uses exactly the same groupings. Only the first N_PARTITIONS partitions are returned."""
    path = os.path.join(OUT_DIR, "design.json")
    if os.path.exists(path):
        design = json.load(open(path))
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        by_scale = {}
        for m in load_meta():
            by_scale.setdefault(m["scale"], []).append(m["prolific_column"].lower())
        design = {"n": N_RESPONDENTS, "max_batch_items": MAX_BATCH_ITEMS, "partitions": []}
        for p in range(1, DESIGN_PARTITIONS + 1):
            rng = np.random.default_rng(100 + p)
            total = sum(len(v) for v in by_scale.values())
            batches = [{"scales": [], "items": []} for _ in range(-(-total // MAX_BATCH_ITEMS))]
            for sc in [sorted(by_scale)[i] for i in rng.permutation(len(by_scale))]:
                size = len(by_scale[sc])
                room = [b for b in batches if len(b["items"]) + size <= MAX_BATCH_ITEMS]
                if not room:                       # no batch has space: open a new one
                    batches.append({"scales": [], "items": []})
                    room = [batches[-1]]
                target = min(room, key=lambda b: len(b["items"]))   # fill the emptiest batch, so sizes stay even
                target["scales"].append(sc)
                target["items"] += by_scale[sc]
            design["partitions"].append([b for b in batches if b["items"]])
        json.dump(design, open(path, "w"), indent=1)
    if N_PARTITIONS > len(design["partitions"]):
        sys.exit(f"design.json only has {len(design['partitions'])} partitions; set N_PARTITIONS to at most that.")
    design = dict(design)
    design["partitions"] = design["partitions"][:N_PARTITIONS]
    return design


def load_generator():
    os.environ.setdefault("OPENAI_API_KEY", "")
    spec = importlib.util.spec_from_file_location("gen", os.path.join(REPO_ROOT, "generate_dataset.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def generate_with_retry(gen, texts, seed, tries=4, wait=20):
    for attempt in range(1, tries + 1):
        try:
            return gen.generate_dataset(items=[{"text": t, "type": "likert"} for t in texts], n=N_RESPONDENTS, seed=seed,
                                        mode="custom", filters=["employed"], likert_scale=OUT_POINTS,
                                        confirm_large_request=True)
        except Exception as e:  # retry only on network trouble; real errors should fail loudly
            if not any(k in type(e).__name__ for k in ("Timeout", "Connection")) or attempt == tries:
                raise
            print(f"  ** network error ({type(e).__name__}); waiting {wait}s, retry {attempt}/{tries} **")
            time.sleep(wait)


def run_generation(gen=None):
    design = make_design()
    text = {m["prolific_column"].lower(): m["text"] for m in load_meta()}
    n_batches = sum(len(b) for b in design["partitions"])
    print(f"Design: {len(design['partitions'])} partitions, {n_batches} batches, "
          f"{sum(len(b['items']) for p in design['partitions'] for b in p)} item generations, N = {N_RESPONDENTS} respondents each.")
    gen = gen or load_generator()
    for p, batches in enumerate(design["partitions"], start=1):
        for b, batch in enumerate(batches, start=1):
            rp, cp = os.path.join(OUT_DIR, f"resp_p{p}_b{b}.csv"), os.path.join(OUT_DIR, f"cov_p{p}_b{b}.csv")
            tag = f"partition {p}/{len(design['partitions'])}, batch {b}/{len(batches)}"
            if os.path.exists(rp) and os.path.exists(cp):
                print(f"[{tag}] already done, skipping")
                continue
            items = batch["items"]
            print(f"[{tag}] generating {len(items)} items from scales {batch['scales']}")
            df, cov = generate_with_retry(gen, [text[i] for i in items], seed=5000 + 100 * p + b)
            resp = pd.DataFrame({it: df[f"item_{i + 1}_response"].to_numpy() for i, it in enumerate(items)})
            cov = cov.copy()
            cov["item_name"] = items
            cov["source"] = [df[f"item_{i + 1}_source"].iloc[0] for i in range(len(items))]
            cov["partition"], cov["batch"] = p, b
            resp.to_csv(rp, index=False)
            cov.to_csv(cp, index=False)
    print("Generation complete.")


def load_outputs():
    resp, cov = {}, {}
    if not os.path.isdir(OUT_DIR):
        return resp, cov
    for f in sorted(os.listdir(OUT_DIR)):
        m = re.match(r"resp_p(\d+)_b(\d+)\.csv$", f)
        cp = os.path.join(OUT_DIR, f"cov_p{m[1]}_b{m[2]}.csv") if m else None
        if m and os.path.exists(cp):
            resp[(int(m[1]), int(m[2]))] = pd.read_csv(os.path.join(OUT_DIR, f))
            cov[(int(m[1]), int(m[2]))] = pd.read_csv(cp)
    return resp, cov


# ------------------------------------------------------------------------------------------ statistics helpers
def fnum(x, nd=2, plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def boot_ci(arrays, stat_fn, n_boot, seed=0, clusters=None):
    """95% percentile CI. Rows are resampled; or whole clusters (e.g. batches) when `clusters` is given."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(arrays.values())))
    rows = []
    if clusters is None:
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            rows.append(stat_fn({k: v[idx] for k, v in arrays.items()}))
    else:
        clusters = np.asarray(clusters)
        members = {c: np.where(clusters == c)[0] for c in np.unique(clusters)}
        keys = list(members)
        for _ in range(n_boot):
            idx = np.concatenate([members[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
            rows.append(stat_fn({k: v[idx] for k, v in arrays.items()}))
    return pd.DataFrame(rows).quantile([0.025, 0.975])


def noise_floor(ref, items, rng):
    """Typical size of each gap between two INDEPENDENT surveys of N respondents drawn from the real distributions."""
    acc = {k: [] for k in ("mean", "sd", "skew", "kurt", "bottom2", "top2", "cdf", "ks")}
    for it in items:
        sh, pos = ref[it]["shares"], ref[it]["pos"]
        a = rng.multinomial(N_RESPONDENTS, sh, size=N_SIM_NOISE) / N_RESPONDENTS
        b = rng.multinomial(N_RESPONDENTS, sh, size=N_SIM_NOISE) / N_RESPONDENTS
        ma, mb = shape_metrics(a, pos), shape_metrics(b, pos)
        for k in ("mean", "sd", "skew", "kurt", "bottom2", "top2"):
            acc[k].append(np.nanmean(np.abs(ma[k] - mb[k])))
        acc["cdf"].append(np.mean(np.abs(ma["cdf"] - mb["cdf"])))
        acc["ks"].append(np.mean(np.max(np.abs(ma["cdf"] - mb["cdf"]), axis=1)))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


# ------------------------------------------------------------------------------------------ item-level table
def build_item_table(ref, resp, cov):
    per = {}
    for key, r in resp.items():
        c = cov[key].set_index("item_name")
        for item in r.columns:
            if item not in ref:
                continue
            x = r[item].to_numpy().astype(int)
            sh = np.bincount(x, minlength=OUT_POINTS + 1)[1:OUT_POINTS + 1] / len(x)
            row = c.loc[item]
            per.setdefault(item, []).append({"sh": sh, "source": row["source"], "tier": row.get("confidence_tier"),
                                             "method": row.get("method"), "matched": row.get("matched_variable"),
                                             "sim": row.get("similarity_top1", np.nan)})
    rows, gshares = [], {}
    for item, lst in per.items():
        sh = np.mean([d["sh"] for d in lst], axis=0)
        gshares[item] = sh
        if len({d["source"] for d in lst}) > 1:
            print(f"  note: {item} was anchored differently in different partitions")
        rm = shape_metrics(ref[item]["shares"], ref[item]["pos"])
        gm = shape_metrics(sh, GRID.astype(float))
        k = ref[item]["k"]
        rows.append({"item": item, "scale": ref[item]["scale"], "rev": ref[item]["rev"], "k": k,
                     "format": "7-point" if k == OUT_POINTS else f"{k}-point",
                     "source": lst[0]["source"], "tier": lst[0]["tier"], "method": lst[0]["method"],
                     "matched": lst[0]["matched"], "similarity": lst[0]["sim"], "partitions": len(lst),
                     "spread_across_partitions": max(abs(float(d["sh"] @ GRID) - float(sh @ GRID)) for d in lst),
                     **{f"r_{m}": float(rm[m][0]) for m in ("mean", "sd", "skew", "kurt", "bottom2", "top2")},
                     **{f"g_{m}": float(gm[m][0]) for m in ("mean", "sd", "skew", "kurt", "bottom2", "top2")},
                     "cdfdist": float(np.mean(np.abs(gm["cdf"][0] - rm["cdf"][0]))),
                     "ks": float(np.max(np.abs(gm["cdf"][0] - rm["cdf"][0]))),
                     "cdfdist_flipped": float(np.mean(np.abs(shape_metrics(sh[::-1], GRID.astype(float))["cdf"][0] - rm["cdf"][0])))})
    df = pd.DataFrame(rows).set_index("item")
    df["group"] = np.where(df.source == "population_calibrated", "no real anchor (fallback)", "real anchor found")
    return df, gshares


METRIC_ROWS = [  # label, key in the stats dict
    ("mean gap", "mean_gap"), ("mean |gap|", "mean_abs"), ("SD gap", "sd_gap"), ("skew gap", "skew_gap"),
    ("kurtosis gap", "kurt_gap"), ("share 1-2 gap", "low_gap"), ("share 6-7 gap", "top_gap"),
    ("distance (CDF)", "cdfdist"), ("KS", "ks")]


def marginal_stats(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # empty groups give NaN, not a stream of warnings
        return _marginal_stats(a)


def _marginal_stats(a):
    d = lambda g, r: a[g] - a[r]
    return {"mean_gap": np.nanmean(d("g_mean", "r_mean")), "mean_abs": np.nanmean(np.abs(d("g_mean", "r_mean"))),
            "sd_gap": np.nanmean(d("g_sd", "r_sd")), "sd_abs": np.nanmean(np.abs(d("g_sd", "r_sd"))),
            "skew_gap": np.nanmean(d("g_skew", "r_skew")), "skew_abs": np.nanmean(np.abs(d("g_skew", "r_skew"))),
            "kurt_gap": np.nanmean(d("g_kurt", "r_kurt")),
            "low_gap": np.nanmean(d("g_bottom2", "r_bottom2")), "low_abs": np.nanmean(np.abs(d("g_bottom2", "r_bottom2"))),
            "top_gap": np.nanmean(d("g_top2", "r_top2")), "top_abs": np.nanmean(np.abs(d("g_top2", "r_top2"))),
            "cdfdist": np.nanmean(a["cdfdist"]), "ks": np.nanmean(a["ks"]),
            "r_means": safe_corr(a["g_mean"], a["r_mean"]), "r_sds": safe_corr(a["g_sd"], a["r_sd"]),
            "sd_of_g_means": np.nanstd(a["g_mean"]), "sd_of_r_means": np.nanstd(a["r_mean"])}


def arrays_of(d):
    cols = ["g_mean", "r_mean", "g_sd", "r_sd", "g_skew", "r_skew", "g_kurt", "r_kurt", "g_bottom2", "r_bottom2",
            "g_top2", "r_top2", "cdfdist", "ks"]
    return {c: d[c].to_numpy(float) for c in cols}


# ------------------------------------------------------------------------------------------ correlation tables
def build_pairs(info, resp, cov, ic, sc_real):
    ipairs, spairs = [], []
    for (p, b), r in resp.items():
        src = dict(zip(cov[(p, b)]["item_name"], cov[(p, b)]["source"]))
        scored = r.copy()
        for c in scored.columns:
            if info[c]["rev"]:
                scored[c] = (OUT_POINTS + 1) - scored[c]
        corr = scored.corr()
        cols = list(scored.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b2 = cols[i], cols[j]
                if a not in ic.index or b2 not in ic.columns:
                    continue
                real, gen = ic.loc[a, b2], corr.loc[a, b2]
                if pd.isna(real) or pd.isna(gen):
                    continue
                ipairs.append({"cluster": f"p{p}_b{b}", "item_1": a, "item_2": b2, "same_scale": info[a]["scale"] == info[b2]["scale"],
                               "real_r": real, "gen_r": gen, "anchors": f"{int(src[a] == 'item_calibrated') + int(src[b2] == 'item_calibrated')} of 2 anchored"})
        by_scale = {}
        for c in cols:
            by_scale.setdefault(info[c]["scale"], []).append(c)
        comp = pd.DataFrame({s: scored[v].mean(axis=1) for s, v in by_scale.items()}).corr()
        names = list(comp.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if names[i] in sc_real.index and names[j] in sc_real.columns and not pd.isna(comp.loc[names[i], names[j]]):
                    spairs.append({"cluster": f"p{p}_b{b}", "scale_1": names[i], "scale_2": names[j],
                                   "real_r": sc_real.loc[names[i], names[j]], "gen_r": comp.loc[names[i], names[j]]})
    return pd.DataFrame(ipairs), pd.DataFrame(spairs)


def corr_stats(a):
    real, gen = a["real_r"], a["gen_r"]
    if len(real) < 3:
        return {}
    d = gen - real
    slope, intercept = np.polyfit(real, gen, 1) if np.std(real) > 1e-12 else (np.nan, np.nan)
    r = safe_corr(real, gen)
    return {"pairs": float(len(real)), "real mean r": real.mean(), "generated mean r": gen.mean(), "bias (gen-real)": d.mean(),
            "MAE": np.abs(d).mean(), "RMSE": np.sqrt((d ** 2).mean()), "pattern r": r,
            "pattern rho": safe_corr(pd.Series(real).rank(), pd.Series(gen).rank()), "slope": slope, "intercept": intercept,
            "R2": r ** 2 if not np.isnan(r) else np.nan, "real SD": real.std(), "generated SD": gen.std(),
            "strong (|r|>=.3) real": (np.abs(real) >= .3).mean(), "strong generated": (np.abs(gen) >= .3).mean(),
            "within +-0.10": (np.abs(d) <= .10).mean(),
            f"MAE after /{CORRECTION_DIVISOR}": np.abs(real - gen / CORRECTION_DIVISOR).mean(),
            f"pairs improved by /{CORRECTION_DIVISOR}": (np.abs(real - gen / CORRECTION_DIVISOR) < np.abs(real - gen)).mean()}


def corr_noise_floor(real):
    return float((0.798 * np.sqrt(2) * (1 - np.asarray(real) ** 2) / np.sqrt(N_RESPONDENTS - 1)).mean())


# ------------------------------------------------------------------------------------------ analysis
def analyze():
    resp, cov = load_outputs()
    if not resp:
        sys.exit(f"No generated data found in {OUT_DIR}. Run without --analyze first.")
    info, ref, nested, mirror, skipped = load_reference()
    ic, sc_real = load_real_corr()
    summary = []   # rows for summary_for_paper.csv

    print("=" * 110)
    print("REAL-DATA PREPARATION")
    print("=" * 110)
    print(f"Generated batches found: {len(resp)}   (partitions: {sorted({k[0] for k in resp})})")
    shifted = [i for i, r in ref.items() if r["note"]]
    print(f"Shifted codes fixed ({len(shifted)} items): "
          + "; ".join(f"{n}: {', '.join(i for i in shifted if ref[i]['note'] == n)}" for n in sorted({ref[i]['note'] for i in shifted})))
    print("Not 7-point (stretched onto 1-7): " + ", ".join(f"{i} ({r['k']}-pt)" for i, r in ref.items() if r["k"] != OUT_POINTS))
    print(f"Reverse-worded items in the JSON are treated as {'reverse-scored (mirrored back)' if mirror else 'stored as asked'} "
          f"({'old flat' if not nested else 'new nested'} format)")
    if skipped:
        print(f"Items without a usable real distribution (left out of the marginal comparison): {skipped}")
    items_df, gshares = build_item_table(ref, resp, cov)
    items_df.to_csv(os.path.join(OUT_DIR, "item_level.csv"))
    n_items = len(items_df)
    print(f"\nItems compared: {n_items}.  Largest change in an item's generated mean between partitions: "
          f"{items_df.spread_across_partitions.max():.4f}")
    print("Anchor status as reported by the tool (tier):", items_df.tier.value_counts().to_dict(),
          "| source:", items_df.source.value_counts().to_dict())

    rng = np.random.default_rng(7)
    floor = noise_floor(ref, list(items_df.index), rng)

    # ---------------------------------------------------------------- marginals
    print("\n" + "=" * 110)
    print("PART A. MARGINALS: mean, SD, shape and tails.  gap = generated minus real; each item counted once; 7-point scale")
    print("=" * 110)
    header = f"{'group':<40}{'n':>4}" + "".join(f"{lab:>15}" for lab, _ in METRIC_ROWS)
    print(header)

    def row(label, d):
        st = marginal_stats(arrays_of(d))
        print(f"{label:<40}{len(d):>4}" + "".join(f"{fnum(st[k], 2, plus=(k.endswith('gap'))):>15}" for _, k in METRIC_ROWS))
        return st

    all_st = row("ALL items", items_df)
    groups = [("by anchor status", "group"), ("by tier", "tier"), ("by wording", None), ("by response format", "format")]
    for title, col in groups:
        print(f"-- {title}")
        if col is None:
            for rv, d in items_df.groupby("rev"):
                row("reverse-worded" if rv else "positively worded", d)
        else:
            for k_, d in items_df.groupby(col):
                row(str(k_), d)
    print("-- anchor status x wording")
    for (g_, rv), d in items_df.groupby(["group", "rev"]):
        row(f"{g_} / {'reverse' if rv else 'positive'}", d)

    print(f"\nNoise floor (typical |gap| if the generated data were a second independent survey of N={N_RESPONDENTS}):")
    print(f"  mean {floor['mean']:.2f} | SD {floor['sd']:.2f} | skew {floor['skew']:.2f} | kurtosis {floor['kurt']:.2f} | "
          f"share 1-2 {floor['bottom2']:.3f} | share 6-7 {floor['top2']:.3f} | CDF distance {floor['cdf']:.3f} | KS {floor['ks']:.3f}")

    ci = boot_ci(arrays_of(items_df), marginal_stats, N_BOOT, seed=1)
    print("\n95% confidence intervals for ALL items (bootstrap over items):")
    for lab, key, bench in [("mean gap", "mean_gap", 0.0), ("mean |gap|", "mean_abs", floor["mean"]), ("SD gap", "sd_gap", 0.0),
                            ("SD |gap|", "sd_abs", floor["sd"]), ("skewness |gap|", "skew_abs", floor["skew"]),
                            ("share answering 1-2, gap", "low_gap", 0.0), ("share answering 6-7, gap", "top_gap", 0.0),
                            ("distribution distance (CDF)", "cdfdist", floor["cdf"]), ("KS", "ks", floor["ks"])]:
        print(f"  {lab:<30}{fnum(all_st[key], 3, plus=key.endswith('gap')):>9}   [{fnum(ci.loc[.025, key], 3, True if key.endswith('gap') else False)}, "
              f"{fnum(ci.loc[.975, key], 3, True if key.endswith('gap') else False)}]   noise floor {bench:.3f}")
        summary.append({"section": "marginals", "metric": lab, "value": all_st[key], "ci_low": ci.loc[.025, key],
                        "ci_high": ci.loc[.975, key], "noise_floor": bench})

    print("\nCan the tool tell items apart?")
    for lab, d in [("ALL", items_df), ("no real anchor", items_df[items_df.source == "population_calibrated"]),
                   ("real anchor found", items_df[items_df.source != "population_calibrated"])]:
        if len(d) == 0:
            continue
        st = marginal_stats(arrays_of(d))
        print(f"  {lab:<18} real item means: SD across items {st['sd_of_r_means']:.2f} | generated: {st['sd_of_g_means']:.2f} | "
              f"correlation of item means {fnum(st['r_means'], 2, True)} | of item SDs {fnum(st['r_sds'], 2, True)}")
    st_all = marginal_stats(arrays_of(items_df))
    summary.append({"section": "marginals", "metric": "correlation across items: generated vs real item mean", "value": st_all["r_means"],
                    "ci_low": ci.loc[.025, "r_means"], "ci_high": ci.loc[.975, "r_means"], "noise_floor": np.nan})
    summary.append({"section": "marginals", "metric": "correlation across items: generated vs real item SD", "value": st_all["r_sds"],
                    "ci_low": ci.loc[.025, "r_sds"], "ci_high": ci.loc[.975, "r_sds"], "noise_floor": np.nan})

    seven = items_df[items_df.k == OUT_POINTS]
    print("\nShare of respondents giving each answer (7-point items only):")
    print(f"  {'':<42}" + "".join(f"{c:>7}" for c in GRID))
    for lab, d in [("no real anchor", seven[seven.source == "population_calibrated"]), ("real anchor found", seven[seven.source != "population_calibrated"])]:
        if len(d):
            print(f"  generated, {lab:<30}" + "".join(f"{x:7.2f}" for x in np.mean([gshares[i] for i in d.index], axis=0)))
            print(f"  real, same items{'':<26}" + "".join(f"{x:7.2f}" for x in np.mean([ref[i]["shares"] for i in d.index], axis=0)))
    pos7 = seven[~seven.rev]
    print(f"  real, positively worded, all anchor groups " + "".join(f"{x:7.2f}" for x in np.mean([ref[i]["shares"] for i in pos7.index], axis=0)))

    print("\nDiagnostic only (not a fix): distribution distance as produced vs generated distribution flipped end to end (1<->7)")
    for (g_, rv), d in items_df.groupby(["group", "rev"]):
        print(f"  {g_:<28} {'reverse' if rv else 'positive':<9} n={len(d):<3} as produced {d.cdfdist.mean():.3f}   flipped {d.cdfdist_flipped.mean():.3f}")

    # ---------------------------------------------------------------- correlations
    ipairs, spairs = build_pairs(info, resp, cov, ic, sc_real)
    ipairs.to_csv(os.path.join(OUT_DIR, "pair_level.csv"), index=False)
    spairs.to_csv(os.path.join(OUT_DIR, "scale_pair_level.csv"), index=False)

    print("\n" + "=" * 110)
    print("PART B. CORRELATIONS (items generated together in a batch; intervals bootstrap over batches)")
    print("=" * 110)
    keys = [("pairs", 0), ("real mean r", 2), ("generated mean r", 2), ("bias (gen-real)", 3), ("MAE", 3), ("pattern r", 2), ("slope", 2),
            ("intercept", 3), ("R2", 2), ("real SD", 2), ("generated SD", 2), ("within +-0.10", 2),
            (f"MAE after /{CORRECTION_DIVISOR}", 3), (f"pairs improved by /{CORRECTION_DIVISOR}", 2)]

    def corr_block(label, d, tag):
        if len(d) < 10:
            print(f"\n{label}: only {len(d)} pairs, skipped")
            return
        arr = {"real_r": d.real_r.to_numpy(float), "gen_r": d.gen_r.to_numpy(float)}
        st = corr_stats(arr)
        c = boot_ci(arr, corr_stats, N_BOOT, seed=2, clusters=d.cluster.to_numpy())
        bench = corr_noise_floor(arr["real_r"])
        print(f"\n{label}   (noise floor for MAE if the generated data were a second independent survey: {bench:.3f})")
        for k, nd in keys:
            plus = k in ("bias (gen-real)", "intercept")
            if k == "pairs":
                print(f"  {k:<30}{int(st[k]):>9}")
            else:
                print(f"  {k:<30}{fnum(st[k], nd, plus):>9}   [{fnum(c.loc[.025, k], nd, plus)}, {fnum(c.loc[.975, k], nd, plus)}]")
        for k in ("bias (gen-real)", "MAE", "pattern r", "slope", "intercept", "R2", f"MAE after /{CORRECTION_DIVISOR}"):
            summary.append({"section": f"correlations: {tag}", "metric": k, "value": st[k], "ci_low": c.loc[.025, k], "ci_high": c.loc[.975, k],
                            "noise_floor": bench if k == "MAE" else np.nan})

    corr_block("ITEM-LEVEL, all pairs", ipairs, "item-level, all pairs")
    corr_block("ITEM-LEVEL, items from DIFFERENT scales (the comparison used for the app note)", ipairs[~ipairs.same_scale], "item-level, different scales")
    corr_block("ITEM-LEVEL, items from the SAME scale", ipairs[ipairs.same_scale], "item-level, same scale")
    for a_, d in ipairs.groupby("anchors"):
        corr_block(f"ITEM-LEVEL by anchoring: {a_}", d, f"item-level, {a_}")
    corr_block("SCALE-LEVEL (scale averages; scales generated together)", spairs, "scale-level")

    pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "summary_for_paper.csv"), index=False)
    print("\n" + "=" * 110)
    print(f"Saved in {os.path.basename(OUT_DIR)}/: summary_for_paper.csv, item_level.csv, pair_level.csv, scale_pair_level.csv")
    print("Read every gap next to its noise floor: a gap close to the noise floor is indistinguishable from a second survey of 398 people.")


# ------------------------------------------------------------------------------------------ entry point
if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--design":
        d = make_design()
        n_batches = sum(len(b) for b in d["partitions"])
        print(f"{len(d['partitions'])} partition(s), {n_batches} batches (subsamples) in total, N = {N_RESPONDENTS} respondents each\n")
        for p, batches in enumerate(d["partitions"], start=1):
            print(f"PARTITION {p}: {len(batches)} batches, {sum(len(x['scales']) for x in batches)} scales, {sum(len(x['items']) for x in batches)} items")
            for b, x in enumerate(batches, start=1):
                print(f"   batch {b}: {len(x['scales']):>2} scales, {len(x['items']):>2} items  -> {', '.join(x['scales'])}")
        print(f"\nAPI calls: about {-(-N_RESPONDENTS // 8)} per batch, {n_batches * -(-N_RESPONDENTS // 8)} in total")
    elif arg == "--analyze":
        analyze()
    else:
        run_generation()
        analyze()
