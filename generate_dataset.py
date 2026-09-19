"""
Synthetic survey response generator calibrated against real U.S. General
Social Survey (GSS) data.

Pipeline: demographic resampling from real GSS respondents -> retrieval of
the nearest real GSS variable(s) for each user-supplied item -> joint LLM
generation per persona batch -> calibration against real weighted
distributions via percentile mapping.

Scope: GSS is a U.S.-only survey; all calibration reflects the U.S. adult
population.

Requirements:
    pip install openai numpy pandas
    OPENAI_API_KEY environment variable must be set.

Required input files (produced by the pipeline/ scripts):
    gss_variable_embeddings.npy
    gss_variable_index.csv
    gss_master_variable_stats.json
    gss_demographics_pool.csv
    gss_demographics_value_labels.json
    generic_likert_shape.npy
    hybrid_validation_results.csv
"""

import csv
import json
import numpy as np
import pandas as pd
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4.1-mini"

client = OpenAI()

# =========================================================================
# PART 0: Empirical error lookup (from held-out validation study)
# =========================================================================

SIMILARITY_BINS = [0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.01]
MIN_VALIDATION_CASES = 10
MIN_ANCHOR_SPECIFIC_CASES = 5  # if THIS SPECIFIC matched variable was used
                                 # as an anchor at least this many times in
                                 # the validation study, report its own
                                 # historical error rather than the broader
                                 # similarity-bin average

def build_error_lookup_table(validation_csv="data/hybrid_validation_results.csv"):
    """
    Empirical expected-error lookup, binned by similarity, built from the
    hybrid validation study (validation/05_validate_hybrid_coverage.py) so
    it reflects the same method choice (single-best / ensemble / none)
    production actually uses at each similarity level. Falls back to the
    older single-best-only validation file if the hybrid one isn't present.
    """
    try:
        df = pd.read_csv(validation_csv)
        sim_col = "top1_similarity"
        error_col = "error"
        if "method" in df.columns:
            # "none" rows have no real anchor and don't represent an actual
            # match outcome -- including them skews the average for bins
            # that do contain real successes.
            before = len(df)
            df = df[df["method"] != "none"]
            print(f"  (error lookup: excluded {before - len(df)} 'none'-method rows)")
    except FileNotFoundError:
        try:
            df = pd.read_csv("data/validation_results.csv")
            sim_col = "similarity"
            error_col = "cdf_distance_error"
        except FileNotFoundError:
            return None
    df["similarity_bin"] = pd.cut(df[sim_col], bins=SIMILARITY_BINS)
    similarity_table = df.groupby("similarity_bin", observed=True).agg(
        n_validation_cases=(sim_col, "count"),
        avg_error=(error_col, "mean"),
        error_std=(error_col, "std"),
    ).reset_index()
    return similarity_table

def lookup_expected_error(similarity, similarity_table):
    if similarity_table is None:
        return None, None, 0
    for interval in similarity_table["similarity_bin"]:
        if similarity in interval:
            row = similarity_table[similarity_table["similarity_bin"] == interval].iloc[0]
            if row["n_validation_cases"] < MIN_VALIDATION_CASES:
                return None, None, int(row["n_validation_cases"])
            return float(row["avg_error"]), float(row["error_std"]), int(row["n_validation_cases"])
    return None, None, 0

# =========================================================================
# PART 1: Demographic resampling (from real GSS respondents)
# =========================================================================

FILTER_DEFINITIONS = {
    "employed": {"variable": "wrkstat", "label_contains": ["working"], "match_type": "contains", "exclude": False},
    "married": {"variable": "marital", "label_contains": ["married"], "match_type": "exact", "exclude": False},
    "religious": {"variable": "relig", "label_contains": ["none"], "match_type": "exact", "exclude": True},
}

CORE_DEMO_FIELDS = ["age", "sex", "race", "marital", "wrkstat", "relig"]

def load_demographics_pool():
    """Load the real-respondent demographic pool, keeping only complete
    cases across the fields describe_persona() needs (real item-level
    non-response otherwise leaves NaNs that break persona construction)."""
    df = pd.read_csv("data/gss_demographics_pool.csv")
    before = len(df)
    df = df.dropna(subset=CORE_DEMO_FIELDS)
    if before - len(df) > 0:
        print(f"  (demographics pool: dropped {before - len(df)} incomplete respondents)")
    with open("data/gss_demographics_value_labels.json", encoding="utf-8") as f:
        value_labels = json.load(f)
    return df, value_labels

def codes_matching(value_labels, variable, substrings, match_type="contains"):
    labels = value_labels.get(variable, {})
    matches = []
    for code, label in labels.items():
        label_lower = label.lower().strip()
        for s in substrings:
            s_lower = s.lower().strip()
            if (match_type == "exact" and label_lower == s_lower) or \
               (match_type == "contains" and s_lower in label_lower):
                matches.append(float(code))
                break
    return matches

def apply_filters(df, value_labels, filter_names):
    for fname in filter_names:
        spec = FILTER_DEFINITIONS[fname]
        var = spec["variable"]
        match_codes = codes_matching(value_labels, var, spec["label_contains"], spec.get("match_type", "contains"))
        if not match_codes:
            print(f"  WARNING: no codes matched for filter '{fname}'")
            continue
        mask = df[var].isin(match_codes)
        if spec["exclude"]:
            mask = ~mask
        df = df[mask]
        print(f"  Applied filter '{fname}': {len(df)} rows remain")
    return df

def get_gender_code(value_labels, label_substring):
    codes = codes_matching(value_labels, "sex", [label_substring], match_type="exact")
    return codes[0] if codes else None

def get_white_codes(value_labels):
    return codes_matching(value_labels, "race", ["white"], match_type="exact")

def allocate_exact_counts(n, cell_probs):
    raw = {cell: p * n for cell, p in cell_probs.items()}
    floors = {cell: int(np.floor(v)) for cell, v in raw.items()}
    remainder = n - sum(floors.values())
    remainders = sorted(raw.items(), key=lambda kv: kv[1] - floors[kv[0]], reverse=True)
    for i in range(remainder):
        floors[remainders[i][0]] += 1
    return floors

def build_sample_pool(mode, n, seed, filters=None, age_min=None, age_max=None,
                       female_pct=None, white_pct=None):
    """mode: 'representative' or 'custom'. See earlier pipeline docs for details."""
    df, value_labels = load_demographics_pool()
    print(f"Starting pool size: {len(df)}")
    rng = np.random.default_rng(seed)

    if mode == "representative":
        weights = df["wtssall"].to_numpy(dtype=float)
        probs = weights / weights.sum()
        idx = rng.choice(df.index.to_numpy(), size=n, replace=True, p=probs)
        return df.loc[idx].reset_index(drop=True), value_labels

    elif mode == "custom":
        if filters:
            df = apply_filters(df, value_labels, filters)
        if age_min is not None:
            df = df[df["age"] >= age_min]
        if age_max is not None:
            df = df[df["age"] <= age_max]
        if len(df) == 0:
            raise ValueError("No respondents remain after filters/age range -- loosen constraints.")

        female_code = get_gender_code(value_labels, "female")
        male_code = get_gender_code(value_labels, "male")
        white_codes = get_white_codes(value_labels)
        df = df.copy()
        df["_is_white"] = df["race"].isin(white_codes)

        if female_pct is None and white_pct is None:
            weights = df["wtssall"].to_numpy(dtype=float)
            probs = weights / weights.sum()
            idx = rng.choice(df.index.to_numpy(), size=n, replace=True, p=probs)
            return df.loc[idx].reset_index(drop=True), value_labels

        f_pct = female_pct if female_pct is not None else (df["sex"] == female_code).mean()
        w_pct = white_pct if white_pct is not None else df["_is_white"].mean()

        # Simplifying assumption: gender and race targets are treated as
        # independent of each other. Disclose this in methods writeups.
        cell_probs = {
            (female_code, True): f_pct * w_pct,
            (female_code, False): f_pct * (1 - w_pct),
            (male_code, True): (1 - f_pct) * w_pct,
            (male_code, False): (1 - f_pct) * (1 - w_pct),
        }
        cell_counts = allocate_exact_counts(n, cell_probs)

        pieces = []
        for (gender_code, is_white), count in cell_counts.items():
            if count == 0:
                continue
            cell_df = df[(df["sex"] == gender_code) & (df["_is_white"] == is_white)]
            if len(cell_df) == 0:
                print(f"  WARNING: no real respondents match cell gender={gender_code}, white={is_white}")
                continue
            cell_weights = cell_df["wtssall"].to_numpy(dtype=float)
            cell_probs_arr = cell_weights / cell_weights.sum()
            idx = rng.choice(cell_df.index.to_numpy(), size=count, replace=True, p=cell_probs_arr)
            pieces.append(df.loc[idx])

        result = pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)
        return result, value_labels
    else:
        raise ValueError("mode must be 'representative' or 'custom'")

def describe_persona(row, value_labels):
    """Turn a resampled respondent's raw codes into a human-readable
    description for the LLM prompt, using GSS's own value labels."""
    def label_for(var, code):
        labels = value_labels.get(var, {})
        return labels.get(str(int(code)), str(code)) if pd.notna(code) else "unknown"

    return {
        "age": int(row["age"]),
        "gender": label_for("sex", row["sex"]),
        "race": label_for("race", row["race"]),
        "marital_status": label_for("marital", row["marital"]),
        "employment": label_for("wrkstat", row["wrkstat"]),
        "religion": label_for("relig", row["relig"]),
    }

def build_data_dictionary(value_labels):
    """
    Builds a plain-language reference for every column a generated dataset
    or coverage report can contain. Demographic value labels are pulled
    directly from value_labels (the same file used at generation time),
    so this can never drift out of sync with what the codes actually mean.
    """
    rows = []

    def add(column, description, values=""):
        rows.append({"column": column, "description": description, "possible_values": values})

    demo_vars = {
        "age": "Respondent's age in years (numeric, not coded).",
        "sex": "Respondent's sex, as coded by GSS.",
        "race": "Respondent's race, as coded by GSS.",
        "marital": "Respondent's marital status, as coded by GSS.",
        "wrkstat": "Respondent's employment status, as coded by GSS.",
        "relig": "Respondent's religious affiliation, as coded by GSS.",
        "region": "Respondent's U.S. census region, as coded by GSS.",
        "educ": "Respondent's years of education (numeric, not coded).",
    }
    for var, desc in demo_vars.items():
        labels = value_labels.get(var, {})
        if labels:
            value_str = "; ".join(f"{k} = {v}" for k, v in sorted(labels.items(), key=lambda x: x[0]))
        else:
            value_str = "(numeric value, no category labels)"
        add(var, desc, value_str)

    add("item_N_raw_propensity",
        "The LLM's raw 0-1 endorsement estimate for item N, before calibration. "
        "Diagnostic only -- not on the response scale, not meant for analysis.")
    add("item_N_response",
        "The final generated response for item N, on the scale you requested "
        "(e.g. 1-7 for Likert, 0/1 for binary). This is the column to analyze.")
    add("item_N_source",
        "How item N's response column was calibrated.",
        "item_calibrated = matched to a specific real GSS variable (or a blend "
        "of several); population_calibrated = no sufficiently strong specific "
        "match was found, so the response was calibrated to the general shape "
        "of real Likert responses instead, without being tied to particular "
        "survey content.")

    add("item", "The exact item text you supplied.")
    add("method",
        "How the item was matched to real data.",
        "single_best = one strong real GSS variable match; ensemble = a "
        "quality-filtered blend of several real variables; none = no real "
        "anchor found, population-level calibration used instead.")
    add("matched_variable", "The real GSS variable name(s) used to calibrate this item (if any).")
    add("similarity_top1", "Semantic similarity (0-1) between your item and its nearest real GSS match.")
    add("confidence_tier",
        "Overall confidence in this item's real-data grounding.",
        "high = single strong match, empirically clears the error bar ~68% of the "
        "time; moderate = blended match, clears it ~50% of the time; "
        "none = population-level calibration only, no specific real-data anchor.")
    add("expected_error",
        "Historically observed calibration error for matches at this "
        "similarity level, from internal validation studies -- a rough "
        "reference, not a guarantee for this specific item.")
    add("expected_error_n_cases", "How many historical validation cases that expected_error estimate is based on.")
    add("real_n_min", "Smallest real-respondent sample size among the GSS variable(s) used to calibrate this item.")
    add("years_fielded_min", "Fewest years any calibrating GSS variable was fielded (more years = more stable estimate).")

    return pd.DataFrame(rows)

# =========================================================================
# PART 2: Item retrieval + real-data calibration
# =========================================================================

def load_item_resources():
    embeddings = np.load("data/gss_variable_embeddings.npy")
    variables, labels = [], []
    with open("data/gss_variable_index.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variables.append(row["variable"])
            labels.append(row["label"])
    with open("data/gss_master_variable_stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    return embeddings, variables, labels, stats

MAX_DOMINANT_CATEGORY_SHARE_LIKERT = 0.85
MAX_DOMINANT_CATEGORY_SHARE_BINARY = 0.95  # binary items are naturally more
                                             # imbalanced (e.g. "ever divorced")
MIN_CATEGORIES_LIKERT = 3   # fewer than this isn't Likert-style
BINARY_CATEGORIES = 2

def is_usable_anchor(stats_entry, response_type="likert"):
    props = stats_entry.get("value_proportions", {})
    real_props = [v for k, v in props.items() if k != "__other__"]
    n_cats = len(real_props)
    if n_cats == 0:
        return False

    if response_type == "binary":
        if n_cats != BINARY_CATEGORIES:
            return False
        return max(real_props) <= MAX_DOMINANT_CATEGORY_SHARE_BINARY
    else:  # "likert"
        if n_cats < MIN_CATEGORIES_LIKERT:
            return False
        return max(real_props) <= MAX_DOMINANT_CATEGORY_SHARE_LIKERT

TOP_K_SEARCH = 15
HYBRID_SWITCH_HIGH = 0.75    # >= this similarity: single-best match
HYBRID_SWITCH_LOW = 0.50     # this to HYBRID_SWITCH_HIGH: quality-filtered
                               # ensemble; below: no real anchor used
ENSEMBLE_MAX_CANDIDATES = 5
ENSEMBLE_MIN_CANDIDATES = 2

def retrieve_calibration_source(item_text, embeddings, variables, labels, stats,
                                 response_type="likert", top_k=TOP_K_SEARCH):
    """
    Hybrid retrieval, validated in validation/05_validate_hybrid_coverage.py:
      - similarity >= 0.75: single-best match
      - 0.50-0.75: quality-filtered ensemble -- searches a wide pool (top 15),
        drops thin/single-year/wrong-type candidates, blends up to 5 of the
        remainder (needs >= 2 qualifying candidates or falls back to "none")
      - < 0.50: no real anchor; falls through to population-level calibration
    """
    response = client.embeddings.create(model=EMBED_MODEL, input=[item_text])
    q_emb = np.array(response.data[0].embedding, dtype=np.float32)
    q_emb = q_emb / np.linalg.norm(q_emb)
    sims = embeddings @ q_emb
    top_idx = np.argsort(-sims)[:top_k]

    candidates = []
    for idx in top_idx:
        var = variables[idx]
        stats_entry = stats.get(var.lower())
        if stats_entry is not None and "error" not in stats_entry:
            candidates.append((var, labels[idx], float(sims[idx]), stats_entry))
    if not candidates:
        raise ValueError(f"No usable candidates found at all for item: {item_text!r}")

    top1_sim = candidates[0][2]

    def is_quality(c):
        _, _, _, s = c
        return is_usable_anchor(s, response_type) and s["n"] >= MIN_REAL_N and len(s["years_fielded"]) > 1

    if top1_sim >= HYBRID_SWITCH_HIGH:
        # Single-best: find the first candidate that's a usable anchor of
        # the right type (mirrors the old retrieve_best_match behavior)
        for var, label, sim, s in candidates:
            if is_usable_anchor(s, response_type):
                cutpoints, k_real = real_distribution_cutpoints(s["value_proportions"])
                return {
                    "method": "single_best",
                    "matched_variable": var, "matched_label": label,
                    "similarity": sim, "top1_similarity": top1_sim,
                    "cutpoints": cutpoints, "k_real": k_real,
                    "stats_n": s["n"], "stats_years": len(s["years_fielded"]),
                    "response_type": response_type, "anchor_forced_degenerate": False,
                }
        # nothing usable -- fall through to forced-degenerate top-1
        var, label, sim, s = candidates[0]
        cutpoints, k_real = real_distribution_cutpoints(s["value_proportions"])
        return {
            "method": "single_best",
            "matched_variable": var, "matched_label": label,
            "similarity": sim, "top1_similarity": top1_sim,
            "cutpoints": cutpoints, "k_real": k_real,
            "stats_n": s["n"], "stats_years": len(s["years_fielded"]),
            "response_type": response_type, "anchor_forced_degenerate": True,
        }

    elif top1_sim >= HYBRID_SWITCH_LOW:
        quality_candidates = [c for c in candidates if is_quality(c)]
        if len(quality_candidates) >= ENSEMBLE_MIN_CANDIDATES:
            used = quality_candidates[:ENSEMBLE_MAX_CANDIDATES]
            weights = np.clip(np.array([c[2] for c in used]), 0, None)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(used)) / len(used)
            cutpoints, k_real = build_ensemble_cutpoints([c[3] for c in used], weights)
            return {
                "method": "ensemble",
                "matched_variable": "+".join(c[0] for c in used),
                "matched_label": "; ".join(c[1] for c in used),
                "similarity": top1_sim, "top1_similarity": top1_sim,
                "cutpoints": cutpoints, "k_real": k_real,
                "stats_n": min(c[3]["n"] for c in used),
                "stats_years": min(len(c[3]["years_fielded"]) for c in used),
                "response_type": response_type, "anchor_forced_degenerate": False,
                "n_candidates_used": len(used),
            }
        # not enough quality candidates -- no real anchor used
        var, label, sim, s = candidates[0]
        return {
            "method": "none",
            "matched_variable": var, "matched_label": label,
            "similarity": sim, "top1_similarity": top1_sim,
            "cutpoints": None, "k_real": None,
            "stats_n": s["n"], "stats_years": len(s["years_fielded"]),
            "response_type": response_type, "anchor_forced_degenerate": True,
        }
    else:
        var, label, sim, s = candidates[0]
        return {
            "method": "none",
            "matched_variable": var, "matched_label": label,
            "similarity": sim, "top1_similarity": top1_sim,
            "cutpoints": None, "k_real": None,
            "stats_n": s["n"], "stats_years": len(s["years_fielded"]),
            "response_type": response_type, "anchor_forced_degenerate": False,
        }

def build_ensemble_cutpoints(stats_list, weights, n_points=20):
    """Blend multiple real variables' cumulative distributions (weighted
    by similarity) into one composite calibration target."""
    curves = np.array([_build_cdf_curve_for_ensemble(s["value_proportions"], n_points) for s in stats_list])
    composite = np.average(curves, axis=0, weights=weights)
    return composite.tolist(), n_points

def _build_cdf_curve_for_ensemble(value_proportions, n_points=20):
    cats = [(float(k), v) for k, v in value_proportions.items() if k != "__other__"]
    cats.sort(key=lambda x: x[0])
    k = len(cats)
    total = sum(v for _, v in cats)
    if k == 0 or total == 0:
        return np.linspace(0, 1, n_points)
    ranks = np.array([i / max(k - 1, 1) for i in range(k)])
    cum = np.cumsum([v / total for _, v in cats])
    query_points = np.linspace(0, 1, n_points)
    return np.interp(query_points, ranks, cum)

MIN_REAL_N = 200  # min sample size for a real variable to count as a usable anchor
MIN_PROPENSITY_STD = 0.06  # Empirically set (validation/09): the natural
                            # distribution of raw_propensity_std across 60
                            # diverse items had median=0.076, 5th pct=0.060 --
                            # a naive 0.08 guess would have flagged ~half of
                            # all healthy items. 0.06 flags genuine outliers
                            # only. Low std can also reflect a real
                            # ceiling/consensus effect, not just weak LLM
                            # differentiation -- treat as a signal to
                            # inspect, not proof of a problem.
MAX_TIE_CONCENTRATION = 0.80  # share of respondents tied at the exact same
                                # raw propensity beyond which any apparent
                                # spread downstream is mostly tie-breaking
                                # noise (std alone can miss this)

def tie_concentration(raw_propensities):
    """Fraction of responses sharing the single most common propensity value."""
    raw = np.array(raw_propensities)
    _, counts = np.unique(raw, return_counts=True)
    return counts.max() / len(raw)

def real_distribution_cutpoints(value_proportions):
    """Real category ranks (1..K) and their renormalized cumulative
    proportions -- the target SHAPE we calibrate onto."""
    cats = [(k, v) for k, v in value_proportions.items() if k != "__other__"]
    cats.sort(key=lambda x: float(x[0]))
    total = sum(v for _, v in cats)
    k = len(cats)
    if k == 0 or total == 0:
        return [1.0], 1
    cum = []
    running = 0.0
    for _, prop in cats:
        running += prop / total
        cum.append(running)
    return cum, k

def calibrate_batch(raw_propensities, cutpoints, k_real, m_target):
    """
    PRIMARY calibration method. Converts each propensity to its PERCENTILE
    RANK within the actual generated batch (not its raw value), then maps
    that percentile through the calibration target's cumulative shape
    (cutpoints, k_real) -- which may come from a SINGLE real variable
    (single-best) or a similarity-weighted BLEND of several (ensemble);
    both are passed in already-computed by retrieve_calibration_source, so
    this function doesn't need to know which case it's handling.

    Why percentile rather than raw value: raw LLM propensities may cluster
    in a narrow range (e.g. 0.4-0.75) that never crosses a real cumulative
    threshold sitting at, say, 0.80 -- causing everyone to collapse into
    one category even though real relative differences exist between them.
    Percentile ranking always spans the full 0-1 range as long as ANY
    difference exists between people (ties are broken so ranks stay
    distinct), so the calibration target's marginal SHAPE is reproduced
    almost exactly regardless of how compressed the raw signal was.
    """
    raw = np.array(raw_propensities)
    n = len(raw)
    # rank(method="first") guarantees distinct ranks even for exact ties,
    # so percentile always spans (0, 1] regardless of input clustering
    ranks = pd.Series(raw).rank(method="first").to_numpy()
    percentiles = ranks / n

    real_ranks = np.searchsorted(cutpoints, percentiles, side="left") + 1
    real_ranks = np.clip(real_ranks, 1, k_real)

    rescaled = [rescale_rank(r, k_real, m_target) for r in real_ranks]
    return rescaled

def rescale_rank(rank, k_real, m_target):
    """Linearly rescale a 1..k_real rank onto a 1..m_target scale."""
    if k_real == 1:
        return int(round((m_target + 1) / 2))
    scaled = 1 + (rank - 1) * (m_target - 1) / (k_real - 1)
    return int(round(scaled))

def load_generic_shape():
    """Generic real Likert response shape, precomputed by
    pipeline/06_build_generic_shape.py from thousands of real GSS items.
    Falls back to a flat/uniform curve if the file isn't present."""
    try:
        return np.load("data/generic_likert_shape.npy")
    except FileNotFoundError:
        print("  NOTE: generic_likert_shape.npy not found -- falling back to "
              "flat/uniform shape for population-level calibration. Run "
              "pipeline/06_build_generic_shape.py to enable the validated "
              "improvement (~30% lower error than uniform).")
        return np.linspace(0, 1, 20)

GENERIC_SHAPE_CURVE = load_generic_shape()

def population_calibrated_batch(raw_propensities, m_target):
    """
    Used when no content-specific real match is trusted enough to
    calibrate against ("population_calibrated"). Rather than assuming a
    flat/uniform response shape (equal-width binning), this maps
    percentile rank through a GENERIC real Likert response shape (averaged
    across thousands of real GSS items, content-agnostic). Validated to
    reduce error by ~30% relative to the flat assumption, though this is
    still NOT a content-specific match, so output stays labeled
    "population_calibrated", not "item_calibrated".
    """
    raw = np.array(raw_propensities)
    n = len(raw)
    ranks = pd.Series(raw).rank(method="first").to_numpy()
    percentiles = ranks / n

    k_generic = len(GENERIC_SHAPE_CURVE)
    real_ranks = np.searchsorted(GENERIC_SHAPE_CURVE, percentiles, side="left") + 1
    real_ranks = np.clip(real_ranks, 1, k_generic)

    return [rescale_rank(r, k_generic, m_target) for r in real_ranks]

def to_binary_output(scaled_values):
    """Convert a 1..2 scaled output to 0/1, for binary-type items."""
    return [v - 1 for v in scaled_values]

def is_degenerate_column(values, min_distinct=2):
    return len(set(values)) < min_distinct

def determine_confidence_tier(info):
    """
    Three-tier confidence, based on validation showing a hard error
    threshold doesn't cleanly separate good/bad matches within the
    ensemble method (every 0.5-0.75 similarity bin clears 0.12 error only
    ~48-56% of the time, fairly uniformly -- no sub-band worth a cutoff).
    Single-best clears it ~68-70% of the time -- a genuinely different tier.

    "high"     -- single_best method, passes N/years/anchor checks
    "moderate" -- ensemble method (candidates already guaranteed
                  N>=200/years>1 by construction) -- reported with its own
                  historical error rate rather than a pass/fail label
    "none"     -- no usable method found, or single_best failed its checks
    """
    n = info["stats_n"]
    years = info["stats_years"]
    method = info["method"]

    if method == "none":
        print(f"  ** NO REAL ANCHOR for '{info['item_text'][:50]}...': "
              f"similarity too low or insufficient quality candidates for ensemble. "
              f"Using population_calibrated estimate. **")
        return "none"

    if method == "single_best":
        issues = []
        if info.get("anchor_forced_degenerate"):
            issues.append("no non-degenerate real anchor found in top candidates")
        if n < MIN_REAL_N:
            issues.append(f"thin real data (n={n} < {MIN_REAL_N})")
        if years <= 1:
            issues.append(f"fielded in only {years} year(s)")
        if issues:
            print(f"  ** LOW CONFIDENCE for '{info['item_text'][:50]}...' -> "
                  f"{info['matched_variable']}: {'; '.join(issues)}. "
                  f"Using population_calibrated estimate instead. **")
            return "none"
        return "high"

    if method == "ensemble":
        print(f"  ** MODERATE CONFIDENCE for '{info['item_text'][:50]}...' -> "
              f"blended match ({info.get('n_candidates_used', '?')} candidates). "
              f"Ensemble matches empirically clear our error bar only ~50% of the "
              f"time (vs ~68% for high-confidence single matches) -- reported "
              f"with its own historical error rate rather than a pass/fail label. **")
        return "moderate"

    return "none"

# =========================================================================
# PART 3: Joint LLM generation per persona batch
# =========================================================================

def build_prompt(personas_batch, items_info):
    item_lines = "\n".join(f"{i+1}. {info['item_text']}" for i, info in enumerate(items_info))
    persona_lines = "\n".join(
        f"Person {i+1}: age {p['age']}, {p['gender']}, {p['race']}, "
        f"{p['marital_status']}, employment: {p['employment']}, religion: {p['religion']}"
        for i, p in enumerate(personas_batch)
    )
    n_items = len(items_info)
    return f"""You are simulating how different U.S. adults would respond to a survey.

Survey items:
{item_lines}

For each person below, output a "propensity" score from 0.0 to 1.0 for EACH item,
reflecting how strongly that specific person would agree/endorse that item given
their demographic profile (0.0 = lowest possible endorsement, 1.0 = highest).
Vary your answers realistically across people -- do not give everyone similar scores.

People:
{persona_lines}

Respond with ONLY a JSON array, one object per person in order, each with keys
"item_1" through "item_{n_items}", each a float between 0 and 1. No other text."""

def call_llm_batch(personas_batch, items_info):
    prompt = build_prompt(personas_batch, items_info)
    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  ** WARNING: LLM response was not valid JSON ({e}). "
              f"First 200 chars of response: {text[:200]!r} "
              f"-- treating as a failed batch (will be retried if a retry "
              f"loop is calling this function). **")
        return []  # empty list deliberately fails any length check downstream,
                    # which triggers the SAME retry path already used for
                    # wrong-count responses -- no separate handling needed
    if not isinstance(parsed, list):
        print(f"  ** WARNING: LLM response was valid JSON but not a list "
              f"(got {type(parsed).__name__}). Treating as a failed batch. **")
        return []
    return parsed

# =========================================================================
# PART 4: Full generation orchestration
# =========================================================================

MAX_ITEMS_SOFT_CAP = 50
MAX_N_SOFT_CAP = 1000
VALIDATED_BATCH_SIZE = 8  # Empirically tested (validation/07, 08): an initial
                            # test (5-30 items) found batch_size=10 had a 0%
                            # mismatch rate; a follow-up at higher item counts
                            # (40-50, near the soft cap) found 10 degrades to a
                            # 33% mismatch rate there. batch_size=8 was the only
                            # value with a 0% mismatch rate across every item
                            # count tested (5-50). An earlier adaptive scheme
                            # ("keep personas*items around 375") was tested and
                            # found unsupported -- fixed batch_size=8 is the
                            # simplest choice actually backed by the evidence.

def generate_dataset(items, n, seed, mode="representative",
                      filters=None, age_min=None, age_max=None,
                      female_pct=None, white_pct=None, batch_size=None,
                      likert_scale=5, confirm_large_request=False,
                      progress_callback=None):
    """
    items: list of items. Each item is either a plain string (defaults to
        "likert" type) or a dict {"text": ..., "type": "likert"|"binary"}.
        Binary items are matched only to real binary (0/1) GSS variables
        and output as 0/1; Likert items require 3+ real categories and
        are output on the requested likert_scale.
    n: sample size
    seed: random seed for reproducibility
    mode: "representative" (no choices) or "custom" (filters + targets)
    filters: list of names from FILTER_DEFINITIONS, e.g. ["employed", "married"]
    age_min, age_max: age range (custom mode only)
    female_pct, white_pct: target proportions 0-1 (custom mode only, optional)
    likert_scale: 5 or 7 -- output scale for Likert-type items (ignored for binary items)
    batch_size: personas per LLM call. Leave as None to use the empirically
        validated default (8). Overriding this is not recommended -- see
        VALIDATED_BATCH_SIZE comment above.
    confirm_large_request: must be True to proceed if items > 50 or n > 1000
        (soft caps -- not hard limits, just a confirmation step against
        accidentally large/expensive/slow requests).
    progress_callback: optional function called after each batch as
        progress_callback(completed, total) -- e.g. to drive a UI progress
        bar. completed/total are respondent counts, not batch counts.
    """
    # Normalize items to a consistent dict form
    normalized_items = []
    for item in items:
        if isinstance(item, str):
            normalized_items.append({"text": item, "type": "likert"})
        else:
            normalized_items.append({"text": item["text"], "type": item.get("type", "likert")})

    n_items = len(normalized_items)
    if (n_items > MAX_ITEMS_SOFT_CAP or n > MAX_N_SOFT_CAP) and not confirm_large_request:
        raise ValueError(
            f"Request exceeds soft limits (items={n_items} vs cap {MAX_ITEMS_SOFT_CAP}, "
            f"n={n} vs cap {MAX_N_SOFT_CAP}). This isn't a hard limit -- it's a check "
            f"against accidentally large/expensive/slow requests. If this is "
            f"intentional, pass confirm_large_request=True to proceed."
        )

    if batch_size is None:
        batch_size = VALIDATED_BATCH_SIZE
        print(f"Using validated batch_size={batch_size} "
              f"(empirically tested as reliable; see VALIDATED_BATCH_SIZE comment)")

    print("=== Step 1: Demographic resampling ===")
    demo_df, value_labels = build_sample_pool(
        mode=mode, n=n, seed=seed, filters=filters,
        age_min=age_min, age_max=age_max,
        female_pct=female_pct, white_pct=white_pct,
    )

    print("\n=== Step 2: Item retrieval ===")
    embeddings, variables, labels, stats = load_item_resources()
    similarity_table = build_error_lookup_table()
    items_info = []
    for item in normalized_items:
        item_text, response_type = item["text"], item["type"]
        match = retrieve_calibration_source(item_text, embeddings, variables, labels, stats,
                                             response_type=response_type)
        items_info.append({"item_text": item_text, **match})
        print(f"  '{item_text[:50]}...' [{response_type}] -> method={match['method']} "
              f"({match['matched_variable']}) (top1_sim={match['top1_similarity']:.3f}, "
              f"real_n(min)={match['stats_n']})")

        avg_err, err_std, n_cases = lookup_expected_error(match["top1_similarity"], similarity_table)
        items_info[-1]["expected_error"] = avg_err
        items_info[-1]["expected_error_std"] = err_std
        items_info[-1]["expected_error_n_cases"] = n_cases
        if avg_err is not None:
            print(f"     Historical reference (hybrid-method validation): avg calibration error "
                  f"{avg_err:.3f} (+/- {err_std:.3f} std), based on {n_cases} comparable cases.")

        tier = determine_confidence_tier(items_info[-1])
        items_info[-1]["confidence_tier"] = tier
        items_info[-1]["confident_match"] = tier != "none"  # kept for simple filtering

    print(f"\n=== Step 3: Generating responses in batches of {batch_size} ===")
    personas = [describe_persona(row, value_labels) for _, row in demo_df.iterrows()]
    all_propensities = []
    MAX_BATCH_RETRIES = 2
    for start in range(0, n, batch_size):
        batch = personas[start:start + batch_size]
        result = None
        candidate = None
        for attempt in range(MAX_BATCH_RETRIES + 1):
            candidate = call_llm_batch(batch, items_info)
            if len(candidate) == len(batch):
                result = candidate
                break
            if attempt < MAX_BATCH_RETRIES:
                print(f"  ** WARNING: batch at position {start} returned {len(candidate)} "
                      f"results, expected {len(batch)}. Retrying "
                      f"(attempt {attempt+2}/{MAX_BATCH_RETRIES+1})... **")
            else:
                print(f"  ** WARNING: batch at position {start} returned {len(candidate)} "
                      f"results, expected {len(batch)} on final attempt "
                      f"({MAX_BATCH_RETRIES+1}/{MAX_BATCH_RETRIES+1}). No retries left. **")
        if result is None:
            # Retries exhausted -- last-resort fallback so the run doesn't
            # crash, but this should be rare and is loudly flagged
            if len(candidate) == 0:
                # All retries failed outright (e.g. malformed JSON every
                # time) -- nothing to pad from. Use a neutral 0.5 propensity
                # for the whole batch rather than crashing; this batch's
                # data should be treated as unreliable.
                print(f"  ** ALL RETRIES FAILED for batch at position {start} "
                      f"(no valid response at all). Filling with neutral (0.5) "
                      f"placeholder propensities for this batch -- treat this "
                      f"batch's results with caution or rerun. **")
                neutral = {f"item_{j+1}": 0.5 for j in range(len(items_info))}
                result = [dict(neutral) for _ in range(len(batch))]
            elif len(candidate) > len(batch):
                print(f"  ** Still mismatched after retries -- truncating "
                      f"{len(candidate)} down to {len(batch)}. **")
                result = candidate[:len(batch)]
            else:
                print(f"  ** Still mismatched after retries -- padding {len(candidate)} "
                      f"up to {len(batch)} by repeating the last entry. This is a "
                      f"last-resort fallback; consider rerunning this generation. **")
                result = candidate + [candidate[-1]] * (len(batch) - len(candidate))
        all_propensities.extend(result)
        completed = min(start + batch_size, n)
        print(f"  generated {completed}/{n}")
        if progress_callback is not None:
            progress_callback(completed, n)

    print(f"\n=== Step 4: Calibrating (percentile-based) ===")
    final_df = demo_df[["age", "sex", "race", "marital", "wrkstat", "relig", "region", "educ"]].copy()
    for i, info in enumerate(items_info):
        col_key = f"item_{i+1}"
        raw = np.array([p[col_key] for p in all_propensities])
        final_df[f"item_{i+1}_raw_propensity"] = raw
        raw_std = raw.std()
        response_type = info["response_type"]
        m_target = 2 if response_type == "binary" else likert_scale

        # Population-level fallback: always produced, percentile-based (see calibrate_batch docstring)
        population_calibrated = population_calibrated_batch(raw, m_target)

        # Item-level calibrated version: ONLY produced for confident matches
        if info["confident_match"]:
            calibrated = calibrate_batch(raw, info["cutpoints"], info["k_real"], m_target)
        else:
            calibrated = None

        if response_type == "binary":
            population_calibrated = to_binary_output(population_calibrated)
            if calibrated is not None:
                calibrated = to_binary_output(calibrated)

        # Backstop kept as defensive programming; should now rarely/never
        # trigger given percentile-based calibration above
        if is_degenerate_column(population_calibrated):
            print(f"  ** BACKSTOP TRIGGERED (unexpected): item_{i+1} population_calibrated "
                  f"column was still constant. This item's wording likely needs revision. **")
        if calibrated is not None and is_degenerate_column(calibrated):
            print(f"  ** BACKSTOP TRIGGERED (unexpected): item_{i+1} item_calibrated column "
                  f"was still constant -- dropping it; using population_calibrated instead. **")
            calibrated = None

        final_df[f"item_{i+1}_response"] = calibrated if calibrated is not None else population_calibrated
        final_df[f"item_{i+1}_source"] = "item_calibrated" if calibrated is not None else "population_calibrated"

        print(f"  item_{i+1}: confidence_tier={info['confidence_tier']} "
              f"calibrated_produced={calibrated is not None} raw propensity std={raw_std:.3f}")
        if raw_std < MIN_PROPENSITY_STD:
            print(f"  ** NOTE: item_{i+1} ('{info['item_text'][:50]}...') shows low raw "
                  f"variance across personas (std={raw_std:.3f} < {MIN_PROPENSITY_STD}). "
                  f"Percentile-based calibration above still preserves the real distribution's "
                  f"marginal shape, but the LLM may not be differentiating personas meaningfully "
                  f"for this item -- relative ordering between people may be weak/arbitrary. "
                  f"Consider rephrasing this item if this persists. **")

        tie_share = tie_concentration(raw)
        if tie_share > MAX_TIE_CONCENTRATION:
            print(f"  ** NOTE: item_{i+1} ('{info['item_text'][:50]}...') has {tie_share*100:.0f}% "
                  f"of respondents tied at the EXACT SAME raw propensity value. Any spread you see "
                  f"in the final response column for this item is mostly arbitrary tie-breaking "
                  f"among those tied respondents, not genuine differentiation. This can happen when "
                  f"the item is near-tautological given the sample's own filters (e.g. asking "
                  f"about employment status in a sample already filtered to employed respondents). "
                  f"Std alone did not catch this (std={raw_std:.3f}). **")

        info["raw_propensity_std"] = float(raw_std)
        info["low_variance_flag"] = bool(raw_std < MIN_PROPENSITY_STD)
        info["tie_concentration"] = float(tie_share)
        info["high_tie_concentration_flag"] = bool(tie_share > MAX_TIE_CONCENTRATION)
        info["calibrated_produced"] = calibrated is not None

    coverage_report = pd.DataFrame([
        {
            "item": info["item_text"],
            "method": info["method"],
            "matched_variable": info["matched_variable"],
            "matched_label": info["matched_label"],
            "similarity_top1": info["top1_similarity"],
            "real_n_min": info["stats_n"],
            "years_fielded_min": info["stats_years"],
            "n_candidates_used": info.get("n_candidates_used", 1),
            "confidence_tier": info["confidence_tier"],
            "confident_match": info["confident_match"],
            "calibrated_produced": info.get("calibrated_produced"),
            "raw_propensity_std": info.get("raw_propensity_std"),
            "low_variance_flag": info.get("low_variance_flag"),
            "tie_concentration": info.get("tie_concentration"),
            "high_tie_concentration_flag": info.get("high_tie_concentration_flag"),
            "anchor_forced_degenerate": info.get("anchor_forced_degenerate"),
            "expected_error": info.get("expected_error"),
            "expected_error_std": info.get("expected_error_std"),
            "expected_error_n_cases": info.get("expected_error_n_cases"),
        }
        for info in items_info
    ])

    return final_df, coverage_report

if __name__ == "__main__":
    items = [
        {"text": "I feel my contributions are valued at this organization", "type": "likert"},
        {"text": "My supervisor gives me clear feedback on my performance", "type": "likert"},
        {"text": "I often feel emotionally exhausted by my work", "type": "likert"},
        {"text": "I volunteer regularly in my community", "type": "binary"},  # not implied by the demographic filters below
        {"text": "I am satisfied with my job", "type": "likert"},  # should closely match SATJOB -- expect confidence_tier="high"
    ]

    df, coverage = generate_dataset(
        items=items, n=50, seed=42, mode="custom",
        filters=["employed", "married"],
        age_min=25, age_max=45,
        female_pct=0.7, white_pct=0.5,
        likert_scale=7,
    )
    df.to_csv("synthetic_dataset_example.csv", index=False)
    coverage.to_csv("synthetic_dataset_coverage_report.csv", index=False)
    print("\nSaved synthetic_dataset_example.csv and synthetic_dataset_coverage_report.csv")
    print(df.head())
