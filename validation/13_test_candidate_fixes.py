"""
Candidate fixes for the failures found by script 12 (exploratory, in-sample; not part of the tool).

Script 12 shows that, for organizational items, the tool's answer distributions are far from the real ones
(means about 1.7 points too low, shapes close to a mirror image). The cause is that the tool maps each
respondent's RANK in the model's agreement ratings onto a fixed anchor shape, so it throws away the model's own
item-level information. This script asks, before anyone changes the tool: is that information really there, and
would simple fixes help?

Marginal distributions. Candidate fixes tested against the real distributions (mean absolute difference between
cumulative distributions):
    * the model's mean rating per item against the REAL item mean (is the information there?)
    * "naive binning": answer = floor(rating x 7) + 1, with no anchor at all
    * "flip": reverse the tool's output when the item's mean rating exceeds 0.5 (the model thinks people agree)

Correlations. Generated correlations run too high. Candidate fix: partly remove each synthetic respondent's
general tendency to agree with everything (strength lambda). It lowers the average bias but also lowers how well
the pattern of correlations is recovered. Compared with dividing correlations by 1.1.

CAVEAT: these 62 items are the ones on which the failure was first seen, and the rules were chosen afterwards, so
treat the numbers as a first look, not a validation. The other 91 items (18 scales) are the hold-out for the planned
follow-up work.

Inputs (in validation/reference_data/): baseline_run_62items.csv (synthetic respondents with the model's raw
ratings), baseline_run_62items_metadata.json (item order, scale, reverse flag), and the real-survey aggregates
used by script 12. No model calls. Run from the repository root:  python validation/13_test_candidate_fixes.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

import importlib.util  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "reference_data")
RESULTS = os.path.join(HERE, "results")

# reuse the real-data loaders of script 12 (a file name starting with a digit cannot be imported normally)
_spec = importlib.util.spec_from_file_location("real_data_test", os.path.join(HERE, "12_validate_against_real_data.py"))
_rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rd)
load_reference, load_real_corr = _rd.load_reference, _rd.load_real_corr


def cdist(a, b):
    return float(np.mean(np.abs(np.cumsum(a) - np.cumsum(b))))


def shares7(x):
    return np.array([(x == c).mean() for c in range(1, 8)])


def main():
    run = pd.read_csv(os.path.join(REFERENCE, "baseline_run_62items.csv"))
    meta = json.load(open(os.path.join(REFERENCE, "baseline_run_62items_metadata.json")))
    _, ref, _, mirror, _ = load_reference()
    item_corr, _ = load_real_corr()
    cats = np.arange(1, 8)

    rows = []
    for i, m in enumerate(meta, 1):
        name = m["prolific_column"].lower()
        real = ref[name]["shares"]
        raw = run[f"item_{i}_raw_propensity"].to_numpy()
        resp = run[f"item_{i}_response"].to_numpy()
        src = run[f"item_{i}_source"].iloc[0]
        prod = shares7(resp)
        naive = shares7(np.clip((raw * 7).astype(int) + 1, 1, 7))
        flip = prod[::-1] if raw.mean() > 0.5 else prod
        rows.append({"item": name, "source": src, "real_mean": float(real @ cats), "rating_mean": raw.mean(),
                     "tool_mean": float(prod @ cats), "naive_mean": float(naive @ cats),
                     "d_tool": cdist(prod, real), "d_naive": cdist(naive, real), "d_flip": cdist(flip, real),
                     "polarity_ok": (raw.mean() > 0.5) == (float(real @ cats) > 4)})
    r = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    r.to_csv(os.path.join(RESULTS, "candidate_fixes_results.csv"), index=False)

    print(f"{len(r)} items; reverse-worded items in the reference file are "
          f"{'mirrored back' if mirror else 'used as stored'}\n")
    print("Q1. Marginal distributions: does the model's own agreement rating carry item-level information?")
    print(f"  correlation with the REAL item mean:  rating {r.rating_mean.corr(r.real_mean):+.2f}"
          f"   |   tool output {r.tool_mean.corr(r.real_mean):+.2f}   |   naive binning {r.naive_mean.corr(r.real_mean):+.2f}")
    print(f"  'rating above 0.5' predicts that real answers lean toward agreement (mean above 4): "
          f"right for {r.polarity_ok.mean() * 100:.0f}% of items")
    print("\n  distance to the real distribution (0 = identical):")
    for lab, d in (("all items", r), ("no anchor", r[r.source == "population_calibrated"]),
                   ("anchor found", r[r.source != "population_calibrated"])):
        print(f"    {lab:<13} n={len(d):<3} tool {d.d_tool.mean():.3f} | naive binning {d.d_naive.mean():.3f}"
              f" | tool flipped {d.d_flip.mean():.3f}")
    print(f"\n  mean answer: real {r.real_mean.mean():.2f} | tool {r.tool_mean.mean():.2f} | naive {r.naive_mean.mean():.2f}")

    # ---- Q2: correlations
    names = [m["prolific_column"].lower() for m in meta]
    scale = {m["prolific_column"].lower(): m["scale"] for m in meta}
    X = pd.DataFrame({n: (8 - run[f"item_{i}_response"] if m["reverse"] else run[f"item_{i}_response"]).astype(float)
                      for i, (n, m) in enumerate(zip(names, meta), 1)})
    Z = (X - X.mean()) / X.std()
    pairs = [(a, b) for k, a in enumerate(names) for b in names[k + 1:]
             if a in item_corr.index and b in item_corr.columns and not np.isnan(item_corr.loc[a, b])]
    real_r = np.array([item_corr.loc[a, b] for a, b in pairs])
    same = np.array([scale[a] == scale[b] for a, b in pairs])

    def evaluate(M, label, divide=1.0):
        C = M.corr()
        g = np.array([C.loc[a, b] for a, b in pairs]) / divide
        d = g - real_r
        out = []
        for sel in (~same, same):
            out.append((d[sel].mean(), np.abs(d[sel]).mean(), np.corrcoef(real_r[sel], g[sel])[0, 1]))
        (bd, md, rd), (bs, ms, rs) = out
        print(f"  {label:<34} different scales: bias {bd:+.3f}  MAE {md:.3f}  pattern r {rd:.2f}"
              f" | same scale: bias {bs:+.3f}  MAE {ms:.3f}  pattern r {rs:.2f}")

    print(f"\nQ2. Correlations: candidate fixes ({int((~same).sum())} different-scale and {int(same.sum())} same-scale pairs)")
    evaluate(Z, "as produced")
    evaluate(Z, "divide by 1.1 (the app note)", divide=1.1)
    person = Z.mean(axis=1)  # each synthetic respondent's overall tendency to agree with everything
    beta = Z.apply(lambda c: np.cov(c, person)[0, 1] / person.var())
    for lam in (0.25, 0.5, 0.75, 1.0):
        evaluate(Z - lam * np.outer(person, beta.values), f"remove general tendency, strength {lam}")
    print("\nSaved: validation/results/candidate_fixes_results.csv")


if __name__ == "__main__":
    main()
