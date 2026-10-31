"""
Does synthetic data overstate scale reliability? Cronbach's alpha, generated against real, scale by scale.

Why it matters. Generated items of the same scale correlate more strongly than real ones (script 12: mean
same-scale r .55 against a real .39). Because alpha rises with the average inter-item correlation, a researcher
who checks reliability on synthetic data would see scales as more reliable than they are. This script measures
by how much.

What it does.
  A. Ground-truth test: for every scale in every saved batch of script 12 (validation/results/real_data_test/
     resp_p*_b*.csv), reverse-scores reverse-worded items, computes Cronbach's alpha on the 398 generated
     respondents, averages over the three groupings, and compares with the scale's real alpha.
  B. Pilot run: the same on the 62-item pilot run (validation/reference_data/baseline_run_62items.csv, 16 scales).

Real alphas come from validation/reference_data/prolific_alpha_benchmark.csv (computed on the real sample's
complete cases; the three leadership scales were answered only by the 147 respondents who supervise others).

No model calls. Run from the repository root:  python validation/13_compare_alpha.py
Writes validation/results/alpha_comparison.csv.
"""

import importlib.util
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "reference_data")
RESULTS = os.path.join(HERE, "results")

_spec = importlib.util.spec_from_file_location("real_data_test", os.path.join(HERE, "12_validate_against_real_data.py"))
_rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rd)


def cronbach_alpha(X):
    X = np.asarray(X, float)
    X = X[~np.isnan(X).any(axis=1)]
    k = X.shape[1]
    if k < 2 or len(X) < 3:
        return np.nan
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return np.nan
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / total_var)


def summarize(df, label):
    d = df.dropna(subset=["generated_alpha", "real_alpha"])
    diff = d.generated_alpha - d.real_alpha
    print(f"\n{label}: {len(d)} scales")
    print(f"  mean alpha, generated {d.generated_alpha.mean():.2f} against real {d.real_alpha.mean():.2f} "
          f"(median {d.generated_alpha.median():.2f} against {d.real_alpha.median():.2f})")
    print(f"  generated higher in {(diff > 0).sum()} of {len(d)} scales; mean difference {diff.mean():+.3f}; "
          f"mean absolute difference {diff.abs().mean():.3f}")
    print(f"  generated alpha .70 or above: {(d.generated_alpha >= .70).sum()} scales; real: {(d.real_alpha >= .70).sum()}")


def main():
    real = pd.read_csv(os.path.join(REFERENCE, "prolific_alpha_benchmark.csv")).set_index("scale")["alpha"]
    pts = _rd.OUT_POINTS
    rows = []

    # A. ground-truth test (saved batches of script 12)
    meta = _rd.load_meta()
    info = {m["prolific_column"].lower(): (m["scale"], bool(m["reverse"])) for m in meta}
    resp, _ = _rd.load_outputs()
    for (p, b), r in sorted(resp.items()):
        r = r.copy()
        r.columns = [c.lower() for c in r.columns]
        for c in r.columns:
            if c in info and info[c][1]:
                r[c] = (pts + 1) - r[c]
        by_scale = {}
        for c in r.columns:
            if c in info:
                by_scale.setdefault(info[c][0], []).append(c)
        for s, cols in by_scale.items():
            rows.append({"test": "ground_truth", "partition": p, "batch": b, "scale": s, "k": len(cols),
                         "generated_alpha": cronbach_alpha(r[cols])})
    gt = pd.DataFrame(rows)
    if len(gt):
        gt_scale = gt.groupby("scale").agg(k=("k", "first"), generated_alpha=("generated_alpha", "mean"),
                                           groupings=("partition", "nunique")).reset_index()
        gt_scale["real_alpha"] = gt_scale.scale.map(real)
        gt_scale.insert(0, "test", "ground_truth")
        summarize(gt_scale, "A. Ground-truth test (153 items, mean over groupings)")
    else:
        gt_scale = pd.DataFrame()
        print("\nA. No saved batches found in validation/results/real_data_test/; run script 12 first.")

    # B. pilot run (62 items, 16 scales)
    run = pd.read_csv(os.path.join(REFERENCE, "baseline_run_62items.csv"))
    pmeta = json.load(open(os.path.join(REFERENCE, "baseline_run_62items_metadata.json")))
    groups = {}
    for i, m in enumerate(pmeta, start=1):
        x = run[f"item_{i}_response"].astype(float)
        groups.setdefault(m["scale"], []).append((pts + 1) - x if m["reverse"] else x)
    pilot = pd.DataFrame([{"test": "pilot_62", "scale": s, "k": len(xs),
                           "generated_alpha": cronbach_alpha(np.column_stack(xs)), "real_alpha": real.get(s, np.nan)}
                          for s, xs in groups.items()])
    summarize(pilot, "B. Pilot run (62 items)")

    out = pd.concat([gt_scale, pilot], ignore_index=True)
    out["difference"] = out.generated_alpha - out.real_alpha
    os.makedirs(RESULTS, exist_ok=True)
    out.round(3).to_csv(os.path.join(RESULTS, "alpha_comparison.csv"), index=False)
    print("\nSaved: validation/results/alpha_comparison.csv")


if __name__ == "__main__":
    main()
