"""
Validate synthetic data from examples/02_generate_bunderson_generic.py
against Bunderson & Thompson (2009)'s REAL Table 1 correlations (N=491),
transcribed and verified from a clean table image.

IMPORTANT CAVEAT (state this explicitly wherever these results are used):
this compares GENERIC item wording on a GENERAL employed population
against real correlations from ZOO-SPECIFIC wording on ACTUAL ZOOKEEPERS.
Two things differ from the original study simultaneously (item wording
AND population), not just one. This tests whether the psychological
RELATIONSHIP between constructs is portable under realistic tool usage
conditions -- it is not a same-wording, same-population replication.

No reverse-scoring needed -- none of the four scales used here contain
reverse-worded items.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR = os.path.join(_REPO_ROOT, "data")

df = pd.read_csv(os.path.join(_DATA_DIR, "bunderson_generic_dataset.csv"))
with open(os.path.join(_DATA_DIR, "bunderson_generic_item_metadata.json")) as f:
    items = json.load(f)

# --- Build the 4 composite scale scores ---
scale_cols = {}
for i, item in enumerate(items):
    col = f"item_{i+1}_response"
    scale_cols.setdefault(item["scale"], []).append(col)

SCALES = ["calling", "occ_identification", "moral_duty", "work_meaningfulness"]
LABELS = ["Calling", "Occ. Identification", "Moral Duty", "Work Meaningfulness"]

scale_scores = {s: df[scale_cols[s]].mean(axis=1) for s in SCALES}
scale_df = pd.DataFrame(scale_scores)

# --- Real correlations, verified from a clean image of Table 1 (N=491) ---
REAL_CORR = np.array([
    [1.00, .36, .47, .37],
    [.36, 1.00, .42, .44],
    [.47, .42, 1.00, .33],
    [.37, .44, .33, 1.00],
])
REAL_ALPHA = {"calling": .91, "occ_identification": .81, "moral_duty": .88, "work_meaningfulness": .88}

gen_corr = scale_df[SCALES].corr().to_numpy()

print("=== Calibrated vs. real correlations ===\n")
rows = []
for i in range(4):
    for j in range(i + 1, 4):
        real_r = REAL_CORR[i, j]
        gen_r = gen_corr[i, j]
        rows.append({
            "var_1": LABELS[i], "var_2": LABELS[j],
            "real_r": real_r, "calibrated_r": round(gen_r, 3),
            "calibrated_abs_diff": round(abs(real_r - gen_r), 3),
        })

corr_comparison = pd.DataFrame(rows)
print(corr_comparison.to_string(index=False))
print(f"\nMean abs difference (calibrated): {corr_comparison['calibrated_abs_diff'].mean():.3f}")

# --- Naive LLM baseline, reusing existing raw propensities (no new API calls) ---
print("\n=== Adding naive LLM baseline (same raw propensities, no calibration) ===\n")

def naive_bin(raw_propensities, k_target=7):
    raw = np.array(raw_propensities)
    return np.clip((raw * k_target).astype(int) + 1, 1, k_target)

naive_scale_scores = {}
for s in SCALES:
    item_indices = [i for i, item in enumerate(items) if item["scale"] == s]
    naive_vals = [naive_bin(df[f"item_{i+1}_raw_propensity"]) for i in item_indices]
    naive_scale_scores[s] = np.mean(naive_vals, axis=0)
naive_df = pd.DataFrame(naive_scale_scores)
naive_corr = naive_df[SCALES].corr().to_numpy()

three_way_rows = []
for i in range(4):
    for j in range(i + 1, 4):
        real_r = REAL_CORR[i, j]
        calib_r = gen_corr[i, j]
        naive_r = naive_corr[i, j]
        three_way_rows.append({
            "var_1": LABELS[i], "var_2": LABELS[j], "real_r": real_r,
            "calibrated_r": round(calib_r, 3), "naive_r": round(naive_r, 3),
            "calibrated_abs_diff": round(abs(real_r - calib_r), 3),
            "naive_abs_diff": round(abs(real_r - naive_r), 3),
        })
three_way = pd.DataFrame(three_way_rows)
three_way.to_csv(os.path.join(_DATA_DIR, "bunderson_generic_three_way_comparison.csv"), index=False)
print(three_way.to_string(index=False))

print(f"\nMean abs error, calibrated: {three_way['calibrated_abs_diff'].mean():.3f}")
print(f"Mean abs error, naive LLM:  {three_way['naive_abs_diff'].mean():.3f}")
improvement = (three_way['naive_abs_diff'].mean() - three_way['calibrated_abs_diff'].mean()) / three_way['naive_abs_diff'].mean() * 100
print(f"Relative improvement from calibration: {improvement:.1f}%")
n_calib_better = (three_way["calibrated_abs_diff"] < three_way["naive_abs_diff"]).sum()
print(f"Pairs where calibrated beats naive: {n_calib_better} / {len(three_way)}")

# --- Reliability check ---
print("\n=== Reliability check ===")
def cronbach_alpha(items_df):
    k = items_df.shape[1]
    item_vars = items_df.var(axis=0, ddof=1).sum()
    total_var = items_df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars / total_var)

for s in SCALES:
    alpha = cronbach_alpha(df[scale_cols[s]])
    print(f"{s}: generated={alpha:.3f}, real={REAL_ALPHA[s]}")
