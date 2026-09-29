"""
Validate synthetic JD-R burnout data (from examples/01_generate_jdr_burnout.py)
against Demerouti, Bakker, Nachreiner, & Schaufeli (2001)'s REAL reported
results:
  - Table 3: individual-level correlations among the 11 demand/resource
    facets, exhaustion, and disengagement (N=374, self-report)
  - Table 5 (Structural model): standardized paths from composite Job
    Demands / Job Resources to Exhaustion / Disengagement

Reverse-scoring: items marked "reverse": true in the item metadata are
positively-worded relative to the construct's scored direction and are
reverse-scored here (5 - x, since this is a 1-4 scale) before ANY mean,
correlation, or regression is computed. This mirrors standard scale-
scoring practice and only affects analysis -- the raw generated
`item_N_response` columns are left untouched.

Approximation note: the original paper's Table 5 structural paths come
from a full structural equation model with latent Job Demands / Job
Resources factors. Here we approximate those latent factors as simple
averages of their facet items and use standardized OLS regression --
a reasonable first-pass comparison, but not identical to SEM factor
scores. This is stated explicitly in the printed output, not just here.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR = os.path.join(_REPO_ROOT, "data")

df = pd.read_csv(os.path.join(_DATA_DIR, "jdr_burnout_dataset.csv"))
with open(os.path.join(_DATA_DIR, "jdr_burnout_item_metadata.json")) as f:
    items = json.load(f)

N_SCALE = 4  # 1-4 response scale

# --- Reverse-score marked items ---
for i, item in enumerate(items):
    col = f"item_{i+1}_response"
    if item["reverse"]:
        df[col] = (N_SCALE + 1) - df[col]

# --- Build the 13 variables needed for Table 3 ---
# 11 demand/resource facets are single-item (matching the paper's use of
# one representative item per facet in the material provided), so their
# "scale score" is just that item's (reverse-scored where applicable) value.
facet_cols = {}
for i, item in enumerate(items):
    col = f"item_{i+1}_response"
    facet = item["facet"]
    facet_cols.setdefault(facet, []).append(col)

variables = {}
demand_facets = ["physical_workload", "time_pressure", "recipient_contact", "shift_work", "physical_environment"]
resource_facets = ["feedback", "rewards", "job_control", "participation", "job_security", "supervisor_support"]

for facet in demand_facets + resource_facets:
    variables[facet] = df[facet_cols[facet]].mean(axis=1)  # mean() is a no-op for single-item facets, correct if ever multi-item

variables["exhaustion"] = df[facet_cols["exhaustion"]].mean(axis=1)
variables["disengagement"] = df[facet_cols["disengagement"]].mean(axis=1)

var_df = pd.DataFrame(variables)

# --- Table 3 comparison ---
# Paper's real correlations (below-diagonal, individual level, N=374),
# transcribed from Table 3. Order matches: 1 physical_workload,
# 2 time_pressure, 3 recipient_contact, 4 shift_work,
# 5 physical_environment, 6 feedback, 7 rewards, 8 job_control,
# 9 participation, 10 job_security, 11 supervisor_support,
# 12 exhaustion, 13 disengagement
ORDER = demand_facets + resource_facets + ["exhaustion", "disengagement"]
LABELS = ["Physical workload", "Time pressure", "Recipient contact", "Shift work",
          "Physical environment", "Feedback", "Rewards", "Job control",
          "Participation", "Job security", "Supervisor support",
          "Exhaustion", "Disengagement"]

REAL_CORR = np.array([
    [1.00, .32, .24, .27, .17, .11, .20, .17, .12, .03, .15, .53, .20],
    [.32, 1.00, .15, .21, .27, .07, .18, .16, .06, .10, .03, .38, .06],
    [.24, .15, 1.00, .25, .23, .22, .33, .08, .10, .01, .27, .39, .28],
    [.27, .21, .25, 1.00, .27, .12, .22, .18, .17, .08, .16, .45, .32],
    [.17, .27, .23, .27, 1.00, .22, .25, .18, .17, -.01, .24, .32, .22],
    [.11, .07, .22, .12, .22, 1.00, .38, .29, .30, .21, .44, .30, .46],
    [.20, .18, .33, .22, .25, .38, 1.00, .13, .19, .22, .26, .39, .37],
    [.17, .16, .08, .18, .18, .29, .13, 1.00, .36, .21, .19, .26, .38],
    [.12, .06, .10, .17, .17, .30, .19, .36, 1.00, .31, .24, .23, .46],
    [.03, .10, .01, .08, -.01, .21, .22, .21, .31, 1.00, .16, .08, .26],
    [.15, .03, .27, .16, .24, .44, .26, .19, .24, .16, 1.00, .23, .36],
    [.53, .38, .39, .45, .32, .30, .39, .26, .23, .08, .23, 1.00, .39],
    [.20, .06, .28, .32, .22, .46, .37, .38, .46, .26, .36, .39, 1.00],
])
# Verified by transcribing and cross-checking the below-diagonal
# (individual-level, N=374) values from the paper's own Table 3 image,
# confirming each row contains exactly one more entry than the previous
# (a necessary property of a valid lower-triangular matrix) before use.

gen_corr = var_df[ORDER].corr().to_numpy()

print("=== Table 3 comparison: real vs. generated correlations ===\n")
diffs = []
rows = []
for i in range(13):
    for j in range(i + 1, 13):
        real_r = REAL_CORR[i, j]
        gen_r = gen_corr[i, j]
        diff = abs(real_r - gen_r)
        diffs.append(diff)
        rows.append({
            "var_1": LABELS[i], "var_2": LABELS[j],
            "real_r": real_r, "generated_r": round(gen_r, 3), "abs_diff": round(diff, 3),
        })

corr_comparison = pd.DataFrame(rows)
corr_comparison.to_csv(os.path.join(_DATA_DIR, "jdr_table3_comparison.csv"), index=False)
print(f"Mean absolute difference across all {len(diffs)} pairwise correlations: {np.mean(diffs):.3f}")
print(f"Median absolute difference: {np.median(diffs):.3f}")
print(f"Max absolute difference: {np.max(diffs):.3f}")
# Correlation between the real and generated correlation matrices themselves
# ("how well does the generated correlational STRUCTURE match the real one")
r_of_r, _ = stats.pearsonr(REAL_CORR[np.triu_indices(13, k=1)], gen_corr[np.triu_indices(13, k=1)])
print(f"Correlation between real and generated correlation patterns (r of r's): {r_of_r:.3f}")
print("Full pairwise comparison saved -> jdr_table3_comparison.csv\n")

# --- Table 5 (Structural model) comparison ---
print("=== Table 5 (Structural model) comparison ===")
print("NOTE: approximation -- real paths are from a full SEM with latent")
print("factors; here Job Demands / Job Resources are simple averages of")
print("their facet items, and paths are standardized OLS regression coefficients.\n")

job_demands = var_df[demand_facets].mean(axis=1)
job_resources = var_df[resource_facets].mean(axis=1)

def standardized_coef(y, x1, x2):
    X = np.column_stack([stats.zscore(x1), stats.zscore(x2)])
    y_z = stats.zscore(y)
    coefs, _, _, _ = np.linalg.lstsq(X, y_z, rcond=None)
    return coefs

coefs_exhaustion = standardized_coef(var_df["exhaustion"], job_demands, job_resources)
coefs_disengagement = standardized_coef(var_df["disengagement"], job_demands, job_resources)

real_paths = {
    ("Job demands", "Exhaustion"): .91, ("Job resources", "Exhaustion"): .04,
    ("Job demands", "Disengagement"): .07, ("Job resources", "Disengagement"): -.72,
}
gen_paths = {
    ("Job demands", "Exhaustion"): coefs_exhaustion[0], ("Job resources", "Exhaustion"): coefs_exhaustion[1],
    ("Job demands", "Disengagement"): coefs_disengagement[0], ("Job resources", "Disengagement"): coefs_disengagement[1],
}

path_rows = []
for key, real_val in real_paths.items():
    gen_val = gen_paths[key]
    path_rows.append({
        "path": f"{key[0]} -> {key[1]}",
        "real_coefficient": real_val,
        "generated_coefficient": round(gen_val, 3),
        "abs_diff": round(abs(real_val - gen_val), 3),
    })
path_comparison = pd.DataFrame(path_rows)
path_comparison.to_csv(os.path.join(_DATA_DIR, "jdr_table5_comparison.csv"), index=False)
print(path_comparison.to_string(index=False))
print("\nSaved -> jdr_table5_comparison.csv")

# --- Naive LLM baseline, for the SAME four structural paths ---
# Reuses the raw propensities already saved in the dataset -- no new API
# calls needed. This tells us whether the off-diagonal path anomaly is
# something calibration introduced, or a generic LLM-knowledge effect
# that would show up regardless of calibration.
print("\n=== Naive LLM baseline comparison (same four paths, no calibration) ===")
print("Uses the SAME raw LLM propensities already in the dataset, reverse-scored")
print("the same way, but binned directly with NO real-data calibration at all.\n")

def naive_bin(raw_propensities, k_target=4):
    raw = np.array(raw_propensities)
    return np.clip((raw * k_target).astype(int) + 1, 1, k_target)

naive_vars = {}
for i, item in enumerate(items):
    raw_col = f"item_{i+1}_raw_propensity"
    naive_val = naive_bin(df[raw_col])
    if item["reverse"]:
        naive_val = (N_SCALE + 1) - naive_val
    naive_vars.setdefault(item["facet"], []).append(naive_val)

naive_facet_scores = {facet: np.mean(vals, axis=0) for facet, vals in naive_vars.items()}
naive_df = pd.DataFrame(naive_facet_scores)

naive_demands = naive_df[demand_facets].mean(axis=1)
naive_resources = naive_df[resource_facets].mean(axis=1)
naive_coefs_exhaustion = standardized_coef(naive_df["exhaustion"], naive_demands, naive_resources)
naive_coefs_disengagement = standardized_coef(naive_df["disengagement"], naive_demands, naive_resources)

naive_paths = {
    ("Job demands", "Exhaustion"): naive_coefs_exhaustion[0],
    ("Job resources", "Exhaustion"): naive_coefs_exhaustion[1],
    ("Job demands", "Disengagement"): naive_coefs_disengagement[0],
    ("Job resources", "Disengagement"): naive_coefs_disengagement[1],
}

three_way_rows = []
for key, real_val in real_paths.items():
    three_way_rows.append({
        "path": f"{key[0]} -> {key[1]}",
        "real": real_val,
        "calibrated": round(gen_paths[key], 3),
        "naive_llm": round(naive_paths[key], 3),
        "calibrated_abs_diff": round(abs(real_val - gen_paths[key]), 3),
        "naive_abs_diff": round(abs(real_val - naive_paths[key]), 3),
    })
three_way_comparison = pd.DataFrame(three_way_rows)
three_way_comparison.to_csv(os.path.join(_DATA_DIR, "jdr_figure3_three_way_comparison.csv"), index=False)
print(three_way_comparison.to_string(index=False))
print("\nSaved -> jdr_figure3_three_way_comparison.csv")
print("\nIf naive_abs_diff is similar to (or worse than) calibrated_abs_diff on the")
print("off-diagonal paths (Demands->Disengagement, Resources->Exhaustion), that")
print("indicates the pattern is generic LLM-knowledge-driven, not something")
print("calibration specifically introduced or could easily fix.")

# --- Bonus: reliability check ---
print("\n=== Reliability check (not in Table 3/5, but a useful sanity check) ===")
exhaustion_items = df[facet_cols["exhaustion"]]
disengagement_items = df[facet_cols["disengagement"]]

def cronbach_alpha(items_df):
    k = items_df.shape[1]
    item_vars = items_df.var(axis=0, ddof=1).sum()
    total_var = items_df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars / total_var)

print(f"Exhaustion alpha: generated={cronbach_alpha(exhaustion_items):.3f}, real=.82")
print(f"Disengagement alpha: generated={cronbach_alpha(disengagement_items):.3f}, real=.83")
