"""
Full validation of the COMBINED confidence rule actually used in
generate_dataset.py: similarity >= MIN_SIMILARITY AND
n >= MIN_REAL_N AND years_fielded > 1 (applied to BOTH the held-out
variable and its match, since either side being thin/single-year data
could produce an unreliable comparison).

This merges in years_fielded (missing from the original validation run)
and directly tests whether the three-factor rule separates low-error from
high-error matches better than similarity alone did.
"""

import json
import pandas as pd

MIN_SIMILARITY = 0.70  # updated to match current production threshold in generate_dataset.py
MIN_REAL_N = 200

df = pd.read_csv("data/validation_results.csv")
print(f"Total pairs: {len(df)}")

with open("data/gss_master_variable_stats.json", encoding="utf-8") as f:
    stats = json.load(f)

def years_count(varname):
    entry = stats.get(varname.lower())
    if entry is None or "error" in entry:
        return 0
    return len(entry.get("years_fielded", []))

df["years_i"] = df["variable"].apply(years_count)
df["years_j"] = df["matched_variable"].apply(years_count)

df["confident"] = (
    (df["similarity"] >= MIN_SIMILARITY) &
    (df["n_j"] >= MIN_REAL_N) &
    (df["years_j"] > 1)
)
# NOTE: previously also required n_i/years_i (the held-out "query"
# variable's own real data). Removed -- this doesn't correspond to real
# production use, where a user's typed item has no real stats at all.
# Only the MATCHED variable's data quality matters in practice.

print(f"\n=== Using current rule: similarity>={MIN_SIMILARITY}, N>={MIN_REAL_N}, years>1 ===")
summary = df.groupby("confident").agg(
    n_items=("variable", "count"),
    avg_cdf_error=("cdf_distance_error", "mean"),
    avg_mean_diff=("mean_diff", "mean"),
).reset_index()
print(summary.to_string(index=False))
print(f"\nCoverage: {df['confident'].mean()*100:.1f}% of items would pass this rule")

# --- Grid search: try a range of threshold combinations to see which
# actually separates low-error from high-error matches best ---
print("\n=== Grid search over threshold combinations ===")
results = []
for sim_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    for n_thresh in [100, 200, 500, 1000]:
        mask = (
            (df["similarity"] >= sim_thresh) &
            (df["n_j"] >= n_thresh) &
            (df["years_j"] > 1)
        )
        if mask.sum() < 10:  # skip combos with too few items to be meaningful
            continue
        results.append({
            "sim_thresh": sim_thresh,
            "n_thresh": n_thresh,
            "n_passing": int(mask.sum()),
            "pct_coverage": round(mask.mean() * 100, 1),
            "avg_error_passing": round(df.loc[mask, "cdf_distance_error"].mean(), 4),
            "avg_error_failing": round(df.loc[~mask, "cdf_distance_error"].mean(), 4),
        })

grid_df = pd.DataFrame(results)
grid_df["error_reduction"] = grid_df["avg_error_failing"] - grid_df["avg_error_passing"]
grid_df = grid_df.sort_values("error_reduction", ascending=False)
print(grid_df.to_string(index=False))
print("\nLook for combinations with a large positive 'error_reduction' AND")
print("reasonable 'pct_coverage' -- that's the best empirically-supported rule.")

grid_df.to_csv("data/threshold_grid_search.csv", index=False)
print("\nSaved -> threshold_grid_search.csv")
