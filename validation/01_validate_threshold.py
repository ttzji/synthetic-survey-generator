"""
VALIDATION SCRIPT: empirically justify the similarity/N threshold.

Method (no API calls needed -- reuses embeddings you already computed):
For each real GSS variable, pretend its own label is a "novel" user item.
Search for its nearest OTHER real variable (excluding itself) using the
existing embedding index. This simulates exactly what happens when a user
types an item with no perfect match in the catalog.

Then compare the held-out variable's REAL distribution to the matched
variable's REAL distribution. If retrieval is working well, high-similarity
matches should show low calibration error (the matched variable is a good
stand-in), and error should grow as similarity drops. This gives you an
empirical curve to justify a threshold, rather than guessing one.

Output:
    validation_results.csv       -- one row per held-out variable
    (printed) binned summary table of error vs. similarity
"""

import csv
import json
import numpy as np
import pandas as pd

def load_resources():
    embeddings = np.load("data/gss_variable_embeddings.npy")
    variables, labels = [], []
    with open("data/gss_variable_index.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variables.append(row["variable"])
            labels.append(row["label"])
    with open("data/gss_master_variable_stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    return embeddings, variables, labels, stats

def mean_rank_normalized(value_proportions):
    """Weighted average rank position, normalized to [0,1]. A simple,
    interpretable summary of where a distribution's mass sits."""
    cats = [(float(k), v) for k, v in value_proportions.items() if k != "__other__"]
    cats.sort(key=lambda x: x[0])
    k = len(cats)
    if k <= 1:
        return 0.5
    total = sum(v for _, v in cats)
    if total == 0:
        return 0.5
    weighted_sum = sum(((i / (k - 1)) * v) for i, (_, v) in enumerate(cats))
    return weighted_sum / total

def cdf_distance(value_proportions_a, value_proportions_b, n_points=9):
    """Compare two distributions' overall SHAPE, not just their mean.
    Builds each distribution's cumulative curve over normalized rank
    position (0 to 1), interpolates both at n_points equally spaced
    positions, and returns the mean absolute difference -- a simple,
    interpretable proxy for how differently these two distributions would
    calibrate a synthetic item."""
    def build_cdf(value_proportions):
        cats = [(float(k), v) for k, v in value_proportions.items() if k != "__other__"]
        cats.sort(key=lambda x: x[0])
        k = len(cats)
        total = sum(v for _, v in cats)
        if k == 0 or total == 0:
            return np.array([0, 1]), np.array([0, 1])
        ranks = np.array([i / max(k - 1, 1) for i in range(k)])
        cum = np.cumsum([v / total for _, v in cats])
        return ranks, cum

    ranks_a, cum_a = build_cdf(value_proportions_a)
    ranks_b, cum_b = build_cdf(value_proportions_b)

    query_points = np.linspace(0, 1, n_points)
    interp_a = np.interp(query_points, ranks_a, cum_a)
    interp_b = np.interp(query_points, ranks_b, cum_b)
    return float(np.mean(np.abs(interp_a - interp_b)))

def run_validation(sample_size=2000, seed=42):
    embeddings, variables, labels, stats = load_resources()
    n_total = len(variables)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n_total, size=min(sample_size, n_total), replace=False)

    results = []
    for i in sample_idx:
        var_i = variables[i]
        stats_i = stats.get(var_i.lower())
        if stats_i is None or "error" in stats_i:
            continue

        # Find nearest OTHER variable, excluding self
        sims = embeddings @ embeddings[i]
        sims[i] = -np.inf  # exclude self
        j = int(np.argmax(sims))
        var_j = variables[j]
        stats_j = stats.get(var_j.lower())
        if stats_j is None or "error" in stats_j:
            continue

        similarity = float(sims[j])
        error = cdf_distance(stats_i["value_proportions"], stats_j["value_proportions"])
        mean_diff = abs(mean_rank_normalized(stats_i["value_proportions"]) -
                         mean_rank_normalized(stats_j["value_proportions"]))

        results.append({
            "variable": var_i,
            "label": labels[i],
            "matched_variable": var_j,
            "matched_label": labels[j],
            "similarity": similarity,
            "cdf_distance_error": error,
            "mean_diff": mean_diff,
            "n_i": stats_i["n"],
            "n_j": stats_j["n"],
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Running validation (no API calls -- uses existing embeddings)...")
    df = run_validation(sample_size=999999)  # effectively the full catalog
    print(f"Validated {len(df)} held-out variables\n")

    df.to_csv("data/validation_results.csv", index=False)
    print("Saved -> validation_results.csv\n")

    # Bin by similarity and show average error per bin -- this is the
    # empirical evidence for picking a threshold
    bins = [0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]
    df["similarity_bin"] = pd.cut(df["similarity"], bins=bins)
    summary = df.groupby("similarity_bin", observed=True).agg(
        n_items=("variable", "count"),
        avg_cdf_error=("cdf_distance_error", "mean"),
        avg_mean_diff=("mean_diff", "mean"),
    ).reset_index()

    print("Calibration error by similarity range:")
    print(summary.to_string(index=False))
    print("\nLook for the similarity range where avg_cdf_error starts climbing")
    print("sharply -- that's your empirically-justified threshold, not a guess.")
