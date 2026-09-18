"""
Test: does blending the top-K nearest real variables into a composite
calibration target (weighted by similarity) produce lower error than using
only the single best match? Uses the same held-out validation methodology
as before, so results are directly comparable.

Method per held-out variable:
  1. Find its top-K nearest OTHER real variables (excluding itself).
  2. SINGLE-BEST baseline: compare held-out variable's real distribution
     to just the #1 match's distribution (same as before).
  3. ENSEMBLE: interpolate each of the top-K candidates' cumulative
     distributions onto a common set of points, weighted-average them by
     similarity, and compare the held-out variable's real distribution to
     THIS composite curve instead.

No API calls -- reuses existing embeddings.
"""

import csv
import json
import numpy as np
import pandas as pd

TOP_K = 5  # how many candidates to blend for the ensemble approach

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

def build_cdf_curve(value_proportions, n_points=20):
    """Interpolate a distribution's cumulative curve onto n_points equally
    spaced positions along normalized rank (0 to 1)."""
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

def curve_distance(curve_a, curve_b):
    return float(np.mean(np.abs(curve_a - curve_b)))

def run_comparison(sample_size=6064, seed=42, top_k=TOP_K, n_points=20):
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

        sims = embeddings @ embeddings[i]
        sims[i] = -np.inf
        top_idx = np.argsort(-sims)[:top_k]

        # Filter to candidates with usable stats
        candidates = []
        for j in top_idx:
            var_j = variables[j]
            stats_j = stats.get(var_j.lower())
            if stats_j is not None and "error" not in stats_j:
                candidates.append((j, float(sims[j]), stats_j))
        if len(candidates) == 0:
            continue

        curve_i = build_cdf_curve(stats_i["value_proportions"], n_points)

        # --- Single-best baseline (top-1 only) ---
        best_j, best_sim, best_stats = candidates[0]
        curve_best = build_cdf_curve(best_stats["value_proportions"], n_points)
        error_single = curve_distance(curve_i, curve_best)

        # --- Ensemble: similarity-weighted blend of all candidates ---
        sim_weights = np.array([c[1] for c in candidates])
        sim_weights = np.clip(sim_weights, 0, None)  # ignore any negative similarities
        if sim_weights.sum() == 0:
            sim_weights = np.ones(len(candidates))  # fallback: equal weight
        sim_weights = sim_weights / sim_weights.sum()

        candidate_curves = np.array([build_cdf_curve(c[2]["value_proportions"], n_points) for c in candidates])
        ensemble_curve = np.average(candidate_curves, axis=0, weights=sim_weights)
        error_ensemble = curve_distance(curve_i, ensemble_curve)

        results.append({
            "variable": var_i,
            "top1_similarity": best_sim,
            "n_candidates_used": len(candidates),
            "error_single_best": error_single,
            "error_ensemble": error_ensemble,
            "ensemble_better": error_ensemble < error_single,
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    print(f"Running single-best vs ensemble (top-{TOP_K}) comparison...")
    df = run_comparison()
    print(f"Compared {len(df)} held-out variables\n")

    df.to_csv("data/ensemble_validation_results.csv", index=False)
    print("Saved -> ensemble_validation_results.csv\n")

    print(f"Overall avg error, single-best match: {df['error_single_best'].mean():.4f}")
    print(f"Overall avg error, ensemble (top-{TOP_K}): {df['error_ensemble'].mean():.4f}")
    print(f"Ensemble was better in {df['ensemble_better'].mean()*100:.1f}% of cases\n")

    bins = [0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.01]
    df["similarity_bin"] = pd.cut(df["top1_similarity"], bins=bins)
    summary = df.groupby("similarity_bin", observed=True).agg(
        n_items=("variable", "count"),
        avg_error_single=("error_single_best", "mean"),
        avg_error_ensemble=("error_ensemble", "mean"),
    ).reset_index()
    summary["improvement"] = summary["avg_error_single"] - summary["avg_error_ensemble"]
    print("By similarity range:")
    print(summary.to_string(index=False))

    # Coverage check: how many items would now pass a 0.12 error threshold
    # under each approach?
    THRESHOLD = 0.12
    pass_single = (df["error_single_best"] <= THRESHOLD).mean()
    pass_ensemble = (df["error_ensemble"] <= THRESHOLD).mean()
    print(f"\nCoverage at error threshold {THRESHOLD}:")
    print(f"  Single-best: {pass_single*100:.1f}% of items would pass")
    print(f"  Ensemble:    {pass_ensemble*100:.1f}% of items would pass")
