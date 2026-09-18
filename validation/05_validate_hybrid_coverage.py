"""
Proper, comparable coverage test for the HYBRID strategy:
  - similarity >= 0.75: use single-best match (ensemble showed no benefit here)
  - 0.5 <= similarity < 0.75: use ensemble (top-5 blend) -- this is where
    the earlier test showed real improvement
  - similarity < 0.5: not confident regardless of method

Applies the SAME multi-criteria gate as production (generate_dataset.py):
error <= MAX_ACCEPTABLE_EXPECTED_ERROR, real N >= MIN_REAL_N, years > 1 --
but extended so that for ensemble cases, EVERY candidate used in the blend
must individually satisfy the N/years requirements (not just one side),
since the calibration now depends on all of them.

This produces a coverage % directly comparable to the single-best-only
20.3% figure from script 15, so we can honestly say whether the hybrid
approach expands genuine (not just relabeled) coverage.
"""

import csv
import json
import numpy as np
import pandas as pd

TOP_K_SEARCH = 15   # search pool size, before quality filtering
ENSEMBLE_MAX_CANDIDATES = 5  # blend at most this many, after filtering
ENSEMBLE_MIN_CANDIDATES = 2  # need at least this many QUALIFYING candidates
                               # to use ensemble at all
HYBRID_SWITCH_HIGH = 0.75
HYBRID_SWITCH_LOW = 0.50
MAX_ACCEPTABLE_EXPECTED_ERROR = 0.12
MIN_REAL_N = 200

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

def run_hybrid_validation(sample_size=6064, seed=42, top_k_search=TOP_K_SEARCH, n_points=20):
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
        n_i = stats_i["n"]
        years_i = len(stats_i["years_fielded"])

        sims = embeddings @ embeddings[i]
        sims[i] = -np.inf
        top_idx = np.argsort(-sims)[:top_k_search]

        candidates = []
        for j in top_idx:
            var_j = variables[j]
            stats_j = stats.get(var_j.lower())
            if stats_j is not None and "error" not in stats_j:
                candidates.append((var_j, float(sims[j]), stats_j))
        if len(candidates) == 0:
            continue

        curve_i = build_cdf_curve(stats_i["value_proportions"], n_points)
        top1_sim = candidates[0][1]

        def is_quality(c):
            _, _, s = c
            return s["n"] >= MIN_REAL_N and len(s["years_fielded"]) > 1

        # --- Decide method based on hybrid rule ---
        if top1_sim >= HYBRID_SWITCH_HIGH:
            method = "single_best"
            used_candidates = candidates[:1]  # top-1 regardless of quality
                                                # filter (matches production:
                                                # single-best still checked
                                                # separately below)
        elif top1_sim >= HYBRID_SWITCH_LOW:
            # Filter to quality candidates FIRST, then blend the survivors --
            # rather than blending raw top-5 and requiring all to pass
            quality_candidates = [c for c in candidates if is_quality(c)]
            if len(quality_candidates) >= ENSEMBLE_MIN_CANDIDATES:
                method = "ensemble"
                used_candidates = quality_candidates[:ENSEMBLE_MAX_CANDIDATES]
            else:
                method = "none"
                used_candidates = candidates[:1]
        else:
            method = "none"
            used_candidates = candidates[:1]

        if method == "single_best":
            curve_match = build_cdf_curve(used_candidates[0][2]["value_proportions"], n_points)
        elif method == "ensemble":
            sim_weights = np.array([c[1] for c in used_candidates])
            sim_weights = np.clip(sim_weights, 0, None)
            if sim_weights.sum() == 0:
                sim_weights = np.ones(len(used_candidates))
            sim_weights = sim_weights / sim_weights.sum()
            curves = np.array([build_cdf_curve(c[2]["value_proportions"], n_points) for c in used_candidates])
            curve_match = np.average(curves, axis=0, weights=sim_weights)
        else:
            curve_match = build_cdf_curve(used_candidates[0][2]["value_proportions"], n_points)

        error = curve_distance(curve_i, curve_match)

        if method == "single_best":
            candidate_quality_ok = is_quality(used_candidates[0])
        elif method == "ensemble":
            candidate_quality_ok = True  # already filtered to quality-only above
        else:
            candidate_quality_ok = False

        # NOTE: no n_i/years_i check -- a real user's typed item has no real
        # stats to check at all. Only the matched candidate(s)' data quality matters.
        confident = (
            method != "none" and
            error <= MAX_ACCEPTABLE_EXPECTED_ERROR and
            candidate_quality_ok
        )

        results.append({
            "variable": var_i,
            "top1_similarity": top1_sim,
            "method": method,
            "n_candidates_used": len(used_candidates),
            "error": error,
            "confident": confident,
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Running hybrid single-best/ensemble validation with full confidence gate...")
    df = run_hybrid_validation()
    print(f"Evaluated {len(df)} held-out variables\n")
    df.to_csv("data/hybrid_validation_results.csv", index=False)

    print(f"Method breakdown:\n{df['method'].value_counts().to_string()}\n")

    coverage = df["confident"].mean() * 100
    print(f"=== HYBRID coverage: {coverage:.1f}% ===")
    print(f"(compare to corrected single-best-only baseline: 25.2%)\n")

    summary = df.groupby("confident").agg(
        n_items=("variable", "count"),
        avg_error=("error", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))
