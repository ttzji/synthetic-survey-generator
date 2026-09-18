"""
For items with similarity < 0.5 (no good content match), current behavior
is pure LLM-only calibration (equal-width binning of propensity -- which
implicitly assumes responses would be uniformly spread across categories).

Test: does calibrating against a GENERIC real Likert shape instead (the
average cumulative curve across many real GSS items, ignoring content
entirely) do better than that flat/uniform assumption?

Method: build one "generic shape" by averaging the cumulative curves of a
large sample of real, usable Likert-type variables. Then, for every
held-out variable, compare its own real distribution against:
  (a) the generic shape (same target for every item, no matching at all)
  (b) a flat/uniform shape (equivalent to what equal-width LLM-only binning
      assumes)
Whichever is closer to real distributions ON AVERAGE is the better
no-match fallback.
"""

import csv
import json
import numpy as np
import pandas as pd

N_POINTS = 20

def load_resources():
    with open("data/gss_master_variable_stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    return stats

def build_cdf_curve(value_proportions, n_points=N_POINTS):
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

def is_likert_like(stats_entry, min_categories=3, max_dominant_share=0.85):
    props = [v for k, v in stats_entry.get("value_proportions", {}).items() if k != "__other__"]
    if len(props) < min_categories:
        return False
    if max(props) > max_dominant_share:
        return False
    return True

def build_generic_shape(stats, sample_size=2000, seed=42):
    likert_vars = [v for v, s in stats.items() if "error" not in s and is_likert_like(s)]
    rng = np.random.default_rng(seed)
    sample = rng.choice(likert_vars, size=min(sample_size, len(likert_vars)), replace=False)
    curves = np.array([build_cdf_curve(stats[v]["value_proportions"]) for v in sample])
    generic_curve = curves.mean(axis=0)
    return generic_curve, len(sample)

def run_test(sample_size=6064, seed=123):
    stats = load_resources()
    generic_curve, n_used_for_generic = build_generic_shape(stats, seed=seed)
    uniform_curve = np.linspace(0, 1, N_POINTS)

    print(f"Generic shape built from {n_used_for_generic} real Likert-type variables")
    print(f"Generic curve (should show central clustering, not a straight diagonal):")
    print(f"  {np.round(generic_curve, 3)}")
    print(f"Uniform/flat curve for comparison:")
    print(f"  {np.round(uniform_curve, 3)}\n")

    all_vars = [v for v, s in stats.items() if "error" not in s]
    rng = np.random.default_rng(seed + 1)
    sample = rng.choice(all_vars, size=min(sample_size, len(all_vars)), replace=False)

    results = []
    for v in sample:
        real_curve = build_cdf_curve(stats[v]["value_proportions"])
        error_generic = curve_distance(real_curve, generic_curve)
        error_uniform = curve_distance(real_curve, uniform_curve)
        results.append({
            "variable": v,
            "error_generic_shape": error_generic,
            "error_uniform_shape": error_uniform,
            "generic_better": error_generic < error_uniform,
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_test()
    print(f"Evaluated {len(df)} real variables (as if none had any content match)\n")

    print(f"Avg error vs GENERIC real shape:  {df['error_generic_shape'].mean():.4f}")
    print(f"Avg error vs UNIFORM/flat shape:  {df['error_uniform_shape'].mean():.4f}")
    print(f"Generic shape was better in {df['generic_better'].mean()*100:.1f}% of cases")

    improvement = df['error_uniform_shape'].mean() - df['error_generic_shape'].mean()
    pct_improvement = improvement / df['error_uniform_shape'].mean() * 100
    print(f"\nRelative improvement from using generic shape instead of uniform: {pct_improvement:.1f}%")
