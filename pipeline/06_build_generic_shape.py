"""
Precompute the generic real Likert response shape ONCE and save it, so
production doesn't need to rescan the full stats file on every run.

Output: generic_likert_shape.npy -- a 20-point cumulative curve
"""

import json
import os
import numpy as np

os.makedirs("data", exist_ok=True)

N_POINTS = 20

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

def is_likert_like(stats_entry, min_categories=3, max_dominant_share=0.85):
    props = [v for k, v in stats_entry.get("value_proportions", {}).items() if k != "__other__"]
    if len(props) < min_categories:
        return False
    if max(props) > max_dominant_share:
        return False
    return True

if __name__ == "__main__":
    with open("data/gss_master_variable_stats.json", encoding="utf-8") as f:
        stats = json.load(f)

    likert_vars = [v for v, s in stats.items() if "error" not in s and is_likert_like(s)]
    print(f"Building generic shape from {len(likert_vars)} real Likert-type variables...")

    curves = np.array([build_cdf_curve(stats[v]["value_proportions"]) for v in likert_vars])
    generic_curve = curves.mean(axis=0)

    np.save("data/generic_likert_shape.npy", generic_curve)
    print(f"Saved -> generic_likert_shape.npy")
    print(f"Curve: {np.round(generic_curve, 3)}")
