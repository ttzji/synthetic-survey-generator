"""
Head-to-head validation: full calibrated pipeline vs. a naive "just ask
the LLM directly" baseline, on held-out real GSS items.

Method: for each held-out item, generate raw LLM propensities ONCE (same
API calls serve both conditions). Condition (a) runs those propensities
through the full pipeline (self-excluded retrieval + percentile
calibration). Condition (b) bins the SAME raw propensities directly with
no real-data anchoring at all -- the naive baseline a reviewer would
reasonably ask "why not just do this?" about. Both are compared against
the held-out item's real distribution using the same shape-based error
metric used throughout this project's other validation studies.

This file is the N=100 run (N_RESPONDENTS below). The sibling files
10_validate_vs_naive_baseline_100/300/500/1000.py are identical except for
that one number. Cost for this run: about $0.03. Results vary somewhat
run to run (LLM sampling). Saves data/naive_baseline_comparison_results_N100.csv.
"""

import os
import importlib.util
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
spec = importlib.util.spec_from_file_location("gen", os.path.join(_REPO_ROOT, "generate_dataset.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

N_HELD_OUT_ITEMS = 40
N_RESPONDENTS = 100
SEED = 42

def build_cdf_curve(value_proportions, n_points=20):
    cats = [(float(k), v) for k, v in value_proportions.items() if k != "__other__"]
    cats.sort(key=lambda x: x[0])
    k = len(cats)
    total = sum(v for _, v in cats)
    if k == 0 or total == 0:
        return np.linspace(0, 1, n_points)
    ranks = np.array([i / max(k - 1, 1) for i in range(k)])
    cum = np.cumsum([v / total for _, v in cats])
    return np.interp(np.linspace(0, 1, n_points), ranks, cum)

def empirical_curve_from_samples(values, k_target, n_points=20):
    """Build the same kind of cumulative curve, but from a set of
    GENERATED category outputs rather than a pre-tabulated distribution."""
    counts = pd.Series(values).value_counts().sort_index()
    proportions = {str(k): v / len(values) for k, v in counts.items()}
    return build_cdf_curve(proportions, n_points)

def curve_distance(curve_a, curve_b):
    return float(np.mean(np.abs(curve_a - curve_b)))

def retrieve_self_excluded(idx, embeddings, variables, labels, stats, response_type="likert"):
    """Same hybrid logic as gen.retrieve_calibration_source, but starting
    from a precomputed embedding index and excluding that item itself from
    the candidate search -- simulating 'if this real item were actually a
    novel user item with no exact match available'."""
    sims = embeddings @ embeddings[idx]
    sims[idx] = -np.inf
    top_idx = np.argsort(-sims)[:gen.TOP_K_SEARCH]

    candidates = []
    for j in top_idx:
        var = variables[j]
        stats_entry = stats.get(var.lower())
        if stats_entry is not None and "error" not in stats_entry:
            candidates.append((var, labels[j], float(sims[j]), stats_entry))
    if not candidates:
        return None

    top1_sim = candidates[0][2]

    def is_quality(c):
        _, _, _, s = c
        return gen.is_usable_anchor(s, response_type) and s["n"] >= gen.MIN_REAL_N and len(s["years_fielded"]) > 1

    if top1_sim >= gen.HYBRID_SWITCH_HIGH:
        for var, label, sim, s in candidates:
            if gen.is_usable_anchor(s, response_type):
                cutpoints, k_real = gen.real_distribution_cutpoints(s["value_proportions"])
                return {"method": "single_best", "cutpoints": cutpoints, "k_real": k_real}
        return None
    elif top1_sim >= gen.HYBRID_SWITCH_LOW:
        quality_candidates = [c for c in candidates if is_quality(c)]
        if len(quality_candidates) >= gen.ENSEMBLE_MIN_CANDIDATES:
            used = quality_candidates[:gen.ENSEMBLE_MAX_CANDIDATES]
            weights = np.clip(np.array([c[2] for c in used]), 0, None)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(used)) / len(used)
            cutpoints, k_real = gen.build_ensemble_cutpoints([c[3] for c in used], weights)
            return {"method": "ensemble", "cutpoints": cutpoints, "k_real": k_real}
        return None
    else:
        return None

def naive_baseline(raw_propensities, k_target):
    """Zero real-data anchoring: bin the raw LLM propensity directly by
    its VALUE (not percentile rank, not any real distribution shape) --
    the simplest possible 'just ask the LLM' approach."""
    raw = np.array(raw_propensities)
    bucket = np.clip((raw * k_target).astype(int) + 1, 1, k_target)
    return bucket.tolist()

def run_comparison():
    embeddings, variables, labels, stats = gen.load_item_resources()
    usable_indices = [i for i, v in enumerate(variables) if "error" not in stats.get(v.lower(), {"error": 1})]

    rng = np.random.default_rng(SEED)
    sample_indices = rng.choice(usable_indices, size=min(N_HELD_OUT_ITEMS, len(usable_indices)), replace=False)

    print(f"Selecting {len(sample_indices)} held-out items and finding self-excluded matches...")
    held_out_items = []
    for idx in sample_indices:
        var = variables[idx]
        true_stats = stats[var.lower()]
        match = retrieve_self_excluded(idx, embeddings, variables, labels, stats)
        held_out_items.append({
            "variable": var,
            "item_text": labels[idx],
            "true_value_proportions": true_stats["value_proportions"],
            "match": match,  # None if no real anchor found (excluding self)
        })
        status = match["method"] if match else "none"
        print(f"  {var}: self-excluded match method = {status}")

    print(f"\nGenerating demographic sample (N={N_RESPONDENTS}, representative)...")
    demo_df, value_labels = gen.build_sample_pool(mode="representative", n=N_RESPONDENTS, seed=SEED)
    personas = [gen.describe_persona(row) for _, row in demo_df.iterrows()]

    items_info_for_prompt = [{"item_text": h["item_text"]} for h in held_out_items]

    print(f"\nGenerating LLM propensities for {len(held_out_items)} items x {N_RESPONDENTS} respondents "
          f"in batches of {gen.VALIDATED_BATCH_SIZE}...")
    all_propensities = []
    batch_size = gen.VALIDATED_BATCH_SIZE
    for start in range(0, N_RESPONDENTS, batch_size):
        batch = personas[start:start + batch_size]
        result = gen.call_llm_batch(batch, items_info_for_prompt)
        if len(result) != len(batch):
            print(f"  WARNING: batch mismatch at {start}, got {len(result)} expected {len(batch)} -- skipping")
            continue
        all_propensities.extend(result)
        print(f"  generated {min(start + batch_size, N_RESPONDENTS)}/{N_RESPONDENTS}")

    print("\nComparing calibrated vs. naive baseline against real distributions...")
    results = []
    for i, h in enumerate(held_out_items):
        col_key = f"item_{i+1}"
        raw = np.array([p[col_key] for p in all_propensities])

        true_cats = [k for k in h["true_value_proportions"] if k != "__other__"]
        k_target = len(true_cats)
        if k_target < 2:
            continue

        if h["match"] is not None:
            percentiles = pd.Series(raw).rank(method="first").to_numpy() / len(raw)
            ranks = np.clip(np.searchsorted(h["match"]["cutpoints"], percentiles, side="left") + 1,
                             1, h["match"]["k_real"])
            calibrated = [gen.rescale_rank(r, h["match"]["k_real"], k_target) for r in ranks]
            calibrated_curve = empirical_curve_from_samples(calibrated, k_target)
            error_calibrated = curve_distance(build_cdf_curve(h["true_value_proportions"]), calibrated_curve)
        else:
            error_calibrated = None  # no real anchor available -- pipeline would use population-level fallback

        naive = naive_baseline(raw, k_target)
        naive_curve = empirical_curve_from_samples(naive, k_target)
        error_naive = curve_distance(build_cdf_curve(h["true_value_proportions"]), naive_curve)

        results.append({
            "variable": h["variable"],
            "method": h["match"]["method"] if h["match"] else "none",
            "error_calibrated": error_calibrated,
            "error_naive": error_naive,
        })

    return pd.DataFrame(results)

def print_summary(df):
    matched = df[df["error_calibrated"].notna()]
    print(f"N_RESPONDENTS used: {N_RESPONDENTS}")
    print(f"Items with a real-data anchor available (excluding self): {len(matched)} / {len(df)}\n")

    print(f"Avg error, CALIBRATED pipeline: {matched['error_calibrated'].mean():.4f}")
    print(f"Avg error, NAIVE baseline (same items):  {matched['error_naive'].mean():.4f}")
    improvement = (matched['error_naive'].mean() - matched['error_calibrated'].mean()) / matched['error_naive'].mean() * 100
    print(f"Relative error reduction from calibration: {improvement:.1f}%")

    # Paired test on the same items (naive minus calibrated error)
    diff = matched["error_naive"] - matched["error_calibrated"]
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    try:
        from scipy import stats
        t_stat = diff.mean() / se
        p_val = 2 * stats.t.sf(abs(t_stat), n - 1)
        half = stats.t.ppf(0.975, n - 1) * se
        print(f"Paired t-test: t({n - 1}) = {t_stat:.2f}, p = {p_val:.4f}; "
              f"mean difference = {diff.mean():.3f}, 95% CI [{diff.mean() - half:.3f}, {diff.mean() + half:.3f}]")
        print(f"Items where calibrated beats naive: {(diff > 0).sum()} / {n}")
    except ImportError:
        print("(install scipy to also print the paired t-test)")

    print(f"\nFor reference, naive baseline error across ALL {len(df)} held-out items "
          f"(including those with no real anchor at all): {df['error_naive'].mean():.4f}")

if __name__ == "__main__":
    df = run_comparison()
    out_name = f"naive_baseline_comparison_results_N{N_RESPONDENTS}.csv"
    df.to_csv(os.path.join(_REPO_ROOT, "data", out_name), index=False)
    print(f"\nSaved -> {out_name}\n")
    print_summary(df)
