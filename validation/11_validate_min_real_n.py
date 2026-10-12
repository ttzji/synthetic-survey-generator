"""
Is a minimum of 200 real respondents (MIN_REAL_N in generate_dataset.py) a sensible floor for using a GSS
variable as an anchor?

Why it matters. The tool copies a GSS variable's answer distribution onto the synthetic data. That distribution
was computed from only the respondents who were asked the question. If that group is small, the distribution is
noisy and the tool would copy the noise. Below some sample size an anchor is not worth using. Until this script,
200 was a round number that nothing had tested.

What it does.
  1. Takes five large GSS questions (happy, satjob, trust, life, health).
  2. Draws 100 random subsamples at each size from 50 to 500 respondents.
  3. Measures how far each subsample's answer distribution drifts from the full-sample distribution (the same
     distance measure the other validation scripts use). This drift is the "sampling noise" of an anchor of
     that size.
  4. Expresses that noise as a share of the benefit of calibration: the mean error of raw model output minus the
     mean error of the calibrated pipeline, from the four runs of script 10. A noise level that is a small
     fraction of that benefit leaves the benefit intact.

What it can and cannot show. It shows whether 200 is defensible. It does not show that 200 is optimal: the
curve has no sharp elbow, so any floor between roughly 150 and 300 is defensible.

Needs: the raw GSS file at data/gss7224_r3a.sav (from NORC, https://gss.norc.org/get-the-data; not in this
repository) and the result files of script 10 (validation/results/naive_baseline_comparison_results_N*.csv).
No model calls. Run from the repository root:  python validation/11_validate_min_real_n.py
"""

import os

import numpy as np
import pandas as pd
import pyreadstat

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
RESULTS_DIR = os.path.join(_REPO_ROOT, "validation", "results")

CANDIDATE_N = [50, 100, 150, 200, 250, 300, 400, 500]
N_REPEATS = 100  # random subsamples drawn per (variable, candidate N) combination
GSS_SAV_PATH = os.path.join(_REPO_ROOT, "data", "gss7224_r3a.sav")  # same file the pipeline/ scripts use
GSS_TEST_VARS = ["happy", "satjob", "trust", "life", "health"]  # large-N, well-known GSS items


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


def curve_distance(curve_a, curve_b):
    return float(np.mean(np.abs(curve_a - curve_b)))


def proportions_from_values(values):
    counts = pd.Series(values).value_counts(normalize=True)
    return {str(k): v for k, v in counts.items()}


def subsampling_error_curve(full_values, candidate_ns, n_repeats, rng):
    """For a variable with all its real respondent values in full_values, returns {candidate_n: mean error across
    repeats}, where error is how far a random subsample of that size drifts from the TRUE full-sample shape."""
    true_curve = build_cdf_curve(proportions_from_values(full_values))
    n_available = len(full_values)
    out = {}
    for n in candidate_ns:
        if n > n_available:
            continue
        errors = []
        for _ in range(n_repeats):
            sub = rng.choice(full_values, size=n, replace=False)
            errors.append(curve_distance(true_curve, build_cdf_curve(proportions_from_values(sub))))
        out[n] = np.mean(errors)
    return out


def main():
    print("=" * 78)
    print("Sampling noise of an anchor, by number of real respondents (GSS)")
    print("=" * 78)
    try:
        _, meta_check = pyreadstat.read_sav(GSS_SAV_PATH, metadataonly=True)
    except FileNotFoundError:
        raise SystemExit(f"GSS file not found at {GSS_SAV_PATH}. Download the GSS cumulative .sav from NORC and "
                         f"place it there (see the README), then rerun.")
    available = {c.lower() for c in meta_check.column_names}
    usable = [v for v in GSS_TEST_VARS if v.lower() in available]
    if len(usable) < len(GSS_TEST_VARS):
        print(f"  NOTE: not found in this release, skipped: {[v for v in GSS_TEST_VARS if v.lower() not in available]}")
    df, _ = pyreadstat.read_sav(GSS_SAV_PATH, usecols=usable)

    rng = np.random.default_rng(42)
    rows = []
    for var in usable:
        full_values = df[var].dropna().to_numpy()
        full_values = full_values[~np.isnan(full_values)]
        if len(full_values) < max(CANDIDATE_N):
            print(f"  {var}: only {len(full_values)} real respondents, skipped (need at least {max(CANDIDATE_N)})")
            continue
        curve = subsampling_error_curve(full_values, CANDIDATE_N, N_REPEATS, rng)
        print(f"  {var} (N={len(full_values)} in total): " + ", ".join(f"n={n}: {e:.4f}" for n, e in curve.items()))
        rows += [{"variable": var, "candidate_n": n, "mean_error": e} for n, e in curve.items()]
    if not rows:
        raise SystemExit("No usable variables; nothing to report.")

    results = pd.DataFrame(rows)
    noise = results.groupby("candidate_n")["mean_error"].mean()

    # ---- the yardstick: how much does calibration help, according to script 10?
    benefits = []
    for n in (100, 300, 500, 1000):
        path = os.path.join(RESULTS_DIR, f"naive_baseline_comparison_results_N{n}.csv")
        if os.path.exists(path):
            d = pd.read_csv(path).dropna(subset=["error_calibrated", "error_naive"])
            benefits.append((d["error_naive"] - d["error_calibrated"]).mean())
    summary = pd.DataFrame({"mean_noise_error": noise})
    if benefits:
        benefit = float(np.mean(benefits))
        summary["noise_as_share_of_benefit"] = noise / benefit
        print(f"\nBenefit of calibration (raw-model error minus calibrated error, mean of {len(benefits)} runs "
              f"of script 10): {benefit:.4f}")
    else:
        print("\nResults of script 10 not found in validation/results/; reporting noise only.")
    print("\nSampling noise of an anchor by number of real respondents:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = summary.reset_index()
    out.to_csv(os.path.join(RESULTS_DIR, "min_real_n_validation_results.csv"), index=False)
    results.to_csv(os.path.join(RESULTS_DIR, "min_real_n_validation_by_variable.csv"), index=False)
    print("\nSaved: validation/results/min_real_n_validation_results.csv and min_real_n_validation_by_variable.csv")
    print("\nCurrent MIN_REAL_N = 200. Look at where the noise stops falling much and at what share of the benefit it")
    print("leaves behind: that is what makes a floor defensible. This does not show 200 is optimal.")


if __name__ == "__main__":
    main()
