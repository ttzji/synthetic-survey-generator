"""
Build a continuous confidence score (0-1) per match, calibrated against
your ACTUAL validation data (validation_results.csv), rather than an
arbitrary hand-picked threshold.

Method: fit a simple linear regression predicting cdf_distance_error from
similarity and log(min(n_i, n_j)). Convert predicted error into a 0-1
confidence score by comparing it to the observed range of errors in your
validation set. This is a modest amount of extra code since the hard part
(the validation dataset) already exists.

Output: a reusable function `confidence_score(similarity, n_i, n_j)` you
can import into generate_dataset.py to attach a score to every item,
instead of (or alongside) the pass/fail threshold.
"""

import numpy as np
import pandas as pd

def fit_confidence_model(validation_csv="data/validation_results.csv"):
    df = pd.read_csv(validation_csv)
    df = df[(df["n_i"] > 0) & (df["n_j"] > 0)].copy()
    df["log_min_n"] = np.log(np.minimum(df["n_i"], df["n_j"]))

    # Simple linear regression via numpy (no sklearn dependency needed)
    X = np.column_stack([np.ones(len(df)), df["similarity"], df["log_min_n"]])
    y = df["cdf_distance_error"].to_numpy()
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    residuals = y - X @ coeffs
    r_squared = 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)

    error_min, error_max = y.min(), y.max()
    print(f"Fitted model: error = {coeffs[0]:.4f} + {coeffs[1]:.4f}*similarity + {coeffs[2]:.4f}*log_min_n")
    print(f"R-squared: {r_squared:.3f}")
    print(f"(A low R-squared confirms what we already found: these features "
          f"only weakly predict error. Report this honestly -- it's a real "
          f"finding about the tool's limits, not a fitting failure.)")

    return {"coeffs": coeffs, "error_min": error_min, "error_max": error_max}

def confidence_score(similarity, n_i, n_j, model):
    """Returns a 0-1 confidence score: 1 = predicted error at the low end
    of what we've observed empirically, 0 = at the high end."""
    log_min_n = np.log(max(min(n_i, n_j), 1))
    x = np.array([1, similarity, log_min_n])
    predicted_error = x @ model["coeffs"]
    predicted_error = np.clip(predicted_error, model["error_min"], model["error_max"])
    score = 1 - (predicted_error - model["error_min"]) / (model["error_max"] - model["error_min"])
    return float(np.clip(score, 0, 1))

if __name__ == "__main__":
    model = fit_confidence_model()

    print("\nExample confidence scores:")
    examples = [
        (0.90, 5000, 5000),
        (0.70, 1000, 1000),
        (0.60, 500, 500),
        (0.55, 200, 200),
        (0.50, 100, 100),
    ]
    for sim, n_i, n_j in examples:
        score = confidence_score(sim, n_i, n_j, model)
        print(f"  similarity={sim}, n_i={n_i}, n_j={n_j} -> confidence={score:.3f}")
