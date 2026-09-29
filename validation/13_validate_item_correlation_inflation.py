"""
Justification for the "divide generated correlations by ~1.1" note shown
in the web interface.

What was tested
---------------
62 items (16 scales) from an independent, real survey of ~400 working
adults (Prolific) were run through this pipeline (N=400 synthetic
respondents, general employed population, 7-point scale). For every pair
of items from DIFFERENT scales (1,788 pairs), we compare:
    real_r       -- correlation among the real respondents
    generated_r  -- correlation among the synthetic respondents
The pair table is data/item_correlation_comparison.csv (aggregate
item-pair correlations only -- no respondent-level data).

Question answered here
----------------------
1. On average, how much larger are generated correlations than real ones
   (the slope of generated_r on real_r)?
2. Does dividing by that slope actually reduce error -- including for
   items the slope was NOT fitted on (held-out-items test below)?
3. Where does it help and where does it not (breakdown by real_r range)?

Limits (read before citing)
---------------------------
- One generation run (one seed, N=400) on one item set. The slope is a
  population-average estimate, not a per-pair fix (R^2 is ~0.4).
- Pairs share items, so pair-level significance tests are optimistic;
  the held-out-items test below is the more honest check.
- This applies to ITEM-level correlations. A scale-level version was
  tested separately with 3 independent replications and was unstable, so
  no scale-level factor is offered.

Runs offline: reads a CSV, makes no API calls.
Requires: pandas, numpy, scipy (pip install scipy).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
PAIRS_CSV = os.path.join(_REPO_ROOT, "data", "item_correlation_comparison.csv")

pairs = pd.read_csv(PAIRS_CSV)
assert {"item_1", "item_2", "real_r", "generated_r"} <= set(pairs.columns)
print(f"Item pairs (cross-scale): {len(pairs)}")

# ---------------------------------------------------------------------
# 1. The overall relationship
# ---------------------------------------------------------------------
slope, intercept, r_value, _, se = stats.linregress(pairs["real_r"], pairs["generated_r"])
lo, hi = slope - 1.96 * se, slope + 1.96 * se
print("\n=== 1. generated_r vs real_r (all pairs) ===")
print(f"generated_r = {intercept:.3f} + {slope:.3f} * real_r   (R^2 = {r_value**2:.3f})")
print(f"slope 95% CI: [{lo:.3f}, {hi:.3f}]  -> generated correlations run ~"
      f"{(slope-1)*100:.0f}% higher on average (CI {(lo-1)*100:.0f}% to {(hi-1)*100:.0f}%)")

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

# ---------------------------------------------------------------------
# 2a. In-sample check (slope fitted on the same pairs -- optimistic)
# ---------------------------------------------------------------------
adj = pairs["generated_r"] / slope
raw_err = (pairs["real_r"] - pairs["generated_r"]).abs()
adj_err = (pairs["real_r"] - adj).abs()
print("\n=== 2a. Divide by the slope: in-sample (optimistic) ===")
print(f"mean abs error  raw: {raw_err.mean():.4f}   divided: {adj_err.mean():.4f}")
print(f"pairs improved: {(adj_err < raw_err).sum()} / {len(pairs)}")

# ---------------------------------------------------------------------
# 2b. Held-out-ITEMS check: the slope is fitted only on pairs that
#     involve none of the held-out items, then applied to pairs that
#     involve at least one held-out item -- i.e., "new items".
# ---------------------------------------------------------------------
items = sorted(set(pairs["item_1"]) | set(pairs["item_2"]))
rng = np.random.default_rng(0)
N_REPEATS, N_GROUPS = 200, 5
raw_maes, adj_maes, frac_improved, fitted_slopes = [], [], [], []
for _ in range(N_REPEATS):
    shuffled = rng.permutation(items)
    for g in range(N_GROUPS):
        held = set(shuffled[g::N_GROUPS])
        in_held = pairs["item_1"].isin(held) | pairs["item_2"].isin(held)
        train, test = pairs[~in_held], pairs[in_held]
        s_fit = stats.linregress(train["real_r"], train["generated_r"]).slope
        fitted_slopes.append(s_fit)
        raw_maes.append(mae(test["real_r"], test["generated_r"]))
        adj_maes.append(mae(test["real_r"], test["generated_r"] / s_fit))
        frac_improved.append(np.mean(
            (test["real_r"] - test["generated_r"] / s_fit).abs()
            < (test["real_r"] - test["generated_r"]).abs()))
print("\n=== 2b. Held-out-items test (slope fitted WITHOUT the tested items) ===")
print(f"{N_REPEATS} random splits x {N_GROUPS} item groups")
print(f"fitted slope across splits: mean {np.mean(fitted_slopes):.3f}, "
      f"range [{np.min(fitted_slopes):.3f}, {np.max(fitted_slopes):.3f}]")
print(f"mean abs error on held-out pairs  raw: {np.mean(raw_maes):.4f}   "
      f"divided: {np.mean(adj_maes):.4f}")
print(f"share of held-out pairs improved: {np.mean(frac_improved)*100:.1f}%")

# ---------------------------------------------------------------------
# 3. Where does it help? (transparency)
# ---------------------------------------------------------------------
bins = [-1, -0.1, 0.1, 0.3, 1.0]
labels = ["real r < -0.1", "-0.1 to 0.1", "0.1 to 0.3", "> 0.3"]
pairs["band"] = pd.cut(pairs["real_r"], bins=bins, labels=labels, right=False)
print("\n=== 3. By size of the REAL correlation (divide by the overall slope) ===")
print(f"{'range':>14} {'n':>5} {'err raw':>9} {'err divided':>12} {'% improved':>11}")
for lab in labels:
    g = pairs[pairs["band"] == lab]
    if len(g) == 0:
        continue
    e_raw = (g["real_r"] - g["generated_r"]).abs()
    e_adj = (g["real_r"] - g["generated_r"] / slope).abs()
    print(f"{lab:>14} {len(g):>5} {e_raw.mean():>9.4f} {e_adj.mean():>12.4f} "
          f"{(e_adj < e_raw).mean()*100:>10.1f}%")
print("\nThe divisor is a population-average correction. Ranges where 'err divided'"
      "\nis not below 'err raw' are ranges where it should not be trusted.")
