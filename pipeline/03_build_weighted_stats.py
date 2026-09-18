"""
STEP 04 (replaces old 05): Compute REAL, WEIGHTED response statistics for
every variable in the clean catalog, using GSS's survey weight (wtssall)
so results reflect the population-representative distribution.

Runs BEFORE the embedding step, so that step 05 can filter out variables
with no usable data before ever embedding them.

For high-cardinality variables, value_proportions is capped to the top 20
categories plus an "other" bucket.

Output: gss_master_variable_stats.json
"""

import csv
import json
import time
import numpy as np
import pandas as pd
import os
import pyreadstat

os.makedirs("data", exist_ok=True)

SAV_PATH = "data/gss7224_r3a.sav"
CATALOG_CSV = "data/gss_variable_catalog.csv"
OUTPUT_JSON = "data/gss_master_variable_stats.json"
MAX_CATEGORIES = 20
WEIGHT_VAR = "wtssall"

CORE_VARS = ["year", "id", WEIGHT_VAR, "age", "sex", "race", "educ"]

def weighted_stats(values, weights):
    """values, weights: numpy arrays, same length, no NaNs."""
    total_w = weights.sum()
    w_mean = float(np.sum(values * weights) / total_w)
    w_var = float(np.sum(weights * (values - w_mean) ** 2) / total_w)
    w_std = float(np.sqrt(w_var))
    return w_mean, w_std

def weighted_value_proportions(values, weights, max_categories=MAX_CATEGORIES):
    df = pd.DataFrame({"value": values, "weight": weights})
    grouped = df.groupby("value")["weight"].sum().sort_values(ascending=False)
    total_w = grouped.sum()
    props = grouped / total_w
    if len(props) > max_categories:
        top = props.iloc[:max_categories]
        other_mass = 1.0 - top.sum()
        result = {str(k): float(v) for k, v in top.items()}
        result["__other__"] = float(other_mass)
        return result, len(props), True
    else:
        return {str(k): float(v) for k, v in props.items()}, len(props), False

def build():
    t0 = time.time()

    print("Loading catalog...")
    target_vars = []
    with open(CATALOG_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target_vars.append(row["variable"].strip().lower())
    print(f"  {len(target_vars)} variables")

    print("Resolving column names against file...")
    _, meta = pyreadstat.read_sav(SAV_PATH, metadataonly=True)
    file_cols = {c.lower(): c for c in meta.column_names}
    needed = set(target_vars) | set(CORE_VARS)
    resolved = [file_cols[v] for v in needed if v in file_cols]
    missing = [v for v in needed if v not in file_cols]
    print(f"  {len(resolved)} resolved, {len(missing)} not found")

    print(f"\nReading full data from {SAV_PATH} ...")
    df, _ = pyreadstat.read_sav(SAV_PATH, usecols=resolved)
    df.columns = [c.lower() for c in df.columns]
    print(f"  Loaded shape: {df.shape} ({time.time()-t0:.0f}s elapsed)")

    if WEIGHT_VAR not in df.columns:
        raise RuntimeError(f"Weight variable '{WEIGHT_VAR}' not found -- cannot compute weighted stats.")

    print("\nComputing weighted stats per variable...")
    results = {}
    n_vars = len(target_vars)
    t1 = time.time()

    for i, v in enumerate(target_vars):
        if v not in df.columns:
            results[v] = {"error": "not found in file"}
            continue

        sub = df[[v, WEIGHT_VAR, "year"]].dropna()
        sub = sub[sub[WEIGHT_VAR] > 0]
        n = len(sub)
        if n == 0:
            results[v] = {"error": "no non-missing weighted data"}
            continue

        values = sub[v].to_numpy(dtype=float) if pd.api.types.is_numeric_dtype(sub[v]) else None
        weights = sub[WEIGHT_VAR].to_numpy(dtype=float)
        years = sorted(sub["year"].unique().tolist())

        entry = {"n": int(n), "years_fielded": years, "weighted": True}

        if values is not None:
            w_mean, w_std = weighted_stats(values, weights)
            entry["mean"] = w_mean
            entry["std"] = w_std
            entry["min"] = float(sub[v].min())
            entry["max"] = float(sub[v].max())
            props, n_cat, truncated = weighted_value_proportions(sub[v], weights)
        else:
            props, n_cat, truncated = weighted_value_proportions(sub[v], weights)

        entry["value_proportions"] = props
        entry["n_categories_total"] = n_cat
        entry["truncated"] = truncated
        results[v] = entry

        if (i + 1) % 500 == 0 or (i + 1) == n_vars:
            print(f"  {i+1}/{n_vars} processed ({time.time()-t1:.0f}s)")

    print(f"\nSaving -> {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")
    print(f"Usable: {sum(1 for r in results.values() if 'error' not in r)}")
    print(f"Errors: {sum(1 for r in results.values() if 'error' in r)}")

if __name__ == "__main__":
    build()
