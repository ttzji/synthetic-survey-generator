"""
STEP 09: Extract a POOL of real GSS respondents' joint demographic profiles
(age, sex, race, marital status, employment status, religion, region, etc.)
-- not just single-variable marginals. This pool is what "general
representative sample" and "custom sample" modes both resample from,
which is what preserves REAL correlations between demographics (e.g.
married people skew older) instead of assuming independence.

Also saves each variable's value labels (e.g. sex=1 -> "Male"), so filters
can be defined by matching label text rather than hardcoded numeric codes
-- more robust across GSS releases.

Output:
    gss_demographics_pool.csv          (one row per real respondent)
    gss_demographics_value_labels.json (variable -> {code: label})
"""

import json
import pyreadstat
import os
import pandas as pd

os.makedirs("data", exist_ok=True)

SAV_PATH = "data/gss7224_r3a.sav"

# Demographic + filter-relevant variables to pull. Extend this list if you
# add more filter options later (e.g. "educ" for education-based filters).
DEMO_VARS = ["year", "id", "wtssall", "age", "sex", "race",
             "marital", "wrkstat", "relig", "region", "educ"]

print("Reading metadata...")
_, meta = pyreadstat.read_sav(SAV_PATH, metadataonly=True)
file_cols = {c.lower(): c for c in meta.column_names}
resolved = [file_cols[v] for v in DEMO_VARS if v in file_cols]
missing = [v for v in DEMO_VARS if v not in file_cols]
print(f"Resolved {len(resolved)} columns, missing: {missing}")

print(f"\nReading data from {SAV_PATH} ...")
df, _ = pyreadstat.read_sav(SAV_PATH, usecols=resolved)
df.columns = [c.lower() for c in df.columns]

# Keep only respondents with a valid weight (needed for representative resampling)
df = df[df["wtssall"].notna() & (df["wtssall"] > 0)]
print(f"Rows with valid weight: {len(df)}")

df.to_csv("data/gss_demographics_pool.csv", index=False)
print("Saved -> gss_demographics_pool.csv")

# Save value labels for each demographic variable (for filter matching)
value_labels = {}
for v in DEMO_VARS:
    actual_name = file_cols.get(v)
    if actual_name and actual_name in meta.variable_value_labels:
        value_labels[v] = {str(k): lbl for k, lbl in meta.variable_value_labels[actual_name].items()}

with open("data/gss_demographics_value_labels.json", "w", encoding="utf-8") as f:
    json.dump(value_labels, f, indent=2)
print("Saved -> gss_demographics_value_labels.json")

print("\nSample value labels found:")
for v, labels in value_labels.items():
    print(f"  {v}: {labels}")
