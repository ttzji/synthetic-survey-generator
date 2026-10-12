"""
STEP 05 (replaces old 04): Build the semantic embedding index, but ONLY for
variables that step 04 confirmed have usable real data (no error entry in
gss_master_variable_stats.json). This guarantees retrieval can never
surface a variable with zero real respondents behind it.

Requirements:
    pip install openai numpy
    OPENAI_API_KEY environment variable must be set.

Input:
    gss_variable_catalog.csv          (from step 03)
    gss_master_variable_stats.json    (from step 04)

Output:
    gss_variable_embeddings.npy
    gss_variable_index.csv
"""

import csv
import json
import time
import numpy as np
import os
from openai import OpenAI

os.makedirs("data", exist_ok=True)

CATALOG_CSV = "data/gss_variable_catalog.csv"
STATS_JSON = "data/gss_master_variable_stats.json"
EMBEDDINGS_OUT = "data/gss_variable_embeddings.npy"
INDEX_OUT = "data/gss_variable_index.csv"
MODEL = "text-embedding-3-small"
BATCH_SIZE = 500

client = OpenAI()

print("Loading catalog...")
all_variables, all_labels = [], []
with open(CATALOG_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        all_variables.append(row["variable"].strip())
        all_labels.append(row["label"].strip())
print(f"  {len(all_variables)} variables in catalog")

print("Loading stats to filter out variables with no usable data...")
with open(STATS_JSON, encoding="utf-8") as f:
    stats = json.load(f)
usable = {k for k, v in stats.items() if "error" not in v}

variables, labels = [], []
for v, l in zip(all_variables, all_labels):
    if v.lower() in usable:
        variables.append(v)
        labels.append(l)

print(f"  Embedding {len(variables)} / {len(all_variables)} variables "
      f"(excluded {len(all_variables) - len(variables)} with no usable data)")

all_embeddings = []
for i in range(0, len(labels), BATCH_SIZE):
    batch = labels[i:i + BATCH_SIZE]
    response = client.embeddings.create(model=MODEL, input=batch)
    all_embeddings.extend([item.embedding for item in response.data])
    print(f"  embedded {min(i + BATCH_SIZE, len(labels))}/{len(labels)}")
    time.sleep(0.2)

embeddings = np.array(all_embeddings, dtype=np.float32)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

np.save(EMBEDDINGS_OUT, embeddings)
with open(INDEX_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["variable", "label"])
    for v, l in zip(variables, labels):
        w.writerow([v, l])

print(f"\nSaved {embeddings.shape[0]} embeddings -> {EMBEDDINGS_OUT}")
print(f"Saved index -> {INDEX_OUT}")
