"""
CONSOLIDATED PIPELINE -- STEP 1 of 3

Build the master variable catalog directly from the .sav file's metadata,
excluding administrative/technical variables (weights, IDs, interviewer
paradata, sample/mode/form codes) from the start.

This replaces the earlier build-then-patch approach: no separate cleanup
step needed downstream.

Output: gss_variable_catalog.csv  (variable, label) -- construct-only,
        ready to feed directly into embedding + stats generation.
"""

import re
import csv
import os
import pyreadstat

os.makedirs("data", exist_ok=True)

SAV_PATH = "data/gss7224_r3a.sav"
OUTPUT_CSV = "data/gss_variable_catalog.csv"

# Self-contained admin/paradata exclusion patterns (no dependency on any
# other file -- reproducible from the .sav file alone)
ADMIN_PATTERNS = [
    r"^wt[a-z0-9_]*$",          # weight variables: wtss, wtssall, wtssps, etc.
    r"weight",
    r"^id$", r"^year$",
    r"^ballot", r"^form$", r"^formwt", r"^version$",
    r"^sample", r"^sampcode$", r"^phase$", r"^subsamprate$", r"^batch$",
    r"^baselinestatus$", r"^amerstatus$",
    r"^vpsu$", r"^vstrat$",
    r"^mode$", r"^modesequence$",
    r"^oversamp$",
    r"^int[a-z]*$",              # interviewer characteristics
    r"^filever", r"^devtype$",
    r"^consent$", r"^adminconsent$",
    r"^coop$", r"^comprend$", r"^huclean$", r"^hlthstrt$",
    r"^huadd", r"^dwellpre$", r"^incuspop$",
    r"^feeused$", r"^feelevel$", r"^totalincentive$",
    r"^lngthinv$", r"^dateintv",
]
ADMIN_RE = re.compile("|".join(ADMIN_PATTERNS))

def is_admin(varname):
    return bool(ADMIN_RE.search(varname.lower()))

print(f"Reading metadata from {SAV_PATH} ...")
_, meta = pyreadstat.read_sav(SAV_PATH, metadataonly=True)
print(f"Total variables in file: {len(meta.column_names)}")

kept = 0
excluded = 0
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["variable", "label"])
    for v in meta.column_names:
        label = (meta.column_names_to_labels.get(v) or "").strip()
        if not label:
            continue
        if is_admin(v):
            excluded += 1
            continue
        w.writerow([v, label])
        kept += 1

print(f"\nKept {kept} construct variables, excluded {excluded} admin/paradata variables.")
print(f"Saved -> {OUTPUT_CSV}")
