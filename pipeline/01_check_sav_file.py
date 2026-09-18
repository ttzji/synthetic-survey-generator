"""
STEP 0: Quick diagnostic. Run this first in any new setup to confirm your
.sav file path is correct and to check how variable names are cased in
your specific file (GSS releases have varied on this).

Reads ONLY metadata -- fast, no full data load.
"""

import pyreadstat

SAV_PATH = "data/gss7224_r3a.sav"  # <-- change to your actual .sav path

_, meta = pyreadstat.read_sav(SAV_PATH, metadataonly=True)

print(f"File opened successfully: {SAV_PATH}")
print(f"Total variables in file: {len(meta.column_names)}\n")

print("First 10 variable names as stored in the file:")
for v in meta.column_names[:10]:
    print(f"  '{v}'")

print("\nCase check for common core variables:")
for candidate in ["year", "YEAR", "id", "ID", "age", "AGE", "wtssall", "WTSSALL"]:
    present = candidate in meta.column_names
    print(f"  '{candidate}': {'FOUND' if present else 'not found'}")

print("\nIf variables show as lowercase 'FOUND' but not uppercase, that's normal --")
print("all later scripts already handle this automatically via case-insensitive matching.")
