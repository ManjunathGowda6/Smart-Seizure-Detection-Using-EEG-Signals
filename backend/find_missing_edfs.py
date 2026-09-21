"""
find_missing_edfs.py
====================
Scans E:\\Downloads\\chb-mit-scalp-eeg-database-1.0.0\\ for all .edf files
physically on disk, compares against dataset_split.csv, finds missing entries,
and saves an updated dataset_split_complete.csv with UNKNOWN rows appended.
"""

import os
import csv
import re

DATA_DIR     = r"E:\Downloads\chb-mit-scalp-eeg-database-1.0.0"
INPUT_CSV    = r"E:\Downloads\Finalyearproject\dataset_split.csv"
OUTPUT_CSV   = r"E:\Downloads\Finalyearproject\dataset_split_complete.csv"

FIELDNAMES = [
    "patient",
    "filename",
    "full_path",
    "seizure_count",
    "label",
    "seizure_start_seconds",
    "seizure_end_seconds",
]

# ── 1. Collect all .edf files physically on disk ──────────────────────────────
print("Scanning disk ...")
disk_files = {}   # filename → full_path  (use filename as key; collision-safe with full_path)
for root, dirs, files in os.walk(DATA_DIR):
    dirs.sort()
    for f in sorted(files):
        if f.lower().endswith(".edf"):
            full_path = os.path.join(root, f)
            disk_files[full_path] = f   # keyed by full path to be collision-safe

disk_paths = set(disk_files.keys())
print(f"  EDF files on disk: {len(disk_paths)}")

# ── 2. Load existing CSV ──────────────────────────────────────────────────────
csv_rows   = []
csv_paths  = set()

with open(INPUT_CSV, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        csv_rows.append(row)
        csv_paths.add(row["full_path"])

print(f"  Files in CSV     : {len(csv_rows)}")

# ── 3. Find missing files (on disk but not in CSV) ───────────────────────────
missing_paths = sorted(disk_paths - csv_paths)
print(f"  Missing files    : {len(missing_paths)}")

# ── 4. Print missing file list ────────────────────────────────────────────────
print()
print("Missing file list:")
print("-" * 60)

missing_rows = []
for full_path in missing_paths:
    fname = os.path.basename(full_path)
    # Derive patient from parent folder name
    patient = os.path.basename(os.path.dirname(full_path))
    print(f"  {fname:<30}  — patient {patient}")
    missing_rows.append({
        "patient":               patient,
        "filename":              fname,
        "full_path":             full_path,
        "seizure_count":         -1,
        "label":                 "UNKNOWN",
        "seizure_start_seconds": "",
        "seizure_end_seconds":   "",
    })

# ── 5. Print summary ──────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"  Total EDF files on disk     : {len(disk_paths)}")
print(f"  Total files in CSV          : {len(csv_rows)}")
print(f"  Missing files (not in CSV)  : {len(missing_paths)}")
print("=" * 55)

# ── 6. Write complete CSV (original + missing) ────────────────────────────────
all_rows = csv_rows + missing_rows

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\n  Complete CSV saved to: {OUTPUT_CSV}")
print(f"  Total rows in new CSV: {len(all_rows)}")
print(f"  Original CSV unchanged: {INPUT_CSV}")
