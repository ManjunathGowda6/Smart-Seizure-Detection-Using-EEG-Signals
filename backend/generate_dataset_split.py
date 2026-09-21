"""
generate_dataset_split.py
=========================
Scans the CHB-MIT dataset folder, parses each patient's summary .txt file,
and produces E:\\Downloads\\Finalyearproject\\dataset_split.csv listing every
EDF file with seizure metadata.

No files are moved or copied. No model inference is run.
"""

import os
import re
import csv

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = r"E:\Downloads\chb-mit-scalp-eeg-database-1.0.0"
OUTPUT_CSV = r"E:\Downloads\Finalyearproject\dataset_split.csv"

# ── CSV column headers ─────────────────────────────────────────────────────────
FIELDNAMES = [
    "patient",
    "filename",
    "full_path",
    "seizure_count",
    "label",
    "seizure_start_seconds",
    "seizure_end_seconds",
]


def parse_summary(patient_dir: str, patient_id: str) -> list[dict]:
    """
    Parse chbXX-summary.txt and return a list of file-record dicts.
    Each dict has: filename, full_path, seizure_count,
                   seizure_starts (list[int]), seizure_ends (list[int])
    """
    summary_path = os.path.join(patient_dir, f"{patient_id}-summary.txt")
    if not os.path.exists(summary_path):
        print(f"  [WARN] No summary file found: {summary_path}")
        return []

    with open(summary_path, "r", errors="ignore") as fh:
        content = fh.read()

    records = []
    # Split on "File Name:" blocks
    blocks = content.split("File Name:")
    for block in blocks[1:]:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        fname = lines[0].strip()
        # Must end in .edf
        if not fname.lower().endswith(".edf"):
            continue

        # Number of seizures
        ns_match = re.search(r"Number of Seizures in File\s*:\s*(\d+)", block, re.IGNORECASE)
        n_seiz = int(ns_match.group(1)) if ns_match else 0

        starts, ends = [], []
        if n_seiz > 0:
            start_matches = re.findall(r"Seizure(?:\s+\d+)?\s+Start\s+Time\s*:\s*(\d+)\s*seconds?", block, re.IGNORECASE)
            end_matches   = re.findall(r"Seizure(?:\s+\d+)?\s+End\s+Time\s*:\s*(\d+)\s*seconds?",   block, re.IGNORECASE)
            starts = [int(s) for s in start_matches]
            ends   = [int(e) for e in end_matches]
            # Reconcile count — summary may list a different number than n_seiz
            if len(starts) != n_seiz or len(ends) != n_seiz:
                n_seiz = min(len(starts), len(ends))
                starts = starts[:n_seiz]
                ends   = ends[:n_seiz]

        full_path = os.path.join(patient_dir, fname)
        records.append({
            "filename":       fname,
            "full_path":      full_path,
            "seizure_count":  n_seiz,
            "seizure_starts": starts,
            "seizure_ends":   ends,
        })

    return records


def main():
    # Discover patient directories (sorted for reproducibility)
    patients = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and re.match(r"chb\d+", d)
    ])

    all_rows = []

    for patient_id in patients:
        patient_dir = os.path.join(DATA_DIR, patient_id)
        records = parse_summary(patient_dir, patient_id)

        if not records:
            # Fallback: list EDF files even without a parseable summary
            for fname in sorted(os.listdir(patient_dir)):
                if fname.lower().endswith(".edf"):
                    records.append({
                        "filename":       fname,
                        "full_path":      os.path.join(patient_dir, fname),
                        "seizure_count":  0,
                        "seizure_starts": [],
                        "seizure_ends":   [],
                    })

        for rec in records:
            # Only include the file if it actually exists on disk
            if not os.path.exists(rec["full_path"]):
                print(f"  [SKIP] File not on disk: {rec['full_path']}")
                continue

            all_rows.append({
                "patient":               patient_id,
                "filename":              rec["filename"],
                "full_path":             rec["full_path"],
                "seizure_count":         rec["seizure_count"],
                "label":                 "SEIZURE" if rec["seizure_count"] > 0 else "NORMAL",
                "seizure_start_seconds": ",".join(str(s) for s in rec["seizure_starts"]),
                "seizure_end_seconds":   ",".join(str(e) for e in rec["seizure_ends"]),
            })

        print(f"  [{patient_id}]  {len(records)} records parsed from summary")

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary statistics
    total        = len(all_rows)
    seizure_cnt  = sum(1 for r in all_rows if r["label"] == "SEIZURE")
    normal_cnt   = total - seizure_cnt
    patients_cov = len(set(r["patient"] for r in all_rows))

    print()
    print("=" * 50)
    print(f"  Total EDF files found : {total}")
    print(f"  Seizure files         : {seizure_cnt}")
    print(f"  Normal files          : {normal_cnt}")
    print(f"  Patients covered      : {patients_cov}")
    print("=" * 50)
    print(f"\n  CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
