"""
predict_unknown_chb24.py
========================
Runs CNN-BiLSTM inference on the 10 UNKNOWN chb24 EDF files from
dataset_split_complete.csv, replaces their labels with model predictions,
and adds a 'source' column to the full CSV.

No retraining. No files moved.
"""

import os
import sys
import csv

# ── Paths ──────────────────────────────────────────────────────────────────────
backend_dir  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

COMPLETE_CSV = r"E:\Downloads\Finalyearproject\dataset_split_complete.csv"
OUTPUT_CSV   = COMPLETE_CSV   # overwrite in place

# ── Load predict pipeline ──────────────────────────────────────────────────────
from ml.predict import predict_eeg_file

FIELDNAMES = [
    "patient",
    "filename",
    "full_path",
    "seizure_count",
    "label",
    "seizure_start_seconds",
    "seizure_end_seconds",
    "source",           # NEW column
]

# ── Load CSV ───────────────────────────────────────────────────────────────────
rows = []
with open(COMPLETE_CSV, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        rows.append(row)

# Separate UNKNOWN rows
unknown_rows    = [r for r in rows if r["label"] == "UNKNOWN"]
annotated_rows  = [r for r in rows if r["label"] != "UNKNOWN"]

print("=" * 60)
print("  CNN-BiLSTM Inference — UNKNOWN chb24 Files")
print("=" * 60)
print(f"  Files to predict: {len(unknown_rows)}")
print()

# ── Run inference ──────────────────────────────────────────────────────────────
predicted_seizure = 0
predicted_normal  = 0

for row in unknown_rows:
    filepath = row["full_path"]
    fname    = row["filename"]

    print(f"File         : {fname}")

    if not os.path.exists(filepath):
        print(f"  ERROR: File not found on disk — skipping")
        row["source"] = "model_predicted"
        row["label"]  = "FILE_MISSING"
        print()
        continue

    try:
        results = predict_eeg_file(filepath, model_name="cnn_bilstm", end_sec=None)

        pred     = results["prediction"]          # "seizure" / "no_seizure"
        conf     = results["confidence_score"]
        segment  = results["seizure_segment"]
        tot_wins = results.get("total_windows", "?")
        sz_wins  = results.get("seizure_wins", "?")

        # Derive longest cluster from the print output (already printed inside
        # predict_eeg_file). We re-compute it here from seizure_wins + segment
        # as a proxy display (actual value is printed inside predict_eeg_file).
        label = "SEIZURE" if pred == "seizure" else "NORMAL"

        if pred == "seizure":
            predicted_seizure += 1
            start_s = f"{segment['start']:.1f}s" if segment and segment["start"] is not None else "?"
            end_s   = f"{segment['end']:.1f}s"   if segment and segment["end"]   is not None else "?"
            interval = f"{start_s} - {end_s}"
        else:
            predicted_normal += 1
            interval = "none"

        print(f"Prediction   : {label}")
        print(f"Confidence   : {conf:.1f}%")
        print(f"Seizure interval: {interval}")
        print(f"Flagged windows : {sz_wins}/{tot_wins}")

        # Update row
        row["label"]  = label
        row["source"] = "model_predicted"

    except Exception as exc:
        print(f"  ERROR during inference: {exc}")
        import traceback; traceback.print_exc()
        row["label"]  = "INFERENCE_ERROR"
        row["source"] = "model_predicted"

    print()

# ── Tag annotated rows with source ────────────────────────────────────────────
for row in annotated_rows:
    row["source"] = "annotated"

# ── Summary ────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  UNKNOWN files predicted as SEIZURE  : {predicted_seizure}")
print(f"  UNKNOWN files predicted as NORMAL   : {predicted_normal}")
print("=" * 60)

# ── Write updated CSV ──────────────────────────────────────────────────────────
all_rows = annotated_rows + unknown_rows   # keep original order roughly

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\n  Updated CSV saved to : {OUTPUT_CSV}")
print(f"  Total rows           : {len(all_rows)}")
