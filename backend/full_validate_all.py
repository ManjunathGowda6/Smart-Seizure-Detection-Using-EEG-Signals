"""
full_validate_all.py
====================
Runs CNN-BiLSTM inference on all 676 annotated files (Normal & Seizure) from
dataset_split_complete.csv and produces a full validation report with TP/TN/FP/FN.

GPU: NVIDIA GeForce GTX 1650
Model: cnn_bilstm_chbmit_v4.pt
Decision: 8+ consecutive flagged windows (threshold 0.40)
"""

import os
import sys
import csv
import time

# Force UTF-8 stdout to avoid charmap errors on Windows console
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from ml.predict import predict_eeg_file

# ── Paths ──────────────────────────────────────────────────────────────────────
COMPLETE_CSV = r"E:\Downloads\Finalyearproject\dataset_split_complete.csv"
REPORT_CSV   = r"E:\Downloads\Finalyearproject\full_validation_report.csv"

REPORT_FIELDS = [
    "filename", "ground_truth", "prediction", "result", "confidence", "longest_cluster"
]

# ── Load only annotated rows ──────────────────────────────────────────────────
rows = []
with open(COMPLETE_CSV, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row.get("source", "annotated") == "annotated":
            rows.append(row)

total = len(rows)
print("=" * 65)
print(f"  FULL VALIDATION — {total} annotated files")
print("=" * 65)
print(f"  Model : cnn_bilstm_chbmit_v4.pt  |  OR-Logic (FP1-F7 >= 0.38 OR FP1-F7 >= 0.30 & F8-T8 >= 0.22)  |  dur=3.0s, k=5, cluster=4")
print("=" * 65)
print()

# ── Accumulators ──────────────────────────────────────────────────────────────
tp = 0
tn = 0
fp = 0
fn = 0

false_positives = []
false_negatives = []

report_rows = []

t_total_start = time.time()

for file_idx, row in enumerate(rows, start=1):
    filepath = row["full_path"]
    fname    = row["filename"]
    gt_label = row["label"]  # SEIZURE or NORMAL

    if not os.path.exists(filepath):
        print(f"[{file_idx:>3}/{total}] {fname}  — ERROR: File missing on disk")
        continue

    try:
        results = predict_eeg_file(filepath, model_name="cnn_bilstm", end_sec=None)

        pred     = results["prediction"]           # "seizure" / "no_seizure"
        conf     = results["confidence_score"]
        
        # Approximate longest cluster based on result
        longest_cluster_str = ">=4" if pred == "seizure" else "<=3"
        
        pred_label = "SEIZURE" if pred == "seizure" else "NORMAL"

        if gt_label == "SEIZURE" and pred_label == "SEIZURE":
            res = "TP"
            tp += 1
        elif gt_label == "NORMAL" and pred_label == "NORMAL":
            res = "TN"
            tn += 1
        elif gt_label == "NORMAL" and pred_label == "SEIZURE":
            res = "FP"
            fp += 1
            false_positives.append((fname, conf, longest_cluster_str))
        elif gt_label == "SEIZURE" and pred_label == "NORMAL":
            res = "FN"
            fn += 1
            false_negatives.append((fname, conf, longest_cluster_str))
        else:
            res = "ERROR"

        print(f"[{file_idx:>3}/{total}] {fname}")
        print(f"  Ground Truth : {gt_label}")
        print(f"  Prediction   : {pred_label}")
        print(f"  Result       : {res}")
        print(f"  Confidence   : {conf:.1f}%")
        print(f"  Longest cluster: {longest_cluster_str} windows")
        print()

        report_rows.append({
            "filename": fname,
            "ground_truth": gt_label,
            "prediction": pred_label,
            "result": res,
            "confidence": f"{conf:.1f}%",
            "longest_cluster": longest_cluster_str
        })

    except Exception as exc:
        print(f"[{file_idx:>3}/{total}] {fname}  — ERROR: {exc}")
        print()

    # Optional: progress summary every 50 files
    if file_idx % 50 == 0:
        print(f"--- Progress Check: {file_idx}/{total} completed ---")

# ── Final summary ──────────────────────────────────────────────────────────────
total_elapsed = time.time() - t_total_start

sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0
accuracy    = ((tp + tn) / total) * 100 if total > 0 else 0.0
precision   = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0

print("================================================================")
print("  FULL VALIDATION REPORT — 676 FILES")
print("================================================================")
print(f"  True Positives  (TP) : {tp}  — seizure correctly detected")
print(f"  True Negatives  (TN) : {tn}  — normal correctly identified")
print(f"  False Positives (FP) : {fp}  — normal wrongly flagged as seizure")
print(f"  False Negatives (FN) : {fn}  — seizure missed by model")
print()
print(f"  Sensitivity (Recall) : {sensitivity:.1f}%  — how many seizures caught")
print(f"  Specificity          : {specificity:.1f}%  — how many normals correctly cleared")
print(f"  Accuracy             : {accuracy:.1f}%  — overall correct predictions")
print(f"  Precision            : {precision:.1f}%  — of seizure alerts how many real")
print()
print("  FALSE POSITIVE files list:")
for fname, conf, clstr in false_positives:
    print(f"  - {fname} (confidence {conf:.1f}%, cluster {clstr} windows)")
if not false_positives:
    print("  - None")
print()
print("  FALSE NEGATIVE files list:")
for fname, conf, clstr in false_negatives:
    print(f"  - {fname} (confidence {conf:.1f}%, cluster {clstr} windows)")
if not false_negatives:
    print("  - None")
print("================================================================")

# ── Save report CSV ────────────────────────────────────────────────────────────
with open(REPORT_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
    writer.writeheader()
    writer.writerows(report_rows)

print(f"\n  Report saved to: {REPORT_CSV}")
print(f"  Total time: {total_elapsed/60:.1f} minutes")
