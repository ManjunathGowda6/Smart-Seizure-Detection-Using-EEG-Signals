"""
batch_validate_seizures.py
==========================
Runs CNN-BiLSTM inference on all 141 annotated SEIZURE files from
dataset_split_complete.csv and produces a detailed validation report.

GPU: NVIDIA GeForce GTX 1650
Model: cnn_bilstm_chbmit_v4.pt
Decision: 3+ consecutive flagged windows (threshold 0.40)
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
COMPLETE_CSV  = r"E:\Downloads\Finalyearproject\dataset_split_complete.csv"
REPORT_CSV    = r"E:\Downloads\Finalyearproject\batch_validation_report.csv"
TIMING_WINDOW = 60   # seconds — tolerance for timing match

REPORT_FIELDS = [
    "filename", "patient", "label", "prediction", "confidence",
    "detected_start", "detected_end",
    "known_start", "known_end",
    "longest_cluster", "timing_match", "correct",
]

# ── Load only annotated SEIZURE rows ──────────────────────────────────────────
rows = []
with open(COMPLETE_CSV, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["label"] == "SEIZURE" and row.get("source", "annotated") == "annotated":
            rows.append(row)

total = len(rows)
print("=" * 65)
print(f"  BATCH VALIDATION — {total} annotated SEIZURE files")
print("=" * 65)
print(f"  Model : cnn_bilstm_chbmit_v4.pt  |  Threshold: 0.40  |  Min cluster: 4 (gap=0, dur=2.5s, k=5)")
print(f"  Timing tolerance: ±{TIMING_WINDOW}s")
print("=" * 65)
print()

# ── Accumulators ──────────────────────────────────────────────────────────────
correctly_detected = 0
missed             = 0
wrong_timing       = 0
timing_match_count = 0

conf_correct = []
conf_missed  = []

report_rows = []

t_total_start = time.time()

for file_idx, row in enumerate(rows, start=1):
    filepath = row["full_path"]
    fname    = row["filename"]
    patient  = row["patient"]

    # Parse known seizure times (take first seizure if multiple)
    starts_raw = [s.strip() for s in row["seizure_start_seconds"].split(",") if s.strip()]
    ends_raw   = [e.strip() for e in row["seizure_end_seconds"].split(",")   if e.strip()]
    known_start = float(starts_raw[0]) if starts_raw else None
    known_end   = float(ends_raw[0])   if ends_raw   else None

    print(f"[{file_idx:>3}/{total}] {fname}")

    if not os.path.exists(filepath):
        print(f"  ERROR: File not found on disk — skipping")
        report_rows.append({
            "filename": fname, "patient": patient, "label": "SEIZURE",
            "prediction": "FILE_MISSING", "confidence": "",
            "detected_start": "", "detected_end": "",
            "known_start": known_start, "known_end": known_end,
            "longest_cluster": "", "timing_match": "MISSING", "correct": "NO",
        })
        missed += 1
        print()
        continue

    try:
        t0 = time.time()
        results = predict_eeg_file(filepath, model_name="cnn_bilstm", end_sec=None)
        elapsed = time.time() - t0

        pred     = results["prediction"]           # "seizure" / "no_seizure"
        conf     = results["confidence_score"]
        segment  = results["seizure_segment"]
        tot_wins = results.get("total_windows", 0)
        sz_wins  = results.get("seizure_wins", 0)

        det_start = segment["start"] if segment and segment["start"] is not None else None
        det_end   = segment["end"]   if segment and segment["end"]   is not None else None

        # Compute longest cluster (re-derived from predictions stored in results;
        # the value is already printed inside predict_eeg_file — we capture from conf display)
        # We approximate via: largest run is what set has_seizure; printed inside the fn.
        # Use a simple heuristic for the report column.
        longest_cluster = ">=3" if pred == "seizure" else "<=2"

        # ── Timing match logic ────────────────────────────────────────────────
        if pred != "seizure":
            timing_match = "MISSED"
            missed       += 1
            conf_missed.append(conf)
            correct_flag = "NO"
        else:
            correctly_detected += 1
            conf_correct.append(conf)
            correct_flag = "YES"
            if known_start is not None and det_start is not None:
                # Overlap check: detected interval vs known annotation
                det_s, det_e = det_start, det_end
                kn_s,  kn_e  = known_start, known_end
                overlap = not (det_e < kn_s - TIMING_WINDOW or det_s > kn_e + TIMING_WINDOW)
                if overlap:
                    timing_match = "YES"
                    timing_match_count += 1
                else:
                    timing_match = "NO"
                    wrong_timing += 1
            else:
                timing_match = "NO_ANNOTATION"
                wrong_timing += 1

        det_start_fmt = f"{det_start:.1f}" if det_start is not None else ""
        det_end_fmt   = f"{det_end:.1f}"   if det_end   is not None else ""
        known_s_fmt   = f"{known_start:.0f}" if known_start is not None else "?"
        known_e_fmt   = f"{known_end:.0f}"   if known_end   is not None else "?"

        print(f"  Prediction       : {'SEIZURE' if pred == 'seizure' else 'NO SEIZURE'}")
        print(f"  Confidence       : {conf:.1f}%")
        print(f"  Detected interval: {det_start_fmt}s – {det_end_fmt}s")
        print(f"  Known interval   : {known_s_fmt}s – {known_e_fmt}s")
        print(f"  Timing match     : {timing_match}")
        print(f"  Longest cluster  : {longest_cluster} windows")
        print(f"  ({elapsed:.1f}s)")

        report_rows.append({
            "filename": fname, "patient": patient, "label": "SEIZURE",
            "prediction": "SEIZURE" if pred == "seizure" else "NO SEIZURE",
            "confidence": f"{conf:.1f}",
            "detected_start": det_start_fmt, "detected_end": det_end_fmt,
            "known_start": known_s_fmt, "known_end": known_e_fmt,
            "longest_cluster": longest_cluster,
            "timing_match": timing_match, "correct": correct_flag,
        })

    except Exception as exc:
        print(f"  ERROR: {exc}")
        import traceback; traceback.print_exc()
        report_rows.append({
            "filename": fname, "patient": patient, "label": "SEIZURE",
            "prediction": "ERROR", "confidence": "",
            "detected_start": "", "detected_end": "",
            "known_start": known_start, "known_end": known_end,
            "longest_cluster": "", "timing_match": "ERROR", "correct": "NO",
        })
        missed += 1

    print()

# ── Final summary ──────────────────────────────────────────────────────────────
total_elapsed = time.time() - t_total_start
avg_conf_correct = sum(conf_correct) / len(conf_correct) if conf_correct else 0.0
avg_conf_missed  = sum(conf_missed)  / len(conf_missed)  if conf_missed  else 0.0

pct = lambda x: f"{x/total*100:.1f}%" if total > 0 else "0.0%"

print()
print("=" * 65)
print("  BATCH VALIDATION REPORT — 141 SEIZURE FILES")
print("=" * 65)
print(f"  Total files tested          : {total}")
print(f"  Correctly detected (SEIZURE): {correctly_detected}  ({pct(correctly_detected)})")
print(f"  Missed (said NO SEIZURE)    : {missed}  ({pct(missed)})")
print(f"  Detected but wrong timing   : {wrong_timing}  ({pct(wrong_timing)})")
print(f"  Timing match (within 60s)   : {timing_match_count}  ({pct(timing_match_count)})")
print()
print(f"  Average confidence (correct): {avg_conf_correct:.1f}%")
print(f"  Average confidence (missed) : {avg_conf_missed:.1f}%")
print()
print(f"  Total time                  : {total_elapsed/60:.1f} minutes")
print("=" * 65)

# ── Save report CSV ────────────────────────────────────────────────────────────
with open(REPORT_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
    writer.writeheader()
    writer.writerows(report_rows)

print(f"\n  Report saved to: {REPORT_CSV}")
