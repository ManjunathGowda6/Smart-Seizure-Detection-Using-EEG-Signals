"""
validate_chbmit_v4.py
======================
Comprehensive evaluation of CNN-BiLSTM (v4) across the CHB-MIT dataset.

Features:
  - Loads models/cnn_bilstm_chbmit_v4.pt
  - Scans representative files using build_file_list_dl from eeg_dataset.py
  - Evaluates full recordings (no 120s crop)
  - Uses the 20% window ratio threshold for file-level classification
  - Outputs detailed file-by-file predictions and overall summary statistics
"""

import os
import sys
import time
import numpy as np
import torch

# Add backend folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.predict import predict_eeg_file
from ml.eeg_dataset import build_file_list_dl

# --- Configurations ---
FAST_MODE = True  # Set to False to run on the full representative subset (268 files), True runs on a quick subset (48 files)

def run_validation():
    print("============================================================")
    print("  CHB-MIT Dataset Validation Suite | CNN-BiLSTM (v4)")
    print("============================================================")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available, running on CPU")
    print("------------------------------------------------------------")
    
    # 1. Gather files
    patients = [f"chb{i:02d}" for i in range(1, 25)]
    all_records = build_file_list_dl(patients)
    
    # Filter files depending on FAST_MODE
    if FAST_MODE:
        print("\n[INFO] Running in FAST_MODE. Selecting up to 2 files per patient.")
        selected_records = []
        for pat in patients:
            pat_records = [r for r in all_records if r['patient'] == pat]
            seizures = [r for r in pat_records if r['num_seizures'] > 0]
            normals = [r for r in pat_records if r['num_seizures'] == 0]
            # Take at most 1 seizure and 1 normal file per patient
            if seizures:
                selected_records.append(seizures[0])
            if normals:
                selected_records.append(normals[0])
        records = selected_records
    else:
        print(f"\n[INFO] Running full validation across the entire representative subset ({len(all_records)} files).")
        records = all_records

    print(f"Total files selected for validation: {len(records)}")
    print("------------------------------------------------------------")
    
    total_files = 0
    correct_preds = 0
    wrong_preds = 0
    
    fp_count = 0  # false positives
    fn_count = 0  # false negatives
    total_normals = 0
    total_seizures = 0
    
    # Header
    print(f"{'File Name':<20} | {'Ground Truth':<12} | {'Prediction':<12} | {'Flagged %':<10} | {'Result':<8}")
    print("-" * 70)
    
    t0_all = time.time()
    
    for idx, rec in enumerate(records):
        filepath = rec['path']
        filename = rec['file_name']
        
        if not os.path.exists(filepath):
            continue
            
        gt_label = "SEIZURE" if rec['num_seizures'] > 0 else "NORMAL"
        
        try:
            # Run prediction on full EDF
            results = predict_eeg_file(filepath, model_name="cnn_bilstm", end_sec=None)
            
            pred_label = "SEIZURE" if results['prediction'] == "seizure" else "NORMAL"
            
            # Compute flagged %
            total_wins = results.get('total_windows')
            flagged_wins = results.get('seizure_wins')
            
            if total_wins is not None and total_wins > 0:
                flagged_pct = (flagged_wins / total_wins) * 100.0
            else:
                flagged_pct = 0.0
                
            is_correct = (gt_label == pred_label)
            result_str = "CORRECT" if is_correct else "WRONG"
            
            print(f"{filename:<20} | {gt_label:<12} | {pred_label:<12} | {flagged_pct:>8.2f}% | {result_str:<8}")
            
            # Statistics
            total_files += 1
            if is_correct:
                correct_preds += 1
            else:
                wrong_preds += 1
                
            if gt_label == "NORMAL":
                total_normals += 1
                if pred_label == "SEIZURE":
                    fp_count += 1
            else:
                total_seizures += 1
                if pred_label == "NORMAL":
                    fn_count += 1
                    
        except Exception as e:
            print(f"{filename:<20} | {gt_label:<12} | Failed to evaluate: {e}")
            
    # Print Summary Report
    elapsed = time.time() - t0_all
    accuracy = (correct_preds / total_files) * 100.0 if total_files > 0 else 0.0
    fpr = (fp_count / total_normals) * 100.0 if total_normals > 0 else 0.0
    fnr = (fn_count / total_seizures) * 100.0 if total_seizures > 0 else 0.0
    recall = ((total_seizures - fn_count) / total_seizures) * 100.0 if total_seizures > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY REPORT")
    print("=" * 60)
    print(f"  Total files tested           : {total_files}")
    print(f"  Correct predictions          : {correct_preds}")
    print(f"  Wrong predictions            : {wrong_preds}")
    print(f"  Overall file-level accuracy  : {accuracy:.2f}%")
    print(f"  False Positive Rate (FPR)    : {fpr:.2f}%  (Normal files called SEIZURE)")
    print(f"  False Negative Rate (FNR)    : {fnr:.2f}%  (Seizure files called NORMAL)")
    print(f"  Seizure Recall (Sensitivity) : {recall:.2f}%")
    print(f"  Total evaluation time        : {elapsed/60:.2f} minutes")
    print("=" * 60)
    
    # Target checks
    print("\n  Target Checklist:")
    print(f"    - Normal files FPR < 10%    : {'[PASSED]' if fpr < 10.0 else '[FAILED]'} (FPR = {fpr:.2f}%)")
    print(f"    - Seizure files Recall > 75%: {'[PASSED]' if recall > 75.0 else '[FAILED]'} (Recall = {recall:.2f}%)")
    print(f"    - Overall accuracy > 85%    : {'[PASSED]' if accuracy > 85.0 else '[FAILED]'} (Accuracy = {accuracy:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    run_validation()
