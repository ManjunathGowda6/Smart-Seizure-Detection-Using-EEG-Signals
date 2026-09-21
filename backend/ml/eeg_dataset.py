"""
eeg_dataset.py
==============
Memory-efficient EEG dataset loader for the CHB-MIT database.

Design principles:
  - Never loads all EDF files simultaneously.
  - Extracts windows from one file at a time, appends to cache.
  - Cache stored as memory-mapped numpy arrays (.npy) for low RAM usage.
  - PyTorch Dataset wraps the cached arrays for DataLoader compatibility.
  - WeightedRandomSampler weights computed from the label distribution.
"""

import os
import re
import gc
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import mne
mne.set_log_level('WARNING')   # suppress per-channel verbose output

# ── local imports ─────────────────────────────────────────────────────────────
try:
    from .preprocess import (
        load_edf, apply_filtering, remove_artifacts_ica, segment_signal
    )
except ImportError:
    from preprocess import (
        load_edf, apply_filtering, remove_artifacts_ica, segment_signal
    )

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SEC   = 5        # seconds per window
OVERLAP_SEC  = 2.5      # 50 % overlap
SFREQ        = 256      # target sampling frequency
N_CHANNELS   = 18       # standard CHB-MIT bipolar montage
WINDOW_SAMP  = int(WINDOW_SEC * SFREQ)   # 1280

DATA_DIR  = r"E:\Downloads\chb-mit-scalp-eeg-database-1.0.0"
CACHE_DIR = r"E:\Downloads\Finalyearproject\backend\models\cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Patient split
TRAIN_PATIENTS = [f"chb{i:02d}" for i in range(1,  19)]
VAL_PATIENTS   = [f"chb{i:02d}" for i in range(19, 22)]
TEST_PATIENTS  = [f"chb{i:02d}" for i in range(22, 25)]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Summary file parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_summary(patient_dir: str, patient_id: str):
    """
    Parses chbXX-summary.txt.
    Returns list of dicts:
        { file_name, num_seizures, seizure_times [(s,e), …], path }
    """
    summary_path = os.path.join(patient_dir, f"{patient_id}-summary.txt")
    if not os.path.exists(summary_path):
        return []

    records = []
    with open(summary_path, 'r', errors='ignore') as f:
        content = f.read()

    for block in content.split("File Name: ")[1:]:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue
        fname = lines[0]
        ns_line = [l for l in lines if "Number of Seizures" in l]
        if not ns_line:
            continue
        n_seiz = int(re.search(r'\d+', ns_line[0]).group())
        times  = []
        if n_seiz > 0:
            starts = [l for l in lines if "Seizure" in l and "Start" in l]
            ends   = [l for l in lines if "Seizure" in l and "End"   in l]
            for s, e in zip(starts, ends):
                times.append((int(re.search(r'\d+', s).group()),
                               int(re.search(r'\d+', e).group())))
        records.append({
            'file_name':    fname,
            'num_seizures': n_seiz,
            'seizure_times': times,
            'path': os.path.join(patient_dir, fname),
        })
    return records


def build_file_list_dl(patients):
    """
    For each patient selects ALL seizure files + balanced non-seizure files.
    Returns list of file-record dicts.
    """
    files = []
    for pat in patients:
        pat_dir = os.path.join(DATA_DIR, pat)
        records = parse_summary(pat_dir, pat)
        seizure = [r for r in records if r['num_seizures'] > 0]
        normal  = [r for r in records if r['num_seizures'] == 0]
        normal  = normal[:len(seizure)]           # balanced
        for r in seizure + normal:
            r['patient'] = pat
            files.append(r)
    return files


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Single-file window extractor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_windows_from_file(record: dict):
    """
    Loads one EDF, preprocesses it, segments into windows, and labels them.

    Pipeline: load -> bandpass filter -> notch -> ICA -> segment

    Returns:
        windows : np.ndarray (N, 18, 1280)  float32
        labels  : np.ndarray (N,)           int8
    """
    path          = record['path']
    seizure_times = record['seizure_times']
    has_seizure   = record['num_seizures'] > 0

    # Determine crop boundaries
    if has_seizure:
        start_sec = max(0, seizure_times[0][0]  - 120)
        end_sec   = seizure_times[-1][1] + 120
    else:
        start_sec, end_sec = 0, 600        # first 10 min of normal file

    try:
        raw   = load_edf(path, start_sec=start_sec, end_sec=end_sec)
        sfreq = raw.info['sfreq']
        raw   = apply_filtering(raw, sfreq)
        raw   = remove_artifacts_ica(raw)
        data  = raw.get_data().astype(np.float32)   # (C, samples)
        del raw; gc.collect()

        # Pad or truncate to exactly N_CHANNELS
        C = data.shape[0]
        if C < N_CHANNELS:
            pad   = np.zeros((N_CHANNELS - C, data.shape[1]), dtype=np.float32)
            data  = np.vstack([data, pad])
        else:
            data  = data[:N_CHANNELS]

        # Segment
        windows, start_indices = segment_signal(
            data, sfreq=sfreq,
            window_sec=WINDOW_SEC, overlap_sec=OVERLAP_SEC
        )
        del data; gc.collect()

        # ── Normalise each window to zero-mean unit-variance ──────────────
        # Per-window, per-channel normalisation prevents amplitude leakage
        mu  = windows.mean(axis=2, keepdims=True)
        std = windows.std(axis=2,  keepdims=True) + 1e-8
        windows = ((windows - mu) / std).astype(np.float32)

        # Label each window
        labels = []
        for si in start_indices:
            w_start = start_sec + si / sfreq
            w_end   = w_start + WINDOW_SEC
            ictal   = any(max(w_start, s) < min(w_end, e)
                          for s, e in seizure_times)
            labels.append(1 if ictal else 0)

        return windows, np.array(labels, dtype=np.int8)

    except Exception as exc:
        print(f"    [SKIP] {record['file_name']}: {exc}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Cache builder
# ─────────────────────────────────────────────────────────────────────────────

def build_or_load_cache(file_list, split_name: str, force_rebuild: bool = False):
    """
    Builds the window cache for one split if not already present.
    Uses memory-mapped numpy files to avoid loading everything into RAM.

    Returns:
        X : np.ndarray (N, 18, 1280) — memory mapped
        y : np.ndarray (N,)
    """
    cx = os.path.join(CACHE_DIR, f"dl_{split_name}_X.npy")
    cy = os.path.join(CACHE_DIR, f"dl_{split_name}_y.npy")

    if os.path.exists(cx) and os.path.exists(cy) and not force_rebuild:
        print(f"  [{split_name}] Loading from cache ...")
        X = np.load(cx, mmap_mode='r')
        y = np.load(cy)
        print(f"  [{split_name}] X={X.shape}  "
              f"seizure={int(y.sum())}  normal={int((y==0).sum())}")
        return X, y

    print(f"\n  Building [{split_name}] cache — {len(file_list)} files ...")
    X_parts, y_parts = [], []

    for idx, rec in enumerate(file_list):
        if not os.path.exists(rec['path']):
            print(f"    [{idx+1}/{len(file_list)}] MISSING — {rec['file_name']}")
            continue
        print(f"    [{idx+1}/{len(file_list)}] {rec['patient']} / "
              f"{rec['file_name']}  (seizures={rec['num_seizures']})")

        wins, labs = _extract_windows_from_file(rec)
        if wins is not None and len(wins) > 0:
            X_parts.append(wins)
            y_parts.append(labs)
        gc.collect()

    if not X_parts:
        raise RuntimeError(f"No windows extracted for split '{split_name}'.")

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int8)
    del X_parts, y_parts; gc.collect()

    np.save(cx, X)
    np.save(cy, y)
    print(f"  [{split_name}] Saved: X={X.shape}  "
          f"seizure={int(y.sum())}  normal={int((y==0).sum())}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EEGWindowDataset(Dataset):
    """
    Wraps cached numpy arrays as a PyTorch Dataset.
    X is kept memory-mapped so only accessed chunks are loaded into RAM.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X        # (N, 18, 1280)  may be mmap
        self.y = y        # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(float(self.y[idx]), dtype=torch.float32)
        return x, label


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Sampler helper
# ─────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler that up-samples the minority (seizure) class.
    This ensures every mini-batch has a balanced seizure/non-seizure ratio.
    """
    n_neg = float((y == 0).sum())
    n_pos = float((y == 1).sum())
    w_neg = 1.0 / n_neg if n_neg > 0 else 0.0
    w_pos = 1.0 / n_pos if n_pos > 0 else 0.0
    weights = np.where(y == 1, w_pos, w_neg)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float64),
        num_samples=len(weights),
        replacement=True,
    )
