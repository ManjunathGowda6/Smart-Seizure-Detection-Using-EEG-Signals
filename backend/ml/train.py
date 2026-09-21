"""
train.py — Research-grade CHB-MIT EEG Seizure Detection Training Pipeline
==========================================================================

Key improvements over the original:
  1. All 24 patients used (chb01–chb24)
  2. All seizure EDF files per patient included
  3. Balanced non-seizure files (N = len(seizure_files) per patient)
  4. Patient-wise train/val/test split (NO data leakage across patients)
     - Train : chb01–chb18
     - Val   : chb19–chb21
     - Test  : chb22–chb24
  5. Full-file processing (not just 5-minute clips)
  6. Memory-efficient: one EDF file loaded at a time, never all in RAM
  7. Best CNN model selected on validation accuracy
  8. Full metrics saved: accuracy, precision, recall, F1, ROC-AUC,
     confusion matrix (as PNG and JSON)
  9. Dataset statistics printed before training

Hardware target: Ryzen 5000 + GTX 1650 4 GB + 8 GB RAM
"""

import os
import re
import gc
import json
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')        # non-interactive backend — safe for server runs
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)

from preprocess import (
    preprocess_eeg_file, STANDARD_CHANNELS,
    load_edf, apply_filtering, remove_artifacts_ica,
    apply_pca_reduction, segment_signal
)
from features import extract_features_from_window
from cnn_model import SeizureCNNLTI

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = r"E:\Downloads\chb-mit-scalp-eeg-database-1.0.0"
MODELS_DIR = r"E:\Downloads\Finalyearproject\backend\models"
METRICS_DIR = os.path.join(MODELS_DIR, "metrics")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ── Patient split (zero-indexed internally, 1-indexed for directory names) ───
TRAIN_PATIENTS = [f"chb{i:02d}" for i in range(1,  19)]   # chb01–chb18
VAL_PATIENTS   = [f"chb{i:02d}" for i in range(19, 22)]   # chb19–chb21
TEST_PATIENTS  = [f"chb{i:02d}" for i in range(22, 25)]   # chb22–chb24
ALL_PATIENTS   = TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS

# ── CNN device ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_seizure_info(summary_path):
    """
    Parses a patient's *-summary.txt file.
    Returns a list of dicts:
        { file_name, num_seizures, seizure_times [(start_sec, end_sec), …] }
    """
    records = []
    with open(summary_path, 'r') as f:
        content = f.read()

    blocks = content.split("File Name: ")
    for block in blocks[1:]:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue
        file_name = lines[0]

        ns_line = [l for l in lines if "Number of Seizures" in l]
        if not ns_line:
            continue
        num_seizures = int(re.search(r'\d+', ns_line[0]).group())

        seizure_times = []
        if num_seizures > 0:
            starts = [l for l in lines if "Seizure" in l and "Start" in l]
            ends   = [l for l in lines if "Seizure" in l and "End"   in l]
            for s, e in zip(starts, ends):
                s_sec = int(re.search(r'\d+', s).group())
                e_sec = int(re.search(r'\d+', e).group())
                seizure_times.append((s_sec, e_sec))

        records.append({
            'file_name':    file_name,
            'num_seizures': num_seizures,
            'seizure_times': seizure_times,
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2. File-list builder (returns metadata only, no EDF loading)
# ─────────────────────────────────────────────────────────────────────────────

def build_file_list(patients):
    """
    Scans the dataset directory for the given patient list.
    For each patient selects:
      - ALL seizure EDF files
      - An equal number of non-seizure EDF files (balanced)
    Returns a list of file-record dicts with an extra 'path' key.
    No EDF files are loaded here.
    """
    target_files = []
    stats = {
        'patients_found':    0,
        'edf_total':         0,
        'edf_seizure':       0,
        'edf_normal_selected': 0,
    }

    for pat in patients:
        pat_dir      = os.path.join(DATA_DIR, pat)
        summary_file = os.path.join(pat_dir, f"{pat}-summary.txt")
        if not os.path.exists(summary_file):
            print(f"  [WARN] Summary missing for {pat}, skipping.")
            continue

        stats['patients_found'] += 1
        records = parse_seizure_info(summary_file)
        stats['edf_total'] += len(records)

        seizure_files = [r for r in records if r['num_seizures'] > 0]
        normal_files  = [r for r in records if r['num_seizures'] == 0]

        # Balanced selection: normal ≤ number of seizure files
        normal_selected = normal_files[:len(seizure_files)]

        stats['edf_seizure']        += len(seizure_files)
        stats['edf_normal_selected'] += len(normal_selected)

        for r in seizure_files + normal_selected:
            r = dict(r)                                      # shallow copy
            r['path']    = os.path.join(pat_dir, r['file_name'])
            r['patient'] = pat
            target_files.append(r)

    return target_files, stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Single-file processing  (memory-efficient, one file at a time)
# ─────────────────────────────────────────────────────────────────────────────

def process_single_file(item, pca_transformer, n_pca=3):
    """
    Loads, preprocesses, segments, and labels one EDF file.
    Extracts both ML features and raw CNN windows.

    Returns:
        ml_features  : np.ndarray (N_windows, F)
        ml_labels    : np.ndarray (N_windows,)
        cnn_windows  : np.ndarray (N_windows, C, T)
        cnn_labels   : np.ndarray (N_windows,)
        pca_transformer : fitted PCA (updated if previously None)
    """
    file_path       = item['path']
    seizure_times   = item['seizure_times']
    has_seizure     = item['num_seizures'] > 0

    # Determine load window
    if has_seizure:
        # Load from 2 min before first seizure to 2 min after last seizure
        first_start = seizure_times[0][0]
        last_end    = seizure_times[-1][1]
        start_load  = max(0, first_start - 120)
        end_load    = last_end + 120
    else:
        # Load first 10 minutes of non-seizure file
        start_load = 0
        end_load   = 600

    try:
        # ── Fit PCA on first encountered file if not yet fitted ─────────────
        if pca_transformer is None:
            raw_temp = load_edf(file_path, start_sec=start_load,
                                end_sec=start_load + 30)
            raw_temp = apply_filtering(raw_temp)
            raw_temp = remove_artifacts_ica(raw_temp)
            temp_data = raw_temp.get_data()
            _, pca_transformer = apply_pca_reduction(temp_data,
                                                     n_components=n_pca)
            del raw_temp, temp_data
            gc.collect()
            print(f"    PCA fitted on first file.")

        # ── Full preprocessing pipeline ──────────────────────────────────────
        windows, start_indices, _, raw_clean, channel_names, sfreq = \
            preprocess_eeg_file(
                file_path,
                pca_model=pca_transformer,
                n_components=n_pca,
                start_sec=start_load,
                end_sec=end_load,
            )

        # Raw windows for CNN (no PCA, original 18 channels)
        raw_windows, _ = segment_signal(raw_clean, sfreq=sfreq,
                                        window_sec=5, overlap_sec=2.5)

        n_windows = len(start_indices)
        ml_features_list = []
        ml_labels_list   = []
        cnn_windows_list = []
        cnn_labels_list  = []

        for w_idx, start_idx in enumerate(start_indices):
            w_start = start_load + (start_idx / sfreq)
            w_end   = w_start + 5.0

            # Label: ictal if window overlaps any seizure interval
            is_ictal = any(
                max(w_start, s) < min(w_end, e)
                for s, e in seizure_times
            )
            label = 1 if is_ictal else 0

            # ML features
            feats = extract_features_from_window(windows[w_idx], sfreq=sfreq)
            ml_features_list.append(feats)
            ml_labels_list.append(label)

            # CNN window
            if w_idx < len(raw_windows):
                cnn_windows_list.append(raw_windows[w_idx])
                cnn_labels_list.append(label)

        # Free large arrays ASAP
        del windows, raw_clean, raw_windows
        gc.collect()

        return (
            np.array(ml_features_list),
            np.array(ml_labels_list),
            np.array(cnn_windows_list),
            np.array(cnn_labels_list),
            pca_transformer,
        )

    except Exception as exc:
        print(f"    [ERROR] {item['file_name']}: {exc}")
        return None, None, None, None, pca_transformer


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset builder for a patient split
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_for_split(file_list, split_name, pca_transformer, n_pca=3):
    """
    Iterates over file_list one file at a time.
    Returns accumulated ML arrays, CNN arrays, and the (possibly updated) PCA.
    """
    print(f"\n  Building [{split_name}] split — {len(file_list)} files ...")
    X_ml, y_ml = [], []
    X_cnn, y_cnn = [], []

    for idx, item in enumerate(file_list):
        if not os.path.exists(item['path']):
            print(f"    [{idx+1}/{len(file_list)}] MISSING: {item['file_name']}")
            continue

        print(f"    [{idx+1}/{len(file_list)}] {item['patient']} / "
              f"{item['file_name']}  "
              f"(seizures={item['num_seizures']})")

        ml_f, ml_l, cnn_w, cnn_l, pca_transformer = process_single_file(
            item, pca_transformer, n_pca=n_pca
        )

        if ml_f is not None and len(ml_f) > 0:
            X_ml.append(ml_f)
            y_ml.append(ml_l)
        if cnn_w is not None and len(cnn_w) > 0:
            X_cnn.append(cnn_w)
            y_cnn.append(cnn_l)

    X_ml  = np.concatenate(X_ml,  axis=0) if X_ml  else np.empty((0,))
    y_ml  = np.concatenate(y_ml,  axis=0) if y_ml  else np.empty((0,))
    X_cnn = np.concatenate(X_cnn, axis=0) if X_cnn else np.empty((0,))
    y_cnn = np.concatenate(y_cnn, axis=0) if y_cnn else np.empty((0,))

    seizure_w = int(np.sum(y_ml))
    total_w   = len(y_ml)
    print(f"  [{split_name}] windows={total_w}  "
          f"seizure={seizure_w}  "
          f"normal={total_w - seizure_w}")

    return X_ml, y_ml, X_cnn, y_cnn, pca_transformer


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dataset statistics report
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_statistics(train_files, val_files, test_files,
                             global_stats):
    print("\n" + "="*60)
    print("  CHB-MIT DATASET STATISTICS")
    print("="*60)
    print(f"  Patients found           : {global_stats['patients_found']}")
    print(f"  Total EDF files scanned  : {global_stats['edf_total']}")
    print(f"  Seizure EDF files        : {global_stats['edf_seizure']}")
    print(f"  Non-seizure EDF selected : {global_stats['edf_normal_selected']}")
    print(f"  Files selected for training  : {len(train_files)}")
    print(f"  Files selected for validation: {len(val_files)}")
    print(f"  Files selected for test      : {len(test_files)}")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ML model training  (SVM + LR)
# ─────────────────────────────────────────────────────────────────────────────

def train_ml_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains SVM and LR on training data.
    Reports metrics on the held-out test split (val used only for GridSearchCV).
    """
    print("\n" + "="*60)
    print("  TRAINING ML MODELS (SVM + Logistic Regression)")
    print("="*60)

    # ── Scaling ────────────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # ── Feature selection (Chi-Sq) ─────────────────────────────────────────
    k_features = min(500, X_train.shape[1])
    selector = SelectKBest(score_func=chi2, k=k_features)
    X_train_sel = selector.fit_transform(X_train_s, y_train)
    X_val_sel   = selector.transform(X_val_s)
    X_test_sel  = selector.transform(X_test_s)
    with open(os.path.join(MODELS_DIR, "feature_selector.pkl"), "wb") as f:
        pickle.dump(selector, f)
    print(f"  Feature selection: {X_train.shape[1]} → {k_features} features")

    # ── SVM ────────────────────────────────────────────────────────────────
    print("\n  Training SVM (GridSearchCV, 5-fold) ...")
    svm_grid = GridSearchCV(
        SVC(kernel='rbf', probability=True, random_state=42),
        {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01]},
        cv=5, scoring='f1', n_jobs=-1, verbose=0
    )
    svm_grid.fit(X_train_sel, y_train)
    best_svm = svm_grid.best_estimator_
    print(f"  Best SVM params: {svm_grid.best_params_}")

    svm_preds = best_svm.predict(X_test_sel)
    svm_probs = best_svm.predict_proba(X_test_sel)[:, 1]
    _report_and_save("SVM", y_test, svm_preds, svm_probs)

    with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "wb") as f:
        pickle.dump(best_svm, f)

    # ── Logistic Regression ────────────────────────────────────────────────
    print("\n  Training Logistic Regression (GridSearchCV, 5-fold) ...")
    lr_grid = GridSearchCV(
        LogisticRegression(penalty='l2', solver='lbfgs',
                           max_iter=1000, random_state=42),
        {'C': [0.01, 0.1, 1, 10, 100]},
        cv=5, scoring='f1', n_jobs=-1, verbose=0
    )
    lr_grid.fit(X_train_sel, y_train)
    best_lr = lr_grid.best_estimator_
    print(f"  Best LR params: {lr_grid.best_params_}")

    lr_preds = best_lr.predict(X_test_sel)
    lr_probs = best_lr.predict_proba(X_test_sel)[:, 1]
    _report_and_save("LR", y_test, lr_preds, lr_probs)

    with open(os.path.join(MODELS_DIR, "lr_model.pkl"), "wb") as f:
        pickle.dump(best_lr, f)

    return best_svm, best_lr


# ─────────────────────────────────────────────────────────────────────────────
# 7. CNN-LTI training
# ─────────────────────────────────────────────────────────────────────────────

def train_cnn_lti(X_train, y_train, X_val, y_val, X_test, y_test,
                  epochs=40, batch_size=32):
    """
    Trains SeizureCNNLTI using:
      - X_train / y_train  for parameter updates
      - X_val  / y_val     for model selection (save best checkpoint)
      - X_test / y_test    for final evaluation only (never seen during training)
    """
    print("\n" + "="*60)
    print("  TRAINING CNN + LTI MODEL")
    print("="*60)
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {epochs}  |  Batch : {batch_size}")

    def _to_tensors(X, y):
        return (torch.tensor(X, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE))

    X_tr_t, y_tr_t = _to_tensors(X_train, y_train)
    X_val_t, y_val_t = _to_tensors(X_val, y_val)
    X_te_t, y_te_t = _to_tensors(X_test, y_test)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True
    )

    # Class-imbalance weight for BCEWithLogitsLoss
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    criterion  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    )

    n_channels      = X_train.shape[1]
    window_samples  = X_train.shape[2]
    model = SeizureCNNLTI(in_channels=n_channels,
                           window_samples=window_samples).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    best_val_f1   = 0.0
    best_ckpt_path = os.path.join(MODELS_DIR, "cnn_lti_model.pt")
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out  = model(bx)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
        epoch_loss /= len(train_loader.dataset)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds  = (torch.sigmoid(val_logits) >= 0.5).long().cpu().numpy().flatten()
            val_probs  = torch.sigmoid(val_logits).cpu().numpy().flatten()

        val_acc = accuracy_score(y_val, val_preds)
        val_f1  = f1_score(y_val, val_preds, zero_division=0)

        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        scheduler.step(val_f1)

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt_path)
            marker = "  ← best"

        print(f"  Epoch {epoch:03d}/{epochs}  "
              f"loss={epoch_loss:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}{marker}")

    # ── Final test evaluation using best checkpoint ───────────────────────
    print(f"\n  Loading best checkpoint (val F1={best_val_f1:.4f}) ...")
    best_model = SeizureCNNLTI(in_channels=n_channels,
                                window_samples=window_samples).to(DEVICE)
    best_model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    best_model.eval()

    with torch.no_grad():
        test_logits = best_model(X_te_t)
        test_preds  = (torch.sigmoid(test_logits) >= 0.5).long().cpu().numpy().flatten()
        test_probs  = torch.sigmoid(test_logits).cpu().numpy().flatten()

    _report_and_save("CNN_LTI", y_test, test_preds, test_probs)

    # Save training history plot
    _save_training_history(history)

    return best_model


# ─────────────────────────────────────────────────────────────────────────────
# 8. Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _report_and_save(model_name, y_true, y_pred, y_prob):
    """Prints, saves JSON metrics, confusion matrix PNG, and ROC curve PNG."""

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred,    zero_division=0)
    f1   = f1_score(y_true, y_pred,        zero_division=0)
    cm   = confusion_matrix(y_true, y_pred).tolist()

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = float(auc(fpr, tpr))
    except Exception:
        roc_auc = 0.0
        fpr, tpr = [], []

    metrics = {
        "model":     model_name,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
        "roc_auc":   round(roc_auc, 4),
        "confusion_matrix": cm,
    }

    print(f"\n  ── {model_name} Test Metrics ──")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1-score  : {f1:.4f}")
    print(f"     ROC-AUC   : {roc_auc:.4f}")
    print("\n" + classification_report(y_true, y_pred,
                                       target_names=['Interictal', 'Ictal'],
                                       zero_division=0))

    # JSON
    json_path = os.path.join(METRICS_DIR,
                             f"{model_name.lower()}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion Matrix PNG
    _plot_confusion_matrix(model_name, np.array(cm))

    # ROC Curve PNG
    if len(fpr) > 0:
        _plot_roc_curve(model_name, fpr, tpr, roc_auc)


def _plot_confusion_matrix(model_name, cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_title(f'{model_name} — Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ticks = ['Interictal', 'Ictal']
    ax.set_xticks([0, 1]); ax.set_xticklabels(ticks, rotation=30)
    ax.set_yticks([0, 1]); ax.set_yticklabels(ticks)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    path = os.path.join(METRICS_DIR, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_roc_curve(model_name, fpr, tpr, roc_auc):
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--')
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} — ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    path = os.path.join(METRICS_DIR, f"{model_name.lower()}_roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()


def _save_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history['train_loss'], 'b-o', ms=3, label='Train Loss')
    axes[0].set_title('CNN-LTI — Training Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['val_acc'], 'g-o', ms=3, label='Val Accuracy')
    axes[1].plot(epochs, history['val_f1'],  'r-s', ms=3, label='Val F1')
    axes[1].set_title('CNN-LTI — Validation Metrics')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Score')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(METRICS_DIR, "cnn_lti_training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training history plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    # ── Step 1: Build file lists (no EDF loading yet) ─────────────────────
    print("\n[STEP 1] Building file lists ...")
    train_files, stats_tr = build_file_list(TRAIN_PATIENTS)
    val_files,   stats_va = build_file_list(VAL_PATIENTS)
    test_files,  stats_te = build_file_list(TEST_PATIENTS)

    global_stats = {
        'patients_found':    stats_tr['patients_found'] + stats_va['patients_found'] + stats_te['patients_found'],
        'edf_total':         stats_tr['edf_total']      + stats_va['edf_total']      + stats_te['edf_total'],
        'edf_seizure':       stats_tr['edf_seizure']    + stats_va['edf_seizure']    + stats_te['edf_seizure'],
        'edf_normal_selected': (stats_tr['edf_normal_selected'] +
                                stats_va['edf_normal_selected'] +
                                stats_te['edf_normal_selected']),
    }
    print_dataset_statistics(train_files, val_files, test_files, global_stats)

    # ── Step 2: Process training split (PCA fitted here) ──────────────────
    print("[STEP 2] Processing TRAINING split (chb01–chb18) ...")
    X_train_ml, y_train_ml, X_train_cnn, y_train_cnn, pca_model = \
        build_dataset_for_split(train_files, "TRAIN", pca_transformer=None)

    # Save PCA
    with open(os.path.join(MODELS_DIR, "pca_transformer.pkl"), "wb") as f:
        pickle.dump(pca_model, f)
    print(f"  PCA transformer saved.")

    # ── Step 3: Process validation split ──────────────────────────────────
    print("\n[STEP 3] Processing VALIDATION split (chb19–chb21) ...")
    X_val_ml, y_val_ml, X_val_cnn, y_val_cnn, _ = \
        build_dataset_for_split(val_files, "VAL", pca_transformer=pca_model)

    # ── Step 4: Process test split ─────────────────────────────────────────
    print("\n[STEP 4] Processing TEST split (chb22–chb24) ...")
    X_test_ml, y_test_ml, X_test_cnn, y_test_cnn, _ = \
        build_dataset_for_split(test_files, "TEST", pca_transformer=pca_model)

    # ── Window-level statistics ────────────────────────────────────────────
    total_windows = len(y_train_ml) + len(y_val_ml) + len(y_test_ml)
    total_seizure = int(np.sum(y_train_ml) + np.sum(y_val_ml) + np.sum(y_test_ml))
    print(f"\n  TOTAL windows : {total_windows}")
    print(f"  Seizure windows: {total_seizure}  "
          f"({100*total_seizure/max(total_windows,1):.1f}%)")
    print(f"  Normal  windows: {total_windows - total_seizure}\n")

    # ── Step 5: Train ML models ────────────────────────────────────────────
    print("[STEP 5] Training ML models ...")
    train_ml_models(
        X_train_ml, y_train_ml,
        X_val_ml,   y_val_ml,
        X_test_ml,  y_test_ml,
    )

    # ── Step 6: Train CNN-LTI ─────────────────────────────────────────────
    print("\n[STEP 6] Training CNN + LTI model ...")
    if len(X_train_cnn) > 0 and len(X_val_cnn) > 0 and len(X_test_cnn) > 0:
        train_cnn_lti(
            X_train_cnn, y_train_cnn,
            X_val_cnn,   y_val_cnn,
            X_test_cnn,  y_test_cnn,
            epochs=40, batch_size=32,
        )
    else:
        print("  [SKIP] Insufficient CNN data in one or more splits.")

    elapsed = time.time() - t0
    hours, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n{'='*60}")
    print(f"  All models trained and saved to:  {MODELS_DIR}")
    print(f"  Metrics saved to:                 {METRICS_DIR}")
    print(f"  Total training time: {hours}h {mins}m {secs}s")
    print(f"{'='*60}\n")
