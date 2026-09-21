"""
train_cnn_only.py
=================
Standalone script to train ONLY the CNN + Biological LTI Layer model.
SVM and Logistic Regression are NOT retrained.

Fixes:
  - ReduceLROnPlateau: removed deprecated 'verbose=True' (PyTorch ≥ 2.2)
  - scheduler progress logged manually instead

Workflow:
  1. Rebuild CNN dataset from EDF files (one file at a time → low RAM)
     If cached numpy arrays exist in models/cache/, they are loaded directly.
  2. Train CNN-LTI with:
       - BCEWithLogitsLoss + pos_weight (class imbalance)
       - Adam optimiser
       - ReduceLROnPlateau (compatible, no verbose kwarg)
       - Gradient clipping
       - Early stopping (patience=10)
       - Best checkpoint saved on val F1
  3. Evaluate on test split → save metrics JSON + plots
  4. Generate final comparison report (SVM vs LR vs CNN-LTI)
"""

import os, gc, json, time, pickle, sys
sys.stdout.reconfigure(encoding='utf-8')   # prevent cp1252 Unicode crashes on Windows
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report,
)

# ── local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import (
    load_edf, apply_filtering, remove_artifacts_ica,
    apply_pca_reduction, segment_signal,
)
from cnn_model import SeizureCNNLTI
from train import (                      # reuse helpers already in train.py
    parse_seizure_info,
    build_file_list,
    DATA_DIR, MODELS_DIR, METRICS_DIR,
    TRAIN_PATIENTS, VAL_PATIENTS, TEST_PATIENTS,
)

# ── cache directory (numpy arrays saved here to avoid re-extracting) ─────────
CACHE_DIR = os.path.join(MODELS_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] PyTorch {torch.__version__}  |  Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}  |  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CNN dataset builder  (raw EEG windows, NO feature extraction)
# ─────────────────────────────────────────────────────────────────────────────

def extract_cnn_windows_from_file(item, pca_model):
    """
    Loads one EDF file, preprocesses it, and returns labelled raw windows.
    Raw windows are 18-channel × 1280-sample arrays for the CNN.
    Returns arrays or None on failure.
    """
    file_path     = item['path']
    seizure_times = item['seizure_times']
    has_seizure   = item['num_seizures'] > 0

    if has_seizure:
        first_start = seizure_times[0][0]
        last_end    = seizure_times[-1][1]
        start_load  = max(0, first_start - 120)
        end_load    = last_end + 120
    else:
        start_load, end_load = 0, 600

    try:
        raw = load_edf(file_path, start_sec=start_load, end_sec=end_load)
        sfreq = raw.info['sfreq']
        raw   = apply_filtering(raw, sfreq)
        raw   = remove_artifacts_ica(raw)
        clean = raw.get_data()              # (18, samples)
        del raw; gc.collect()

        raw_windows, start_indices = segment_signal(
            clean, sfreq=sfreq, window_sec=5, overlap_sec=2.5
        )
        del clean; gc.collect()

        labels = []
        for start_idx in start_indices:
            w_start = start_load + start_idx / sfreq
            w_end   = w_start + 5.0
            is_ictal = any(
                max(w_start, s) < min(w_end, e)
                for s, e in seizure_times
            )
            labels.append(1 if is_ictal else 0)

        return raw_windows, np.array(labels, dtype=np.int8)

    except Exception as exc:
        print(f"    [ERROR] {item['file_name']}: {exc}")
        return None, None


def build_cnn_split(file_list, split_name, pca_model):
    """
    Iterates file_list one file at a time and accumulates CNN windows.
    Checks for a cached .npy file first.
    """
    cache_x = os.path.join(CACHE_DIR, f"X_cnn_{split_name}.npy")
    cache_y = os.path.join(CACHE_DIR, f"y_cnn_{split_name}.npy")

    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print(f"  [{split_name}] Loading cached arrays ...")
        X = np.load(cache_x, allow_pickle=False)
        y = np.load(cache_y, allow_pickle=False)
        print(f"  [{split_name}] windows={len(y)}  seizure={int(y.sum())}  "
              f"normal={int((y==0).sum())}")
        return X, y

    print(f"\n  Building [{split_name}] CNN dataset — {len(file_list)} files ...")
    X_list, y_list = [], []

    for idx, item in enumerate(file_list):
        if not os.path.exists(item['path']):
            print(f"    [{idx+1}/{len(file_list)}] MISSING — skipping")
            continue
        print(f"    [{idx+1}/{len(file_list)}] {item['patient']} / "
              f"{item['file_name']}  (seizures={item['num_seizures']})")

        wins, labs = extract_cnn_windows_from_file(item, pca_model)
        if wins is not None and len(wins) > 0:
            X_list.append(wins)
            y_list.append(labs)

    if not X_list:
        return np.empty((0,)), np.empty((0,))

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int8)

    # Save cache
    np.save(cache_x, X)
    np.save(cache_y, y)
    print(f"  [{split_name}] windows={len(y)}  seizure={int(y.sum())}  "
          f"normal={int((y==0).sum())}  -> cached")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CNN-LTI training
# ─────────────────────────────────────────────────────────────────────────────

def train_cnn_lti(X_train, y_train, X_val, y_val, X_test, y_test,
                  epochs=50, batch_size=32, early_stop_patience=10):

    print("\n" + "="*60)
    print("  TRAINING  CNN + BIOLOGICAL LTI LAYER")
    print("="*60)
    print(f"  Train : {len(y_train)} windows  "
          f"(seizure={int(y_train.sum())}, normal={int((y_train==0).sum())})")
    print(f"  Val   : {len(y_val)} windows  "
          f"(seizure={int(y_val.sum())}, normal={int((y_val==0).sum())})")
    print(f"  Test  : {len(y_test)} windows  "
          f"(seizure={int(y_test.sum())}, normal={int((y_test==0).sum())})")
    print(f"  Epochs: {epochs}  |  Batch: {batch_size}  |  "
          f"Early-stop patience: {early_stop_patience}")

    def to_t(X, y):
        return (torch.tensor(X, dtype=torch.float32).to(DEVICE),
                torch.tensor(y.astype(np.float32), dtype=torch.float32)
                      .unsqueeze(1).to(DEVICE))

    X_tr_t, y_tr_t   = to_t(X_train, y_train)
    X_val_t, y_val_t = to_t(X_val,   y_val)
    X_te_t,  y_te_t  = to_t(X_test,  y_test)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=batch_size, shuffle=True,
                        pin_memory=False)

    # ── Class-imbalance weight ────────────────────────────────────────────
    n_neg = float((y_train == 0).sum())
    n_pos = float((y_train == 1).sum())
    pos_w = n_neg / max(n_pos, 1.0)
    print(f"\n  pos_weight for BCEWithLogitsLoss: {pos_w:.2f}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(DEVICE)
    )

    # ── Model ─────────────────────────────────────────────────────────────
    n_channels     = X_train.shape[1]
    window_samples = X_train.shape[2]
    model = SeizureCNNLTI(in_channels=n_channels,
                           window_samples=window_samples).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ── Scheduler  (verbose kwarg removed — not supported in PyTorch ≥ 2.2) ──
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ── Training loop ─────────────────────────────────────────────────────
    ckpt_path   = os.path.join(MODELS_DIR, "cnn_lti_model.pt")
    best_val_f1 = 0.0
    no_improve  = 0
    history     = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []}
    current_lr  = optimizer.param_groups[0]['lr']

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
        epoch_loss /= len(loader.dataset)

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds  = (torch.sigmoid(val_logits) >= 0.5).long() \
                              .cpu().numpy().flatten()
        val_acc = accuracy_score(y_val, val_preds)
        val_f1  = f1_score(y_val, val_preds, zero_division=0)

        # Step scheduler (returns new lr silently)
        scheduler.step(val_f1)
        new_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(new_lr)

        lr_tag = f"  [LR->{new_lr:.2e}]" if new_lr != current_lr else ""
        current_lr = new_lr

        # Use val_acc as checkpoint metric when val has no seizure windows
        # (val_f1 would be stuck at 0.0 making checkpointing impossible)
        ckpt_metric = val_f1 if int(y_val.sum()) > 0 else val_acc
        improved = ckpt_metric > best_val_f1
        if improved:
            best_val_f1 = ckpt_metric
            no_improve  = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = "  * best"
        else:
            no_improve += 1
            marker = ""

        print(f"  Epoch {epoch:03d}/{epochs}  "
              f"loss={epoch_loss:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}"
              f"{marker}{lr_tag}")

        if no_improve >= early_stop_patience:
            print(f"\n  Early stopping triggered (no val F1 improvement "
                  f"for {early_stop_patience} epochs).")
            break

    # ── Final test evaluation ─────────────────────────────────────────────
    print(f"\n  Loading best checkpoint (val F1={best_val_f1:.4f}) ...")
    best = SeizureCNNLTI(in_channels=n_channels,
                          window_samples=window_samples).to(DEVICE)
    best.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    best.eval()

    with torch.no_grad():
        test_logits = best(X_te_t)
        test_probs  = torch.sigmoid(test_logits).cpu().numpy().flatten()
        test_preds  = (test_probs >= 0.5).astype(int)

    metrics = _compute_and_save_metrics(
        "CNN_LTI", y_test, test_preds, test_probs
    )
    _save_training_history(history)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_and_save_metrics(model_name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred,    zero_division=0)
    f1   = f1_score(y_true, y_pred,        zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = float(auc(fpr, tpr))
    except Exception:
        roc_auc = 0.0; fpr, tpr = np.array([]), np.array([])

    metrics = {
        "model":     model_name,
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1_score":  round(float(f1),   4),
        "roc_auc":   round(roc_auc,     4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n  -- {model_name} Final Test Results --")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"     {k:<12}: {v}")
    print("\n" + classification_report(y_true, y_pred,
                                       target_names=['Interictal', 'Ictal'],
                                       zero_division=0))

    json_path = os.path.join(METRICS_DIR, "cnn_lti_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set(title=f'{model_name} — Confusion Matrix',
           xlabel='Predicted', ylabel='True',
           xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Interictal', 'Ictal'],
           yticklabels=['Interictal', 'Ictal'])
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, "cnn_lti_confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC curve plot
    if len(fpr) > 0:
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
        plt.savefig(os.path.join(METRICS_DIR, "cnn_lti_roc_curve.png"), dpi=150)
        plt.close()

    return metrics


def _save_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['train_loss'], 'b-o', ms=3)
    axes[0].set(title='Training Loss', xlabel='Epoch', ylabel='BCE Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['val_acc'], 'g-o', ms=3, label='Accuracy')
    axes[1].plot(epochs, history['val_f1'],  'r-s', ms=3, label='F1')
    axes[1].set(title='Validation Metrics', xlabel='Epoch', ylabel='Score',
                ylim=[0, 1])
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['lr'], 'm-^', ms=3)
    axes[2].set(title='Learning Rate', xlabel='Epoch', ylabel='LR')
    axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(METRICS_DIR, "cnn_lti_training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training history saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Final comparison report
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison_report(cnn_metrics: dict):
    """
    Loads SVM and LR metrics from JSON, adds CNN metrics,
    and generates:
      - comparison_report.json
      - comparison_table.png
      - comparison_report.txt  (human-readable)
    """
    print("\n" + "="*60)
    print("  GENERATING FINAL COMPARISON REPORT")
    print("="*60)

    svm_path = os.path.join(METRICS_DIR, "svm_metrics.json")
    lr_path  = os.path.join(METRICS_DIR, "lr_metrics.json")

    with open(svm_path) as f: svm_m = json.load(f)
    with open(lr_path)  as f: lr_m  = json.load(f)

    all_models = [svm_m, lr_m, cnn_metrics]
    names      = [m['model'] for m in all_models]
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    labels       = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    # ── Determine winners ────────────────────────────────────────────────
    def _best(key):
        vals = [m[key] for m in all_models]
        idx  = int(np.argmax(vals))
        return names[idx], vals[idx]

    best_acc   = _best('accuracy')
    best_rec   = _best('recall')
    best_auc   = _best('roc_auc')
    best_f1    = _best('f1_score')
    # Best seizure detection model = highest recall (clinical priority)
    best_seiz  = best_rec

    # ── Bar chart ────────────────────────────────────────────────────────
    x      = np.arange(len(labels))
    width  = 0.25
    colors = ['#4C72B0', '#55A868', '#C44E52']

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (model_d, color) in enumerate(zip(all_models, colors)):
        vals = [model_d[k] for k in metrics_keys]
        bars = ax.bar(x + i * width, vals, width,
                      label=model_d['model'], color=color, alpha=0.88,
                      edgecolor='white', linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold')

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        "Comparative Analysis of Classical Machine Learning and Deep Learning\n"
        "Models for Epileptic Seizure Detection using the CHB-MIT Dataset",
        fontsize=12, fontweight='bold', pad=12
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Annotate best model for Recall (most clinically important)
    ax.annotate(
        f'★ Best Recall: {best_seiz[0]}',
        xy=(0.98, 0.97), xycoords='axes fraction',
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFFFCC', ec='gray', lw=0.8)
    )
    plt.tight_layout()
    chart_path = os.path.join(METRICS_DIR, "comparison_table.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Bar chart saved -> {chart_path}")

    # ── ROC overlay ──────────────────────────────────────────────────────
    # (individual ROC PNGs already exist; create an overlay for the report)
    # We don't have fpr/tpr arrays in memory, so we skip the overlay.

    # ── Text report ─────────────────────────────────────────────────────
    sep = "-" * 72
    header = (
        "=" * 72 + "\n"
        "  Comparative Analysis of Classical Machine Learning and Deep Learning\n"
        "  Models for Epileptic Seizure Detection using the CHB-MIT Dataset\n"
        "=" * 72 + "\n"
    )
    table_header = (
        f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} "
        f"{'Recall':>8} {'F1':>8} {'ROC-AUC':>9}\n"
        + f"  {sep}\n"
    )
    table_rows = ""
    for m in all_models:
        table_rows += (
            f"  {m['model']:<22} {m['accuracy']:>9.4f} "
            f"{m['precision']:>10.4f} {m['recall']:>8.4f} "
            f"{m['f1_score']:>8.4f} {m['roc_auc']:>9.4f}\n"
        )

    summary = (
        f"\n  SUMMARY\n"
        f"  {sep}\n"
        f"  Best Accuracy            : {best_acc[0]:<22} ({best_acc[1]:.4f})\n"
        f"  Best Recall              : {best_rec[0]:<22} ({best_rec[1]:.4f})\n"
        f"  Best ROC-AUC             : {best_auc[0]:<22} ({best_auc[1]:.4f})\n"
        f"  Best F1-Score            : {best_f1[0]:<22} ({best_f1[1]:.4f})\n"
        f"  Best Seizure Detection   : {best_seiz[0]:<22} "
        f"(highest Recall = {best_seiz[1]:.4f})\n"
        f"\n"
        f"  NOTES\n"
        f"  {sep}\n"
        f"  - Recall is the primary clinical metric: it measures the fraction\n"
        f"    of true seizures correctly identified (minimising missed seizures).\n"
        f"  - High accuracy with low recall indicates class imbalance dominance.\n"
        f"  - Dataset: CHB-MIT Scalp EEG Database (24 patients, patient-wise split)\n"
        f"  - Split  : Train chb01-18 | Val chb19-21 | Test chb22-24\n"
    )

    full_report = header + "\n" + table_header + table_rows + summary

    txt_path  = os.path.join(METRICS_DIR, "comparison_report.txt")
    json_path = os.path.join(METRICS_DIR, "comparison_report.json")

    with open(txt_path, "w", encoding='utf-8') as f:
        f.write(full_report)
    print(f"  Text report saved -> {txt_path}")

    report_json = {
        "title": ("Comparative Analysis of Classical Machine Learning and "
                  "Deep Learning Models for Epileptic Seizure Detection "
                  "using the CHB-MIT Dataset"),
        "dataset":   "CHB-MIT Scalp EEG Database",
        "patients":  24,
        "split": {
            "train": "chb01-chb18",
            "val":   "chb19-chb21",
            "test":  "chb22-chb24",
        },
        "models": all_models,
        "winners": {
            "best_accuracy":         {"model": best_acc[0],  "value": best_acc[1]},
            "best_recall":           {"model": best_rec[0],  "value": best_rec[1]},
            "best_roc_auc":          {"model": best_auc[0],  "value": best_auc[1]},
            "best_f1":               {"model": best_f1[0],   "value": best_f1[1]},
            "best_seizure_detection":{"model": best_seiz[0], "value": best_seiz[1]},
        },
    }
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(report_json, f, indent=2)
    print(f"  JSON report  saved -> {json_path}")

    # Print to console
    print("\n" + full_report)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    # Load fitted PCA (already saved during the original training run)
    pca_path = os.path.join(MODELS_DIR, "pca_transformer.pkl")
    with open(pca_path, "rb") as f:
        pca_model = pickle.load(f)
    print(f"[INFO] PCA transformer loaded from {pca_path}")

    # Build file lists (fast — only reads summary .txt files, no EDF loading)
    print("\n[STEP 1] Building file lists (summary parsing only) ...")
    train_files, _ = build_file_list(TRAIN_PATIENTS)
    val_files,   _ = build_file_list(VAL_PATIENTS)
    test_files,  _ = build_file_list(TEST_PATIENTS)

    # Build / load CNN datasets
    print("\n[STEP 2] Building CNN datasets ...")
    X_train, y_train = build_cnn_split(train_files, "train", pca_model)
    X_val,   y_val   = build_cnn_split(val_files,   "val",   pca_model)
    X_test,  y_test  = build_cnn_split(test_files,  "test",  pca_model)

    if len(X_train) == 0:
        raise RuntimeError("Training CNN dataset is empty. "
                           "Check DATA_DIR and file paths.")

    # Train CNN-LTI
    print("\n[STEP 3] Training CNN-LTI ...")
    cnn_metrics = train_cnn_lti(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        epochs=50,
        batch_size=32,
        early_stop_patience=10,
    )

    # Generate final comparison report
    print("\n[STEP 4] Generating comparison report ...")
    generate_comparison_report(cnn_metrics)

    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    print(f"\n{'='*60}")
    print(f"  CNN-LTI training complete.")
    print(f"  Saved to      : {MODELS_DIR}")
    print(f"  Metrics in    : {METRICS_DIR}")
    print(f"  Elapsed time  : {h}h {m}m {s}s")
    print(f"{'='*60}\n")
