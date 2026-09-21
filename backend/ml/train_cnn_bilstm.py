"""
train_cnn_bilstm.py
====================
CNN-BiLSTM training pipeline — optimized for LOW RAM (1 GB free).

Hardware target:
  GTX 1650  4 GB VRAM
  Ryzen 5000
  ~1 GB free RAM  (rest used by OS + browser + backend)

Memory strategy:
  - Data stays on disk as memory-mapped numpy files (never fully loaded)
  - SequentialChunkSampler reads data in 512-window sequential chunks
  - Each chunk occupies ~47 MB of RAM (512 x 18 x 1280 x 4 bytes)
  - RAM footprint during training: < 200 MB total
  - VRAM footprint per batch (8 windows): ~11 MB + model ~90 MB = ~100 MB

No bulk loading. No WeightedRandomSampler. No pin_memory.
"""

import os, sys, gc, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from cnn_bilstm_model import CNNBiLSTM
from eeg_dataset import (
    build_or_load_cache, build_file_list_dl,
    TRAIN_PATIENTS, VAL_PATIENTS, TEST_PATIENTS,
    N_CHANNELS, WINDOW_SAMP, CACHE_DIR,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = r"E:\Downloads\Finalyearproject\backend\models"
METRICS_DIR = os.path.join(MODELS_DIR, "metrics")
CKPT_PATH   = os.path.join(MODELS_DIR, "cnn_bilstm_model.pt")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Cache paths (used directly to avoid double mmap)
CACHE_X_TRAIN = os.path.join(CACHE_DIR, "dl_bilstm_train_X.npy")
CACHE_Y_TRAIN = os.path.join(CACHE_DIR, "dl_bilstm_train_y.npy")
CACHE_X_VAL   = os.path.join(CACHE_DIR, "dl_bilstm_val_X.npy")
CACHE_Y_VAL   = os.path.join(CACHE_DIR, "dl_bilstm_val_y.npy")
CACHE_X_TEST  = os.path.join(CACHE_DIR, "dl_bilstm_test_X.npy")
CACHE_Y_TEST  = os.path.join(CACHE_DIR, "dl_bilstm_test_y.npy")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")

print("=" * 65)
print("  CNN-BiLSTM  |  Low-RAM Chunk-Stream Training")
print("=" * 65)
print(f"  PyTorch : {torch.__version__}")
print(f"  Device  : {DEVICE}")
if DEVICE.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU     : {props.name}")
    print(f"  VRAM    : {props.total_memory // 1024**2} MB")
print(f"  AMP     : {USE_AMP}")
print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Chunk-streaming Dataset + Sampler
#     Keeps RAM usage to ~50 MB at any time regardless of dataset size.
# ─────────────────────────────────────────────────────────────────────────────

class MmapEEGDataset(Dataset):
    """
    Wraps memory-mapped numpy arrays.
    Labels (y) are tiny (<1 MB) and fully loaded.
    Windows (X) are mmap — only accessed pages are in RAM.
    """
    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path, mmap_mode='r')   # stays on disk
        self.y = np.load(y_path).astype(np.float32)  # tiny — load fully
        assert len(self.X) == len(self.y), "X/y length mismatch"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # .copy() forces mmap page into RAM only for this one window
        x = torch.tensor(self.X[idx].copy(), dtype=torch.float32)
        return x, torch.tensor(self.y[idx], dtype=torch.float32)


class SequentialChunkSampler(Sampler):
    """
    Divides dataset into fixed-size sequential chunks.
    Each epoch: shuffles chunk ORDER, shuffles indices WITHIN each chunk.

    Why this works:
      - OS page cache loads pages for chunk[i] while we process chunk[i-1]
      - Random access is contained within a 47 MB window (not 3 GB)
      - RAM pressure: only 1-2 chunks in page cache at once

    chunk_size = 512:
      512 x 18 x 1280 x 4 bytes = ~47 MB per chunk
    """
    def __init__(self, n: int, chunk_size: int = 512):
        self.n          = n
        self.chunk_size = chunk_size

    def __iter__(self):
        rng    = np.random.default_rng()
        starts = np.arange(0, self.n, self.chunk_size)
        rng.shuffle(starts)                     # shuffle chunk order each epoch
        for s in starts:
            e       = min(s + self.chunk_size, self.n)
            indices = np.arange(s, e)
            rng.shuffle(indices)                # shuffle within chunk
            yield from indices.tolist()

    def __len__(self):
        return self.n


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Training / evaluation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, n_seen = 0.0, 0
    for Xb, yb in loader:
        Xb = Xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with autocast(device_type='cuda'):
                loss = criterion(model(Xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * Xb.size(0)
        n_seen     += Xb.size(0)

    return total_loss / max(n_seen, 1)


@torch.no_grad()
def evaluate(model, x_path: str, y: np.ndarray, batch_size: int = 32):
    """
    Evaluates model by streaming mmap file sequentially.
    RAM per call: batch_size x 18 x 1280 x 4 bytes = ~3 MB (batch=32)
    """
    model.eval()
    X_mmap    = np.load(x_path, mmap_mode='r')
    all_probs = []
    n         = len(y)

    for start in range(0, n, batch_size):
        chunk = X_mmap[start : start + batch_size].copy()   # ~3 MB
        Xb    = torch.tensor(chunk, dtype=torch.float32).to(DEVICE)
        if USE_AMP:
            with autocast(device_type='cuda'):
                logits = model(Xb)
        else:
            logits = model(Xb)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.append(probs)
        del Xb, chunk

    probs = np.concatenate(all_probs)
    preds = (probs >= 0.5).astype(int)
    return preds, probs


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train_cnn_bilstm(
    y_train: np.ndarray,
    y_val:   np.ndarray,
    y_test:  np.ndarray,
    epochs:     int   = 80,
    batch_size: int   = 8,
    chunk_size: int   = 512,
    early_stop: int   = 15,
    lr:         float = 1e-3,
):
    n_train = len(y_train)
    n_val   = len(y_val)
    n_test  = len(y_test)
    n_seiz  = int(y_train.sum())
    n_norm  = int((y_train == 0).sum())

    print("\n" + "=" * 65)
    print("  TRAINING CNN-BiLSTM  (chunk-stream, low-RAM mode)")
    print("=" * 65)
    print(f"  Train : {n_train:>7,}  (ictal={n_seiz:,}  interictal={n_norm:,})")
    print(f"  Val   : {n_val:>7,}  (ictal={int(y_val.sum()):,}  "
          f"interictal={int((y_val==0).sum()):,})")
    print(f"  Test  : {n_test:>7,}  (ictal={int(y_test.sum()):,}  "
          f"interictal={int((y_test==0).sum()):,})")
    print(f"  Batch={batch_size}  Chunk={chunk_size}  "
          f"Epochs={epochs}  EarlyStop={early_stop}  LR={lr}")
    chunk_mb = chunk_size * N_CHANNELS * WINDOW_SAMP * 4 / 1e6
    print(f"  RAM per chunk : ~{chunk_mb:.0f} MB  (rest stays on disk)")
    print()

    # ── STEP A: Dataset + Sampler + DataLoader ────────────────────────────
    t0 = time.time()
    print(f"  [A] Building DataLoader from mmap ...")
    train_ds      = MmapEEGDataset(CACHE_X_TRAIN, CACHE_Y_TRAIN)
    chunk_sampler = SequentialChunkSampler(n_train, chunk_size=chunk_size)
    train_loader  = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = chunk_sampler,
        num_workers = 0,       # Windows-safe
        pin_memory  = False,   # keep RAM pressure low
        drop_last   = True,
    )
    print(f"  [A] DataLoader ready: {len(train_loader)} batches/epoch  "
          f"[{time.time()-t0:.1f}s]")

    # ── STEP B: First-batch smoke test ────────────────────────────────────
    t0 = time.time()
    print(f"  [B] First-batch test ...")
    it      = iter(train_loader)
    Xb, yb  = next(it)
    del it
    print(f"  [B] OK — X={Xb.shape}  y={yb.shape}  [{time.time()-t0:.2f}s]")

    # ── STEP C: Model ─────────────────────────────────────────────────────
    t0 = time.time()
    print(f"  [C] Building model on {DEVICE} ...")
    model = CNNBiLSTM(
        in_channels    = N_CHANNELS,
        window_samples = WINDOW_SAMP,
        lstm_hidden    = 128,
        lstm_layers    = 2,
        cnn_dropout    = 0.25,
        lstm_dropout   = 0.35,
        fc_dropout     = 0.50,
    ).to(DEVICE)
    print(f"  [C] Parameters: {model.count_params():,}  [{time.time()-t0:.2f}s]")
    if DEVICE.type == "cuda":
        used  = torch.cuda.memory_allocated(0) / 1e6
        total = props.total_memory / 1e6
        print(f"  [C] VRAM: {used:.0f} MB / {total:.0f} MB")

    # ── STEP D: Forward-pass sanity check ─────────────────────────────────
    t0 = time.time()
    print(f"  [D] Forward-pass sanity check ...")
    model.eval()
    with torch.no_grad():
        tb = Xb.to(DEVICE)
        if USE_AMP:
            with autocast(device_type='cuda'):
                _ = model(tb)
        else:
            _ = model(tb)
    del tb, _
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated(0) / 1e6
        print(f"  [D] Forward OK  VRAM={used:.0f} MB  [{time.time()-t0:.3f}s]")

    # ── Loss + Optimiser ──────────────────────────────────────────────────
    pos_w     = n_norm / max(n_seiz, 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(DEVICE)
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6
    )
    scaler = GradScaler(enabled=USE_AMP)

    print(f"\n  pos_weight = {pos_w:.2f}")
    print(f"\n  --- Starting Epoch Training ---\n")

    # ── Training loop ─────────────────────────────────────────────────────
    best_metric = 0.0
    no_improve  = 0
    history     = {'loss': [], 'val_acc': [], 'val_f1': [], 'val_rec': [], 'lr': []}
    current_lr  = lr

    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        train_loss = train_epoch(model, train_loader, criterion,
                                  optimizer, scaler)

        # Validate — streams mmap file sequentially (low RAM)
        val_preds, val_probs = evaluate(model, CACHE_X_VAL, y_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1  = f1_score(y_val, val_preds, zero_division=0)
        val_rec = recall_score(y_val, val_preds, zero_division=0)

        # Use val_acc when val split has no seizure windows
        ckpt_metric = val_f1 if int(y_val.sum()) > 0 else val_acc
        scheduler.step(ckpt_metric)
        new_lr = optimizer.param_groups[0]['lr']

        history['loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_rec'].append(val_rec)
        history['lr'].append(new_lr)

        lr_tag = f" [LR->{new_lr:.1e}]" if new_lr != current_lr else ""
        current_lr = new_lr

        if ckpt_metric > best_metric:
            best_metric = ckpt_metric
            no_improve  = 0
            torch.save(model.state_dict(), CKPT_PATH)
            marker = " <- best"
        else:
            no_improve += 1
            marker = ""

        ep_s = time.time() - t_ep
        if DEVICE.type == "cuda":
            vram_mb = torch.cuda.memory_allocated(0) / 1e6
            vram_tag = f"  VRAM={vram_mb:.0f}MB"
        else:
            vram_tag = ""

        print(f"  Ep {epoch:03d}/{epochs}  "
              f"loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}  "
              f"val_rec={val_rec:.4f}  "
              f"[{ep_s:.0f}s]{vram_tag}{marker}{lr_tag}")

        if no_improve >= early_stop:
            print(f"\n  Early stopping ({early_stop} epochs without improvement).")
            break

        gc.collect()

    # ── Final test evaluation ─────────────────────────────────────────────
    print(f"\n  Loading best checkpoint (metric={best_metric:.4f}) ...")
    best = CNNBiLSTM(in_channels=N_CHANNELS, window_samples=WINDOW_SAMP).to(DEVICE)
    best.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

    y_test_f    = y_test.astype(np.float32)
    test_preds, test_probs = evaluate(best, CACHE_X_TEST, y_test_f)
    metrics = _save_metrics("CNN_BiLSTM", y_test_f, test_preds, test_probs)
    _save_training_history(history)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Metrics + plots
# ─────────────────────────────────────────────────────────────────────────────

def _save_metrics(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred,    zero_division=0)
    f1   = f1_score(y_true, y_pred,        zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = float(auc(fpr, tpr))
    except Exception:
        roc_auc = 0.0; fpr, tpr = np.array([]), np.array([])

    metrics = {
        "model":       name,
        "accuracy":    round(float(acc),  4),
        "precision":   round(float(prec), 4),
        "recall":      round(float(rec),  4),
        "f1_score":    round(float(f1),   4),
        "roc_auc":     round(roc_auc,     4),
        "sensitivity": round(float(tp / max(tp+fn, 1)), 4),
        "specificity": round(float(tn / max(tn+fp, 1)), 4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n  ------ {name} Test Results ------")
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "model"):
            print(f"     {k:<14}: {v}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0, target_names=['Interictal','Ictal'])}")

    with open(os.path.join(METRICS_DIR, "cnn_bilstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(title=f'{name} Confusion Matrix',
           xlabel='Predicted', ylabel='True',
           xticks=[0,1], yticks=[0,1],
           xticklabels=['Interictal','Ictal'],
           yticklabels=['Interictal','Ictal'])
    thr = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white' if cm[i,j] > thr else 'black', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, "cnn_bilstm_confusion_matrix.png"), dpi=150)
    plt.close()

    if len(fpr) > 0:
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.4f}')
        plt.plot([0,1],[0,1],'navy',lw=1.5,linestyle='--')
        plt.xlim([0,1]); plt.ylim([0,1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve'); plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(METRICS_DIR, "cnn_bilstm_roc_curve.png"), dpi=150)
        plt.close()

    return metrics


def _save_training_history(history):
    epochs = range(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['loss'], 'b-', lw=1.5)
    axes[0].set(title='Training Loss', xlabel='Epoch', ylabel='BCE Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['val_f1'],  'r-', lw=1.5, label='F1')
    axes[1].plot(epochs, history['val_acc'], 'g-', lw=1.5, label='Acc')
    axes[1].plot(epochs, history['val_rec'], 'm-', lw=1.5, label='Recall')
    axes[1].set(title='Validation Metrics', xlabel='Epoch',
                ylabel='Score', ylim=[0, 1.05])
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['lr'], 'k-', lw=1.5)
    axes[2].set(title='Learning Rate', xlabel='Epoch', ylabel='LR')
    axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(METRICS_DIR, "cnn_bilstm_training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training history -> {path}")


def generate_comparison_report(cnn_bilstm_metrics: dict):
    print("\n" + "=" * 65)
    print("  GENERATING FINAL COMPARISON REPORT")
    print("=" * 65)

    metric_files = {
        "SVM":        "svm_metrics.json",
        "LR":         "lr_metrics.json",
        "CNN_LTI":    "cnn_lti_metrics.json",
        "CNN_BiLSTM": "cnn_bilstm_metrics.json",
    }
    model_types = {
        "SVM": "Classical ML", "LR": "Classical ML",
        "CNN_LTI": "Deep Learning", "CNN_BiLSTM": "Deep Learning",
    }
    all_models = []
    for key, fname in metric_files.items():
        fpath = os.path.join(METRICS_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                m = json.load(f)
            m['model']      = key
            m['model_type'] = model_types[key]
            all_models.append(m)

    if not all_models:
        print("  No metric files found."); return

    keys   = ['accuracy','precision','recall','f1_score','roc_auc']
    labels = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']
    palette = ['#4C72B0','#55A868','#C44E52','#8172B2']

    x     = np.arange(len(labels))
    width = 0.85 / len(all_models)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (m, color) in enumerate(zip(all_models, palette)):
        vals = [m.get(k, 0) for k in keys]
        bars = ax.bar(x + i*width, vals, width,
                      label=f"{m['model']} ({m['model_type']})",
                      color=color, alpha=0.87, edgecolor='white', lw=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height()+0.004, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x + width*(len(all_models)-1)/2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        "Comparative Analysis: Classical ML vs Deep Learning\n"
        "Epileptic Seizure Detection - CHB-MIT Dataset",
        fontsize=12, fontweight='bold', pad=14)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    chart = os.path.join(METRICS_DIR, "comparison_table.png")
    plt.savefig(chart, dpi=150); plt.close()
    print(f"  Bar chart -> {chart}")

    def _best(key):
        vals = [m.get(key, 0) for m in all_models]
        idx  = int(np.argmax(vals))
        return all_models[idx]['model'], vals[idx]

    best_acc = _best('accuracy'); best_rec = _best('recall')
    best_auc = _best('roc_auc'); best_f1  = _best('f1_score')

    sep = "-" * 80
    hdr = (
        "=" * 80 + "\n"
        "  Comparative Analysis: Classical Machine Learning vs Deep Learning\n"
        "  Epileptic Seizure Detection using the CHB-MIT Scalp EEG Dataset\n"
        "=" * 80 + "\n\n"
        "  Dataset : CHB-MIT Scalp EEG Database (24 patients)\n"
        "  Split   : Train chb01-18 | Val chb19-21 | Test chb22-24\n\n"
    )
    tbl = (
        f"  {'Model':<16} {'Type':<16} "
        f"{'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}\n"
        f"  {sep}\n"
    )
    for m in all_models:
        tbl += (
            f"  {m['model']:<16} {m['model_type']:<16} "
            f"{m.get('accuracy',0):>8.4f} {m.get('precision',0):>8.4f} "
            f"{m.get('recall',0):>8.4f} {m.get('f1_score',0):>8.4f} "
            f"{m.get('roc_auc',0):>8.4f}\n"
        )
    summ = (
        f"\n  WINNERS\n  {sep}\n"
        f"  Best Accuracy  : {best_acc[0]} ({best_acc[1]:.4f})\n"
        f"  Best Recall    : {best_rec[0]} ({best_rec[1]:.4f})\n"
        f"  Best ROC-AUC   : {best_auc[0]} ({best_auc[1]:.4f})\n"
        f"  Best F1-Score  : {best_f1[0]}  ({best_f1[1]:.4f})\n\n"
        f"  CLINICAL NOTE\n  {sep}\n"
        f"  Recall/Sensitivity is the primary metric for seizure detection.\n"
        f"  A missed seizure is more dangerous than a false alarm.\n"
    )
    full = hdr + tbl + summ

    txt = os.path.join(METRICS_DIR, "comparison_report.txt")
    with open(txt, "w", encoding='utf-8') as f:
        f.write(full)
    print(f"  Text report -> {txt}")

    rep = {
        "title":    "Comparative Analysis: Classical ML vs Deep Learning - CHB-MIT",
        "dataset":  "CHB-MIT Scalp EEG Database",
        "patients": 24,
        "split":    {"train":"chb01-18","val":"chb19-21","test":"chb22-24"},
        "models":   all_models,
        "winners": {
            "best_accuracy":  {"model":best_acc[0],"value":best_acc[1]},
            "best_recall":    {"model":best_rec[0],"value":best_rec[1]},
            "best_roc_auc":   {"model":best_auc[0],"value":best_auc[1]},
            "best_f1_score":  {"model":best_f1[0], "value":best_f1[1]},
        },
    }
    jpath = os.path.join(METRICS_DIR, "comparison_report.json")
    with open(jpath, "w", encoding='utf-8') as f:
        json.dump(rep, f, indent=2)
    print(f"  JSON report -> {jpath}")
    print("\n" + full)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    # ── Step 1: Build file lists (summary .txt parsing only) ─────────────
    print("[STEP 1] Parsing CHB-MIT summary files ...")
    train_files = build_file_list_dl(TRAIN_PATIENTS)
    val_files   = build_file_list_dl(VAL_PATIENTS)
    test_files  = build_file_list_dl(TEST_PATIENTS)
    print(f"  Train={len(train_files)}  Val={len(val_files)}  Test={len(test_files)}")

    # ── Step 2: Build caches if not present (EDF loading phase) ──────────
    print("\n[STEP 2] Building / verifying window caches ...")
    _, y_train = build_or_load_cache(train_files, "bilstm_train")
    _, y_val   = build_or_load_cache(val_files,   "bilstm_val")
    _, y_test  = build_or_load_cache(test_files,  "bilstm_test")

    y_train = y_train.astype(np.float32)
    y_val   = y_val.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    total      = len(y_train) + len(y_val) + len(y_test)
    total_seiz = int(y_train.sum() + y_val.sum() + y_test.sum())
    print(f"\n  Total windows : {total:,}")
    print(f"  Seizure       : {total_seiz:,}  ({100*total_seiz/max(total,1):.1f}%)")
    print(f"  Normal        : {total-total_seiz:,}")

    # ── Step 3: Train CNN-BiLSTM (chunk-stream mode, low RAM) ────────────
    print("\n[STEP 3] Training CNN-BiLSTM ...")
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    metrics = train_cnn_bilstm(
        y_train    = y_train,
        y_val      = y_val,
        y_test     = y_test,
        epochs     = 80,
        batch_size = 8,        # 8 x 18 x 1280 x 4 = 1.17 MB per batch
        chunk_size = 512,      # 512 windows = ~47 MB in RAM per chunk
        early_stop = 15,
        lr         = 1e-3,
    )

    # ── Step 4: Comparison report ─────────────────────────────────────────
    print("\n[STEP 4] Generating comparison report ...")
    generate_comparison_report(metrics)

    elapsed = time.time() - t0
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    print(f"\n{'='*65}")
    print(f"  Done.  Model -> {CKPT_PATH}")
    print(f"  Time  -> {h}h {m}m {s}s")
    print(f"{'='*65}\n")
