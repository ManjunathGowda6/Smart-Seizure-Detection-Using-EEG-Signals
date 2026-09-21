"""
train_bilstm.py  (PyTorch version — Python 3.13 compatible)
=============================================================
Phase 1 — CNN-BiLSTM on Bonn University EEG CSV + CHB-MIT.

Fix for previous models: all predicted 100% "no seizure" due to 97.7% class imbalance.
Solution: SMOTE oversampling to 1:1 balance before training.

Note: TensorFlow does NOT support Python 3.13.
      This script uses PyTorch (already installed and CUDA-enabled).
      Model saved as cnn_bilstm_v1.pt (equivalent to .h5 for PyTorch).

Output files:
  backend/models/cnn_bilstm_v1.pt        <- trained model
  backend/models/bilstm_metadata.json    <- metrics + config
  backend/models/bilstm_label_encoder.pkl
  backend/ml/plots/confusion_matrix_bilstm.png
  backend/ml/plots/roc_curve_bilstm.png
  backend/ml/plots/training_history_bilstm.png
"""

import os, sys, json, pickle, time, warnings, subprocess
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Auto-install missing packages (except tensorflow — incompatible with Py3.13)
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED = {"imbalanced-learn": "imblearn", "seaborn": "seaborn",
            "scipy": "scipy", "scikit-learn": "sklearn"}

for pkg, imp in REQUIRED.items():
    try:
        __import__(imp)
    except ImportError:
        print(f"  Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Imports
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import resample as scipy_resample

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, recall_score, precision_score,
    accuracy_score,
)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = r"E:\Downloads\Finalyearproject\backend"
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "ml", "plots")
CACHE_DIR  = os.path.join(MODELS_DIR, "cache")
BONN_CSV   = r"E:\Downloads\Epileptic Seizure Recognition.csv"

MODEL_SAVE = os.path.join(MODELS_DIR, "cnn_bilstm_v1.pt")
LABEL_ENC  = os.path.join(MODELS_DIR, "bilstm_label_encoder.pkl")
META_JSON  = os.path.join(MODELS_DIR, "bilstm_metadata.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

TIMESTEPS  = 178
N_FEATURES = 1
BATCH      = 32
EPOCHS     = 100
SEED       = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Device
# ─────────────────────────────────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

print("=" * 62)
print("  Phase 1 — CNN-BiLSTM (PyTorch, Python 3.13 compatible)")
print("=" * 62)
print(f"  PyTorch  : {torch.__version__}")
print(f"  Device   : {DEVICE}")
if DEVICE.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU      : {p.name}  |  VRAM: {p.total_memory//1024**2} MB")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Load Bonn University EEG CSV
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 1]  Loading Bonn University EEG CSV ...")

df = pd.read_csv(BONN_CSV)
print(f"  Raw shape  : {df.shape}")

# Detect label column (last non-unnamed column named 'y' or last column)
label_col    = "y" if "y" in df.columns else df.columns[-1]
feature_cols = [c for c in df.columns
                if c != label_col and not str(c).lower().startswith("unnamed")]

X_bonn = df[feature_cols].values.astype(np.float32)  # (n, 178)
y_raw  = df[label_col].values

# y==1 -> seizure (Set E), y==2,3,4,5 -> normal
y_bonn = (y_raw == 1).astype(np.int32)

print(f"  Features   : {X_bonn.shape[1]}  (expecting 178)")
print(f"  Samples    : {len(y_bonn):,}")
print(f"  Seizure(1) : {y_bonn.sum():,}")
print(f"  Normal (0) : {(y_bonn==0).sum():,}")

# Per-sample z-score normalisation
mu  = X_bonn.mean(axis=1, keepdims=True)
sg  = X_bonn.std(axis=1,  keepdims=True) + 1e-8
X_bonn = (X_bonn - mu) / sg

# Resample to TIMESTEPS if needed
if X_bonn.shape[1] != TIMESTEPS:
    print(f"  Resampling {X_bonn.shape[1]} -> {TIMESTEPS} ...")
    X_bonn = scipy_resample(X_bonn, TIMESTEPS, axis=1).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Load CHB-MIT (channel 0, resampled 1280->178, low-RAM chunks)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2]  Loading CHB-MIT cache (channel 0, chunk-resampled) ...")

CHB_X = os.path.join(CACHE_DIR, "X_cnn_train.npy")
CHB_Y = os.path.join(CACHE_DIR, "y_cnn_train.npy")

X_chb, y_chb = None, None
datasets_used = ["bonn_csv"]

if os.path.exists(CHB_X) and os.path.exists(CHB_Y):
    y_chb = np.load(CHB_Y).astype(np.int32)
    n_chb = len(y_chb)
    print(f"  CHB-MIT windows : {n_chb:,}  "
          f"seizure={y_chb.sum():,}  normal={(y_chb==0).sum():,}")

    X_mmap   = np.load(CHB_X, mmap_mode='r')
    CHUNK    = 500                            # ~23 MB per chunk (ch0 resamp)
    chunks   = []

    print(f"  Extracting ch0 + resample 1280->{TIMESTEPS} in chunks of {CHUNK} ...")
    for s in range(0, n_chb, CHUNK):
        e     = min(s + CHUNK, n_chb)
        c     = X_mmap[s:e, 0, :].copy()     # (chunk, 1280)
        c     = scipy_resample(c, TIMESTEPS, axis=1).astype(np.float32)
        mu_c  = c.mean(axis=1, keepdims=True)
        sg_c  = c.std(axis=1,  keepdims=True) + 1e-8
        chunks.append((c - mu_c) / sg_c)
        if (s // CHUNK) % 10 == 0:
            print(f"    {s}-{e} / {n_chb}")

    X_chb = np.vstack(chunks)
    del chunks, X_mmap
    print(f"  CHB-MIT X shape : {X_chb.shape}")
    datasets_used.append("chbmit")
else:
    print("  CHB-MIT preprocessed arrays not found — training on Bonn dataset only")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SMOTE — 1:1 class balance
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3]  Applying SMOTE (1:1 balance) ...")

def smote_balance(X2d, y, tag):
    print(f"  [{tag}] BEFORE  seizure={y.sum():,}  normal={(y==0).sum():,}")
    sm = SMOTE(random_state=SEED, k_neighbors=min(5, int(y.sum())-1))
    Xr, yr = sm.fit_resample(X2d, y)
    print(f"  [{tag}] AFTER   seizure={yr.sum():,}  normal={(yr==0).sum():,}")
    return Xr.astype(np.float32), yr.astype(np.int32)

X_bonn_bal, y_bonn_bal = smote_balance(X_bonn, y_bonn, "Bonn")
del X_bonn, y_bonn

if X_chb is not None:
    X_chb_bal, y_chb_bal = smote_balance(X_chb, y_chb, "CHB-MIT")
    del X_chb, y_chb

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Merge + shuffle + 70/15/15 split
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 4]  Merging, shuffling, splitting 70/15/15 ...")

if "chbmit" in datasets_used:
    X_all = np.vstack([X_bonn_bal, X_chb_bal])
    y_all = np.concatenate([y_bonn_bal, y_chb_bal])
    del X_bonn_bal, X_chb_bal, y_bonn_bal, y_chb_bal
else:
    X_all, y_all = X_bonn_bal, y_bonn_bal

rng = np.random.default_rng(SEED)
idx = rng.permutation(len(y_all))
X_all, y_all = X_all[idx], y_all[idx]

print(f"  Total  : {len(y_all):,}  seizure={y_all.sum():,}  normal={(y_all==0).sum():,}")

X_tmp,  X_test, y_tmp,  y_test  = train_test_split(
    X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
X_train, X_val, y_train, y_val  = train_test_split(
    X_tmp, y_tmp, test_size=round(0.15/0.85, 6),
    random_state=SEED, stratify=y_tmp)
del X_all, y_all, X_tmp, y_tmp

# Reshape -> (n, 1, timesteps)  [PyTorch: batch, channels, time]
X_train_t = torch.tensor(X_train[:, np.newaxis, :])   # (n, 1, 178)
X_val_t   = torch.tensor(X_val  [:, np.newaxis, :])
X_test_t  = torch.tensor(X_test [:, np.newaxis, :])
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

print(f"  Train  : {X_train_t.shape}  seizure={int(y_train.sum()):,}")
print(f"  Val    : {X_val_t.shape}    seizure={int(y_val.sum()):,}")
print(f"  Test   : {X_test_t.shape}   seizure={int(y_test.sum()):,}")

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t,   y_val_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                          num_workers=0)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  CNN-BiLSTM model (PyTorch)
#     Same architecture as Keras spec but in PyTorch syntax.
#     Input shape: (batch, 1, 178)  — channel-first
# ─────────────────────────────────────────────────────────────────────────────
class CNNBiLSTMPhase1(nn.Module):
    """
    Conv1D(64,5) -> BN -> MaxPool -> Dropout
    Conv1D(128,3) -> BN -> MaxPool -> Dropout
    BiLSTM(64, seq=True) -> Dropout
    BiLSTM(32, seq=False) -> Dropout
    Dense(64) -> Dropout -> Dense(1, sigmoid)
    """
    def __init__(self, timesteps: int = TIMESTEPS):
        super().__init__()
        # CNN block 1
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )
        # CNN block 2
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )
        # After 2x MaxPool(2): timesteps -> 178//4 = 44
        lstm_in = timesteps // 4
        # BiLSTM block 1
        self.bilstm1 = nn.LSTM(128, 64, batch_first=True,
                               bidirectional=True)   # out: (batch, T, 128)
        self.drop3 = nn.Dropout(0.3)
        # BiLSTM block 2
        self.bilstm2 = nn.LSTM(128, 32, batch_first=True,
                               bidirectional=True)   # out: (batch, T, 64)
        self.drop4 = nn.Dropout(0.3)
        # FC head
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            # No sigmoid here — use BCEWithLogitsLoss for numerical stability
        )

    def forward(self, x):
        # x: (batch, 1, 178)
        x = self.cnn1(x)           # (batch, 64, 89)
        x = self.cnn2(x)           # (batch, 128, 44)
        x = x.permute(0, 2, 1)    # (batch, 44, 128) — LSTM expects (B, T, F)

        x, _ = self.bilstm1(x)    # (batch, 44, 128)
        x = self.drop3(x)
        x, _ = self.bilstm2(x)    # (batch, 44, 64)
        x = self.drop4(x)

        x = x[:, -1, :]           # take last timestep: (batch, 64)
        return self.fc(x).squeeze(1)   # (batch,)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

model = CNNBiLSTMPhase1(TIMESTEPS).to(DEVICE)
print(f"\n[STEP 5]  Model parameters: {model.count_params():,}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Loss + Optimiser
#     class_weight {0:1.0, 1:2.0} -> pos_weight=2.0 in BCEWithLogitsLoss
# ─────────────────────────────────────────────────────────────────────────────
pos_weight = torch.tensor([2.0], dtype=torch.float32).to(DEVICE)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = optim.Adam(model.parameters(), lr=1e-3)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
)
scaler = GradScaler(enabled=USE_AMP)

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Training loop  (EarlyStopping on val_recall, patience=15)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[STEP 6]  Training  Epochs={EPOCHS}  Batch={BATCH}  "
      f"pos_weight=2.0  EarlyStop=15 on val_recall\n")

def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            if train and USE_AMP:
                with autocast(device_type='cuda'):
                    logits = model(Xb)
                    loss   = criterion(logits, yb)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                logits = model(Xb)
                loss   = criterion(logits, yb)
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item() * Xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.long().cpu().numpy())

    n = len(all_labels)
    avg_loss  = total_loss / max(n, 1)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    acc  = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, rec, prec

best_val_recall = 0.0
patience_count  = 0
PATIENCE        = 15
hist = {"loss":[], "val_loss":[], "acc":[], "val_acc":[],
        "recall":[], "val_recall":[], "precision":[], "val_precision":[]}

t_start = time.time()
for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()

    tr_loss, tr_acc, tr_rec, tr_prec = run_epoch(train_loader, train=True)
    vl_loss, vl_acc, vl_rec, vl_prec = run_epoch(val_loader,   train=False)

    scheduler.step(vl_loss)

    hist["loss"].append(tr_loss);         hist["val_loss"].append(vl_loss)
    hist["acc"].append(tr_acc);           hist["val_acc"].append(vl_acc)
    hist["recall"].append(tr_rec);        hist["val_recall"].append(vl_rec)
    hist["precision"].append(tr_prec);    hist["val_precision"].append(vl_prec)

    ep_s = time.time() - t_ep

    # Save best on val_recall
    if vl_rec > best_val_recall:
        best_val_recall = vl_rec
        patience_count  = 0
        torch.save(model.state_dict(), MODEL_SAVE)
        ckpt_tag = "  <- best"
    else:
        patience_count += 1
        ckpt_tag = ""

    print(f"  Ep {epoch:03d}/{EPOCHS}"
          f"  loss={tr_loss:.4f}  val_loss={vl_loss:.4f}"
          f"  val_rec={vl_rec:.4f}  val_prec={vl_prec:.4f}"
          f"  [{ep_s:.0f}s]{ckpt_tag}")

    if patience_count >= PATIENCE:
        print(f"\n  EarlyStopping: val_recall did not improve for {PATIENCE} epochs.")
        break

elapsed = time.time() - t_start
h, r = divmod(int(elapsed), 3600)
m, s = divmod(r, 60)
print(f"\n  Training time: {h}h {m}m {s}s")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Test evaluation (best checkpoint)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 7]  Evaluating on test set (best checkpoint) ...")

model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
model.eval()

all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(DEVICE)
        logits = model(Xb)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(yb.numpy().astype(int))

y_prob_arr  = np.array(all_probs)
y_pred_arr  = np.array(all_preds)
y_true_arr  = np.array(all_labels)

cm               = confusion_matrix(y_true_arr, y_pred_arr)
tn, fp, fn, tp   = cm.ravel()
roc_auc          = roc_auc_score(y_true_arr, y_prob_arr)
report           = classification_report(
    y_true_arr, y_pred_arr,
    target_names=["Normal (0)", "Seizure (1)"],
    output_dict=True, zero_division=0)

seizure_recall    = report["Seizure (1)"]["recall"]
seizure_precision = report["Seizure (1)"]["precision"]
seizure_f1        = report["Seizure (1)"]["f1-score"]
accuracy          = report["accuracy"]

print(f"\n  Confusion Matrix:   TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
print()
print(classification_report(y_true_arr, y_pred_arr,
      target_names=["Normal (0)", "Seizure (1)"], zero_division=0))
print(f"  ROC-AUC : {roc_auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  Save plots
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 8]  Saving plots ...")

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Normal","Seizure"], yticklabels=["Normal","Seizure"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("CNN-BiLSTM Phase1 - Confusion Matrix")
plt.tight_layout()
p1 = os.path.join(PLOTS_DIR, "confusion_matrix_bilstm.png")
plt.savefig(p1, dpi=150); plt.close()
print(f"  -> {p1}")

# ROC curve
fpr_a, tpr_a, _ = roc_curve(y_true_arr, y_prob_arr)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr_a, tpr_a, color="darkorange", lw=2, label=f"AUC={roc_auc:.4f}")
ax.plot([0,1],[0,1],"navy",lw=1.5,linestyle="--")
ax.set(xlim=[0,1], ylim=[0,1.05],
       xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="CNN-BiLSTM Phase1 - ROC Curve")
ax.legend(loc="lower right")
plt.tight_layout()
p2 = os.path.join(PLOTS_DIR, "roc_curve_bilstm.png")
plt.savefig(p2, dpi=150); plt.close()
print(f"  -> {p2}")

# Training history
eps = range(1, len(hist["loss"]) + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(eps, hist["loss"],     "b-", lw=1.5, label="Train")
axes[0].plot(eps, hist["val_loss"], "r-", lw=1.5, label="Val")
axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE"); axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(eps, hist["acc"],     "b-", lw=1.5, label="Train")
axes[1].plot(eps, hist["val_acc"], "r-", lw=1.5, label="Val")
axes[1].set(title="Accuracy", xlabel="Epoch", ylim=[0,1.05]); axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(eps, hist["recall"],         "b-",  lw=1.5, label="Train Recall")
axes[2].plot(eps, hist["val_recall"],     "r-",  lw=1.5, label="Val Recall")
axes[2].plot(eps, hist["precision"],      "b--", lw=1.5, label="Train Prec")
axes[2].plot(eps, hist["val_precision"],  "r--", lw=1.5, label="Val Prec")
axes[2].set(title="Recall & Precision", xlabel="Epoch", ylim=[0,1.05])
axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
p3 = os.path.join(PLOTS_DIR, "training_history_bilstm.png")
plt.savefig(p3, dpi=150); plt.close()
print(f"  -> {p3}")

# ─────────────────────────────────────────────────────────────────────────────
# 13.  Save metadata + label encoder
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 9]  Saving metadata ...")

with open(LABEL_ENC, "wb") as f:
    pickle.dump({0:"normal", 1:"seizure"}, f)

metadata = {
    "model_name":       "cnn_bilstm_v1",
    "framework":        f"PyTorch {torch.__version__}",
    "timesteps":        TIMESTEPS,
    "n_features":       N_FEATURES,
    "input_shape":      [1, TIMESTEPS],
    "datasets_used":    datasets_used,
    "seizure_recall":   round(seizure_recall,    4),
    "seizure_precision": round(seizure_precision, 4),
    "f1_score":         round(seizure_f1,         4),
    "roc_auc":          round(roc_auc,            4),
    "accuracy":         round(accuracy,           4),
    "threshold":        0.5,
    "class_weight":     {"0": 1.0, "1": 2.0},
    "trained_on":       datetime.now().isoformat(),
    "training_time_sec": int(elapsed),
    "epochs_run":       len(hist["loss"]),
    "confusion_matrix": cm.tolist(),
}
with open(META_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved: {META_JSON}")

# ─────────────────────────────────────────────────────────────────────────────
# 14.  Phase 1 pass/fail + summary
# ─────────────────────────────────────────────────────────────────────────────
NEED_REC  = 0.75; NEED_PREC = 0.70; NEED_ACC = 0.85
passed = (seizure_recall >= NEED_REC and
          seizure_precision >= NEED_PREC and
          accuracy >= NEED_ACC)

print("\n" + "=" * 62)
print("  PHASE 1 TRAINING COMPLETE")
print("=" * 62)
print(f"  Dataset        : {' + '.join(d.upper() for d in datasets_used)} (SMOTE balanced)")
print(f"  Model          : CNN-BiLSTM v1  ({model.count_params():,} params)")
print(f"  Framework      : PyTorch {torch.__version__}  (Python 3.13 compatible)")
print(f"  Training time  : {h}h {m}m {s}s")
print(f"  Epochs run     : {len(hist['loss'])}")
print()
print(f"  TEST RESULTS:")
print(f"  Accuracy       : {accuracy*100:.2f}%")
print(f"  Seizure Recall : {seizure_recall*100:.2f}%   <- did we catch seizures?")
print(f"  Seizure Precis : {seizure_precision*100:.2f}%")
print(f"  F1-Score       : {seizure_f1*100:.2f}%")
print(f"  ROC-AUC        : {roc_auc*100:.2f}%")
print()
print(f"  Model saved    : {MODEL_SAVE}")
print(f"  Plots saved    : {PLOTS_DIR}")
print("=" * 62)

if passed:
    print("  PHASE 1 TARGETS MET:")
    print(f"    Recall    {seizure_recall:.2f} >= {NEED_REC}  OK")
    print(f"    Precision {seizure_precision:.2f} >= {NEED_PREC}  OK")
    print(f"    Accuracy  {accuracy:.2f} >= {NEED_ACC}  OK")
else:
    print("  Phase 1 target not met — suggestions:")
    if seizure_recall < NEED_REC:
        print(f"    Recall {seizure_recall:.2f} < {NEED_REC}"
              f"  -> increase pos_weight from 2.0 to 3.0 (line ~170)")
    if seizure_precision < NEED_PREC:
        print(f"    Precision {seizure_precision:.2f} < {NEED_PREC}"
              f"  -> reduce pos_weight or lower threshold below 0.5")
    if accuracy < NEED_ACC:
        print(f"    Accuracy {accuracy:.2f} < {NEED_ACC}"
              f"  -> train more epochs or increase BATCH")

print("=" * 62 + "\n")
