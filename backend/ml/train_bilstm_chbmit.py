"""
train_bilstm_chbmit.py
======================
Retrain CNN-BiLSTM on CHB-MIT preprocessed numpy arrays.

WHY THIS EXISTS
  cnn_bilstm_v1.pt was trained on Bonn University CSV data (single-channel,
  different sampling rate, different signal morphology).  It consistently
  outputs NORMAL on CHB-MIT EDF uploads because the signal characteristics
  are completely foreign to it.

  This script trains the SAME architecture on the cached CHB-MIT numpy arrays
  that were produced during the original CNN/BiLSTM training pipeline.

SAVED MODEL
  backend/models/cnn_bilstm_chbmit.pt

SUCCESS CRITERION
  Seizure Recall > 0.75 on CHB-MIT test set at sub-window level.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, recall_score,
    precision_score, f1_score, accuracy_score
)
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(_THIS_DIR, '..', 'models', 'cache')
MODELS_DIR = os.path.join(_THIS_DIR, '..', 'models')
SAVE_PATH  = os.path.join(MODELS_DIR, 'cnn_bilstm_chbmit_v2.pt')

# Make sure the architecture module is importable
sys.path.insert(0, _THIS_DIR)
from cnn_bilstm_v1_arch import CNNBiLSTMPhase1

# ── Hyperparameters ────────────────────────────────────────────────────────────
WIN_SIZE             = 178     # sub-window samples — must match inference code
WIN_STEP             = 89      # 50 % overlap
BATCH_SIZE           = 512
EPOCHS               = 30
LR                   = 5e-4
WEIGHT_DECAY         = 1e-4
POS_WEIGHT           = 5.0     # BCEWithLogitsLoss weight for seizure class
SMOTE_RATIO          = 1.0     # target seizure:normal = 1:1 after SMOTE
SMOTE_K              = 5       # k-nearest neighbours for SMOTE
THRESHOLD            = 0.30    # sigmoid threshold for classification
EARLY_STOP_PATIENCE  = 20
DEVICE               = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════════════════════════
print('=' * 65)
print('  CNN-BiLSTM  |  CHB-MIT Domain Retraining')
print('=' * 65)
print(f'  PyTorch  : {torch.__version__}')
print(f'  Device   : {DEVICE}')
if DEVICE.type == 'cuda':
    vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM     : {vram} MB')
print(f'  WIN_SIZE : {WIN_SIZE}  WIN_STEP: {WIN_STEP}')
print(f'  POS_WEIGHT: {POS_WEIGHT}  SMOTE_RATIO: {SMOTE_RATIO}')
print('=' * 65)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load cached CHB-MIT numpy arrays
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 1] Loading CHB-MIT cached numpy arrays ...')

X_train_raw = np.load(
    os.path.join(CACHE_DIR, 'dl_bilstm_train_X.npy'), mmap_mode='r'
)  # (32834, 18, 1280)
y_train_raw = np.load(os.path.join(CACHE_DIR, 'dl_bilstm_train_y.npy'))

X_test_raw  = np.load(
    os.path.join(CACHE_DIR, 'dl_bilstm_test_X.npy'), mmap_mode='r'
)  # (4968, 18, 1280)
y_test_raw  = np.load(os.path.join(CACHE_DIR, 'dl_bilstm_test_y.npy'))

n_tr_sz  = int((y_train_raw == 1).sum())
n_tr_nm  = int((y_train_raw == 0).sum())
n_te_sz  = int((y_test_raw  == 1).sum())
n_te_nm  = int((y_test_raw  == 0).sum())

print(f'  Train : X={X_train_raw.shape}  seizure={n_tr_sz:,}  normal={n_tr_nm:,}  '
      f'ratio={n_tr_sz/(n_tr_sz+n_tr_nm)*100:.1f}%')
print(f'  Test  : X={X_test_raw.shape}   seizure={n_te_sz:,}   normal={n_te_nm:,}  '
      f'ratio={n_te_sz/(n_te_sz+n_te_nm)*100:.1f}%')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Sliding-window extraction
#   CHB-MIT shape: (N, 18, 1280)
#   → take channel 0   → (N, 1280)
#   → slide WIN_SIZE=178 step=89  → (N×K, 178) sub-windows
#   → z-score per sub-window, clip ±6σ
#   → remove flat windows (std < 1e-3)
# ══════════════════════════════════════════════════════════════════════════════
def extract_subwindows(X_raw, y_raw, tag=''):
    """
    Vectorised sub-window extraction.
    Returns X_sub (M, 178) float32, y_sub (M,) int32.
    """
    starts  = list(range(0, X_raw.shape[2] - WIN_SIZE + 1, WIN_STEP))
    n_wins  = len(starts)   # sub-windows per parent segment
    N       = len(X_raw)

    print(f'  [{tag}] parent segments={N:,}  '
          f'sub-windows per segment={n_wins}  '
          f'expected total={N*n_wins:,}')

    # Load channel 0 from mmap into RAM: (N, 1280) float32 (~169 MB for train)
    ch0 = np.array(X_raw[:, 0, :], dtype=np.float32)      # (N, 1280)

    # Vectorised stack: list of (N, WIN_SIZE) arrays → (N, K, WIN_SIZE)
    subs_raw  = np.stack(
        [ch0[:, s:s + WIN_SIZE] for s in starts], axis=1
    )                                                       # (N, K, 178)
    subs_flat = subs_raw.reshape(-1, WIN_SIZE)              # (N*K, 178)
    y_sub     = np.repeat(y_raw, n_wins).astype(np.int32)  # (N*K,)

    del ch0, subs_raw   # free memory

    # ── z-score normalise per sub-window ──────────────────────────────────────
    mean      = subs_flat.mean(axis=1, keepdims=True)
    raw_std   = subs_flat.std(axis=1)                      # (N*K,)
    std_col   = raw_std[:, None] + 1e-8
    subs_norm = np.clip((subs_flat - mean) / std_col, -6.0, 6.0).astype(np.float32)

    # ── drop flat / artifact windows ─────────────────────────────────────────
    mask      = raw_std >= 1e-3
    X_out     = subs_norm[mask]
    y_out     = y_sub[mask]

    n_sz  = int((y_out == 1).sum())
    n_nm  = int((y_out == 0).sum())
    print(f'  [{tag}] after extraction: {len(X_out):,} sub-windows  '
          f'seizure={n_sz:,}  normal={n_nm:,}')
    return X_out, y_out


print('\n[STEP 2] Extracting 178-sample sliding sub-windows (50% overlap) ...')
t0 = time.time()
X_sub_train, y_sub_train = extract_subwindows(X_train_raw, y_train_raw, 'train')
X_sub_test,  y_sub_test  = extract_subwindows(X_test_raw,  y_test_raw,  'test')
print(f'  Extraction done in {time.time()-t0:.1f}s')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SMOTE to balance training sub-windows (seizure:normal = 1:1)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n[STEP 3] Applying SMOTE  (target ratio {SMOTE_RATIO:.0%}) ...')
print(f'  Before: seizure={y_sub_train.sum():,}  normal={(y_sub_train==0).sum():,}')

t0 = time.time()
smote = SMOTE(
    sampling_strategy=SMOTE_RATIO,
    k_neighbors=SMOTE_K,
    random_state=42,
)
X_bal, y_bal = smote.fit_resample(X_sub_train, y_sub_train)
del X_sub_train, y_sub_train   # free ~300 MB

print(f'  After : seizure={y_bal.sum():,}  normal={(y_bal==0).sum():,}')
print(f'  Total : {len(X_bal):,} training sub-windows')
print(f'  SMOTE done in {time.time()-t0:.1f}s')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Train / Validation split (stratified 90 / 10)
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 4] Stratified 90/10 train-val split ...')
X_tr, X_val, y_tr, y_val = train_test_split(
    X_bal, y_bal, test_size=0.10, stratify=y_bal, random_state=42
)
del X_bal, y_bal

print(f'  Train: {len(X_tr):,}  seizure={y_tr.sum():,}  normal={(y_tr==0).sum():,}')
print(f'  Val  : {len(X_val):,}  seizure={y_val.sum():,}  normal={(y_val==0).sum():,}')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Build DataLoaders
#   Input shape: (N, 178) → unsqueeze(1) → (N, 1, 178)  [Conv1D: B, C, T]
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 5] Building DataLoaders ...')

X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32).unsqueeze(1)
y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

train_ds     = TensorDataset(X_tr_t,  y_tr_t)
val_ds       = TensorDataset(X_val_t, y_val_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(DEVICE.type=='cuda'))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE.type=='cuda'))
print(f'  Train batches: {len(train_loader):,}')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Model, loss, optimiser
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 6] Building CNN-BiLSTM model ...')
model = CNNBiLSTMPhase1(timesteps=WIN_SIZE).to(DEVICE)
print(f'  Parameters: {model.count_params():,}')

# BCEWithLogitsLoss + pos_weight = {0:1.0, 1:POS_WEIGHT}
# Seizure (positive class) gets POS_WEIGHT=3.0x higher gradient contribution.
# This is ON TOP of SMOTE balancing — extra emphasis so recall stays high.
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT], dtype=torch.float32).to(DEVICE)
)
optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Training loop
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 7] Training ...')
print(f'  {"Epoch":>5}  {"tr_loss":>9}  {"val_loss":>9}  {"val_recall":>10}  {"val_prec":>9}')
print('  ' + '-'*52)

best_val_loss  = float('inf')
best_state     = None
patience_count = 0

for epoch in range(1, EPOCHS + 1):
    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    tr_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient clip
        optimizer.step()
        tr_loss += loss.item() * len(Xb)
    tr_loss /= len(train_ds)

    # ── Validate ───────────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    vp = []; vl = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            val_loss += criterion(logits, yb).item() * len(Xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            vp.extend((probs >= THRESHOLD).astype(int).tolist())
            vl.extend(yb.cpu().numpy().tolist())
    val_loss  /= len(val_ds)
    val_recall = recall_score(vl, vp, zero_division=0)
    val_prec   = precision_score(vl, vp, zero_division=0)

    scheduler.step(val_loss)

    # ── Early stopping ──────────────────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_count = 0
        marker = ' *'
    else:
        patience_count += 1
        marker = ''

    lr_now = optimizer.param_groups[0]['lr']
    print(f'  {epoch:>5}  {tr_loss:>9.4f}  {val_loss:>9.4f}  '
          f'{val_recall:>10.4f}  {val_prec:>9.4f}  lr={lr_now:.2e}{marker}')

    if patience_count >= EARLY_STOP_PATIENCE:
        print(f'\n  Early stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})')
        break

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Evaluate on CHB-MIT test set (NO SMOTE — real distribution)
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 8] Evaluating on CHB-MIT test sub-windows ...')
model.load_state_dict(best_state)
model.eval()

X_ts_t   = torch.tensor(X_sub_test, dtype=torch.float32).unsqueeze(1)
y_ts_arr = y_sub_test
test_ds  = TensorDataset(X_ts_t, torch.tensor(y_ts_arr, dtype=torch.float32))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

all_preds  = []
all_labels = []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(DEVICE)
        probs = torch.sigmoid(model(Xb)).cpu().numpy()
        all_preds.extend((probs >= THRESHOLD).astype(int).tolist())
        all_labels.extend(yb.numpy().tolist())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

recall    = recall_score(all_labels, all_preds, zero_division=0)
precision = precision_score(all_labels, all_preds, zero_division=0)
f1        = f1_score(all_labels, all_preds, zero_division=0)
accuracy  = accuracy_score(all_labels, all_preds)

print()
print('=' * 60)
print('  TEST RESULTS  (CHB-MIT sub-window level)')
print('=' * 60)
print(classification_report(
    all_labels, all_preds,
    target_names=['Normal', 'Seizure'],
    digits=4
))
print(f'  Seizure Recall   : {recall:.4f}')
print(f'  Seizure Precision: {precision:.4f}')
print(f'  Seizure F1       : {f1:.4f}')
print(f'  Accuracy         : {accuracy:.4f}')
print()
if recall >= 0.75:
    print(f'  [SUCCESS] Recall {recall:.2%} meets > 75% target')
else:
    print(f'  [WARNING] Recall {recall:.2%} < 75% target')
    print('    Options: lower THRESHOLD (try 0.20), increase POS_WEIGHT (try 5.0),')
    print('    or increase EPOCHS (try 50).')
print('=' * 60)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Save model
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n[STEP 9] Saving model to {SAVE_PATH}')
torch.save(best_state, SAVE_PATH)
size_kb = os.path.getsize(SAVE_PATH) // 1024
print(f'  Saved: {size_kb} KB')
print('\n  Existing models are intact:')
for f in ['cnn_bilstm_v1.pt', 'cnn_bilstm_chbmit.pt', 'cnn_bilstm_chbmit_v2.pt', 'cnn_lti_model.pt']:
    p = os.path.join(MODELS_DIR, f)
    if os.path.exists(p):
        print(f'    {f}  ({os.path.getsize(p)//1024} KB)')

print('\n=== Done. Run backend and select CNN-BiLSTM on Quick Scan page. ===')
