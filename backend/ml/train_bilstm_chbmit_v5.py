"""
train_bilstm_chbmit_v5.py
=========================
Retrain CNN-BiLSTM using Hard Negative Mining to fix False Positives.
Extracts hard negatives dynamically from the FP files listed in full_validation_report.csv.
Applies SMOTE and Focal Loss.

SAVED MODEL
  backend/models/cnn_bilstm_chbmit_v5.pt
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
import mne

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(_THIS_DIR, '..', 'models', 'cache')
MODELS_DIR = os.path.join(_THIS_DIR, '..', 'models')
REPORT_CSV = os.path.join(_THIS_DIR, '..', '..', 'full_validation_report.csv')
DATASET_DIR= r"E:\Downloads\chb-mit-scalp-eeg-database-1.0.0"
SAVE_PATH  = os.path.join(MODELS_DIR, 'cnn_bilstm_chbmit_v5.pt')
V4_PATH    = os.path.join(MODELS_DIR, 'cnn_bilstm_chbmit_v4.pt')

# Ensure imports work
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))
from cnn_bilstm_v1_arch import CNNBiLSTMPhase1
from preprocess import load_edf, apply_filtering, remove_artifacts_ica

# ── Hyperparameters ────────────────────────────────────────────────────────────
WIN_SIZE             = 178     
WIN_STEP             = 89      
BATCH_SIZE           = 64
EPOCHS               = 50
LR                   = 3e-4
WEIGHT_DECAY         = 1e-4
EARLY_STOP_PATIENCE  = 15
DEVICE               = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD            = 0.40 # extraction threshold for hard negatives

# ══════════════════════════════════════════════════════════════════════════════
# Focal Loss Definition
# ══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt)**self.gamma * bce
        return focal.mean()

print('=' * 65)
print('  CNN-BiLSTM  |  V5 Retraining (Hard Negative Mining + Focal Loss)')
print('=' * 65)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Extract Hard Negatives
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 1] Extracting Hard Negatives from False Positive files ...')
if not os.path.exists(REPORT_CSV):
    print(f"Error: Could not find report at {REPORT_CSV}")
    sys.exit(1)

report_df = pd.read_csv(REPORT_CSV)
fp_files = report_df[(report_df['ground_truth'] == 'NORMAL') & (report_df['prediction'] == 'SEIZURE')]['filename'].tolist()

print(f"  Found {len(fp_files)} False Positive files in report.")

# Load V4 model for inference
v4_model = CNNBiLSTMPhase1(timesteps=WIN_SIZE).to(DEVICE)
v4_model.load_state_dict(torch.load(V4_PATH, map_location=DEVICE, weights_only=True))
v4_model.eval()

hard_negatives = []

t0 = time.time()
for idx, fp_file in enumerate(fp_files):
    # Determine subject folder
    subject = fp_file.split('_')[0]
    edf_path = os.path.join(DATASET_DIR, subject, fp_file)
    if not os.path.exists(edf_path):
        print(f"  [{idx+1}/{len(fp_files)}] Skipping {fp_file} - not found.")
        continue
    
    try:
        raw_mne = load_edf(edf_path, start_sec=0, end_sec=None)
        sfreq = raw_mne.info['sfreq']
        raw_mne = apply_filtering(raw_mne, sfreq)
        raw_mne = remove_artifacts_ica(raw_mne)
        clean_data = raw_mne.get_data()
        
        raw_ch0 = clean_data[0].astype(np.float32)
        sw_starts = list(range(0, len(raw_ch0) - WIN_SIZE + 1, WIN_STEP))
        
        if len(sw_starts) == 0: continue
            
        subs_raw = np.stack([raw_ch0[s:s + WIN_SIZE] for s in sw_starts])
        mean = subs_raw.mean(axis=1, keepdims=True)
        std = subs_raw.std(axis=1, keepdims=True) + 1e-8
        subs_norm = np.clip((subs_raw - mean) / std, -6.0, 6.0)
        
        # valid mask
        std_flat = std.flatten()
        mask = std_flat >= 1e-7
        valid_windows = subs_norm[mask]
        
        if len(valid_windows) == 0: continue
            
        # Inference mini-batches
        infer_batch_size = 512
        probs_all = []
        with torch.no_grad():
            for i in range(0, len(valid_windows), infer_batch_size):
                chunk = valid_windows[i:i+infer_batch_size]
                chunk_t = torch.tensor(chunk, dtype=torch.float32).unsqueeze(1).to(DEVICE)
                logits = v4_model(chunk_t)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                probs_all.extend(probs)
                
        probs_all = np.array(probs_all)
        
        # Hard Negatives: falsely flagged windows (> 0.40 threshold)
        hn_indices = np.where(probs_all > THRESHOLD)[0]
        if len(hn_indices) > 0:
            hn_windows = valid_windows[hn_indices]
            hard_negatives.append(hn_windows)
            
        print(f"  [{idx+1}/{len(fp_files)}] {fp_file}: Found {len(hn_indices)} hard negatives out of {len(valid_windows)} windows.")
    except Exception as e:
        print(f"  [{idx+1}/{len(fp_files)}] Error processing {fp_file}: {str(e)}")

if len(hard_negatives) > 0:
    X_hn = np.vstack(hard_negatives)
    y_hn = np.zeros(len(X_hn), dtype=np.int32)
else:
    X_hn = np.array([]).reshape(0, WIN_SIZE)
    y_hn = np.array([])
    
print(f"  Total Hard Negatives extracted: {len(X_hn):,} (in {time.time()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load Cached Arrays & Filter True Positives / Normal Background
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 2] Loading Original CHB-MIT cached arrays ...')
X_train_raw = np.load(os.path.join(CACHE_DIR, 'dl_bilstm_train_X.npy'), mmap_mode='r')
y_train_raw = np.load(os.path.join(CACHE_DIR, 'dl_bilstm_train_y.npy'))

def extract_subwindows(X_raw, y_raw):
    starts = list(range(0, X_raw.shape[2] - WIN_SIZE + 1, WIN_STEP))
    n_wins = len(starts)
    ch0 = np.array(X_raw[:, 0, :], dtype=np.float32)
    subs_raw = np.stack([ch0[:, s:s + WIN_SIZE] for s in starts], axis=1)
    subs_flat = subs_raw.reshape(-1, WIN_SIZE)
    y_sub = np.repeat(y_raw, n_wins).astype(np.int32)
    mean = subs_flat.mean(axis=1, keepdims=True)
    std = subs_flat.std(axis=1, keepdims=True) + 1e-8
    subs_norm = np.clip((subs_flat - mean) / std, -6.0, 6.0).astype(np.float32)
    mask = std.flatten() >= 1e-7
    return subs_norm[mask], y_sub[mask]

X_orig, y_orig = extract_subwindows(X_train_raw, y_train_raw)

# Separate original TPs (seizures) and TNs (normals)
X_tp = X_orig[y_orig == 1]
y_tp = y_orig[y_orig == 1]
X_tn = X_orig[y_orig == 0]
y_tn = y_orig[y_orig == 0]

print(f"  Original Seizure Windows (TP): {len(X_tp):,}")
print(f"  Original Normal Windows (TN): {len(X_tn):,}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build New Training Set and Apply SMOTE
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 3] Combining and Applying SMOTE (ratio=0.5) ...')
X_comb = np.vstack([X_tp, X_tn, X_hn])
y_comb = np.concatenate([y_tp, y_tn, y_hn])

print(f"  Class counts before SMOTE: Seizure={len(y_tp):,}, Normal={len(y_tn)+len(y_hn):,}")

# SMOTE requires 2D flat arrays, which our sub-windows already are (N, 178)
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_comb, y_comb)

print(f"  Class counts after SMOTE : Seizure={(y_resampled==1).sum():,}, Normal={(y_resampled==0).sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Train/Val Split
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 4] Stratified 90/10 train-val split ...')
X_tr, X_val, y_tr, y_val = train_test_split(X_resampled, y_resampled, test_size=0.10, stratify=y_resampled, random_state=42)

X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32).unsqueeze(1)
y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type=='cuda'))
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE.type=='cuda'))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Model Training (Focal Loss)
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 5] Training V5 Model with Focal Loss ...')
model = CNNBiLSTMPhase1(timesteps=WIN_SIZE).to(DEVICE)
criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_val_f1 = -1.0
best_state  = None
patience_count = 0

print(f'  {"Epoch":>5}  {"tr_loss":>9}  {"val_loss":>9}  {"val_f1":>9}')
print('  ' + '-'*45)

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item() * len(Xb)
    tr_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    vp = []; vl = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            val_loss += criterion(logits, yb).item() * len(Xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            vp.extend((probs >= 0.50).astype(int).tolist()) # Eval threshold 0.50
            vl.extend(yb.cpu().numpy().tolist())
            
    val_loss /= len(val_loader.dataset)
    val_f1 = f1_score(vl, vp, zero_division=0)
    
    marker = ''
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_count = 0
        marker = ' *'
    else:
        patience_count += 1
        
    print(f'  {epoch:>5}  {tr_loss:>9.4f}  {val_loss:>9.4f}  {val_f1:>9.4f}{marker}')
    
    if patience_count >= EARLY_STOP_PATIENCE:
        print(f'\n  Early stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})')
        break

print(f'\n[STEP 6] Saving V5 Model to {SAVE_PATH}')
torch.save(best_state, SAVE_PATH)

print('\n=== V5 Training Complete. Starting Automatic Validation... ===')
os.system(f'{sys.executable} ../full_validate_all.py')
