"""
eval_cnn_lti.py
===============
Evaluate the already-trained CNN-LTI checkpoint on the test split.
Saves cnn_lti_metrics.json and regenerates the comparison report.
No retraining needed — runs in < 2 minutes.
"""
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from cnn_model import SeizureCNNLTI

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = r"E:\Downloads\Finalyearproject\backend\models"
METRICS_DIR = os.path.join(MODELS_DIR, "metrics")
CACHE_DIR   = os.path.join(MODELS_DIR, "cache")
CKPT        = os.path.join(MODELS_DIR, "cnn_lti_model.pt")

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")

print("=" * 60)
print("  CNN-LTI  |  Test Evaluation (no retraining)")
print("=" * 60)
print(f"  Device   : {DEVICE}")
print(f"  Checkpoint: {CKPT}")

# ── Load test data from cache ──────────────────────────────────────────────
print("\n[1] Loading test cache ...")
X_test = np.load(os.path.join(CACHE_DIR, "X_cnn_test.npy"), mmap_mode='r')
y_test = np.load(os.path.join(CACHE_DIR, "y_cnn_test.npy"))
print(f"  Test shape : {X_test.shape}")
print(f"  Seizure    : {int(y_test.sum()):,}  /  Normal: {int((y_test==0).sum()):,}")

# ── Load model ────────────────────────────────────────────────────────────
print("\n[2] Loading CNN-LTI checkpoint ...")
model = SeizureCNNLTI(in_channels=X_test.shape[1],
                      window_samples=X_test.shape[2]).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()
params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {params:,}")

# ── Evaluate on test set (batch streaming — low RAM) ──────────────────────
print("\n[3] Running inference on test set ...")
from torch.amp import autocast

@torch.no_grad()
def evaluate(model, X_np, batch_size=32):
    all_probs = []
    n = len(X_np)
    for s in range(0, n, batch_size):
        chunk = X_np[s:s+batch_size].copy()
        Xb    = torch.tensor(chunk, dtype=torch.float32).to(DEVICE)
        if USE_AMP:
            with autocast(device_type='cuda'):
                logits = model(Xb)
        else:
            logits = model(Xb)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.append(probs)
        del Xb, chunk
    return np.concatenate(all_probs)

probs = evaluate(model, X_test)
preds = (probs >= 0.5).astype(int)

# ── Compute metrics ───────────────────────────────────────────────────────
print("\n[4] Computing metrics ...")
acc  = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec  = recall_score(y_test, preds,    zero_division=0)
f1   = f1_score(y_test, preds,        zero_division=0)
cm   = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()

try:
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = float(auc(fpr, tpr))
except Exception:
    roc_auc = 0.0; fpr, tpr = np.array([]), np.array([])

metrics = {
    "model":           "CNN_LTI",
    "accuracy":        round(float(acc),  4),
    "precision":       round(float(prec), 4),
    "recall":          round(float(rec),  4),
    "f1_score":        round(float(f1),   4),
    "roc_auc":         round(roc_auc,     4),
    "sensitivity":     round(float(tp / max(tp+fn, 1)), 4),
    "specificity":     round(float(tn / max(tn+fp, 1)), 4),
    "confusion_matrix": cm.tolist(),
}

print(f"\n  -- CNN-LTI Test Results --")
for k, v in metrics.items():
    if k not in ("confusion_matrix", "model"):
        print(f"     {k:<14}: {v}")
print(f"\n{classification_report(y_test, preds, zero_division=0, target_names=['Interictal','Ictal'])}")

# Save JSON
with open(os.path.join(METRICS_DIR, "cnn_lti_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved -> cnn_lti_metrics.json")

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set(title='CNN-LTI Confusion Matrix', xlabel='Predicted', ylabel='True',
       xticks=[0,1], yticks=[0,1],
       xticklabels=['Interictal','Ictal'],
       yticklabels=['Interictal','Ictal'])
thr = cm.max() / 2
for i, j in np.ndindex(cm.shape):
    ax.text(j, i, str(cm[i,j]), ha='center', va='center',
            color='white' if cm[i,j] > thr else 'black', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "cnn_lti_confusion_matrix.png"), dpi=150)
plt.close()

# ROC curve
if len(fpr) > 0:
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.4f}')
    plt.plot([0,1],[0,1],'navy',lw=1.5,linestyle='--')
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('CNN-LTI ROC Curve'); plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, "cnn_lti_roc_curve.png"), dpi=150)
    plt.close()

# ── Regenerate comparison report ──────────────────────────────────────────
print("\n[5] Regenerating comparison report ...")
metric_files = {
    "SVM":        "svm_metrics.json",
    "LR":         "lr_metrics.json",
    "CNN_LTI":    "cnn_lti_metrics.json",
    "CNN_BiLSTM": "cnn_bilstm_metrics.json",
}
model_types = {
    "SVM":"Classical ML","LR":"Classical ML",
    "CNN_LTI":"Deep Learning","CNN_BiLSTM":"Deep Learning",
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

keys   = ['accuracy','precision','recall','f1_score','roc_auc']
labels = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']
palette= ['#4C72B0','#55A868','#C44E52','#8172B2']

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
ax.set_title("Comparative Analysis: Classical ML vs Deep Learning\n"
             "Epileptic Seizure Detection - CHB-MIT Dataset",
             fontsize=12, fontweight='bold', pad=14)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
chart = os.path.join(METRICS_DIR, "comparison_table.png")
plt.savefig(chart, dpi=150); plt.close()
print(f"  Saved -> comparison_table.png")

def _best(key):
    vals = [m.get(key, 0) for m in all_models]
    return all_models[int(np.argmax(vals))]['model'], max(vals)

best_acc = _best('accuracy'); best_rec = _best('recall')
best_auc = _best('roc_auc'); best_f1  = _best('f1_score')

sep = "-" * 80
hdr = (
    "=" * 80 + "\n"
    "  Comparative Analysis: Classical ML vs Deep Learning\n"
    "  Epileptic Seizure Detection - CHB-MIT Scalp EEG Dataset\n"
    "=" * 80 + "\n\n"
    "  Dataset : CHB-MIT Scalp EEG Database (24 patients)\n"
    "  Split   : Train chb01-18 | Val chb19-21 | Test chb22-24\n\n"
)
tbl = (f"  {'Model':<16} {'Type':<16} "
       f"{'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}\n"
       f"  {sep}\n")
for m in all_models:
    tbl += (f"  {m['model']:<16} {m['model_type']:<16} "
            f"{m.get('accuracy',0):>8.4f} {m.get('precision',0):>8.4f} "
            f"{m.get('recall',0):>8.4f} {m.get('f1_score',0):>8.4f} "
            f"{m.get('roc_auc',0):>8.4f}\n")
summ = (f"\n  WINNERS\n  {sep}\n"
        f"  Best Accuracy  : {best_acc[0]} ({best_acc[1]:.4f})\n"
        f"  Best Recall    : {best_rec[0]} ({best_rec[1]:.4f})\n"
        f"  Best ROC-AUC   : {best_auc[0]} ({best_auc[1]:.4f})\n"
        f"  Best F1-Score  : {best_f1[0]}  ({best_f1[1]:.4f})\n\n"
        f"  CLINICAL NOTE\n  {sep}\n"
        f"  Recall/Sensitivity is the primary metric for seizure detection.\n"
        f"  A missed seizure is more dangerous than a false alarm.\n")

full = hdr + tbl + summ
txt  = os.path.join(METRICS_DIR, "comparison_report.txt")
with open(txt, "w", encoding='utf-8') as f:
    f.write(full)
print(f"  Saved -> comparison_report.txt")

rep = {
    "title":    "Comparative Analysis: Classical ML vs Deep Learning - CHB-MIT",
    "dataset":  "CHB-MIT Scalp EEG Database",
    "patients": 24,
    "split":    {"train":"chb01-18","val":"chb19-21","test":"chb22-24"},
    "models":   all_models,
    "winners": {
        "best_accuracy": {"model":best_acc[0],"value":best_acc[1]},
        "best_recall":   {"model":best_rec[0],"value":best_rec[1]},
        "best_roc_auc":  {"model":best_auc[0],"value":best_auc[1]},
        "best_f1_score": {"model":best_f1[0], "value":best_f1[1]},
    },
}
with open(os.path.join(METRICS_DIR, "comparison_report.json"), "w", encoding='utf-8') as f:
    json.dump(rep, f, indent=2)
print(f"  Saved -> comparison_report.json\n")
print(full)
print("=" * 60)
print("  CNN-LTI evaluation complete")
print("=" * 60)
