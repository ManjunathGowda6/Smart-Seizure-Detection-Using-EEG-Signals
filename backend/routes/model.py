"""
routes/model.py
================
FastAPI endpoints for CNN-BiLSTM v1 model demo and status.

Routes:
  GET  /model/status              — all model availability + metadata
  POST /model/bilstm-predict      — run inference on 178 EEG values
  GET  /model/sample/{kind}       — generate synthetic EEG (normal / seizure / random)
"""

import os, sys, json
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/model", tags=["model"])

MODELS_DIR = r"E:\Downloads\Finalyearproject\backend\models"
ML_DIR     = os.path.join(os.path.dirname(__file__), "..", "ml")

# ── Lazy model loader ─────────────────────────────────────────────────────────
_bilstm_v1 = None

def _load_bilstm_v1():
    global _bilstm_v1
    if _bilstm_v1 is None:
        import torch
        sys.path.insert(0, ML_DIR)
        from cnn_bilstm_v1_arch import CNNBiLSTMPhase1

        ckpt = os.path.join(MODELS_DIR, "cnn_bilstm_v1.pt")
        if not os.path.exists(ckpt):
            return None

        model = CNNBiLSTMPhase1(timesteps=178)
        model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                         weights_only=True))
        model.eval()
        _bilstm_v1 = model
    return _bilstm_v1


# ── Schemas ───────────────────────────────────────────────────────────────────
class EEGInput(BaseModel):
    eeg_values: List[float]   # exactly 178 floats


# ── Helpers ───────────────────────────────────────────────────────────────────
def _zscore(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean()
    sg = arr.std() + 1e-8
    return (arr - mu) / sg


BONN_CSV = r"E:\Downloads\Epileptic Seizure Recognition.csv"

# Cache a few real Bonn samples at startup so the demo uses real data
_bonn_seizure_rows: list = []
_bonn_normal_rows:  list = []

def _preload_bonn():
    global _bonn_seizure_rows, _bonn_normal_rows
    if _bonn_seizure_rows:
        return          # already loaded
    if not os.path.exists(BONN_CSV):
        return
    import pandas as pd
    df = pd.read_csv(BONN_CSV)
    label_col = "y" if "y" in df.columns else df.columns[-1]
    feat_cols  = [c for c in df.columns
                  if c != label_col and not str(c).lower().startswith("unnamed")]
    is_seizure = df[label_col] == 1
    # Keep 200 samples of each for random draw
    _bonn_seizure_rows = df[is_seizure][feat_cols].values.astype(np.float32).tolist()[:200]
    _bonn_normal_rows  = df[~is_seizure][feat_cols].values.astype(np.float32).tolist()[:200]

def _real_sample(kind: str, rng: np.random.Generator) -> np.ndarray:
    """Return a real row from the Bonn CSV (z-score NOT applied — API will apply it)."""
    _preload_bonn()
    if kind == "seizure" and _bonn_seizure_rows:
        row = _bonn_seizure_rows[int(rng.integers(0, len(_bonn_seizure_rows)))]
    elif kind == "normal" and _bonn_normal_rows:
        row = _bonn_normal_rows[int(rng.integers(0, len(_bonn_normal_rows)))]
    else:
        # Fallback: simple synthetic if CSV not found
        t   = np.linspace(0, 0.7, 178, dtype=np.float32)
        amp = 80.0 if kind == "seizure" else 20.0
        row = (amp * np.sin(2 * np.pi * (5.0 if kind == "seizure" else 10.0) * t)
               + rng.normal(0, amp * 0.2, 178).astype(np.float32)).tolist()
    return np.array(row, dtype=np.float32)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/status")
def model_status():
    """
    Returns availability and metadata for all trained models.
    """
    def _check(fname):
        return os.path.exists(os.path.join(MODELS_DIR, fname))

    # Load CNN-BiLSTM v1 metadata
    meta_path = os.path.join(MODELS_DIR, "bilstm_metadata.json")
    bilstm_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            bilstm_meta = json.load(f)

    # Load comparison report metrics
    cmp_path = os.path.join(MODELS_DIR, "metrics", "comparison_report.json")
    cmp = {}
    if os.path.exists(cmp_path):
        with open(cmp_path, encoding="utf-8") as f:
            cmp = json.load(f)

    def _metric(model_key, metric):
        if not cmp: return None
        for m in cmp.get("models", []):
            if m.get("model") == model_key:
                return m.get(metric)
        return None

    return {
        "models": [
            {
                "id": "svm",
                "name": "Support Vector Machine",
                "type": "Classical ML",
                "available": _check("svm_model.pkl"),
                "dataset": "CHB-MIT",
                "accuracy": _metric("SVM", "accuracy"),
                "recall":   _metric("SVM", "recall"),
                "f1":       _metric("SVM", "f1_score"),
                "roc_auc":  _metric("SVM", "roc_auc"),
            },
            {
                "id": "lr",
                "name": "Logistic Regression",
                "type": "Classical ML",
                "available": _check("lr_model.pkl"),
                "dataset": "CHB-MIT",
                "accuracy": _metric("LR", "accuracy"),
                "recall":   _metric("LR", "recall"),
                "f1":       _metric("LR", "f1_score"),
                "roc_auc":  _metric("LR", "roc_auc"),
            },
            {
                "id": "cnn_lti",
                "name": "CNN + Biological LTI",
                "type": "Deep Learning",
                "available": _check("cnn_lti_model.pt"),
                "dataset": "CHB-MIT",
                "accuracy": _metric("CNN_LTI", "accuracy"),
                "recall":   _metric("CNN_LTI", "recall"),
                "f1":       _metric("CNN_LTI", "f1_score"),
                "roc_auc":  _metric("CNN_LTI", "roc_auc"),
            },
            {
                "id": "cnn_bilstm_v1",
                "name": "CNN-BiLSTM v1  (Phase 1)",
                "type": "Deep Learning",
                "available": _check("cnn_bilstm_v1.pt"),
                "dataset": "Bonn + CHB-MIT (SMOTE balanced)",
                "accuracy":  bilstm_meta.get("accuracy"),
                "recall":    bilstm_meta.get("seizure_recall"),
                "f1":        bilstm_meta.get("f1_score"),
                "roc_auc":   bilstm_meta.get("roc_auc"),
                "epochs":    bilstm_meta.get("epochs_run"),
                "trained_on": bilstm_meta.get("trained_on"),
                "framework": bilstm_meta.get("framework"),
            },
        ]
    }


@router.post("/bilstm-predict")
def bilstm_predict(body: EEGInput):
    """
    Run CNN-BiLSTM v1 inference on 178 EEG values.
    Returns: prediction, probability, confidence.
    """
    import torch

    if len(body.eeg_values) < 10:
        raise HTTPException(400, "Need at least 10 EEG values.")

    # Pad or truncate to 178
    vals = np.array(body.eeg_values, dtype=np.float32)
    if len(vals) < 178:
        vals = np.pad(vals, (0, 178 - len(vals)))
    else:
        vals = vals[:178]

    # Per-sample z-score (same normalisation used during training)
    vals = _zscore(vals)

    model = _load_bilstm_v1()
    if model is None:
        raise HTTPException(
            503,
            "CNN-BiLSTM v1 model not found. "
            "Run `python train_bilstm.py` to train it first."
        )

    x = torch.tensor(vals, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,178)
    with torch.no_grad():
        logit = model(x)
        prob  = float(torch.sigmoid(logit).item())

    pred        = "seizure" if prob >= 0.5 else "normal"
    confidence  = prob if pred == "seizure" else (1.0 - prob)
    risk_level  = ("High" if prob >= 0.7 else
                   "Moderate" if prob >= 0.4 else "Low")

    return {
        "prediction":   pred,
        "probability":  round(prob,        4),
        "confidence":   round(confidence,  4),
        "seizure_risk": risk_level,
        "eeg_values":   vals.tolist(),      # normalised, for waveform display
    }


@router.get("/sample/{kind}")
def get_sample_eeg(kind: str):
    """
    Return a real EEG sample from the Bonn University dataset.
    kind: 'normal' | 'seizure' | 'random'
    """
    if kind not in ("normal", "seizure", "random"):
        raise HTTPException(400, "kind must be 'normal', 'seizure', or 'random'")

    if kind == "random":
        kind = "seizure" if np.random.rand() > 0.5 else "normal"

    rng = np.random.default_rng()
    sig = _real_sample(kind, rng)
    return {
        "kind":       kind,
        "eeg_values": sig.tolist(),
        "length":     len(sig),
        "source":     "Bonn University EEG dataset (real sample)",
    }
