"""
cnn_model.py — SeizureCNNLTI architecture
==========================================
Changes vs original:
  - AdaptiveAvgPool1d replaces fixed MaxPool so the model works for any
    window length (the FC layer input size is now always 128 * 16).
  - Forward uses sigmoid-free output (raw logits) for BCEWithLogitsLoss
    during training; sigmoid is applied at inference time.
  - Added a helper `predict_proba` method used by predict.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LTILayer(nn.Module):
    """
    Fixed-weight depthwise 1-D convolution that simulates biological signal
    smoothing (Gaussian low-pass filter).  Weights are frozen (not learnable).
    """
    def __init__(self, in_channels: int, kernel_size: int = 15):
        super().__init__()
        t      = np.linspace(-3, 3, kernel_size)
        kernel = np.exp(-0.5 * t ** 2)
        kernel = kernel / kernel.sum()                     # normalise

        # shape: (in_channels, 1, kernel_size) for depthwise conv
        weight = torch.tensor(kernel, dtype=torch.float32) \
                      .view(1, 1, -1).repeat(in_channels, 1, 1)
        self.weight  = nn.Parameter(weight, requires_grad=False)
        self.padding = kernel_size // 2
        self.groups  = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight, padding=self.padding,
                        groups=self.groups)


class SeizureCNNLTI(nn.Module):
    """
    CNN + biological LTI layer for multi-channel EEG seizure classification.

    Input : (Batch, Channels, Samples)   e.g. (B, 18, 1280)
    Output: (Batch, 1)  — raw logit (use sigmoid for probability)

    The model is agnostic to `window_samples` because it uses
    AdaptiveAvgPool1d to reduce the temporal dimension to a fixed size (16)
    before the fully-connected head.
    """

    ADAPTIVE_OUT = 16      # fixed temporal size after adaptive pooling

    def __init__(self, in_channels: int = 18, window_samples: int = 1280):
        super().__init__()
        self.in_channels    = in_channels
        self.window_samples = window_samples

        # ── Conv block 1 ────────────────────────────────────────────────
        self.conv1    = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.bn1      = nn.BatchNorm1d(64)
        self.pool1    = nn.MaxPool1d(kernel_size=4)    # /4
        self.dropout1 = nn.Dropout(0.25)

        # ── Conv block 2 ────────────────────────────────────────────────
        self.conv2    = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2      = nn.BatchNorm1d(128)
        self.pool2    = nn.MaxPool1d(kernel_size=4)    # /4
        self.dropout2 = nn.Dropout(0.25)

        # ── LTI biological filter ────────────────────────────────────────
        self.lti = LTILayer(in_channels=128, kernel_size=15)

        # ── Adaptive pool → fixed 128×16 = 2048 features ────────────────
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.ADAPTIVE_OUT)

        fc_in = 128 * self.ADAPTIVE_OUT          # 2 048

        # ── Classifier head ──────────────────────────────────────────────
        self.fc1      = nn.Linear(fc_in, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2      = nn.Linear(128, 1)

    # ── Forward (returns RAW LOGIT) ──────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(self.pool1(x))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(self.pool2(x))

        x = self.lti(x)
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        return self.fc2(x)                  # logit — NO sigmoid here

    # ── Convenience: probability (for inference / predict.py) ────────────
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
