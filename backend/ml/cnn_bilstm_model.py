"""
cnn_bilstm_model.py
====================
CNN + BiLSTM + Attention architecture for raw EEG seizure detection.

Input  : (Batch, Channels=18, Samples=1280)  — raw EEG windows
Output : (Batch, 1)  — raw logit (apply sigmoid for probability)

Pipeline inside the model:
  Conv Block 1-4  →  spatial / frequency feature extraction
  Reshape         →  (Batch, Time-steps, CNN-features)
  BiLSTM (2L)     →  temporal seizure dynamics
  Attention       →  focus on seizure-relevant time steps
  Classifier head →  Dense → logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool with optional Dropout."""
    def __init__(self, in_ch, out_ch, kernel=3, pool=2, drop=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over the time dimension.
    Input : (Batch, Time, Features)
    Output: (Batch, Features)  — context vector
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):                        # x: (B, T, H)
        weights = torch.softmax(self.score(x), dim=1)   # (B, T, 1)
        context = (weights * x).sum(dim=1)              # (B, H)
        return context


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class CNNBiLSTM(nn.Module):
    """
    CNN + Bidirectional LSTM + Attention for EEG seizure detection.

    Architecture summary (default params, input 18×1280):
      Conv Block 1 : 18  → 64  ch,  1280 → 640  time steps
      Conv Block 2 : 64  → 128 ch,   640 → 320
      Conv Block 3 : 128 → 256 ch,   320 → 160
      Conv Block 4 : 256 → 256 ch,   160 → 80
      Reshape      : (B, 256, 80) → (B, 80, 256)  [time, features]
      BiLSTM × 2   : hidden=128  →  output 256 per step
      Attention    : (B, 80, 256) → (B, 256)
      FC head      : 256 → 128 → 1  (logit)
    """

    def __init__(self,
                 in_channels:    int = 18,
                 window_samples: int = 1280,
                 lstm_hidden:    int = 128,
                 lstm_layers:    int = 2,
                 cnn_dropout:    float = 0.25,
                 lstm_dropout:   float = 0.35,
                 fc_dropout:     float = 0.50):
        super().__init__()

        self.in_channels    = in_channels
        self.window_samples = window_samples

        # ── CNN Encoder ───────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            ConvBlock(in_channels,  64,  kernel=7, pool=2, drop=cnn_dropout),
            ConvBlock(64,          128,  kernel=5, pool=2, drop=cnn_dropout),
            ConvBlock(128,         256,  kernel=3, pool=2, drop=cnn_dropout),
            ConvBlock(256,         256,  kernel=3, pool=2, drop=cnn_dropout),
        )

        # Compute CNN output length dynamically (avoids hard-coding)
        with torch.no_grad():
            dummy   = torch.zeros(1, in_channels, window_samples)
            cnn_out = self.cnn(dummy)                 # (1, 256, T')
            self._cnn_out_ch  = cnn_out.shape[1]     # 256
            self._cnn_out_len = cnn_out.shape[2]     # ~80 for 1280 input

        # ── BiLSTM ────────────────────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size  = self._cnn_out_ch,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = lstm_dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2      # bidirectional doubles the dim

        # ── Attention ─────────────────────────────────────────────────────
        self.attention = AdditiveAttention(lstm_out_dim)

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(128, 1),          # raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)  — raw EEG windows
        returns logit  (B, 1)
        """
        # CNN
        feat = self.cnn(x)                       # (B, 256, T')
        feat = feat.permute(0, 2, 1)             # (B, T', 256)  LSTM wants (B,T,F)

        # BiLSTM
        lstm_out, _ = self.bilstm(feat)          # (B, T', 256)

        # Attention
        context = self.attention(lstm_out)       # (B, 256)

        # Classify
        return self.classifier(context)          # (B, 1)

    # ── Convenience helpers ───────────────────────────────────────────────
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probability — use for inference."""
        return torch.sigmoid(self.forward(x))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
