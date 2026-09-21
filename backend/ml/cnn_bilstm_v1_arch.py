"""
cnn_bilstm_v1_arch.py
======================
Standalone model architecture definition for CNN-BiLSTM v1.
Extracted from train_bilstm.py so it can be imported by the backend API
without pulling in training dependencies (SMOTE, scipy, etc.).
"""
import torch
import torch.nn as nn


class CNNBiLSTMPhase1(nn.Module):
    """
    CNN-BiLSTM for single-channel EEG seizure detection.
    Input  : (batch, 1, 178)   — 1 channel, 178 EEG timesteps
    Output : (batch,)           — raw logit; apply sigmoid for probability

    Architecture:
        Conv1D(64, k=5) → BN → ReLU → MaxPool(2) → Dropout(0.3)
        Conv1D(128, k=3) → BN → ReLU → MaxPool(2) → Dropout(0.3)
        BiLSTM(64, seq=True) → Dropout(0.3)
        BiLSTM(32, seq=False, last-step) → Dropout(0.3)
        Dense(64) → Dropout(0.2)
        Dense(1)   ← logit
    """
    def __init__(self, timesteps: int = 178):
        super().__init__()
        self.timesteps = timesteps

        # --- CNN block 1
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )
        # --- CNN block 2
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )
        # After 2× MaxPool(2): timesteps // 4  (178 → 44)
        # --- BiLSTM 1
        self.bilstm1 = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.drop3   = nn.Dropout(0.3)
        # --- BiLSTM 2
        self.bilstm2 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        self.drop4   = nn.Dropout(0.3)
        # --- FC head
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.cnn1(x)           # (B, 64,  T//2)
        x = self.cnn2(x)           # (B, 128, T//4)
        x = x.permute(0, 2, 1)    # (B, T//4, 128)

        x, _ = self.bilstm1(x)    # (B, T//4, 128)
        x = self.drop3(x)
        x, _ = self.bilstm2(x)    # (B, T//4, 64)
        x = self.drop4(x)

        x = x[:, -1, :]           # last timestep → (B, 64)
        return self.fc(x).squeeze(1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
