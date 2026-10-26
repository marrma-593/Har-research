"""LSTM baseline (DESIGN.md §3.3.2).

The ``[batch, 9, 128]`` windows are reshaped to ``[batch, 128, 9]`` (128 time
steps, 9 features each). A stack of LSTM layers models the sequence and the
last time step of the final layer feeds a fully-connected head.

Input  : [batch, 9, 128]
Output : [batch, 6]
"""

import torch.nn as nn

from models import BaseModel


class LSTMClassifier(BaseModel):
    """Multi-layer LSTM over the raw sensor sequence."""

    def __init__(
        self,
        in_channels=9,
        num_classes=6,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: [batch, 9, 128] -> [batch, 128, 9]
        seq = x.transpose(1, 2)
        out, _ = self.lstm(seq)
        # last time step of the last layer: [batch, hidden_size]
        return self.head(out[:, -1])
