"""CNN-LSTM (DESIGN.md §3.3.4).

A small 1D-CNN front-end extracts local features; the resulting feature
sequence is fed to an LSTM for temporal modelling, and the last hidden state
goes through a fully-connected head.

Input  : [batch, 9, 128]
Output : [batch, 6]
"""

import torch.nn as nn

from models import BaseModel


class ConvPoolBlock(nn.Module):
    """``Conv1d -> BatchNorm -> ReLU -> MaxPool`` feature block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, pool=2):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x):
        return self.block(x)


class CNNLSTM(BaseModel):
    """Convolutional feature extractor followed by an LSTM."""

    def __init__(
        self,
        in_channels=9,
        num_classes=6,
        conv_channels=(64, 128),
        hidden_size=128,
        num_layers=2,
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()
        blocks = []
        prev = in_channels
        for out_channels in conv_channels:
            blocks.append(ConvPoolBlock(prev, out_channels, kernel_size=kernel_size))
            prev = out_channels
        self.features = nn.Sequential(*blocks)

        self.lstm = nn.LSTM(
            input_size=prev,
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
        # x: [batch, 9, 128] -> features [batch, C, T] -> sequence [batch, T, C]
        features = self.features(x)
        seq = features.transpose(1, 2)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1, :])
