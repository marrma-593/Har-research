"""1D-CNN baseline (DESIGN.md §3.3.1).

A stack of ``Conv1d -> BatchNorm -> ReLU -> MaxPool`` blocks, followed by
global average pooling and a fully-connected classification head.

Input  : [batch, 9, 128]
Output : [batch, 6]
"""

import torch.nn as nn

from models import BaseModel


class ConvBlock(nn.Module):
    """A single ``Conv1d -> BatchNorm -> ReLU -> MaxPool`` unit."""

    def __init__(self, in_channels, out_channels, kernel_size=3, pool=2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class CNN1D(BaseModel):
    """Baseline 1D convolutional network for HAR."""

    def __init__(self, in_channels=9, num_classes=6, channels=(64, 128, 256), kernel_size=3):
        super().__init__()
        layers = []
        prev = in_channels
        for c in channels:
            layers.append(ConvBlock(prev, c, kernel_size=kernel_size))
            prev = c

        self.features = nn.Sequential(*layers)
        # global average pooling over the time axis -> [batch, prev, 1] -> flatten -> logits
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(prev, num_classes),
        )

    def forward(self, x):
        # x: [batch, 9, 128] -> logits: [batch, 6]
        return self.head(self.features(x))
