"""ResNet1D — residual 1D convolution (DESIGN.md §3.3.3).

Stacked 1D residual blocks (``Conv1d + BatchNorm + ReLU`` with a residual
connection and optional downsampling), followed by global average pooling and
a fully-connected head.

``ResNet1DBackbone`` returns the feature sequence ``[batch, channels, time]``
and is reused as the front-end of ResNet1D-LSTM and ResNet1D-TCN.

Input  : [batch, 9, 128]
Output : [batch, 6]
"""

import torch.nn as nn

from models import BaseModel


class ResidualBlock1D(nn.Module):
    """A basic 1D residual block with two convolutions."""

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return self.dropout(out)


class ResNet1DBackbone(nn.Module):
    """Residual convolutional feature extractor: [B, 9, 128] -> [B, C, T]."""

    def __init__(
        self,
        in_channels=9,
        channels=(64, 128, 256),
        blocks_per_stage=2,
        strides=(1, 2, 2),
        stem_channels=64,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()
        if len(strides) != len(channels):
            raise ValueError("strides must have one entry per stage")

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, 5, padding=2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )

        stages = []
        prev = stem_channels
        for stage_idx, (out_channels, stride) in enumerate(zip(channels, strides)):
            for block_idx in range(blocks_per_stage):
                block_stride = stride if block_idx == 0 else 1
                stages.append(
                    ResidualBlock1D(
                        prev,
                        out_channels,
                        stride=block_stride,
                        kernel_size=kernel_size,
                        dropout=dropout,
                    )
                )
                prev = out_channels
        self.stages = nn.Sequential(*stages)
        self.out_channels = prev

    def forward(self, x):
        return self.stages(self.stem(x))


class ResNet1D(BaseModel):
    """ResNet1D classifier: residual backbone + global average pooling head."""

    def __init__(
        self,
        in_channels=9,
        num_classes=6,
        channels=(64, 128, 256),
        blocks_per_stage=2,
        strides=(1, 2, 2),
        stem_channels=64,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()
        self.backbone = ResNet1DBackbone(
            in_channels=in_channels,
            channels=channels,
            blocks_per_stage=blocks_per_stage,
            strides=strides,
            stem_channels=stem_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, num_classes),
        )

    def extract_features(self, x):
        """Feature sequence before pooling: [batch, channels, time]."""
        return self.backbone(x)

    def forward(self, x):
        # x: [batch, 9, 128] -> logits: [batch, 6]
        return self.head(self.pool(self.backbone(x)))
