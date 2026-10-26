"""ResNet1D-TCN (DESIGN.md §3.3.7).

ResNet1D front-end followed by a Temporal Convolutional Network instead of an
LSTM:

- dilated convolutions with exponentially growing dilation, widening the
  temporal receptive field,
- temporal residual blocks (two dilated convolutions + non-linearity +
  residual connection),
- causal convolution (left padding only) so no future information leaks;
  ``causal=False`` switches to symmetric padding.

TCN is still a convolutional model; whether it beats the LSTM is an empirical
question, not an assumption.

Input  : [batch, 9, 128]
Output : [batch, 6]
"""

import torch.nn as nn
import torch.nn.functional as F

from models import BaseModel
from models.resnet1d import ResNet1DBackbone


def _temporal_pad(x, padding, causal):
    if padding == 0:
        return x
    if causal:
        return F.pad(x, (padding, 0))
    return F.pad(x, (padding, padding))


class TemporalBlock(nn.Module):
    """Two dilated convolutions with a residual connection."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.2,
        causal=True,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.causal = causal

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = _temporal_pad(x, self.padding, self.causal)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.dropout(out)

        out = _temporal_pad(out, self.padding, self.causal)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """Stack of temporal blocks with exponentially growing dilations."""

    def __init__(
        self,
        in_channels,
        channels=128,
        num_blocks=4,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.2,
        causal=True,
    ):
        super().__init__()
        blocks = []
        prev = in_channels
        for i in range(num_blocks):
            dilation = dilations[i % len(dilations)]
            blocks.append(
                TemporalBlock(
                    prev,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    causal=causal,
                )
            )
            prev = channels
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels

    def forward(self, x):
        return self.blocks(x)


class ResNet1DTCN(BaseModel):
    """ResNet1D front-end + dilated temporal convolutional backend."""

    def __init__(
        self,
        in_channels=9,
        num_classes=6,
        channels=(64, 128, 256),
        blocks_per_stage=2,
        strides=(1, 2, 2),
        stem_channels=64,
        kernel_size=3,
        tcn_channels=128,
        tcn_blocks=4,
        tcn_kernel_size=3,
        tcn_dilations=(1, 2, 4, 8),
        dropout=0.2,
        causal=True,
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
        self.tcn = TemporalConvNet(
            in_channels=self.backbone.out_channels,
            channels=tcn_channels,
            num_blocks=tcn_blocks,
            kernel_size=tcn_kernel_size,
            dilations=tcn_dilations,
            dropout=dropout,
            causal=causal,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.tcn.out_channels, num_classes),
        )

    def forward(self, x):
        # x: [batch, 9, 128] -> features [batch, C, T] -> TCN [batch, C, T]
        features = self.tcn(self.backbone(x))
        # With causal convolutions the last time step sees the whole history.
        return self.head(features[:, :, -1])
