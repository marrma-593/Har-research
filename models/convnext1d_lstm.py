"""ConvNeXt1D-LSTM (DESIGN.md §3.3.6).

A from-scratch 1D implementation of the ConvNeXt block, used as the
convolutional front-end instead of ResNet1D:

- depthwise ``Conv1d`` (one filter per channel) for spatial mixing,
- ``LayerNorm`` instead of ``BatchNorm``,
- ``GELU`` instead of ``ReLU``,
- an inverted bottleneck with pointwise ``1x1`` convolutions,
- a residual connection.

The resulting feature sequence feeds an LSTM and a classification head.

Input  : [batch, 9, 128]
Output : [batch, 6]
"""

import torch.nn as nn

from models import BaseModel


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a ``[B, C, T]`` tensor."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt block adapted to one-dimensional sensor sequences."""

    def __init__(self, dim, kernel_size=7, expansion=4, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        hidden_dim = dim * expansion
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = ChannelLayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.pointwise_out = nn.Conv1d(hidden_dim, dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.depthwise(x)
        out = self.norm(out)
        out = self.pointwise_in(out)
        out = self.act(out)
        out = self.pointwise_out(out)
        out = self.dropout(out)
        return out + residual


class ConvNeXt1DLSTM(BaseModel):
    """ConvNeXt-style backbone + LSTM temporal backend."""

    def __init__(
        self,
        in_channels=9,
        num_classes=6,
        dims=(64, 128, 256),
        blocks_per_stage=(2, 2, 2),
        kernel_size=7,
        expansion=4,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        if len(blocks_per_stage) != len(dims):
            raise ValueError("blocks_per_stage must have one entry per stage")

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, dims[0], 3, padding=1),
            ChannelLayerNorm(dims[0]),
        )

        stages = []
        prev = dims[0]
        for stage_idx, (dim, n_blocks) in enumerate(zip(dims, blocks_per_stage)):
            if stage_idx > 0:
                # Non-overlapping 2x temporal downsampling, channels doubled.
                stages.append(
                    nn.Sequential(
                        nn.Conv1d(prev, dim, kernel_size=2, stride=2),
                        ChannelLayerNorm(dim),
                    )
                )
            stages.append(
                nn.Sequential(
                    *[
                        ConvNeXtBlock1D(
                            dim,
                            kernel_size=kernel_size,
                            expansion=expansion,
                            dropout=dropout,
                        )
                        for _ in range(n_blocks)
                    ]
                )
            )
            prev = dim
        self.stages = nn.Sequential(*stages)
        self.out_channels = prev

        self.lstm = nn.LSTM(
            input_size=self.out_channels,
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
        # x: [batch, 9, 128] -> [batch, C, T] -> [batch, T, C]
        features = self.stages(self.stem(x))
        out, _ = self.lstm(features.transpose(1, 2))
        return self.head(out[:, -1])
