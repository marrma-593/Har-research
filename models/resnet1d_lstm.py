"""ResNet1D-LSTM — core model (DESIGN.md §3.3.5).

ResNet1D backbone (front-end) produces a feature sequence ``[batch, C, T]``;
the sequence is transposed to ``[batch, T, C]`` and modelled by an LSTM; the
last hidden state is mapped to six logits.

This model is the anchor for the two extension comparisons:
- ConvNeXt1D-LSTM  -> convolutional backbone difference
- ResNet1D-TCN     -> temporal-modelling module difference

Input  : [batch, 9, 128]
Output : [batch, 6]
"""

import torch.nn as nn

from models import BaseModel
from models.resnet1d import ResNet1DBackbone


class ResNet1DLSTM(BaseModel):
    """ResNet1D front-end + LSTM temporal backend."""

    def __init__(
        self,
        in_channels=9,
        num_classes=6,
        channels=(64, 128, 256),
        blocks_per_stage=2,
        strides=(1, 2, 2),
        stem_channels=64,
        kernel_size=3,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
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
        self.lstm = nn.LSTM(
            input_size=self.backbone.out_channels,
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
        features = self.backbone(x)
        seq = features.transpose(1, 2)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1])
