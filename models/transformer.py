"""Transformer baseline (DESIGN.md §4 附加实验 / §7 第 7 周可选对照).

A decoder-free Transformer encoder over the raw sensor sequence, mirroring the
LSTM baseline's tokenisation:

- ``[batch, 9, 128]`` -> ``[batch, 128, 9]`` (128 time steps, 9 features each),
- a linear projection to ``d_model`` plus a learnable positional embedding,
- a learnable ``[CLS]`` token whose encoder output feeds the classifier.

The point of this model is a controlled architecture comparison: it keeps the
same input representation, training loop, budget and classification head as the
other models, so any difference reflects attention-based temporal modelling
rather than extra preprocessing.

Input  : [batch, 9, 128]
Output : [batch, 6]
"""

import torch
import torch.nn as nn

from models import BaseModel


class TransformerClassifier(BaseModel):
    """Transformer encoder classifier over the raw sensor sequence."""

    def __init__(
        self,
        in_channels=9,
        num_classes=6,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        max_len=128,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.max_len = max_len
        self.input_proj = nn.Linear(in_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.input_dropout = nn.Dropout(dropout)

        # Pre-norm + GELU is the stable default for small datasets.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: [batch, 9, 128] -> [batch, 128, 9]
        seq = x.transpose(1, 2)
        if seq.size(1) > self.max_len:
            raise ValueError(
                f"sequence length {seq.size(1)} exceeds max_len {self.max_len}"
            )

        tokens = self.input_proj(seq)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        tokens = self.input_dropout(tokens)

        out = self.encoder(tokens)
        # [CLS] token aggregates the sequence for classification.
        return self.head(self.norm(out[:, 0]))
