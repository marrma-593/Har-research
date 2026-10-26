"""Model registry and shared base interface.

All models follow the unified interface defined in DESIGN.md §3.2:

    input  : [batch, 9, 128]   (9 channels x 128 timesteps)
    output : [batch, 6]        (logits over the 6 activity classes)
"""

import importlib

import torch.nn as nn


class BaseModel(nn.Module):
    """Common interface for every model in the project."""

    def forward(self, x):
        # x: [batch, 9, 128] -> logits: [batch, 6]
        raise NotImplementedError


# name -> (module, class). DESIGN.md §3.1 lists all seven models.
_MODEL_REGISTRY = {
    "cnn1d": ("models.cnn1d", "CNN1D"),
    "lstm": ("models.lstm", "LSTMClassifier"),
    "resnet1d": ("models.resnet1d", "ResNet1D"),
    "cnn_lstm": ("models.cnn_lstm", "CNNLSTM"),
    "resnet1d_lstm": ("models.resnet1d_lstm", "ResNet1DLSTM"),
    "convnext1d_lstm": ("models.convnext1d_lstm", "ConvNeXt1DLSTM"),
    "resnet1d_tcn": ("models.resnet1d_tcn", "ResNet1DTCN"),
    # Optional comparison (DESIGN.md §4 附加实验 / §7 week 7).
    "transformer": ("models.transformer", "TransformerClassifier"),
}


def get_model(name, **kwargs):
    """Instantiate a registered model by name."""
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model: {name!r}. Available: {available}")
    module_name, class_name = _MODEL_REGISTRY[name]
    model_cls = getattr(importlib.import_module(module_name), class_name)
    return model_cls(**kwargs)
