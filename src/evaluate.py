"""Unified evaluation (DESIGN.md §5.1, §6)."""

import json
from pathlib import Path

import numpy as np
import torch

from src import metrics
from src.data import CLASS_ABBREV, CLASS_NAMES


def predict(model, loader, device):
    """Return (preds, targets) as numpy arrays over the whole loader."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y.cpu().numpy())
    if preds:
        return (
            np.concatenate(preds).astype(np.int64),
            np.concatenate(targets).astype(np.int64),
        )
    return np.array([], dtype=np.int64), np.array([], dtype=np.int64)


def evaluate(model, loader, device):
    """Compute the full metric dict for a model over a loader."""
    preds, targets = predict(model, loader, device)
    return metrics.compute_metrics(targets, preds)


def format_confusion_matrix(cm, width=8):
    """Render a confusion matrix as an aligned text table."""
    n = len(cm)
    header = f"{'':>{width}}" + "".join(f"{CLASS_ABBREV[j]:>{width}}" for j in range(n))
    lines = [header]
    for i in range(n):
        row = f"{CLASS_ABBREV[i]:>{width}}" + "".join(f"{int(cm[i][j]):>{width}}" for j in range(n))
        lines.append(row)
    lines.append("")
    lines.append("Class index -> label:")
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    return "\n".join(lines)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path
