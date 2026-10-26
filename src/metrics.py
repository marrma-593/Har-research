"""Classification metrics (DESIGN.md §5.1).

Accuracy / Macro-F1 / per-class Recall / confusion matrix, implemented with
numpy only so the project has no dependency on scikit-learn.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=6):
    """Return a [C, C] matrix where cm[i, j] counts true=i, pred=j."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def per_class_recall(cm):
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(row_sums > 0, np.diagonal(cm) / row_sums, 0.0)
    return recall


def per_class_precision(cm):
    cm = np.asarray(cm, dtype=np.float64)
    col_sums = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(col_sums > 0, np.diagonal(cm) / col_sums, 0.0)
    return precision


def per_class_f1(cm):
    p = per_class_precision(cm)
    r = per_class_recall(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    return f1


def macro_f1(y_true, y_pred, num_classes=6):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return float(per_class_f1(cm).mean())


def compute_metrics(y_true, y_pred, num_classes=6):
    """Return a dict of all classification metrics (JSON-serialisable)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": float(per_class_f1(cm).mean()),
        "per_class_precision": per_class_precision(cm).tolist(),
        "per_class_recall": per_class_recall(cm).tolist(),
        "per_class_f1": per_class_f1(cm).tolist(),
        "confusion_matrix": cm.tolist(),
    }
