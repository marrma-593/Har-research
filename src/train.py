"""Unified training loop (DESIGN.md §4, §5.3)."""

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.evaluate import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested="auto"):
    if requested in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return requested


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, total_correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    cfg,
    device="auto",
    seed=42,
    checkpoint_path=None,
):
    """Train ``model`` and return history, best val metrics and checkpoint path.

    ``cfg`` keys: ``epochs``, ``lr``, ``weight_decay``, ``lr_factor``,
    ``lr_patience``. The best checkpoint is selected by validation Macro-F1.
    """
    device = get_device(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.get("lr_factor", 0.5),
        patience=cfg.get("lr_patience", 5),
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_state = None
    best_val_metrics = None

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["macro_f1"])

        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "lr": lr,
            }
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics

        print(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"val_acc {val_metrics['accuracy']:.4f} val_f1 {val_metrics['macro_f1']:.4f} "
            f"lr {lr:.2e} | {time.time() - t0:.1f}s"
        )

    checkpoint = {
        "model_name": cfg.get("model_name", "unknown"),
        "seed": seed,
        "config": cfg,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "state_dict": best_state,
    }
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint": checkpoint,
    }
