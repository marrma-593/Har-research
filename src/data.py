"""Data loading / preprocessing / participant-grouped split (DESIGN.md §2).

Loads the raw inertial signal windows (not the pre-extracted 561-dim feature
table) and organises them as ``[batch, 9, 128]``:

| channels | signal                          |
| -------- | ------------------------------- |
| 1-3      | body_acc_x / y / z              |
| 4-6      | body_gyro_x / y / z             |
| 7-9      | total_acc_x / y / z             |
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES = 6
NUM_CHANNELS = 9
SEQ_LEN = 128

# Class names in the order of activity_labels.txt (label -> 1-indexed).
CLASS_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

# Short labels used for rendering the confusion matrix.
CLASS_ABBREV = ["Walk", "WalkUp", "WalkDn", "Sit", "Stand", "Lay"]

# Channel order fixed by DESIGN.md §2.2.
_CHANNEL_FILES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _load_signal(path):
    """Load one raw inertial signal file: [N, 128] float32."""
    return np.loadtxt(path, dtype=np.float32)


def _load_split(split_dir, split_name):
    """Load a single train/test split into (x, y, subjects)."""
    split_dir = Path(split_dir)
    channels = [
        _load_signal(split_dir / "Inertial Signals" / f"{name}_{split_name}.txt")
        for name in _CHANNEL_FILES
    ]
    x = np.stack(channels, axis=1).astype(np.float32)  # [N, 9, 128]

    y = np.loadtxt(split_dir / f"y_{split_name}.txt", dtype=np.int64)
    y = y - 1  # 1..6 -> 0..5 class indices

    subjects = np.loadtxt(split_dir / f"subject_{split_name}.txt", dtype=np.int64)
    return x, y, subjects


def load_raw_signals(dataset_dir):
    """Return the full dataset as a dict of numpy arrays."""
    dataset_dir = Path(dataset_dir)
    x_train, y_train, s_train = _load_split(dataset_dir / "train", "train")
    x_test, y_test, s_test = _load_split(dataset_dir / "test", "test")
    return {
        "x_train": x_train,
        "y_train": y_train,
        "subject_train": s_train,
        "x_test": x_test,
        "y_test": y_test,
        "subject_test": s_test,
    }


def split_train_val_by_subject(subjects, val_ratio=0.2, seed=42):
    """Split train indices into train/val, grouped by participant (DESIGN.md §2.3).

    ``seed`` is the *data-split* seed and should stay fixed across model seeds,
    otherwise seed-to-seed differences mix model randomness with a different
    validation-subject set.

    Returns ``(train_idx, val_idx, val_subjects)`` so no participant appears in
    both splits.
    """
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    val_subjects = np.sort(order[:n_val])
    val_mask = np.isin(subjects, val_subjects)
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    return train_idx, val_idx, val_subjects


def stratified_subsample_indices(y, fraction, seed=42):
    """Return class-stratified training indices for the small-sample study.

    ``fraction`` is applied *per class*, so every activity keeps the same
    relative representation. Subsets are **nested** across fractions: for a
    fixed ``seed`` the 10% indices are a subset of the 25% indices, which are a
    subset of the 50% indices. This removes "which windows were drawn" as a
    confound when comparing fractions.

    ``seed`` should be fixed across model seeds (see ``build_dataloaders``).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")
    y = np.asarray(y)
    if fraction == 1.0:
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        # Draw one permutation per class; take a prefix so subsets nest.
        order = rng.permutation(cls_idx)
        n = max(1, int(round(len(cls_idx) * fraction)))
        selected.append(order[: min(n, len(cls_idx))])
    return np.sort(np.concatenate(selected))


class HARDataset(Dataset):
    """Torch dataset wrapping raw windows [N, 9, 128] and integer labels."""

    def __init__(self, x, y):
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def build_dataloaders(
    dataset_dir,
    val_ratio=0.2,
    batch_size=64,
    split_seed=42,
    subset_seed=42,
    train_fraction=1.0,
    num_workers=0,
):
    """Build train/val/test DataLoaders with a subject-grouped validation split.

    Seeds are deliberately separate:

    * ``split_seed`` fixes the participant-grouped train/val split;
    * ``subset_seed`` fixes which windows form the small-sample training subset;
    * the *model* seed is applied by the caller (``set_seed``) and controls
      weight initialisation, dropout and batch order only.

    Keeping the two data seeds fixed across model seeds means the reported
    mean/std across seeds measures training stochasticity, not a different data
    split. The small-sample subsets are class-stratified and nested across
    fractions (see ``stratified_subsample_indices``).
    """
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")

    data = load_raw_signals(dataset_dir)

    train_idx, val_idx, val_subjects = split_train_val_by_subject(
        data["subject_train"], val_ratio=val_ratio, seed=split_seed
    )

    num_train_full = len(train_idx)
    if train_fraction < 1.0:
        sub = stratified_subsample_indices(
            data["y_train"][train_idx], train_fraction, seed=subset_seed
        )
        train_idx = train_idx[sub]

    train_ds = HARDataset(data["x_train"][train_idx], data["y_train"][train_idx])
    val_ds = HARDataset(data["x_train"][val_idx], data["y_train"][val_idx])
    test_ds = HARDataset(data["x_test"], data["y_test"])

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return {
        "loaders": loaders,
        "data_split": {
            "val_ratio": val_ratio,
            "split_seed": split_seed,
            "subset_seed": subset_seed,
            "train_fraction": train_fraction,
            "train_subjects": np.unique(data["subject_train"][train_idx]).tolist(),
            "val_subjects": np.sort(val_subjects).tolist(),
            "num_train_full": num_train_full,
            "num_train": len(train_ds),
            "num_val": len(val_ds),
            "num_test": len(test_ds),
        },
        "val_subjects": np.sort(val_subjects).tolist(),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
    }
