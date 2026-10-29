"""Training entry point (DESIGN.md §6).

Usage:
    conda run -n mamba_env python train.py --config configs/cnn1d.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from models import get_model
from src import benchmark, data
from src.evaluate import evaluate, format_confusion_matrix, save_json
from src.train import get_device, set_seed, train_model

DEFAULT_DATA_DIR = (
    "data/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a HAR model.")
    parser.add_argument("--config", default="configs/cnn1d.yaml")
    parser.add_argument("--model", default=None, help="override model_name in config")
    parser.add_argument("--seed", type=int, default=None, help="model seed")
    parser.add_argument("--split-seed", type=int, default=None, help="participant-split seed")
    parser.add_argument("--subset-seed", type=int, default=None, help="small-sample subset seed")
    parser.add_argument("--train-fraction", type=float, default=None, help="fraction of training windows (0,1]")
    parser.add_argument("--tag-suffix", default="", help="suffix appended to result/checkpoint names")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    model_name = args.model or cfg["model_name"]
    model_params = cfg.get("model_params", {})

    training = dict(cfg.get("training", {}))
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.lr is not None:
        training["lr"] = args.lr

    data_cfg = cfg.get("data", {})

    # Model seed and data seeds are deliberately independent: the train/val
    # participant split (and the small-sample subset) stay fixed across the
    # three model seeds, so seed-to-seed variance reflects training
    # stochasticity rather than a different validation-subject set.
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    split_seed = (
        args.split_seed if args.split_seed is not None else data_cfg.get("split_seed", 42)
    )
    subset_seed = (
        args.subset_seed if args.subset_seed is not None else data_cfg.get("subset_seed", split_seed)
    )
    train_fraction = (
        args.train_fraction
        if args.train_fraction is not None
        else data_cfg.get("train_fraction", 1.0)
    )
    val_ratio = args.val_ratio if args.val_ratio is not None else data_cfg.get("val_ratio", 0.2)
    batch_size = training.get("batch_size", 64)

    set_seed(seed)
    device = get_device(args.device)

    # ---- data ----
    data_dir = Path(args.data_dir)
    split = data.build_dataloaders(
        data_dir,
        val_ratio=val_ratio,
        batch_size=batch_size,
        split_seed=split_seed,
        subset_seed=subset_seed,
        train_fraction=train_fraction,
    )
    loaders = split["loaders"]
    print(
        f"Data: fraction={train_fraction:g} train={split['num_train']}/{split['data_split']['num_train_full']} "
        f"val={split['num_val']} test={split['num_test']} | "
        f"split_seed={split_seed} subset_seed={subset_seed} | "
        f"val_subjects={split['val_subjects']} | device={device}"
    )

    # ---- model ----
    model = get_model(model_name, **model_params)
    n_params = benchmark.count_parameters(model)
    print(f"Model: {model_name} | trainable={n_params['trainable']:,} total={n_params['total']:,}")

    # ---- train ----
    run_cfg = {
        "model_name": model_name,
        "model_params": model_params,
        "data": {
            **data_cfg,
            "val_ratio": val_ratio,
            "split_seed": split_seed,
            "subset_seed": subset_seed,
            "train_fraction": train_fraction,
        },
        "seed": seed,
        **training,
    }
    resolved_cfg = {**cfg, "model_name": model_name, "seed": seed, "data": run_cfg["data"]}
    tag = f"{model_name}_seed{seed}{args.tag_suffix}"
    ckpt_path = Path(args.checkpoint_dir) / f"{tag}_best.pt"
    result = train_model(
        model,
        loaders["train"],
        loaders["val"],
        run_cfg,
        device=device,
        seed=seed,
        checkpoint_path=ckpt_path,
    )

    # ---- test ----
    model.load_state_dict(result["checkpoint"]["state_dict"])
    model = model.to(device)
    test_metrics = evaluate(model, loaders["test"], device)

    # ---- resource efficiency ----
    size_path = Path(args.checkpoint_dir) / f"{tag}_weights.pt"
    size_bytes = benchmark.model_size_bytes(model, size_path)
    latency = benchmark.measure_cpu_latency(model)
    bench = {
        "parameters": n_params,
        "model_size_bytes": size_bytes,
        "cpu_latency_ms": latency,
    }

    # ---- save results ----
    results_dir = Path(args.results_dir)
    summary = {
        "model_name": model_name,
        "seed": seed,
        "split_seed": split_seed,
        "subset_seed": subset_seed,
        "train_fraction": train_fraction,
        "config": resolved_cfg,
        "data_split": split["data_split"],
        "val_subjects": split["val_subjects"],
        "best_epoch": result["best_epoch"],
        "best_val_metrics": result["best_val_metrics"],
        "test_metrics": test_metrics,
        "benchmark": bench,
    }
    save_json(summary, results_dir / f"{tag}_summary.json")
    save_json(result["history"], results_dir / f"{tag}_history.json")

    cm_text = format_confusion_matrix(test_metrics["confusion_matrix"])
    (results_dir / f"{tag}_confusion_matrix.txt").write_text(cm_text + "\n")

    # ---- console summary ----
    print("\n================ Best validation ================")
    print(json.dumps(result["best_val_metrics"], indent=2))
    print("\n================ Test metrics ================")
    print(json.dumps({k: v for k, v in test_metrics.items() if k != "confusion_matrix"}, indent=2))
    print(f"\nConfusion matrix (rows=true, cols=pred):\n{cm_text}")
    print("\n================ Resource efficiency ================")
    print(
        f"params: trainable={n_params['trainable']:,} total={n_params['total']:,} | "
        f"size={size_bytes / 1024:.2f} KB | "
        f"cpu_latency mean={latency['mean_ms']:.3f} ms (p95 {latency['p95_ms']:.3f} ms)"
    )
    print(f"\nSaved to: {results_dir}/ and {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
