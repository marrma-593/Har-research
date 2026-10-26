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
    parser.add_argument("--seed", type=int, default=None)
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

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    val_ratio = args.val_ratio if args.val_ratio is not None else cfg.get("data", {}).get("val_ratio", 0.2)
    batch_size = training.get("batch_size", 64)

    set_seed(seed)
    device = get_device(args.device)

    # ---- data ----
    data_dir = Path(args.data_dir)
    split = data.build_dataloaders(
        data_dir,
        val_ratio=val_ratio,
        batch_size=batch_size,
        seed=seed,
    )
    loaders = split["loaders"]
    print(
        f"Data: train={split['num_train']} val={split['num_val']} test={split['num_test']} | "
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
        "data": cfg.get("data", {}),
        **training,
    }
    ckpt_path = Path(args.checkpoint_dir) / f"{model_name}_seed{seed}_best.pt"
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
    size_path = Path(args.checkpoint_dir) / f"{model_name}_seed{seed}_weights.pt"
    size_bytes = benchmark.model_size_bytes(model, size_path)
    latency = benchmark.measure_cpu_latency(model)
    bench = {
        "parameters": n_params,
        "model_size_bytes": size_bytes,
        "cpu_latency_ms": latency,
    }

    # ---- save results ----
    results_dir = Path(args.results_dir)
    tag = f"{model_name}_seed{seed}"
    summary = {
        "model_name": model_name,
        "seed": seed,
        "config": cfg,
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
