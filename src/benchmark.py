"""Resource-efficiency measurements (DESIGN.md §5.2).

Reports:
- number of trainable / total parameters,
- model file size on disk (state_dict only),
- CPU single-sample inference latency.

Run all models through the same controlled measurement:

    conda run -n mamba_env python -m src.benchmark --num-runs 200
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def count_parameters(model):
    """Return trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


def model_size_bytes(model, path):
    """Save the state_dict to ``path`` and return its size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path.stat().st_size


def measure_cpu_latency(model, input_shape=(1, 9, 128), num_warmup=10, num_runs=100):
    """Measure CPU single-sample forward latency (mean/std/p95 in ms)."""
    model = model.cpu().eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        for _ in range(num_warmup):
            model(x)
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    times = np.asarray(times) * 1000.0  # seconds -> ms
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "p95_ms": float(np.percentile(times, 95)),
        "num_runs": num_runs,
    }


def benchmark_model(model, checkpoint_dir, model_name, seed=42, num_runs=200):
    """Load the best checkpoint (if present) and measure all resource metrics."""
    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / f"{model_name}_seed{seed}_best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print(f"[warn] {best_path} not found; benchmarking random weights")

    weights_path = checkpoint_dir / f"{model_name}_seed{seed}_weights.pt"
    return {
        "parameters": count_parameters(model),
        "model_size_bytes": model_size_bytes(model, weights_path),
        "cpu_latency_ms": measure_cpu_latency(model, num_runs=num_runs),
    }


def main(argv=None):
    import yaml

    from models import get_model

    parser = argparse.ArgumentParser(description="Benchmark parameters / size / CPU latency.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=None,
        help="config files (default: all configs/*.yaml)",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=200)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU thread count")
    parser.add_argument("--output", default="results/benchmark_summary.json")
    args = parser.parse_args(argv)

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    config_paths = (
        [Path(p) for p in args.config]
        if args.config
        else sorted(Path("configs").glob("*.yaml"))
    )

    results = {}
    for config_path in config_paths:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg["model_name"]
        model = get_model(model_name, **cfg.get("model_params", {}))
        bench = benchmark_model(
            model,
            args.checkpoint_dir,
            model_name,
            seed=args.seed,
            num_runs=args.num_runs,
        )
        results[model_name] = bench
        print(f"benchmarked {model_name}")

    summary = {
        "num_runs": args.num_runs,
        "torch_threads": torch.get_num_threads(),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCPU threads: {torch.get_num_threads()} | runs per model: {args.num_runs}\n")
    print(f"{'model':18s}{'params':>12s}{'size_KB':>11s}{'latency_ms':>12s}{'p95_ms':>10s}")
    for name, bench in results.items():
        print(
            f"{name:18s}"
            f"{bench['parameters']['trainable']:>12,d}"
            f"{bench['model_size_bytes'] / 1024:>11.1f}"
            f"{bench['cpu_latency_ms']['mean_ms']:>12.3f}"
            f"{bench['cpu_latency_ms']['p95_ms']:>10.3f}"
        )
    print(f"\nSaved to: {output}")
    return results


if __name__ == "__main__":
    main()
