"""Week 6 small-sample experiments (DESIGN.md §4 实验三, §5.3).

Runs every model over the training-data fractions {10%, 25%, 50%, 100%} with
three model seeds, then aggregates mean ± std and renders the Macro-F1 learning
curve.

Seed policy
-----------
``--split-seed`` fixes the participant-grouped train/val split and
``--subset-seed`` fixes the class-stratified, nested small-sample subset. Both
stay **fixed across model seeds**, so the spread across seeds measures training
stochasticity (initialisation / dropout / batch order), not a different data
split. This is why the runner does not pass the model seed to the data code.

Usage
-----
    conda run -n mamba_env python -m src.small_sample
    conda run -n mamba_env python -m src.small_sample --aggregate-only
    conda run -n mamba_env python -m src.small_sample --models resnet1d_lstm --jobs 1
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.plot import render_learning_curve

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    "data/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
)

# Fixed order: also fixes the categorical colour slot of each model.
MODEL_ORDER = [
    "cnn1d",
    "lstm",
    "resnet1d",
    "cnn_lstm",
    "resnet1d_lstm",
    "convnext1d_lstm",
    "resnet1d_tcn",
    "transformer",
]
MODEL_DISPLAY = {
    "cnn1d": "1D-CNN",
    "lstm": "LSTM",
    "resnet1d": "ResNet1D",
    "cnn_lstm": "CNN-LSTM",
    "resnet1d_lstm": "ResNet1D-LSTM",
    "convnext1d_lstm": "ConvNeXt1D-LSTM",
    "resnet1d_tcn": "ResNet1D-TCN",
    "transformer": "Transformer",
}
DEFAULT_MODELS = MODEL_ORDER[:7]
DEFAULT_FRACTIONS = [0.10, 0.25, 0.50, 1.00]
DEFAULT_SEEDS = [42, 43, 44]

# Rough per-epoch cost on CPU (seconds) from the week 1-5 logs. Used only to
# schedule long jobs first (LPT), never for reporting.
_EPOCH_COST = {
    "cnn1d": 1.0,
    "lstm": 7.2,
    "resnet1d": 5.4,
    "cnn_lstm": 2.3,
    "resnet1d_lstm": 8.6,
    "convnext1d_lstm": 13.0,
    "resnet1d_tcn": 7.7,
    "transformer": 14.5,
}


def _job_cost(model, fraction):
    # Validation runs on the full val split every epoch, so cost is not purely
    # proportional to the training fraction.
    return _EPOCH_COST.get(model, 8.0) * (0.2 + 0.8 * float(fraction))


def frac_tag(fraction):
    """0.1 -> 'frac0p1', 0.25 -> 'frac0p25', 1.0 -> 'frac1'."""
    return "frac" + f"{float(fraction):g}".replace(".", "p")


def summary_path(results_dir, model, seed, fraction):
    return Path(results_dir) / f"{model}_seed{seed}_{frac_tag(fraction)}_summary.json"


def build_command(args, model, seed, fraction):
    tag = "_" + frac_tag(fraction)
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config", str(Path(args.config_dir) / f"{model}.yaml"),
        "--seed", str(seed),
        "--split-seed", str(args.split_seed),
        "--subset-seed", str(args.subset_seed),
        "--train-fraction", str(fraction),
        "--tag-suffix", tag,
        "--device", args.device,
        "--data-dir", args.data_dir,
        "--results-dir", str(Path(args.results_dir)),
        "--checkpoint-dir", str(Path(args.checkpoint_dir)),
    ]
    return cmd


def run_one(args, model, seed, fraction):
    out_path = summary_path(args.results_dir, model, seed, fraction)
    if out_path.exists() and not args.force:
        return {"model": model, "seed": seed, "fraction": fraction, "status": "skipped"}
    if out_path.exists():
        # Do not let a failed rerun leave a stale summary for aggregation.
        out_path.unlink()

    log_dir = Path(args.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model}_seed{seed}_{frac_tag(fraction)}.log"
    cmd = build_command(args, model, seed, fraction)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(args.threads)

    t0 = time.time()
    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 and out_path.exists() else "failed"
    return {
        "model": model,
        "seed": seed,
        "fraction": fraction,
        "status": status,
        "seconds": elapsed,
        "log": str(log_path),
        "returncode": proc.returncode,
    }


def run_grid(args):
    jobs = [
        (model, seed, fraction)
        for model in args.models
        for fraction in args.fractions
        for seed in args.seeds
    ]
    # Longest-processing-time-first keeps the tail of the schedule short.
    jobs.sort(key=lambda job: _job_cost(job[0], job[2]), reverse=True)
    print(
        f"Small-sample grid: {len(args.models)} models × {len(args.seeds)} seeds × "
        f"{len(args.fractions)} fractions = {len(jobs)} runs | jobs={args.jobs} "
        f"threads/job={args.threads}"
    )
    print(
        f"Fixed data seeds: split_seed={args.split_seed} subset_seed={args.subset_seed} | "
        f"model seeds={args.seeds}"
    )
    if args.dry_run:
        for model, seed, fraction in jobs:
            print(" ".join(build_command(args, model, seed, fraction)))
        return []

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, args, *job): job for job in jobs}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            elapsed = time.time() - t0
            if res["status"] == "skipped":
                detail = "summary exists"
            elif res["status"] == "ok":
                detail = f"{res['seconds']:.0f}s -> {res['log']}"
            else:
                detail = f"returncode={res.get('returncode')} -> {res['log']}"
            print(
                f"[{len(results):>3}/{len(jobs)}] {res['status']:<7} "
                f"{res['model']} seed={res['seed']} frac={res['fraction']:g} | "
                f"{detail} | total {elapsed / 60:.1f} min"
            )

    failed = [r for r in results if r["status"] == "failed"]
    ran = [r for r in results if r["status"] == "ok"]
    print(
        f"\nGrid finished: {len(ran)} ran, "
        f"{len(results) - len(ran) - len(failed)} skipped, {len(failed)} failed "
        f"in {(time.time() - t0) / 60:.1f} min"
    )
    return failed


def load_runs(results_dir, models, fractions, seeds, split_seed):
    """Load per-run summaries from a small-sample results directory."""
    runs = []
    for path in sorted(Path(results_dir).glob("*_summary.json")):
        try:
            with open(path) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if "train_fraction" not in s or "data_split" not in s:
            continue
        if s["model_name"] not in models:
            continue
        if split_seed is not None and s.get("split_seed") != split_seed:
            continue
        if fractions is not None and float(s["train_fraction"]) not in fractions:
            continue
        if seeds is not None and int(s.get("seed", -1)) not in seeds:
            continue
        runs.append(
            {
                "model": s["model_name"],
                "seed": int(s["seed"]),
                "train_fraction": float(s["train_fraction"]),
                "split_seed": s.get("split_seed"),
                "subset_seed": s.get("subset_seed"),
                "num_train": s.get("data_split", {}).get("num_train"),
                "num_val": s.get("data_split", {}).get("num_val"),
                "best_epoch": s.get("best_epoch"),
                "val_macro_f1": s.get("best_val_metrics", {}).get("macro_f1"),
                "test_accuracy": s.get("test_metrics", {}).get("accuracy"),
                "test_macro_f1": s.get("test_metrics", {}).get("macro_f1"),
                "params_trainable": s.get("benchmark", {})
                .get("parameters", {})
                .get("trainable"),
                "model_size_kb": None
                if s.get("benchmark", {}).get("model_size_bytes") is None
                else s["benchmark"]["model_size_bytes"] / 1024.0,
                "cpu_latency_ms": s.get("benchmark", {})
                .get("cpu_latency_ms", {})
                .get("mean_ms"),
                "path": str(path),
            }
        )
    return runs


def _stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(values) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def aggregate(results_dir, models, fractions, seeds, split_seed, model_order=None):
    """Aggregate runs into mean ± std per (model, fraction)."""
    model_order = model_order or [m for m in MODEL_ORDER if m in models] + [
        m for m in models if m not in MODEL_ORDER
    ]
    runs = load_runs(results_dir, set(models), list(fractions), list(seeds), split_seed)

    grouped = {}
    for run in runs:
        key = (run["model"], f"{run['train_fraction']:g}")
        grouped.setdefault(key, []).append(run)

    summary = {
        "meta": {
            "source": "UCI HAR raw inertial windows",
            "metric": "test_macro_f1",
            "split_seed": split_seed,
            "subset_seed": None,
            "seeds": sorted(seeds),
            "fractions": [float(f) for f in fractions],
            "num_runs": len(runs),
            "seed_policy": (
                "split_seed and subset_seed fixed across model seeds; "
                "seed-to-seed variance reflects training stochasticity only"
            ),
        },
        "model_order": model_order,
        "model_display": {m: MODEL_DISPLAY.get(m, m) for m in model_order},
        "fractions": [float(f) for f in fractions],
        "models": {},
    }
    if runs:
        subset_seeds = sorted({r["subset_seed"] for r in runs if r["subset_seed"] is not None})
        summary["meta"]["subset_seed"] = subset_seeds[0] if len(subset_seeds) == 1 else subset_seeds

    for model in model_order:
        summary["models"][model] = {}
        for fraction in fractions:
            group = grouped.get((model, f"{float(fraction):g}"), [])
            if not group:
                continue
            entry = {
                "n": len(group),
                "seeds": sorted(r["seed"] for r in group),
                "num_train": group[0]["num_train"],
            }
            for metric in ("test_macro_f1", "test_accuracy", "val_macro_f1"):
                stats = _stats([r[metric] for r in group])
                entry[f"{metric}_mean"] = stats["mean"]
                entry[f"{metric}_std"] = stats["std"]
                entry[f"{metric}_min"] = stats["min"]
                entry[f"{metric}_max"] = stats["max"]
            entry["best_epoch_mean"] = float(np.mean([r["best_epoch"] for r in group]))
            summary["models"][model][str(float(fraction))] = entry

    return summary, runs


def write_runs_csv(runs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "seed",
        "train_fraction",
        "num_train",
        "num_val",
        "split_seed",
        "subset_seed",
        "best_epoch",
        "val_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "params_trainable",
        "model_size_kb",
        "cpu_latency_ms",
    ]
    runs = sorted(runs, key=lambda r: (r["model"], r["train_fraction"], r["seed"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({k: run.get(k) for k in fields})
    return path


def _cell(entry, metric):
    mean = entry.get(f"{metric}_mean")
    std = entry.get(f"{metric}_std")
    if mean is None:
        return "–"
    return f"{mean:.4f} ± {std:.4f}"


def write_markdown(summary, path):
    models = summary["models"]
    fractions = summary["fractions"]
    order = summary["model_order"]
    display = summary["model_display"]
    lines = [
        "# 第 6 周：小样本学习曲线",
        "",
        f"- 数据划分种子（split_seed）：{summary['meta']['split_seed']}（固定）",
        f"- 小样本子集种子（subset_seed）：{summary['meta']['subset_seed']}（固定）",
        f"- 模型种子：{', '.join(str(s) for s in summary['meta']['seeds'])}",
        f"- 训练比例：{', '.join(f'{f * 100:g}%' for f in fractions)}",
        f"- 已完成运行：{summary['meta']['num_runs']}",
        "",
        "> 单元格为 **均值 ± 标准差**（跨模型种子）。划分与子集种子固定，"
        "因此标准差只反映训练随机性。官方测试集全程固定。",
        "",
        "## 测试集 Macro-F1",
        "",
        "| 模型 | " + " | ".join(f"{f * 100:g}%" for f in fractions) + " |",
        "| --- | " + " | ".join("---:" for _ in fractions) + " |",
    ]
    for model in order:
        cells = []
        for fraction in fractions:
            entry = models.get(model, {}).get(str(float(fraction)))
            cells.append(_cell(entry, "test_macro_f1") if entry else "–")
        lines.append(f"| {display.get(model, model)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 测试集 Accuracy",
        "",
        "| 模型 | " + " | ".join(f"{f * 100:g}%" for f in fractions) + " |",
        "| --- | " + " | ".join("---:" for _ in fractions) + " |",
    ]
    for model in order:
        cells = []
        for fraction in fractions:
            entry = models.get(model, {}).get(str(float(fraction)))
            cells.append(_cell(entry, "test_accuracy") if entry else "–")
        lines.append(f"| {display.get(model, model)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 验证集 Macro-F1（用于选择最佳 epoch，不参与测试调参）",
        "",
        "| 模型 | " + " | ".join(f"{f * 100:g}%" for f in fractions) + " |",
        "| --- | " + " | ".join("---:" for _ in fractions) + " |",
    ]
    for model in order:
        cells = []
        for fraction in fractions:
            entry = models.get(model, {}).get(str(float(fraction)))
            cells.append(_cell(entry, "val_macro_f1") if entry else "–")
        lines.append(f"| {display.get(model, model)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "曲线图：`macro_f1_curve.png`；逐次运行记录：`runs.csv`；完整统计：`summary.json`。",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def write_aggregate_outputs(summary, runs, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": summary["meta"], **{k: v for k, v in summary.items() if k != "meta"}}
    json_path = results_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    csv_path = write_runs_csv(runs, results_dir / "runs.csv")
    md_path = write_markdown(summary, results_dir / "summary.md")

    plot_summary = {
        "model_order": summary["model_order"],
        "model_display": summary["model_display"],
        "fractions": summary["fractions"],
        "models": summary["models"],
    }
    plot_path = zoom_path = None
    if runs:
        plot_path = render_learning_curve(plot_summary, results_dir / "macro_f1_curve.png")

        # A zoom view without curves that are clearly step-limited: a model
        # whose small-fraction means collapse while its 100% mean is healthy is
        # under-trained by the fixed 30-epoch budget, not data-limited. Keeping
        # it in the main figure is honest, but it compresses everyone else's
        # y-scale, so ship a zoom as well.
        display = summary["model_display"]
        limited = []
        for model in summary["model_order"]:
            means = [
                summary["models"].get(model, {}).get(str(float(f)), {}).get("test_macro_f1_mean")
                for f in summary["fractions"]
            ]
            means = [m for m in means if m is not None]
            if means and min(means) < 0.75 and max(means) > 0.85:
                limited.append(model)
        zoom_models = [m for m in summary["model_order"] if m not in limited]
        if limited and len(zoom_models) >= 2:
            names = "、".join(display.get(m, m) for m in limited)
            zoom_path = render_learning_curve(
                plot_summary,
                results_dir / "macro_f1_curve_zoom.png",
                title="小样本学习曲线（放大）：测试集 Macro-F1",
                subtitle=f"UCI HAR · 不含 {names}（受固定 30 epoch 步数预算限制，见主图）",
                plot_models=zoom_models,
            )

    return {
        "summary": json_path,
        "runs": csv_path,
        "markdown": md_path,
        "plot": plot_path,
        "plot_zoom": zoom_path,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Week 6 small-sample experiments.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=MODEL_ORDER)
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--subset-seed", type=int, default=None,
                        help="defaults to --split-seed (fixed across model seeds)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4, help="torch threads per job")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--results-dir", default="results/small_sample")
    parser.add_argument("--checkpoint-dir", default="checkpoints/small_sample")
    parser.add_argument("--logs-dir", default="logs/small_sample")
    parser.add_argument("--force", action="store_true", help="rerun existing summaries")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.subset_seed is None:
        args.subset_seed = args.split_seed
    for fraction in args.fractions:
        if not 0.0 < fraction <= 1.0:
            parser.error(f"--fractions values must be in (0, 1], got {fraction}")
    return args


def main(argv=None):
    args = parse_args(argv)
    failed = []
    if not args.aggregate_only:
        failed = run_grid(args)
        if args.dry_run:
            return 0

    summary, runs = aggregate(
        args.results_dir,
        args.models,
        args.fractions,
        args.seeds,
        args.split_seed,
        model_order=[m for m in MODEL_ORDER if m in args.models],
    )
    paths = write_aggregate_outputs(summary, runs, args.results_dir)

    print(f"\nAggregated {len(runs)} runs.")
    print(f"  summary : {paths['summary']}")
    print(f"  runs    : {paths['runs']}")
    print(f"  table   : {paths['markdown']}")
    if paths["plot"]:
        print(f"  curve   : {paths['plot']}")
    if paths.get("plot_zoom"):
        print(f"  zoom    : {paths['plot_zoom']}")

    if failed:
        print(f"\n{len(failed)} run(s) failed:")
        for item in failed:
            print(f"  {item['model']} seed={item['seed']} frac={item['fraction']}: {item['log']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
