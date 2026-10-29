"""Matplotlib learning-curve renderer (DESIGN.md §4 实验三).

Renders the small-sample Macro-F1 curve (mean ± std across model seeds) with
``matplotlib``. Colour follows the model entity via ``model_order`` — never the
plotted / filtered order — so a model keeps its hue in both the main and the
zoom figure, and distinct marker shapes add a second identity channel. The
CSV / Markdown tables emitted by ``src.small_sample`` are the table view for
the same values.

The output is a static figure (PNG by default); interactive hover and the
dark-mode variant of the previous hand-written SVG renderer are not part of a
static matplotlib image.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file rendering only; safe on headless machines

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import rc_context
from matplotlib.lines import Line2D

# Fixed categorical order (validated light palette). Slot follows the model's
# position in ``model_order``, so filtering series never repaints survivors.
# Three slots sit below 3:1 contrast on the light surface; the visible legend /
# direct label plus the accompanying Markdown table are the relief rule.
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# Secondary encoding so identity never rests on colour alone.
_MARKERS = ["o", "s", "^", "D", "h", "X", "v", "*"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _fmt_fraction(f):
    return f"{float(f) * 100:g}%"


def _nice_bounds(values):
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 0.02)
    pad = max(0.004, span * 0.06)
    ymin = max(0.0, math.floor((lo - pad) * 50.0) / 50.0)
    ymax = min(1.0, math.ceil((hi + pad) * 50.0) / 50.0)
    if ymax - ymin < 0.10:
        mid = (ymax + ymin) / 2.0
        ymin = max(0.0, mid - 0.05)
        ymax = min(1.0, mid + 0.05)
    return ymin, ymax


def render_learning_curve(
    summary,
    output_path,
    metric="test_macro_f1",
    title="小样本学习曲线：测试集 Macro-F1",
    subtitle="UCI HAR · 每个点为 3 个模型种子的均值，误差棒为 ±1 标准差",
    plot_models=None,
):
    """Render ``summary`` (from ``src.small_sample.aggregate``) to a figure file.

    ``plot_models`` restricts the plotted series (e.g. a zoom view) while colour
    slots stay tied to the global ``model_order``, so a model keeps its hue in
    every panel. The format follows the suffix of ``output_path``; a path with
    no suffix gets ``.png``. Explicit ``.svg`` / ``.pdf`` / ... still work.
    """
    model_order = summary["model_order"]
    plot_models = list(plot_models) if plot_models is not None else list(model_order)
    display = summary.get("model_display", {})
    fractions = [float(f) for f in summary["fractions"]]
    models = summary["models"]
    metric_label = {
        "test_macro_f1": "测试集 Macro-F1",
        "test_accuracy": "测试集 Accuracy",
        "val_macro_f1": "验证集 Macro-F1",
    }.get(metric, metric)

    if len(model_order) > len(SERIES_COLORS):
        raise ValueError("more than 8 series: fold the tail into 'Other' or facet")

    # Collect all mean ± std values for a common y-scale.
    values = []
    n_seeds = 0
    for model in plot_models:
        for frac in fractions:
            stats = models.get(model, {}).get(str(frac))
            if not stats or stats.get(f"{metric}_mean") is None:
                continue
            mean = stats[f"{metric}_mean"]
            std = stats.get(f"{metric}_std") or 0.0
            values.append(mean - std)
            values.append(mean + std)
            n_seeds = max(n_seeds, int(stats.get("n", 0)))
    if not values:
        raise ValueError("no data to plot")
    ymin, ymax = _nice_bounds(values)

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "svg.fonttype": "path",  # keep CJK glyphs self-contained in the SVG
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }

    with rc_context(style):
        fig = plt.figure(figsize=(11, 7), dpi=100)
        fig.patch.set_facecolor(SURFACE)
        # Fixed layout mirrors the previous renderer: plot on the left, legend
        # in the right margin, title block above, provenance notes below.
        ax = fig.add_axes([0.078, 0.164, 0.695, 0.704])
        ax.set_facecolor(SURFACE)

        fig.text(
            0.078, 0.965, title,
            fontsize=16, fontweight="semibold", color=INK, ha="left", va="top",
        )
        fig.text(
            0.078, 0.925, subtitle,
            fontsize=9.5, color=INK_2, ha="left", va="top",
        )

        # Recessive chrome: horizontal hairlines, no top/right spines.
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(
            axis="both", colors=AXIS, labelcolor=MUTED, labelsize=8.5,
            length=4, width=0.8,
        )

        n_ticks = 5
        yticks = [ymin + (ymax - ymin) * i / (n_ticks - 1) for i in range(n_ticks)]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:.2f}" for v in yticks])
        ax.set_ylim(ymin, ymax)

        n_frac = len(fractions)
        ax.set_xticks(range(n_frac))
        ax.set_xticklabels([_fmt_fraction(f) for f in fractions])
        ax.set_xlim(-0.12, max(0.12, n_frac - 1 + 0.12))

        ax.set_xlabel("训练数据比例", fontsize=9.5, color=INK_2, labelpad=8)
        ax.set_ylabel(metric_label, fontsize=9.5, color=INK_2, labelpad=8)

        # Series, in fixed model order (colour identity follows the model).
        handles = []
        labels = []
        plotted_endpoints = []
        for model in plot_models:
            slot = model_order.index(model) + 1
            color = SERIES_COLORS[slot - 1]
            marker = _MARKERS[slot - 1]

            points = []
            for idx, frac in enumerate(fractions):
                stats = models.get(model, {}).get(str(frac))
                if not stats or stats.get(f"{metric}_mean") is None:
                    continue
                points.append(
                    (
                        idx,
                        frac,
                        float(stats[f"{metric}_mean"]),
                        float(stats.get(f"{metric}_std") or 0.0),
                        int(stats.get("n", 0)),
                    )
                )
            if not points:
                continue

            xs = [p[0] for p in points]
            means = [p[2] for p in points]

            ax.plot(
                xs, means, color=color, linewidth=2.0,
                solid_capstyle="round", solid_joinstyle="round", zorder=3,
            )

            err = [p for p in points if p[3] > 0]
            if err:
                ax.errorbar(
                    [p[0] for p in err],
                    [p[2] for p in err],
                    yerr=[
                        [p[2] - max(0.0, p[2] - p[3]) for p in err],
                        [min(1.0, p[2] + p[3]) - p[2] for p in err],
                    ],
                    fmt="none", ecolor=color, elinewidth=1.1,
                    capsize=3.0, capthick=1.1, alpha=0.55, zorder=2,
                )

            ax.plot(
                xs, means, linestyle="none", marker=marker,
                markersize=8, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
            )

            handles.append(
                Line2D(
                    [], [], color=color, linewidth=2.0, marker=marker,
                    markersize=8, markerfacecolor=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                )
            )
            labels.append(display.get(model, model))
            plotted_endpoints.append((model, points[-1]))

        # Selective direct label: the highest final mean only.
        if plotted_endpoints:
            best_model, best_point = max(
                plotted_endpoints, key=lambda item: item[1][2]
            )
            bx, bmean = best_point[0], best_point[2]
            ax.annotate(
                f"{display.get(best_model, best_model)} {bmean:.3f}",
                xy=(bx, bmean), xytext=(-8, 10), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.5, color=INK, zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground=SURFACE)],
            )

        # Right-side legend: needed for ≥ 2 series; a lone series is named by
        # the direct label instead (identity is never left implicit).
        if len(handles) >= 2:
            legend = fig.legend(
                handles, labels, title="模型",
                loc="upper left", bbox_to_anchor=(0.797, 0.868),
                frameon=False, fontsize=9, title_fontsize=9.5,
                labelcolor=INK_2, handlelength=2.0, handletextpad=0.6,
                borderaxespad=0.0, labelspacing=0.9,
            )
            legend.get_title().set_color(INK)
            legend.get_title().set_fontweight("semibold")

        # Footer / provenance.
        seed_note = f"（n={n_seeds} 个模型种子）" if n_seeds else ""
        fig.text(
            0.078, 0.070,
            f"误差棒：±1 标准差{seed_note}；数据划分与子集种子固定，"
            f"种子差异仅来自训练随机性。",
            fontsize=8, color=MUTED, ha="left", va="top",
        )
        fig.text(
            0.078, 0.038,
            f"X 轴为 {n_frac} 个训练比例条件（等距排列）；"
            f"Y 轴非零起点（{ymin:.2f}–{ymax:.2f}）。",
            fontsize=8, color=MUTED, ha="left", va="top",
        )

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, facecolor=SURFACE, edgecolor="none")
        plt.close(fig)

    return output_path
