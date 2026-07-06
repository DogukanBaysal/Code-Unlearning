#!/usr/bin/env python3
import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "grid_search_plot_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 28,
        "axes.titlesize": 42,
        "axes.labelsize": 36,
        "xtick.labelsize": 31,
        "ytick.labelsize": 31,
        "legend.fontsize": 28,
    }
)


def parse_float(value: str) -> float:
    return float(value) if value not in ("", "None", None) else float("nan")


def read_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["checkpoint"] = int(row["checkpoint"])
            row["learning_rate_float"] = float(row["learning_rate"])
            row["utility_retention"] = parse_float(row["utility_retention"])
            row["average_similarity_score_reduction"] = parse_float(
                row["average_similarity_score_reduction"]
            )
            rows.append(row)
    return rows


def group_by_technique_and_lr(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["technique"]][row["learning_rate"]].append(row)
    for technique_rows in grouped.values():
        for lr_rows in technique_rows.values():
            lr_rows.sort(key=lambda row: row["checkpoint"])
    return grouped


def y_limits(by_lr: dict[str, list[dict]]) -> tuple[float, float]:
    all_y = [
        row["average_similarity_score_reduction"]
        for lr_rows in by_lr.values()
        for row in lr_rows
    ]
    y_min = min(0.0, min(all_y))
    y_max = max(all_y)
    padding = max((y_max - y_min) * 0.08, 0.03)
    return y_min - padding, y_max + padding


def x_limits(by_lr: dict[str, list[dict]]) -> tuple[float, float]:
    all_x = [
        row["utility_retention"]
        for lr_rows in by_lr.values()
        for row in lr_rows
    ]
    all_x.append(1.0)
    x_min = min(all_x)
    x_max = max(all_x)
    padding = max((x_max - x_min) * 0.12, 0.025)
    return max(0.0, x_min - padding), min(1.05, x_max + padding)


def draw_technique(
    ax,
    technique: str,
    by_lr: dict[str, list[dict]],
    annotate_checkpoints: bool,
    show_ylabel: bool = True,
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    panel_label: str | None = None,
) -> None:
    sorted_lrs = sorted(by_lr, key=float)
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    ax.scatter(
        [1.0],
        [0.0],
        marker="s",
        s=76,
        color="black",
        label="Baseline",
        zorder=4,
    )

    for idx, lr in enumerate(sorted_lrs):
        lr_rows = by_lr[lr]
        xs = [row["utility_retention"] for row in lr_rows]
        ys = [row["average_similarity_score_reduction"] for row in lr_rows]
        color = cmap(idx % cmap.N)
        marker = markers[idx % len(markers)]

        ax.plot(
            xs,
            ys,
            marker=marker,
            markersize=7,
            linewidth=2.0,
            color=color,
            label=f"lr={lr}",
        )

        if annotate_checkpoints:
            for row, x, y in zip(lr_rows, xs, ys):
                ax.annotate(
                    str(row["checkpoint"]),
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )

    title = technique.upper()
    if panel_label:
        title = f"({panel_label}) {title}"
    ax.set_title(title)
    ax.set_xlabel("UtilityEval Retention")
    if show_ylabel:
        ax.set_ylabel("Forget Quality")
    ax.grid(True, alpha=0.28)
    ax.set_xlim(*(xlim or x_limits(by_lr)))
    ax.set_ylim(*(ylim or y_limits(by_lr)))
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    x_span = current_xlim[1] - current_xlim[0]
    y_span = current_ylim[1] - current_ylim[0]
    ax.axvspan(
        current_xlim[1] - 0.22 * x_span,
        current_xlim[1],
        ymin=0.72,
        ymax=1.0,
        color="#2ca02c",
        alpha=0.10,
        zorder=0,
    )
    ax.text(
        current_xlim[1] - 0.11 * x_span,
        current_ylim[1] - 0.07 * y_span,
        "Optimal region",
        fontsize=28,
        fontweight="bold",
        color="#1b6e1b",
        ha="center",
        va="center",
    )
    ax.legend(frameon=True, loc="best")


def save_figure(fig, output_stem: Path, dpi: int) -> list[Path]:
    output_paths = []
    for suffix in (".png", ".pdf"):
        output_path = output_stem.with_suffix(suffix)
        save_kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        output_paths.append(output_path)
    return output_paths


def plot_technique(
    technique: str,
    by_lr: dict[str, list[dict]],
    output_dir: Path,
    dpi: int,
    annotate_checkpoints: bool,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.5, 7.8))
    draw_technique(ax, technique, by_lr, annotate_checkpoints)
    fig.tight_layout()

    output_paths = save_figure(fig, output_dir / f"{technique}_tradeoff_by_lr", dpi)
    plt.close(fig)
    return output_paths


def plot_combined(
    grouped: dict[str, dict[str, list[dict]]],
    output_dir: Path,
    dpi: int,
    annotate_checkpoints: bool,
) -> list[Path]:
    techniques = sorted(grouped)
    combined_by_lr = {
        "all": [
            row
            for technique in techniques
            for lr_rows in grouped[technique].values()
            for row in lr_rows
        ]
    }
    ylim = y_limits(combined_by_lr)
    fig, axes = plt.subplots(1, len(techniques), figsize=(30.0, 8.8), sharey=True)
    if len(techniques) == 1:
        axes = [axes]

    for idx, (ax, technique) in enumerate(zip(axes, techniques)):
        draw_technique(
            ax,
            technique,
            grouped[technique],
            annotate_checkpoints,
            show_ylabel=idx == 0,
            ylim=ylim,
            xlim=x_limits(grouped[technique]),
            panel_label=chr(ord("a") + idx),
        )

    fig.tight_layout()
    output_paths = save_figure(fig, output_dir / "all_techniques_tradeoff_by_lr", dpi)
    plt.close(fig)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one utility-vs-forget-quality plot per unlearning technique."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("unlearning_similarity_utility_metrics.csv"),
        help="Input CSV produced by extract_unlearning_utility_metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where technique plots will be written.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--annotate-checkpoints",
        action="store_true",
        help="Label points with checkpoint numbers.",
    )
    args = parser.parse_args()

    rows = read_rows(args.csv)
    grouped = group_by_technique_and_lr(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for technique in sorted(grouped):
        output_paths.extend(
            plot_technique(
                technique=technique,
                by_lr=grouped[technique],
                output_dir=args.output_dir,
                dpi=args.dpi,
                annotate_checkpoints=args.annotate_checkpoints,
            )
        )
    output_paths.extend(
        plot_combined(
            grouped=grouped,
            output_dir=args.output_dir,
            dpi=args.dpi,
            annotate_checkpoints=args.annotate_checkpoints,
        )
    )

    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
