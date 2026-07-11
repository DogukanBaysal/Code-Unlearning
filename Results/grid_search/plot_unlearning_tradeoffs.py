#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "grid_search_plot_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import thesis_style  # noqa: E402

thesis_style.apply_rcparams()

EXCLUDED_TECHNIQUE_LRS = {("npo", "5e-6")}


def parse_float(value: str) -> float:
    return float(value) if value not in ("", "None", None) else float("nan")


def read_rows(csv_path: Path) -> list[dict]:
    """Read the filtered grid-search metrics.

    Uses the same filtering as the secret/code-unit analyses: forget quality is
    the chrF reduction on baseline exact-match rows, utility retention is
    measured on UtilityEval tasks the baseline passed.
    """
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["technique"], row["learning_rate"]) in EXCLUDED_TECHNIQUE_LRS:
                continue
            row["checkpoint"] = int(row["checkpoint"])
            row["learning_rate_float"] = float(row["learning_rate"])
            row["utility_retention"] = parse_float(
                row["utility_retention_baseline_passed"]
            )
            row["forget_quality"] = parse_float(
                row["chrf_similarity_score_reduction"]
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
        row["forget_quality"]
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
    color_by_lr: dict[str, str],
    marker_by_lr: dict[str, str],
    annotate_checkpoints: bool,
    show_ylabel: bool = True,
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    panel_label: str | None = None,
) -> None:
    sorted_lrs = sorted(by_lr, key=float)

    ax.scatter(
        [1.0],
        [0.0],
        marker="X",
        s=90,
        color="#0b0b0b",
        label="baseline",
        zorder=5,
    )

    for lr in sorted_lrs:
        lr_rows = by_lr[lr]
        xs = [row["utility_retention"] for row in lr_rows]
        ys = [row["forget_quality"] for row in lr_rows]
        color = color_by_lr[lr]
        marker = marker_by_lr[lr]

        ax.plot(
            xs,
            ys,
            marker=marker,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.8,
            color=color,
            label=f"lr={lr}",
        )

        if annotate_checkpoints:
            for row, x, y in zip(lr_rows, xs, ys):
                digit = ax.annotate(
                    str(row["checkpoint"]),
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9.5,
                    color=color,
                )
                digit.set_path_effects(thesis_style.halo())

    title = technique.upper()
    if panel_label:
        title = f"({panel_label}) {title}"
    ax.set_title(title)
    ax.set_xlabel("UtilityEval pass@1")
    if show_ylabel:
        ax.set_ylabel("Forget Quality (1 − chrF)")
    ax.grid(True, alpha=0.5)
    ax.set_xlim(*(xlim or x_limits(by_lr)))
    ax.set_ylim(*(ylim or y_limits(by_lr)))
    # Same acceptance treatment as the secret/code-unit tradeoff figures.
    thesis_style.draw_acceptance_quadrant(ax, 0.90, 0.90, label=show_ylabel)
    ax.legend(frameon=True, loc="lower left", framealpha=0.9)


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
    color_by_lr: dict[str, str],
    marker_by_lr: dict[str, str],
    output_dir: Path,
    dpi: int,
    annotate_checkpoints: bool,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    draw_technique(
        ax,
        technique,
        by_lr,
        color_by_lr,
        marker_by_lr,
        annotate_checkpoints,
    )
    fig.tight_layout()

    output_paths = save_figure(fig, output_dir / f"{technique}_tradeoff_by_lr", dpi)
    plt.close(fig)
    return output_paths


def plot_combined(
    grouped: dict[str, dict[str, list[dict]]],
    color_by_lr: dict[str, str],
    marker_by_lr: dict[str, str],
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
    # Same canvas as the secret-suite 3-panel tradeoff figures so fonts render
    # at the same effective size when scaled to text width.
    fig, axes = plt.subplots(
        1, len(techniques), figsize=(12.6, 5.8), sharey=False, sharex=False
    )
    if len(techniques) == 1:
        axes = [axes]

    for idx, (ax, technique) in enumerate(zip(axes, techniques)):
        draw_technique(
            ax,
            technique,
            grouped[technique],
            color_by_lr,
            marker_by_lr,
            annotate_checkpoints,
            show_ylabel=idx == 0,
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
        default=Path("filtered_baseline_exact_utility_metrics.csv"),
        help=(
            "Input CSV produced by plot_filtered_baseline_tradeoffs.py "
            "(baseline exact-match rows, baseline-passed UtilityEval tasks)."
        ),
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
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Label points with checkpoint numbers.",
    )
    args = parser.parse_args()

    rows = read_rows(args.csv)
    grouped = group_by_technique_and_lr(rows)
    color_by_lr, marker_by_lr = thesis_style.lr_series_style_maps(
        row["learning_rate"] for row in rows
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for technique in sorted(grouped):
        output_paths.extend(
            plot_technique(
                technique=technique,
                by_lr=grouped[technique],
                color_by_lr=color_by_lr,
                marker_by_lr=marker_by_lr,
                output_dir=args.output_dir,
                dpi=args.dpi,
                annotate_checkpoints=args.annotate_checkpoints,
            )
        )
    output_paths.extend(
        plot_combined(
            grouped=grouped,
            color_by_lr=color_by_lr,
            marker_by_lr=marker_by_lr,
            output_dir=args.output_dir,
            dpi=args.dpi,
            annotate_checkpoints=args.annotate_checkpoints,
        )
    )

    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
