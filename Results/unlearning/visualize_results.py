#!/usr/bin/env python3
"""Visualize average similarity scores for unlearning runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "average_similarity_scores.pdf"
UNLEARNING_STYLES = {
    "code": {
        "label": "Code-unit unlearning",
        "color": "#1f77b4",
    },
    "secret": {
        "label": "Secret unlearning",
        "color": "#d62728",
    },
}


def checkpoint_sort_key(checkpoint: str) -> tuple[int, int | str]:
    if checkpoint == "base":
        return (0, 0)
    try:
        return (1, int(checkpoint))
    except ValueError:
        return (2, checkpoint)


def load_scores(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    scores: dict[str, dict[str, dict[str, float]]] = {}

    for aggregate_path in sorted(root.glob("*/*/*/aggregate_results.json")):
        model, unlearning_type, checkpoint = aggregate_path.parts[-4:-1]

        if model.upper() == "META":
            model = "Llama 3.2 3B"
        else:
            model = "Qwen2.5 Coder 3B"

        if unlearning_type not in UNLEARNING_STYLES:
            continue

        with aggregate_path.open("r", encoding="utf-8") as handle:
            aggregate = json.load(handle)

        reported_mode = aggregate.get("mode")
        if reported_mode and reported_mode != unlearning_type:
            print(
                "Warning: folder mode and JSON mode differ for "
                f"{aggregate_path.relative_to(root)} "
                f"({unlearning_type!r} vs {reported_mode!r})."
            )

        score = aggregate.get("average_similarity_score")
        if score is None:
            print(f"Warning: skipping {aggregate_path.relative_to(root)} without average_similarity_score.")
            continue

        scores.setdefault(model, {}).setdefault(unlearning_type, {})[checkpoint] = round(float(score), 2)

    return scores


def configure_matplotlib(root: Path) -> None:
    cache_dir = root / ".matplotlib-cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    xdg_cache_dir = root / ".cache"
    xdg_cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def plot_scores(scores: dict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not scores:
        raise ValueError("No aggregate_results.json files with average_similarity_score were found.")

    plt.rcParams.update(
        {
            "font.size": 34,
            "axes.titlesize": 44,
            "axes.labelsize": 40,
            "xtick.labelsize": 36,
            "ytick.labelsize": 36,
            "legend.fontsize": 34,
        }
    )

    models = sorted(scores)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(models),
        figsize=(17 * len(models), 12),
        sharey=True,
        squeeze=False,
    )

    legend_handles = [
        Line2D([0], [0], color=style["color"], marker="o", linewidth=6, markersize=18, label=style["label"])
        for style in UNLEARNING_STYLES.values()
    ]

    for col, model in enumerate(models):
        ax = axes[0][col]

        for unlearning_type in ("code", "secret"):
            style = UNLEARNING_STYLES[unlearning_type]
            model_scores = scores.get(model, {}).get(unlearning_type, {})

            base_score = model_scores.get("base")
            base_x = -0.08 if unlearning_type == "code" else 0.08
            epoch_items = [
                (int(checkpoint), score)
                for checkpoint, score in sorted(model_scores.items(), key=lambda item: checkpoint_sort_key(item[0]))
                if checkpoint != "base" and checkpoint.isdigit()
            ]

            if base_score is not None:
                ax.scatter(
                    [base_x],
                    [base_score],
                    s=420,
                    color=style["color"],
                    marker="D",
                    zorder=3,
                )
                base_offset = (14, 18) if unlearning_type == "code" else (14, 10)
                base_ha = "left"
                base_va = "bottom"
                ax.annotate(
                    f"{base_score:.2f}",
                    (base_x, base_score),
                    textcoords="offset points",
                    xytext=base_offset,
                    ha=base_ha,
                    va=base_va,
                    fontsize=32,
                    weight="bold",
                )

            if epoch_items:
                epochs = [epoch for epoch, _ in epoch_items]
                epoch_scores = [score for _, score in epoch_items]
                ax.plot(
                    epochs,
                    epoch_scores,
                    color=style["color"],
                    marker="o",
                    linewidth=6,
                    markersize=18,
                )

                for epoch, score in epoch_items:
                    if unlearning_type == "secret" and score >= 0.2:
                        label_offset = (0, -18)
                        label_va = "top"
                    else:
                        label_offset = (0, 14)
                        label_va = "bottom"
                    ax.annotate(
                        f"{score:.2f}",
                        (epoch, score),
                        textcoords="offset points",
                        xytext=label_offset,
                        ha="center",
                        va=label_va,
                        fontsize=32,
                        weight="bold",
                    )

        panel_label = chr(ord("a") + col)
        ax.set_title(f"({panel_label}) {model}", pad=26)
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("Average similarity score")
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(["Base", "1", "2", "3", "4", "5"])
        ax.set_xlim(-0.25, 5.25)
        ax.set_ylim(-0.05, 1.16)
        ax.grid(True, axis="y", alpha=0.35, linewidth=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 0.98), w_pad=2.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Directory containing model/unlearning/checkpoint result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    configure_matplotlib(root)
    scores = load_scores(root)
    plot_scores(scores, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
