#!/usr/bin/env python3
"""Find average-direction pairwise CodeBLEU + token similarity in dbaysal/all-content.

For each unordered pair (i, j), this script computes:

    forward = CodeBLEU(i as prediction, j as reference)
    reverse = CodeBLEU(j as prediction, i as reference)
    average = (forward + reverse) / 2

It also computes tokenizer-based Jaccard token similarity:

    token_similarity = |tokens_i ∩ tokens_j| / |tokens_i ∪ tokens_j|

It reports the pair with the highest average CodeBLEU and stores every pair
whose average CodeBLEU is at least the threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from codebleu import bleu, calc_codebleu, weighted_ngram_match
from codebleu.dataflow_match import (
    Parser,
    dfg_function,
    get_data_flow,
    get_tree_sitter_language,
    normalize_dataflow,
    remove_comments_and_docstrings,
)
from datasets import load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "token_codebleu_reports"
DEFAULT_DATASET = "dbaysal/all-content"
DEFAULT_SPLIT = "train"
DEFAULT_TOKENIZER = "openai/gpt-oss-120b"
DEFAULT_REPORT = OUT / "codebleu_all_content_report.json"
DEFAULT_PAIRS_REPORT = OUT / "codebleu_pairs_above_threshold.jsonl"
DEFAULT_PLOT = OUT / "codebleu_diversity.pdf"
PLOT_FONT_SIZES = {
    "title": 30,
    "label": 28,
    "tick": 24,
    "legend": 24,
}
DEFAULT_THRESHOLD = 0.6

CODEBLEU_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


@dataclass(frozen=True)
class CodeFeatures:
    tokens: list[str]
    weighted_ref: list[Any]
    sexps: list[str]
    sexp_set: set[str]
    dataflows: list[Any]
    tokenizer_token_ids: list[int]
    tokenizer_token_set: set[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute exact average-direction pairwise CodeBLEU for dbaysal/all-content."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset name. Default: dbaysal/all-content",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to load. Default: train",
    )
    parser.add_argument(
        "--content-column",
        default="content",
        help="Column containing code/content. Default: content",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Hugging Face tokenizer name. Default: {DEFAULT_TOKENIZER}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path for the JSON summary report.",
    )
    parser.add_argument(
        "--pairs-report",
        type=Path,
        default=DEFAULT_PAIRS_REPORT,
        help="Path for JSONL report of all pairs above threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Report pairs with average directional CodeBLEU >= this threshold.",
    )
    parser.add_argument(
        "--token-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Optional token similarity threshold. If set, pairs are written when "
            "average CodeBLEU >= --threshold OR token_similarity >= --token-threshold."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every N unordered pairs.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=DEFAULT_PLOT,
        help="Path for the diversity distribution plot (PDF).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable the diversity distribution plot.",
    )
    parser.add_argument(
        "--plot-bins",
        type=int,
        default=50,
        help="Number of histogram bins over [0, 1] for the diversity plot.",
    )
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=20,
        help="Number of directed scores to validate against codebleu.calc_codebleu.",
    )
    return parser.parse_args()


def load_hf_rows(dataset_name: str, config: str | None, split: str) -> list[dict[str, Any]]:
    if config:
        dataset = load_dataset(dataset_name, config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    return [dict(row) for row in dataset]


def get_row_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    return default if value is None else value


def codebleu_keywords(lang: str) -> set[str]:
    import codebleu as codebleu_pkg

    keywords_path = (
        Path(codebleu_pkg.__file__).resolve().parent
        / "keywords"
        / f"{lang}.txt"
    )
    return set(keywords_path.read_text(encoding="utf-8").splitlines())


def all_subtrees(root_node: Any) -> list[str]:
    node_stack = [(root_node, 1)]
    subtrees: list[str] = []

    while node_stack:
        node, depth = node_stack.pop()
        subtrees.append(str(node))

        for child in node.children:
            if child.children:
                node_stack.append((child, depth + 1))

    return subtrees


def stripped_code(code: str, lang: str) -> str:
    try:
        return remove_comments_and_docstrings(code, lang)
    except Exception:
        return code


def tokenizer_token_ids(code: str, tokenizer: Any) -> list[int]:
    if not isinstance(code, str) or not code:
        return []

    return tokenizer.encode(
        code,
        add_special_tokens=False,
    )


def token_jaccard_similarity(left: CodeFeatures, right: CodeFeatures) -> float:
    left_set = left.tokenizer_token_set
    right_set = right.tokenizer_token_set

    if not left_set and not right_set:
        return 1.0

    if not left_set or not right_set:
        return 0.0

    return len(left_set & right_set) / len(left_set | right_set)


def token_overlap_counts(left: CodeFeatures, right: CodeFeatures) -> dict[str, int]:
    left_set = left.tokenizer_token_set
    right_set = right.tokenizer_token_set

    return {
        "left_unique_tokens": len(left_set),
        "right_unique_tokens": len(right_set),
        "shared_unique_tokens": len(left_set & right_set),
        "union_unique_tokens": len(left_set | right_set),
        "left_total_tokens": len(left.tokenizer_token_ids),
        "right_total_tokens": len(right.tokenizer_token_ids),
    }


def build_features(
    contents: list[str],
    lang: str,
    tokenizer: Any,
) -> list[CodeFeatures]:
    keywords = codebleu_keywords(lang)

    tree_sitter_language = get_tree_sitter_language(lang)
    parser = Parser()
    parser.language = tree_sitter_language
    dataflow_parser = [parser, dfg_function[lang]]

    features: list[CodeFeatures] = []

    for code in contents:
        if not isinstance(code, str):
            code = ""

        stripped = stripped_code(code, lang)

        tree = parser.parse(bytes(stripped, "utf8")).root_node
        sexps = all_subtrees(tree)

        try:
            dataflows = normalize_dataflow(get_data_flow(stripped, dataflow_parser))
        except Exception:
            dataflows = []

        tokens = code.strip().split()
        weights = {token: 1 if token in keywords else 0.2 for token in tokens}

        token_ids = tokenizer_token_ids(code, tokenizer)

        features.append(
            CodeFeatures(
                tokens=tokens,
                weighted_ref=[tokens, weights],
                sexps=sexps,
                sexp_set=set(sexps),
                dataflows=dataflows,
                tokenizer_token_ids=token_ids,
                tokenizer_token_set=set(token_ids),
            )
        )

    return features


def syntax_score(reference: CodeFeatures, prediction: CodeFeatures) -> float:
    if not reference.sexps:
        return 0.0

    return sum(
        1 for subtree in reference.sexps if subtree in prediction.sexp_set
    ) / len(reference.sexps)


def dataflow_score(reference: CodeFeatures, prediction: CodeFeatures) -> float:
    if not reference.dataflows:
        return 0.0

    prediction_dataflows = list(prediction.dataflows)
    match_count = 0

    for dataflow in reference.dataflows:
        if dataflow in prediction_dataflows:
            match_count += 1
            prediction_dataflows.remove(dataflow)

    return match_count / len(reference.dataflows)


def directed_score(reference: CodeFeatures, prediction: CodeFeatures) -> dict[str, float]:
    ngram = bleu.sentence_bleu([reference.tokens], prediction.tokens)
    weighted_ngram = weighted_ngram_match.sentence_bleu(
        [reference.weighted_ref],
        prediction.tokens,
    )

    syntax = syntax_score(reference, prediction)
    dataflow = dataflow_score(reference, prediction)

    alpha, beta, gamma, theta = CODEBLEU_WEIGHTS

    codebleu = (
        alpha * ngram
        + beta * weighted_ngram
        + gamma * syntax
        + theta * (dataflow or 1)
    )

    return {
        "codebleu": codebleu,
        "ngram_match_score": ngram,
        "weighted_ngram_match_score": weighted_ngram,
        "syntax_match_score": syntax,
        "dataflow_match_score": dataflow,
    }


def average_components(
    forward: dict[str, float],
    reverse: dict[str, float],
) -> dict[str, float]:
    return {
        key: (forward[key] + reverse[key]) / 2
        for key in forward
    }


def assert_close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(
            f"{label}: optimized={actual} calc_codebleu={expected}"
        )


def validate_scores(
    rows: list[dict[str, Any]],
    features: list[CodeFeatures],
    *,
    sample_count: int,
    lang: str,
    content_column: str,
) -> None:
    checked = 0
    row_count = len(rows)

    for i in range(row_count):
        for j in range(i + 1, row_count):
            for ref_i, pred_i in ((i, j), (j, i)):
                optimized = directed_score(features[ref_i], features[pred_i])

                package = calc_codebleu(
                    [rows[ref_i][content_column]],
                    [rows[pred_i][content_column]],
                    lang=lang,
                )

                for key in optimized:
                    assert_close(
                        optimized[key],
                        package[key],
                        label=f"{key} {ref_i}->{pred_i}",
                    )

                checked += 1

                if checked >= sample_count:
                    print(
                        f"Validated {checked} directed scores "
                        "against codebleu.calc_codebleu"
                    )
                    return


def row_summary(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "uuid": get_row_value(row, "uuid"),
        "id": get_row_value(row, "id"),
        "task_id": get_row_value(row, "task_id"),
        "entry_point": get_row_value(row, "entry_point"),
        "source_dataset": get_row_value(row, "source_dataset"),
        "source_file": get_row_value(row, "source_file"),
        "type": get_row_value(row, "type"),
    }


def should_write_pair(
    avg_codebleu: float,
    token_similarity: float,
    codebleu_threshold: float,
    token_threshold: float | None,
) -> bool:
    if avg_codebleu >= codebleu_threshold:
        return True

    if token_threshold is not None and token_similarity >= token_threshold:
        return True

    return False


def bin_index(value: float, n_bins: int, lo: float = 0.0, hi: float = 1.0) -> int:
    """Map a value in [lo, hi] to a histogram bin index in [0, n_bins - 1]."""
    if value <= lo:
        return 0
    if value >= hi:
        return n_bins - 1
    return int((value - lo) / (hi - lo) * n_bins)


def write_histogram_csvs(
    codebleu_hist: "np.ndarray",
    token_hist: "np.ndarray",
    joint_hist: "np.ndarray",
    n_bins: int,
    plot_path: Path,
) -> tuple[Path, Path]:
    """Dump raw bin counts so figures can be regenerated without re-running the
    O(N^2) scan. Writes a 1D CSV (CodeBLEU + token marginals) next to a long-form
    joint CSV (codebleu_bin x token_bin counts). Returns both paths."""
    import csv

    width = 1.0 / n_bins
    base = plot_path.with_suffix("")

    hist_path = base.parent / f"{base.name}_histogram.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bin_left", "bin_right", "bin_center",
                    "codebleu_count", "token_count"])
        for k in range(n_bins):
            left = k * width
            right = (k + 1) * width
            w.writerow([f"{left:.6f}", f"{right:.6f}", f"{(left + right) / 2:.6f}",
                        int(codebleu_hist[k]), int(token_hist[k])])
    print(f"Wrote {hist_path}")

    joint_path = base.parent / f"{base.name}_joint.csv"
    with joint_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["codebleu_bin_center", "token_bin_center", "count"])
        for ci in range(n_bins):
            cb_center = (ci + 0.5) * width
            for ti in range(n_bins):
                count = int(joint_hist[ci, ti])
                if count:
                    w.writerow([f"{cb_center:.6f}", f"{(ti + 0.5) * width:.6f}",
                                count])
    print(f"Wrote {joint_path}")

    return hist_path, joint_path


def plot_codebleu_diversity(
    codebleu_hist: "np.ndarray",
    token_hist: "np.ndarray",
    joint_hist: "np.ndarray",
    n_bins: int,
    threshold: float,
    token_threshold: float | None,
    total_pairs: int,
    out_path: Path,
) -> Path | None:
    """Plot pairwise CodeBLEU / token-similarity distributions as diversity evidence.

    Two panels: the CodeBLEU distribution and the token-Jaccard distribution.
    A diverse pool concentrates mass at low similarity, with only a thin tail
    crossing the duplicate thresholds. Bin counts are accumulated incrementally
    during the pair scan, so no per-pair scores are stored. Returns the PDF path,
    or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[plot] skipped ({exc}); install matplotlib to enable plots")
        return None

    centers = (np.arange(n_bins) + 0.5) / n_bins
    width = 1.0 / n_bins

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    axes[0].bar(centers, codebleu_hist, width=width, color="steelblue")
    axes[0].axvline(
        threshold,
        color="red",
        ls="--",
        linewidth=2.2,
        label=f"threshold = {threshold}",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(
        "average directional CodeBLEU",
        fontsize=PLOT_FONT_SIZES["label"],
    )
    axes[0].set_ylabel(
        "number of pairs (log scale)",
        fontsize=PLOT_FONT_SIZES["label"],
    )
    axes[0].set_title(
        "Pairwise CodeBLEU distribution",
        fontsize=PLOT_FONT_SIZES["title"],
    )
    axes[0].tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
    axes[0].legend(fontsize=PLOT_FONT_SIZES["legend"])

    axes[1].bar(centers, token_hist, width=width, color="seagreen")
    if token_threshold is not None:
        axes[1].axvline(
            token_threshold,
            color="red",
            ls="--",
            linewidth=2.2,
            label=f"threshold = {token_threshold}",
        )
        axes[1].legend(fontsize=PLOT_FONT_SIZES["legend"])

    axes[1].set_yscale("log")
    axes[1].set_xlabel("token Jaccard similarity", fontsize=PLOT_FONT_SIZES["label"])
    axes[1].set_ylabel(
        "number of pairs (log scale)",
        fontsize=PLOT_FONT_SIZES["label"],
    )
    axes[1].set_title(
        "Pairwise token similarity distribution",
        fontsize=PLOT_FONT_SIZES["title"],
    )
    axes[1].tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    args = parse_args()

    lang = "python"
    start = time.perf_counter()

    rows = load_hf_rows(args.dataset, args.config, args.split)

    if not rows:
        raise ValueError(f"No rows found in {args.dataset}, split={args.split}")

    if args.content_column not in rows[0]:
        available = ", ".join(rows[0].keys())
        raise KeyError(
            f"Column '{args.content_column}' not found. Available columns: {available}"
        )

    contents = [
        row[args.content_column] if isinstance(row.get(args.content_column), str) else ""
        for row in rows
    ]

    row_count = len(rows)
    pair_count = row_count * (row_count - 1) // 2

    print(f"Loaded {row_count} rows from Hugging Face dataset {args.dataset}, split={args.split}")
    print(f"Using content column: {args.content_column}")
    print(f"Loading tokenizer: {args.tokenizer}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Building reusable CodeBLEU and tokenizer features for {lang}")

    features = build_features(contents, lang, tokenizer)

    if args.validate_samples:
        validate_scores(
            rows,
            features,
            sample_count=args.validate_samples,
            lang=lang,
            content_column=args.content_column,
        )

    best_codebleu_pair: dict[str, Any] | None = None
    best_token_similarity_pair: dict[str, Any] | None = None
    processed = 0
    pairs_above_threshold = 0
    pairs_above_token_threshold = 0

    # Diversity histograms: fixed bins over [0, 1] accumulated per pair so we
    # never store O(N^2) scores. joint_hist is indexed [codebleu_bin, token_bin].
    n_bins = args.plot_bins
    codebleu_hist = np.zeros(n_bins, dtype=np.int64)
    token_hist = np.zeros(n_bins, dtype=np.int64)
    joint_hist = np.zeros((n_bins, n_bins), dtype=np.int64)

    print(f"Scanning {pair_count} unordered pairs ({pair_count * 2} directed scores)")
    print(f"CodeBLEU threshold: average directional CodeBLEU >= {args.threshold}")

    if args.token_threshold is not None:
        print(f"Token threshold: token_similarity >= {args.token_threshold}")

    print(f"Writing matching pairs to {args.pairs_report}")

    args.pairs_report.parent.mkdir(parents=True, exist_ok=True)

    with args.pairs_report.open("w", encoding="utf-8") as pairs_file:
        for i in range(row_count):
            left = features[i]

            for j in range(i + 1, row_count):
                right = features[j]

                forward = directed_score(left, right)
                reverse = directed_score(right, left)

                avg_score = (forward["codebleu"] + reverse["codebleu"]) / 2
                max_score = max(forward["codebleu"], reverse["codebleu"])
                min_score = min(forward["codebleu"], reverse["codebleu"])

                token_similarity = token_jaccard_similarity(left, right)
                token_counts = token_overlap_counts(left, right)

                cb_bin = bin_index(avg_score, n_bins)
                tok_bin = bin_index(token_similarity, n_bins)
                codebleu_hist[cb_bin] += 1
                token_hist[tok_bin] += 1
                joint_hist[cb_bin, tok_bin] += 1

                pair_report = {
                    "average_codebleu": avg_score,
                    "max_direction_codebleu": max_score,
                    "min_direction_codebleu": min_score,
                    "forward_codebleu": forward["codebleu"],
                    "reverse_codebleu": reverse["codebleu"],
                    "token_similarity": token_similarity,
                    "token_similarity_definition": (
                        "Jaccard similarity over unique tokenizer token ids"
                    ),
                    "token_overlap_counts": token_counts,
                    "average_components": average_components(forward, reverse),
                    "forward_components": forward,
                    "reverse_components": reverse,
                    "row_indices": [i, j],
                    "rows": [
                        row_summary(rows[i], i),
                        row_summary(rows[j], j),
                    ],
                }

                if (
                    best_codebleu_pair is None
                    or avg_score > best_codebleu_pair["average_codebleu"]
                ):
                    best_codebleu_pair = pair_report

                if (
                    best_token_similarity_pair is None
                    or token_similarity > best_token_similarity_pair["token_similarity"]
                ):
                    best_token_similarity_pair = pair_report

                if avg_score >= args.threshold:
                    pairs_above_threshold += 1

                if (
                    args.token_threshold is not None
                    and token_similarity >= args.token_threshold
                ):
                    pairs_above_token_threshold += 1

                if should_write_pair(
                    avg_codebleu=avg_score,
                    token_similarity=token_similarity,
                    codebleu_threshold=args.threshold,
                    token_threshold=args.token_threshold,
                ):
                    pairs_file.write(
                        json.dumps(pair_report, ensure_ascii=False) + "\n"
                    )

                processed += 1

                if args.progress_every and processed % args.progress_every == 0:
                    elapsed = time.perf_counter() - start

                    current_best_codebleu = (
                        best_codebleu_pair["average_codebleu"]
                        if best_codebleu_pair is not None
                        else 0.0
                    )

                    current_best_token = (
                        best_token_similarity_pair["token_similarity"]
                        if best_token_similarity_pair is not None
                        else 0.0
                    )

                    print(
                        f"  processed {processed}/{pair_count} pairs; "
                        f"current max average CodeBLEU={current_best_codebleu:.12f}; "
                        f"current max token similarity={current_best_token:.12f}; "
                        f"pairs above CodeBLEU threshold={pairs_above_threshold}; "
                        f"pairs above token threshold={pairs_above_token_threshold}; "
                        f"elapsed={elapsed:.1f}s"
                    )

    elapsed = time.perf_counter() - start

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    hist_csv_path, joint_csv_path = write_histogram_csvs(
        codebleu_hist, token_hist, joint_hist, n_bins, args.plot
    )

    plot_path = None
    if not args.no_plot:
        plot_path = plot_codebleu_diversity(
            codebleu_hist,
            token_hist,
            joint_hist,
            n_bins=n_bins,
            threshold=args.threshold,
            token_threshold=args.token_threshold,
            total_pairs=pair_count,
            out_path=args.plot,
        )

    report = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "content_column": args.content_column,
        "tokenizer": args.tokenizer,
        "token_similarity_definition": (
            "Jaccard similarity over unique tokenizer token ids"
        ),
        "language": lang,
        "row_count": row_count,
        "source_dataset_counts": dict(
            Counter(get_row_value(row, "source_dataset", "missing") for row in rows)
        ),
        "type_counts": dict(
            Counter(get_row_value(row, "type", "missing") for row in rows)
        ),
        "unordered_pair_count": pair_count,
        "directed_score_count": pair_count * 2,
        "similarity_definition": (
            "average(forward_codebleu, reverse_codebleu) per unordered pair"
        ),
        "threshold": args.threshold,
        "threshold_definition": (
            "average(forward_codebleu, reverse_codebleu) >= threshold"
        ),
        "token_threshold": args.token_threshold,
        "token_threshold_definition": (
            "token_similarity >= token_threshold"
            if args.token_threshold is not None
            else None
        ),
        "pairs_above_threshold": pairs_above_threshold,
        "pairs_above_token_threshold": pairs_above_token_threshold,
        "pairs_report": str(args.pairs_report),
        "plot": str(plot_path) if plot_path else None,
        "histogram_csv": str(hist_csv_path),
        "joint_histogram_csv": str(joint_csv_path),
        "plot_bins": n_bins,
        "weights": CODEBLEU_WEIGHTS,
        "max_average_pairwise_codebleu": best_codebleu_pair,
        "max_pairwise_token_similarity": best_token_similarity_pair,
        "elapsed_seconds": elapsed,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {args.report}")
    print(f"Wrote {args.pairs_report}")
    print(f"Pairs above CodeBLEU threshold: {pairs_above_threshold}")
    print(f"Pairs above token threshold: {pairs_above_token_threshold}")

    print("\nMax average CodeBLEU pair:")
    print(json.dumps(report["max_average_pairwise_codebleu"], indent=2))

    print("\nMax token similarity pair:")
    print(json.dumps(report["max_pairwise_token_similarity"], indent=2))


if __name__ == "__main__":
    main()
