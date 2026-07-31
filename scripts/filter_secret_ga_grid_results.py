#!/usr/bin/env python3
"""Post-filter secret/UtilityEval results and rank a completed GA grid search.

Raw evaluation files are never changed. For each grid point this script writes:

* secret_forget_pass1/aggregate_results_filtered.json
* utilityeval_pass1/utilityeval.filtered.eval_results.json

It also writes grid_summary_filtered.csv, grid_search_filtered.json, and
grid_search_filtered.md at the results root.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECRET_FILTER = REPO_ROOT / "UnlearningEvaluation/non_exact_matches.csv"
DEFAULT_UTILITY_FILTER = (
    REPO_ROOT / "evalplus/evalplus/baseline_failed_test_ids.csv"
)
UTILITY_FILTER_TOOL = REPO_ROOT / "evalplus/tools/filter_baseline_failed_results.py"
EPOCH_RE = re.compile(r"^epoch-(?P<epoch>\d+)_checkpoint-(?P<step>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root containing model/lr-*/evaluations/epoch-*_checkpoint-* results",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional separate output root; defaults to --results-root",
    )
    parser.add_argument(
        "--secret-filter-csv",
        type=Path,
        default=DEFAULT_SECRET_FILTER,
        help="UUID exclusion CSV used by UnlearningEvaluation",
    )
    parser.add_argument(
        "--utility-filter-csv",
        type=Path,
        default=DEFAULT_UTILITY_FILTER,
        help="Baseline-failed task exclusion CSV used by EvalPlus",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            rows.append(row)
    return rows


def load_secret_exclusions(
    path: Path,
) -> dict[tuple[str, str, str], frozenset[str]]:
    grouped: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"model_dir", "split", "eval_mode", "uuid"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            uuid = row["uuid"].strip()
            if uuid:
                key = (
                    row["model_dir"].strip(),
                    row["split"].strip(),
                    row["eval_mode"].strip(),
                )
                grouped[key].add(uuid)
    return {key: frozenset(values) for key, values in grouped.items()}


def average(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def aggregate_secret_rows(
    original: dict[str, Any],
    rows: list[dict[str, Any]],
    excluded_uuids: frozenset[str],
    filter_csv: Path,
    model_dir: str,
) -> dict[str, Any]:
    pass_k = original.get("pass_k")
    if pass_k != 1:
        raise ValueError(
            "Post-hoc secret filtering currently requires pass@1 row_results; "
            f"found pass_k={pass_k!r}"
        )

    original_count = original.get("num_evaluated_examples")
    if original_count != len(rows):
        raise ValueError(
            f"row_results has {len(rows)} rows but aggregate reports "
            f"{original_count} examples"
        )
    uuids = [str(row.get("uuid", "")) for row in rows]
    if len(set(uuids)) != len(uuids):
        raise ValueError("Duplicate UUIDs found in pass@1 row_results")

    retained = [row for row in rows if row.get("uuid") not in excluded_uuids]
    if not retained:
        raise ValueError("The UUID filter removed every secret-evaluation example")

    group_columns = tuple(original.get("grouped_averages", {}).keys())
    grouped_scores: dict[str, dict[str, list[float]]] = {
        column: defaultdict(list) for column in group_columns
    }
    scores: list[float] = []
    for row in retained:
        score = row.get("score_value")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"Invalid score_value for UUID {row.get('uuid')}: {score!r}")
        numeric_score = float(score)
        scores.append(numeric_score)
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        for column in group_columns:
            value = metadata.get(column)
            key = "<missing>" if value is None else str(value)
            grouped_scores[column][key].append(numeric_score)

    grouped_averages = {
        column: {
            value: average(values)
            for value, values in sorted(value_scores.items())
            if values
        }
        for column, value_scores in grouped_scores.items()
    }
    filtered_average = average(scores)
    filtered = deepcopy(original)
    filtered["num_evaluated_examples"] = len(retained)
    filtered["num_generated_results"] = len(retained)
    filtered["average_similarity_score"] = filtered_average
    filtered["pass_at_k"] = {
        "pass@1": {
            "average_similarity_score": filtered_average,
            "grouped_averages": grouped_averages,
        }
    }
    filtered["score_failures"] = sum("score_error" in row for row in retained)
    filtered["grouped_averages"] = grouped_averages
    filtered["uuid_filter"] = {
        "source_csv": str(filter_csv.resolve()),
        "operation": "exclude",
        "model_dir": model_dir,
        "split": "forget",
        "eval_mode": "secret",
        "num_excluded_uuids_in_csv": len(excluded_uuids),
        "num_excluded_examples": len(rows) - len(retained),
        "num_included_examples": len(retained),
    }
    return filtered


def import_utility_filter_tool(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("baseline_result_filter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import EvalPlus filter tool from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_grid_point(
    results_root: Path, aggregate_path: Path
) -> tuple[str, str, int, int, Path]:
    evaluation_dir = aggregate_path.parent.parent
    match = EPOCH_RE.fullmatch(evaluation_dir.name)
    if match is None:
        raise ValueError(f"Unexpected evaluation directory name: {evaluation_dir}")
    relative_parts = evaluation_dir.relative_to(results_root).parts
    lr_indexes = [i for i, part in enumerate(relative_parts) if part.startswith("lr-")]
    if len(lr_indexes) != 1 or lr_indexes[0] == 0:
        raise ValueError(f"Could not infer model and learning rate from {evaluation_dir}")
    lr_index = lr_indexes[0]
    model_dir = relative_parts[lr_index - 1]
    learning_rate = relative_parts[lr_index][len("lr-") :]
    return (
        model_dir,
        learning_rate,
        int(match.group("epoch")),
        int(match.group("step")),
        evaluation_dir,
    )


def extract_utility_pass1(result: dict[str, Any]) -> float:
    try:
        value = result["pass_at_k"]["base"]["pass@1"]
    except (KeyError, TypeError) as exc:
        raise ValueError("UtilityEval result has no pass_at_k.base.pass@1") from exc
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Invalid UtilityEval pass@1 value: {value!r}")
    return float(value)


def is_pareto_efficient(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    secret = candidate["filtered_secret_similarity_pass1"]
    utility = candidate["filtered_utilityeval_pass1"]
    return not any(
        other is not candidate
        and other["filtered_secret_similarity_pass1"] <= secret
        and other["filtered_utilityeval_pass1"] >= utility
        and (
            other["filtered_secret_similarity_pass1"] < secret
            or other["filtered_utilityeval_pass1"] > utility
        )
        for other in rows
    )


def read_source_models(results_root: Path) -> dict[tuple[str, str, int], str]:
    summary = results_root / "grid_summary.csv"
    if not summary.is_file():
        return {}
    values: dict[tuple[str, str, int], str] = {}
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (row["model"], row["learning_rate"], int(row["epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
            values[key] = row.get("source_full_model", "")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "balanced_rank",
        "pareto_efficient",
        "model",
        "source_full_model",
        "learning_rate",
        "epoch",
        "checkpoint",
        "global_step",
        "raw_secret_similarity_pass1",
        "filtered_secret_similarity_pass1",
        "secret_excluded_examples",
        "raw_utilityeval_pass1",
        "filtered_utilityeval_pass1",
        "utility_excluded_tasks",
        "balanced_error",
        "evaluation_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fieldnames} for row in rows)


def display_lr(value: str) -> str:
    try:
        return f"{float(value):g}"
    except ValueError:
        return value


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Filtered secret GA grid search",
        "",
        "Lower secret similarity and higher UtilityEval pass@1 are better. "
        "Balanced error is `(secret similarity + 1 - utility pass@1) / 2`; lower is better.",
        "",
        "| Rank | LR | Epoch | Step | Secret pass@1 | Utility pass@1 | Balanced error | Pareto |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        lines.append(
            "| {balanced_rank} | {lr} | {epoch} | {global_step} | {secret:.6f} | "
            "{utility:.6f} | {balanced:.6f} | {pareto} |".format(
                balanced_rank=row["balanced_rank"],
                lr=display_lr(row["learning_rate"]),
                epoch=row["epoch"],
                global_step=row["global_step"],
                secret=row["filtered_secret_similarity_pass1"],
                utility=row["filtered_utilityeval_pass1"],
                balanced=row["balanced_error"],
                pareto="yes" if row["pareto_efficient"] else "",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_grid_point(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "balanced_rank",
        "pareto_efficient",
        "model",
        "learning_rate",
        "epoch",
        "checkpoint",
        "global_step",
        "filtered_secret_similarity_pass1",
        "filtered_utilityeval_pass1",
        "balanced_error",
        "evaluation_dir",
    )
    return {key: row[key] for key in keys}


def main() -> int:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else results_root
    )
    secret_filter_csv = args.secret_filter_csv.expanduser().resolve()
    utility_filter_csv = args.utility_filter_csv.expanduser().resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    for required_path in (secret_filter_csv, utility_filter_csv, UTILITY_FILTER_TOOL):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    secret_exclusions = load_secret_exclusions(secret_filter_csv)
    utility_tool = import_utility_filter_tool(UTILITY_FILTER_TOOL)
    utility_exclusions = utility_tool.load_excluded_tasks(utility_filter_csv)
    source_models = read_source_models(results_root)
    aggregate_paths = sorted(
        results_root.glob(
            "*/lr-*/evaluations/epoch-*_checkpoint-*/"
            "secret_forget_pass1/aggregate_results.json"
        )
    )
    if not aggregate_paths:
        raise FileNotFoundError(f"No secret grid results found under {results_root}")

    rows: list[dict[str, Any]] = []
    for aggregate_path in aggregate_paths:
        model_dir, learning_rate, epoch, global_step, evaluation_dir = infer_grid_point(
            results_root, aggregate_path
        )
        key = (model_dir, "forget", "secret")
        if key not in secret_exclusions:
            raise ValueError(
                "No matching secret UUID exclusions for "
                f"model_dir={model_dir}, split=forget, eval_mode=secret"
            )
        excluded_uuids = secret_exclusions[key]
        original_secret = load_json(aggregate_path)
        row_results_path = aggregate_path.with_name("row_results.jsonl")
        filtered_secret = aggregate_secret_rows(
            original_secret,
            load_jsonl(row_results_path),
            excluded_uuids,
            secret_filter_csv,
            model_dir,
        )

        utility_path = evaluation_dir / "utilityeval_pass1/utilityeval.eval_results.json"
        if not utility_path.is_file():
            raise FileNotFoundError(f"Missing matching UtilityEval result: {utility_path}")
        original_utility = load_json(utility_path)
        filtered_utility = utility_tool.filter_result(
            original_utility, utility_exclusions, "utilityeval"
        )
        filtered_utility["baseline_filter_source"] = str(utility_filter_csv)

        output_evaluation_dir = output_root / evaluation_dir.relative_to(results_root)
        filtered_secret_path = (
            output_evaluation_dir
            / "secret_forget_pass1/aggregate_results_filtered.json"
        )
        filtered_utility_path = (
            output_evaluation_dir
            / "utilityeval_pass1/utilityeval.filtered.eval_results.json"
        )
        write_json(filtered_secret_path, filtered_secret)
        write_json(filtered_utility_path, filtered_utility)

        raw_secret_score = float(original_secret["average_similarity_score"])
        filtered_secret_score = float(filtered_secret["average_similarity_score"])
        raw_utility_score = extract_utility_pass1(original_utility)
        filtered_utility_score = extract_utility_pass1(filtered_utility)
        utility_filter_summary = filtered_utility["baseline_filter"]
        row = {
            "model": model_dir,
            "source_full_model": source_models.get(
                (model_dir, learning_rate, epoch), ""
            ),
            "learning_rate": learning_rate,
            "epoch": epoch,
            "checkpoint": f"checkpoint-{global_step}",
            "global_step": global_step,
            "raw_secret_similarity_pass1": raw_secret_score,
            "filtered_secret_similarity_pass1": filtered_secret_score,
            "secret_excluded_examples": filtered_secret["uuid_filter"][
                "num_excluded_examples"
            ],
            "raw_utilityeval_pass1": raw_utility_score,
            "filtered_utilityeval_pass1": filtered_utility_score,
            "utility_excluded_tasks": utility_filter_summary["excluded_task_count"],
            "balanced_error": (
                filtered_secret_score + 1.0 - filtered_utility_score
            )
            / 2.0,
            "evaluation_dir": str(evaluation_dir),
        }
        if not all(
            math.isfinite(row[name])
            for name in (
                "raw_secret_similarity_pass1",
                "filtered_secret_similarity_pass1",
                "raw_utilityeval_pass1",
                "filtered_utilityeval_pass1",
                "balanced_error",
            )
        ):
            raise ValueError(f"Non-finite metric found at {evaluation_dir}")
        rows.append(row)

    for row in rows:
        row["pareto_efficient"] = is_pareto_efficient(row, rows)
    rows.sort(
        key=lambda row: (
            row["balanced_error"],
            -row["filtered_utilityeval_pass1"],
            row["filtered_secret_similarity_pass1"],
            row["global_step"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["balanced_rank"] = rank

    csv_path = output_root / "grid_summary_filtered.csv"
    json_path = output_root / "grid_search_filtered.json"
    markdown_path = output_root / "grid_search_filtered.md"
    write_csv(csv_path, rows)
    write_markdown(markdown_path, rows)
    best_unlearning = min(
        rows,
        key=lambda row: (
            row["filtered_secret_similarity_pass1"],
            -row["filtered_utilityeval_pass1"],
        ),
    )
    best_utility = max(
        rows,
        key=lambda row: (
            row["filtered_utilityeval_pass1"],
            -row["filtered_secret_similarity_pass1"],
        ),
    )
    report = {
        "results_root": str(results_root),
        "output_root": str(output_root),
        "grid_point_count": len(rows),
        "objective": {
            "secret_similarity": "minimize",
            "utilityeval_pass_at_1": "maximize",
            "balanced_error": (
                "(filtered_secret_similarity_pass1 + 1 - "
                "filtered_utilityeval_pass1) / 2; minimize"
            ),
        },
        "filters": {
            "secret_uuid_filter_csv": str(secret_filter_csv),
            "utility_baseline_filter_csv": str(utility_filter_csv),
        },
        "best_balanced": compact_grid_point(rows[0]),
        "best_unlearning": compact_grid_point(best_unlearning),
        "best_utility": compact_grid_point(best_utility),
        "pareto_front": [
            compact_grid_point(row) for row in rows if row["pareto_efficient"]
        ],
        "ranked_grid": [compact_grid_point(row) for row in rows],
    }
    write_json(json_path, report)

    print(f"Processed {len(rows)} grid points")
    print(f"Filtered CSV: {csv_path}")
    print(f"Grid report: {json_path}")
    print(f"Readable table: {markdown_path}")
    print(
        "Best balanced: "
        f"lr={rows[0]['learning_rate']}, epoch={rows[0]['epoch']}, "
        f"secret={rows[0]['filtered_secret_similarity_pass1']:.6f}, "
        f"utility={rows[0]['filtered_utilityeval_pass1']:.6f}, "
        f"balanced_error={rows[0]['balanced_error']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
