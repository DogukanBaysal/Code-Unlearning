#!/usr/bin/env python3
"""Exclude baseline-failed tasks and recompute EvalPlus pass@k metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_FILTER_CSV = (
    Path(__file__).resolve().parents[1] / "evalplus" / "baseline_failed_test_ids.csv"
)


def normalize_dataset(value: str) -> str:
    normalized = value.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "forgeteval": "forgeteval",
        "utilityeval": "utilityeval",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported component dataset: {value!r}")
    return aliases[normalized]


def load_excluded_tasks(path: Path) -> dict[str, set[str]]:
    excluded = {"forgeteval": set(), "utilityeval": set()}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "test_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Filter CSV is missing required column(s): {', '.join(sorted(missing))}"
            )
        for row_number, row in enumerate(reader, start=2):
            dataset = normalize_dataset(row["dataset"])
            task_id = row["test_id"].strip()
            if not task_id:
                raise ValueError(f"Empty test_id in {path} at row {row_number}")
            excluded[dataset].add(task_id)
    return excluded


def estimate_pass_at_k(total: int, correct: int, k: int) -> float:
    if total < k:
        raise ValueError(f"Cannot calculate pass@{k} from only {total} samples")
    if total - correct < k:
        return 1.0
    return 1.0 - math.prod(
        1.0 - k / value for value in range(total - correct + 1, total + 1)
    )


def calculate_pass_at_k(task_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if not task_results:
        raise ValueError("Filtering removed every task; pass@k is undefined")

    sample_counts = [len(results) for results in task_results.values()]
    if any(count <= 0 for count in sample_counts):
        raise ValueError("Every retained task must contain at least one generation")

    min_samples = min(sample_counts)
    report_k_values = sorted({1, (min_samples + 1) // 2, min_samples})
    base_metrics: dict[str, float] = {}
    for k in report_k_values:
        estimates = []
        for results in task_results.values():
            correct = sum(result.get("base_status") == "pass" for result in results)
            estimates.append(estimate_pass_at_k(len(results), correct, k))
        base_metrics[f"pass@{k}"] = sum(estimates) / len(estimates)

    return {"base": base_metrics}


def filter_component(
    component: dict[str, Any],
    dataset: str,
    excluded_task_ids: set[str],
) -> dict[str, Any]:
    task_results = component.get("eval")
    if not isinstance(task_results, dict):
        raise ValueError(f"The {dataset} component has no task-level 'eval' mapping")

    present_exclusions = sorted(excluded_task_ids.intersection(task_results))
    retained_results = {
        task_id: results
        for task_id, results in task_results.items()
        if task_id not in excluded_task_ids
    }

    filtered = deepcopy(component)
    filtered["eval"] = retained_results
    filtered["pass_at_k"] = calculate_pass_at_k(retained_results)
    filtered["baseline_filter"] = {
        "dataset": dataset,
        "excluded_task_ids": present_exclusions,
        "excluded_task_count": len(present_exclusions),
        "original_task_count": len(task_results),
        "retained_task_count": len(retained_results),
    }
    return filtered


def infer_component_dataset(result: dict[str, Any]) -> str:
    task_results = result.get("eval")
    if not isinstance(task_results, dict) or not task_results:
        raise ValueError("Cannot infer dataset from an empty or missing 'eval' mapping")
    first_task_id = next(iter(task_results))
    if first_task_id.startswith("ForgetEval"):
        return "forgeteval"
    if first_task_id.startswith("FunctionalCorrectness"):
        return "utilityeval"
    raise ValueError(
        "Cannot infer component dataset from task IDs; pass --dataset explicitly"
    )


def filter_result(
    result: dict[str, Any],
    excluded: dict[str, set[str]],
    component_dataset: str | None = None,
) -> dict[str, Any]:
    evaluation = result.get("eval")
    if not isinstance(evaluation, dict):
        raise ValueError("Result has no top-level 'eval' mapping")

    combined_components: dict[str, dict[str, Any]] = {}
    for name, component in evaluation.items():
        try:
            normalized = normalize_dataset(name)
        except ValueError:
            continue
        if isinstance(component, dict) and isinstance(component.get("eval"), dict):
            combined_components[normalized] = component

    if combined_components:
        filtered = deepcopy(result)
        filter_summary: dict[str, Any] = {}
        for dataset, component in combined_components.items():
            filtered_component = filter_component(component, dataset, excluded[dataset])
            filtered["eval"][dataset] = filtered_component
            filtered.setdefault("pass_at_k", {})[dataset] = filtered_component[
                "pass_at_k"
            ]
            filter_summary[dataset] = filtered_component["baseline_filter"]
        filtered["baseline_filter"] = filter_summary
        return filtered

    dataset = (
        normalize_dataset(component_dataset)
        if component_dataset is not None
        else infer_component_dataset(result)
    )
    return filter_component(result, dataset, excluded[dataset])


def default_output_path(input_path: Path) -> Path:
    suffix = ".eval_results.json"
    if input_path.name.endswith(suffix):
        return input_path.with_name(
            input_path.name[: -len(suffix)] + ".filtered.eval_results.json"
        )
    return input_path.with_name(input_path.stem + ".filtered.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="EvalPlus result JSON to filter")
    parser.add_argument(
        "--filter-csv",
        type=Path,
        default=DEFAULT_FILTER_CSV,
        help="CSV containing dataset and test_id columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Filtered JSON destination (default: beside the input result)",
    )
    parser.add_argument(
        "--dataset",
        choices=("forgeteval", "utilityeval"),
        default=None,
        help="Dataset name for a single-component result; inferred when omitted",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    filter_csv = args.filter_csv.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(input_path)
    )

    with input_path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    excluded = load_excluded_tasks(filter_csv)
    filtered = filter_result(result, excluded, args.dataset)
    filtered["baseline_filter_source"] = str(filter_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle)
        handle.write("\n")

    print(f"Filtered result: {output_path}")
    for dataset, summary in filtered.get("baseline_filter", {}).items():
        if not isinstance(summary, dict):
            continue
        metrics = filtered["pass_at_k"][dataset]["base"]
        metrics_text = ", ".join(f"{name}={value:.6f}" for name, value in metrics.items())
        print(
            f"{dataset}: excluded={summary['excluded_task_count']}, "
            f"retained={summary['retained_task_count']}, {metrics_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
