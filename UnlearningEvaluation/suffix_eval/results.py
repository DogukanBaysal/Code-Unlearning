from __future__ import annotations

import json
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from suffix_eval.config import DatasetEvalConfig
from suffix_eval.generation import GeneratedRow
from suffix_eval.scoring import ScoreResult


def pass_at_k_cutoffs(pass_k: int) -> tuple[int, ...]:
    if pass_k <= 0:
        raise ValueError("pass_k must be greater than 0")
    cutoffs = {1, pass_k}
    cutoffs.update(cutoff for cutoff in (5, 10) if cutoff <= pass_k)
    return tuple(sorted(cutoffs))


def max_similarity_at_k(scores: list[float]) -> dict[int, float]:
    if not scores:
        raise ValueError("scores must not be empty")
    return {
        cutoff: max(scores[:cutoff])
        for cutoff in pass_at_k_cutoffs(len(scores))
    }


@dataclass
class AggregateTracker:
    group_columns: tuple[str, ...]
    scores: list[float] = field(default_factory=list)
    grouped_scores: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )

    def add(self, score: float, metadata: dict[str, Any]) -> None:
        self.scores.append(score)
        for column in self.group_columns:
            value = metadata.get(column)
            key = "<missing>" if value is None else str(value)
            self.grouped_scores[column][key].append(score)

    def averages(self) -> tuple[float, dict[str, dict[str, float]]]:
        average = sum(self.scores) / len(self.scores) if self.scores else 0.0
        grouped = {
            column: {
                value: sum(values) / len(values)
                for value, values in sorted(value_scores.items())
                if values
            }
            for column, value_scores in self.grouped_scores.items()
        }
        return average, grouped


class JsonlWriter(AbstractContextManager["JsonlWriter"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        if self._handle is not None:
            self._handle.close()

    def write_row(
        self,
        generated: GeneratedRow,
        score: ScoreResult,
        generation_params: dict[str, Any],
        score_error: str | None = None,
        is_worst_case: bool = False,
    ) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlWriter must be used as a context manager")
        row = {
            "uuid": generated.row.uuid,
            "prefix": generated.row.prefix,
            "real_suffix": generated.row.suffix,
            "generated_suffix": generated.generated_suffix,
            "score_type": score.metric,
            "score_value": score.value,
            "pass_index": generated.pass_index,
            "is_worst_case": is_worst_case,
            "metadata": generated.row.metadata,
            "generation": {
                **generation_params,
                "target_token_count": generated.target_token_count,
                "max_new_tokens_used": generated.max_new_tokens_used,
            },
        }
        if score_error is not None:
            row["score_error"] = score_error
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_aggregate_results(
    path: Path,
    config: DatasetEvalConfig,
    metric: str,
    num_examples: int,
    average_score: float,
    grouped_averages: dict[str, dict[str, float]],
    pass_k: int = 1,
    num_generated_results: int | None = None,
    score_failures: int = 0,
    pass_at_k_results: (
        dict[int, tuple[float, dict[str, dict[str, float]]]] | None
    ) = None,
    uuid_filter: dict[str, Any] | None = None,
) -> None:
    if num_generated_results is None:
        num_generated_results = num_examples * pass_k
    if pass_at_k_results is None:
        pass_at_k_results = {pass_k: (average_score, grouped_averages)}
    payload = {
        "mode": config.mode,
        "num_evaluated_examples": num_examples,
        "num_generated_results": num_generated_results,
        "average_similarity_score": average_score,
        "score_metric": metric,
        "pass_k": pass_k,
        "pass_k_aggregation": "max_similarity",
        "pass_at_k": {
            f"pass@{cutoff}": {
                "average_similarity_score": cutoff_average,
                "grouped_averages": cutoff_grouped_averages,
            }
            for cutoff, (cutoff_average, cutoff_grouped_averages) in sorted(
                pass_at_k_results.items()
            )
        },
        "score_failures": score_failures,
        "grouped_averages": grouped_averages,
        "config": config.as_dict(),
    }
    if uuid_filter is not None:
        payload["uuid_filter"] = uuid_filter
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
