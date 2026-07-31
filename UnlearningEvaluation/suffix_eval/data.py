from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from suffix_eval.config import DatasetEvalConfig


CODE_GROUP_COLUMNS = ("difficulty", "type")
SECRET_GROUP_COLUMNS = ("secret_location", "secret_type")


@dataclass(frozen=True)
class EvaluationRow:
    uuid: str
    prefix: str
    suffix: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AggregateUuidFilter:
    source_csv: Path
    model_dir: str
    split: str
    eval_mode: str
    excluded_uuids: frozenset[str]

    def as_dict(
        self,
        evaluated_examples: int,
        included_examples: int,
    ) -> dict[str, Any]:
        return {
            "source_csv": str(self.source_csv),
            "operation": "exclude",
            "model_dir": self.model_dir,
            "split": self.split,
            "eval_mode": self.eval_mode,
            "num_excluded_uuids_in_csv": len(self.excluded_uuids),
            "num_excluded_examples": evaluated_examples - included_examples,
            "num_included_examples": included_examples,
        }


def load_aggregate_uuid_filter(
    csv_path: Path,
    config: DatasetEvalConfig,
    model_name: str,
) -> AggregateUuidFilter:
    model_dir = aggregate_filter_model_dir(model_name)
    split, eval_mode = aggregate_filter_selector(config)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"model_dir", "split", "eval_mode", "uuid"}
        missing = sorted(required_columns - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                f"Aggregate filter CSV {csv_path} is missing column(s): "
                f"{', '.join(missing)}"
            )
        excluded_uuids = frozenset(
            row["uuid"].strip()
            for row in reader
            if row["model_dir"].strip() == model_dir
            and row["split"].strip() == split
            and row["eval_mode"].strip() == eval_mode
            and row["uuid"].strip()
        )
    return AggregateUuidFilter(
        source_csv=csv_path,
        model_dir=model_dir,
        split=split,
        eval_mode=eval_mode,
        excluded_uuids=excluded_uuids,
    )


def aggregate_filter_model_dir(model_name: str) -> str:
    normalized = model_name.lower().replace("_", "-")
    if "qwen2.5-coder-3b" in normalized or "qwen2-5-coder-3b" in normalized:
        return "qwen2_5_coder_3b"
    if any(
        alias in normalized
        for alias in ("llama-3.2-3b", "llama3.2-3b", "llama3-2-3b")
    ):
        return "meta_llama3_2_3b"
    raise ValueError(
        f"Could not map model {model_name!r} to a model_dir in the aggregate filter CSV"
    )


def aggregate_filter_selector(config: DatasetEvalConfig) -> tuple[str, str]:
    dataset_key = (config.label or config.dataset_name.rsplit("/", 1)[-1]).lower()
    dataset_key = dataset_key.replace("-", "_")
    if dataset_key == "forget":
        eval_mode = "secret" if config.mode == "secret" else "code-unit"
        return "forget", eval_mode
    if dataset_key in {"retain", "retain_half", "retain_full"}:
        return "retain", "code"
    if dataset_key in {"approximate", "held_out_approximate"}:
        return "held_out_approximate", "code"
    raise ValueError(
        "Could not map aggregate filter rows for dataset "
        f"{config.label or config.dataset_name!r}; expected forget, retain, or approximate"
    )


def load_evaluation_dataset(config: DatasetEvalConfig) -> Dataset:
    dataset_path = Path(config.dataset_name)
    if dataset_path.exists():
        try:
            loaded = load_from_disk(str(dataset_path))
        except (FileNotFoundError, OSError, ValueError):
            dataset = load_dataset(config.dataset_name, split=config.dataset_split)
        else:
            if isinstance(loaded, DatasetDict):
                if config.dataset_split not in loaded:
                    available = ", ".join(sorted(loaded))
                    raise ValueError(
                        f"Dataset split {config.dataset_split!r} not found. "
                        f"Available splits: {available}"
                    )
                dataset = loaded[config.dataset_split]
            elif isinstance(loaded, Dataset):
                dataset = loaded
            else:
                raise TypeError(f"Unsupported dataset loaded from disk: {type(loaded).__name__}")
    else:
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    validate_dataset(dataset, config)
    return dataset


def validate_dataset(dataset: Dataset, config: DatasetEvalConfig) -> None:
    if len(dataset) == 0:
        raise ValueError(f"Dataset {config.dataset_name!r} split {config.dataset_split!r} is empty")

    required_columns = {config.prefix_column, config.suffix_column, config.uuid_column}
    missing = sorted(required_columns - set(dataset.column_names))
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing)}")

    if config.mode == "code":
        return

    group_columns = SECRET_GROUP_COLUMNS
    missing_groups = [column for column in group_columns if column not in dataset.column_names]
    if missing_groups:
        raise ValueError(
            f"mode={config.mode!r} requires grouping column(s): {', '.join(missing_groups)}"
        )


def available_group_columns(dataset: Dataset, mode: str) -> tuple[str, ...]:
    group_columns = CODE_GROUP_COLUMNS if mode == "code" else SECRET_GROUP_COLUMNS
    return tuple(column for column in group_columns if column in dataset.column_names)


def iter_evaluation_rows(dataset: Dataset, config: DatasetEvalConfig) -> Iterable[EvaluationRow]:
    metadata_columns = (*CODE_GROUP_COLUMNS, *SECRET_GROUP_COLUMNS)
    for raw_row in dataset:
        metadata = {column: raw_row.get(column) for column in metadata_columns if column in raw_row}
        yield EvaluationRow(
            uuid=str(raw_row[config.uuid_column]),
            prefix=coerce_text(raw_row[config.prefix_column], config.prefix_column),
            suffix=coerce_text(raw_row[config.suffix_column], config.suffix_column),
            metadata=metadata,
        )


def coerce_text(value: Any, column_name: str) -> str:
    if value is None:
        raise ValueError(f"Column {column_name!r} contains None; expected text")
    if isinstance(value, str):
        return value
    return str(value)
