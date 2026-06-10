from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from suffix_eval.config import EvalConfig


CODE_GROUP_COLUMNS = ("difficulty", "type")
SECRET_GROUP_COLUMNS = ("secret_location", "secret_type")


@dataclass(frozen=True)
class EvaluationRow:
    uuid: str
    prefix: str
    suffix: str
    metadata: dict[str, Any]


def load_evaluation_dataset(config: EvalConfig) -> Dataset:
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


def validate_dataset(dataset: Dataset, config: EvalConfig) -> None:
    if len(dataset) == 0:
        raise ValueError(f"Dataset {config.dataset_name!r} split {config.dataset_split!r} is empty")

    required_columns = {config.prefix_column, config.suffix_column, config.uuid_column}
    missing = sorted(required_columns - set(dataset.column_names))
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing)}")

    group_columns = CODE_GROUP_COLUMNS if config.mode == "code" else SECRET_GROUP_COLUMNS
    missing_groups = [column for column in group_columns if column not in dataset.column_names]
    if missing_groups:
        raise ValueError(
            f"mode={config.mode!r} requires grouping column(s): {', '.join(missing_groups)}"
        )


def iter_evaluation_rows(dataset: Dataset, config: EvalConfig) -> Iterable[EvaluationRow]:
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
