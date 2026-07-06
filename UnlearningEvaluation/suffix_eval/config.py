from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


VALID_MODES = {"code", "secret"}
GENERATION_KEYS = {
    "max_new_tokens",
    "temperature",
    "top_p",
    "do_sample",
    "batch_size",
    "device",
    "dtype",
    "greedy",
}
MODEL_KEYS = {
    "model_name",
    "tokenizer_name",
    "peft_name",
    "peft_subfolder",
    "trust_remote_code",
}
DATASET_KEYS = {
    "label",
    "dataset_name",
    "dataset_split",
    "prefix_column",
    "suffix_column",
    "uuid_column",
    "mode",
    "output_dir",
    "code_language",
}


@dataclass
class GenerationSettings:
    max_new_tokens: int = 128
    temperature: float | None = None
    top_p: float | None = None
    do_sample: bool = False
    batch_size: int = 1
    device: str = "auto"
    dtype: str = "auto"
    greedy: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GenerationSettings":
        if not isinstance(raw, dict):
            raise ValueError("generation must be a YAML mapping")
        unknown = sorted(set(raw) - GENERATION_KEYS)
        if unknown:
            raise ValueError(f"Unknown generation config key(s): {', '.join(unknown)}")

        settings = cls(**raw)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("generation.max_new_tokens must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("generation.batch_size must be greater than 0")
        if self.temperature is not None and self.temperature <= 0:
            raise ValueError("generation.temperature must be greater than 0 when set")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("generation.top_p must be in the range (0, 1]")

        if self.greedy:
            self.do_sample = False
            self.temperature = None
            self.top_p = None

    def to_generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
        return kwargs


@dataclass
class DatasetEvalConfig:
    dataset_name: str
    prefix_column: str
    suffix_column: str
    uuid_column: str
    mode: str
    output_dir: Path
    label: str | None = None
    dataset_split: str = "test"
    code_language: str = "python"

    @classmethod
    def from_dict(cls, raw: dict[str, Any], defaults: dict[str, Any] | None = None) -> "DatasetEvalConfig":
        if not isinstance(raw, dict):
            raise ValueError("Each datasets item must be a YAML mapping")

        values = dict(defaults or {})
        values.update(raw)

        required = {
            "dataset_name",
            "prefix_column",
            "suffix_column",
            "uuid_column",
            "mode",
            "output_dir",
        }
        missing = sorted(key for key in required if key not in values or values[key] in (None, ""))
        if missing:
            raise ValueError(f"Missing required dataset config key(s): {', '.join(missing)}")

        unknown = sorted(set(values) - DATASET_KEYS)
        if unknown:
            raise ValueError(f"Unknown dataset config key(s): {', '.join(unknown)}")

        values["output_dir"] = Path(values["output_dir"])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {self.mode!r}")
        if not self.dataset_split:
            raise ValueError("dataset_split must not be empty")
        if not self.code_language:
            raise ValueError("code_language must not be empty")

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass
class EvalConfig:
    model_name: str
    tokenizer_name: str | None = None
    peft_name: str | None = None
    peft_subfolder: str | None = None
    trust_remote_code: bool = False
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    datasets: list[DatasetEvalConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        if not isinstance(raw, dict):
            raise ValueError("Configuration file must contain a YAML mapping")

        generation_value = raw.pop("generation", {}) or {}
        if not isinstance(generation_value, dict):
            raise ValueError("generation must be a YAML mapping")
        generation_raw = dict(generation_value)
        for key in list(raw):
            if key in GENERATION_KEYS:
                generation_raw[key] = raw.pop(key)

        missing = sorted(key for key in {"model_name"} if key not in raw or raw[key] in (None, ""))
        if missing:
            raise ValueError(f"Missing required config key(s): {', '.join(missing)}")

        datasets = cls._parse_datasets(raw)
        model_values = {key: raw.pop(key) for key in list(raw) if key in MODEL_KEYS}
        unknown = sorted(set(raw))
        if unknown:
            raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")

        config = cls(
            **model_values,
            generation=GenerationSettings.from_dict(generation_raw),
            datasets=datasets,
        )
        config.validate()
        return config

    @classmethod
    def _parse_datasets(cls, raw: dict[str, Any]) -> list[DatasetEvalConfig]:
        dataset_defaults = {
            key: raw.pop(key)
            for key in list(raw)
            if key in {"dataset_split", "uuid_column", "code_language"}
        }

        if "datasets" in raw:
            datasets_raw = raw.pop("datasets")
            if not isinstance(datasets_raw, list) or not datasets_raw:
                raise ValueError("datasets must be a non-empty YAML list")
            return [
                DatasetEvalConfig.from_dict(dataset_raw, defaults=dataset_defaults)
                for dataset_raw in datasets_raw
            ]

        dataset_raw = {
            key: raw.pop(key)
            for key in list(raw)
            if key in DATASET_KEYS
        }
        return [DatasetEvalConfig.from_dict(dataset_raw, defaults=dataset_defaults)]

    def validate(self) -> None:
        if not self.datasets:
            raise ValueError("At least one dataset evaluation config is required")

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["datasets"] = [dataset.as_dict() for dataset in self.datasets]
        return data
