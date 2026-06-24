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
class EvalConfig:
    model_name: str
    dataset_name: str
    prefix_column: str
    suffix_column: str
    uuid_column: str
    mode: str
    output_dir: Path
    tokenizer_name: str | None = None
    peft_name: str | None = None
    peft_subfolder: str | None = None
    dataset_split: str = "test"
    code_language: str = "python"
    trust_remote_code: bool = False
    generation: GenerationSettings = field(default_factory=GenerationSettings)

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

        required = {
            "model_name",
            "dataset_name",
            "prefix_column",
            "suffix_column",
            "uuid_column",
            "mode",
            "output_dir",
        }
        missing = sorted(key for key in required if key not in raw or raw[key] in (None, ""))
        if missing:
            raise ValueError(f"Missing required config key(s): {', '.join(missing)}")

        allowed = {
            "model_name",
            "tokenizer_name",
            "peft_name",
            "peft_subfolder",
            "dataset_name",
            "dataset_split",
            "prefix_column",
            "suffix_column",
            "uuid_column",
            "mode",
            "output_dir",
            "code_language",
            "trust_remote_code",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")

        init_values = dict(raw)
        init_values["output_dir"] = Path(raw["output_dir"])
        config = cls(**init_values, generation=GenerationSettings.from_dict(generation_raw))
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
