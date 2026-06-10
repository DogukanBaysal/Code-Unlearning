from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from suffix_eval.config import EvalConfig


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


def load_model_and_tokenizer(config: EvalConfig) -> LoadedModel:
    tokenizer_name = config.tokenizer_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.trust_remote_code,
    )
    ensure_padding_token(tokenizer)
    tokenizer.padding_side = "left"

    device = resolve_device(config.generation.device)
    torch_dtype = resolve_dtype(config.generation.dtype)

    model_kwargs = {"trust_remote_code": config.trust_remote_code}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    if config.peft_name:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "peft_name was provided, but the 'peft' package is not installed. "
                "Install it with: pip install peft"
            ) from exc
        model = PeftModel.from_pretrained(model, config.peft_name)

    model.to(device)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, device=device)


def ensure_padding_token(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if device.type == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            raise RuntimeError("MPS was requested, but it is not available")
    return device


def resolve_dtype(dtype: str) -> torch.dtype | str | None:
    normalized = dtype.lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32", "full"}:
        return torch.float32
    raise ValueError("generation.dtype must be one of: auto, float16, bfloat16, float32")
