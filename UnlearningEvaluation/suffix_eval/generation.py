from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from suffix_eval.config import GenerationSettings
from suffix_eval.data import EvaluationRow


@dataclass(frozen=True)
class GeneratedRow:
    row: EvaluationRow
    generated_suffix: str
    target_token_count: int
    max_new_tokens_used: int


@torch.inference_mode()
def generate_suffix_batch(
    rows: list[EvaluationRow],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    settings: GenerationSettings,
) -> list[GeneratedRow]:
    if not rows:
        return []

    inputs = tokenizer(
        [row.prefix for row in rows],
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(device)

    target_token_counts = [
        max(1, len(tokenizer(row.suffix, add_special_tokens=False).input_ids)) for row in rows
    ]
    max_new_tokens_used = min(settings.max_new_tokens, max(target_token_counts))

    generation_kwargs = settings.to_generation_kwargs()
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens_used,
        **generation_kwargs,
    )

    input_length = inputs["input_ids"].shape[1]
    generated_rows: list[GeneratedRow] = []
    for index, row in enumerate(rows):
        generated_ids = output_ids[index][input_length:]
        capped_ids = generated_ids[: target_token_counts[index]]
        generated_text = tokenizer.decode(capped_ids, skip_special_tokens=True)
        generated_rows.append(
            GeneratedRow(
                row=row,
                generated_suffix=generated_text,
                target_token_count=target_token_counts[index],
                max_new_tokens_used=min(settings.max_new_tokens, target_token_counts[index]),
            )
        )
    return generated_rows
