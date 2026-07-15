from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from suffix_eval.config import DatasetEvalConfig, GenerationSettings
from suffix_eval.data import EvaluationRow
from suffix_eval.generation import GeneratedRow, generate_suffix_batch
from suffix_eval.results import (
    JsonlWriter,
    max_similarity_at_k,
    pass_at_k_cutoffs,
    write_aggregate_results,
)
from suffix_eval.scoring import ScoreResult


class FakeBatch(dict):
    def to(self, device: torch.device) -> "FakeBatch":
        return FakeBatch({key: value.to(device) for key, value in self.items()})


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, value, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(value, str):
            return SimpleNamespace(input_ids=[1] * len(value))
        return FakeBatch(
            {
                "input_ids": torch.tensor([[0, 1], [2, 3]]),
                "attention_mask": torch.tensor([[0, 1], [1, 1]]),
            }
        )

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:  # type: ignore[no-untyped-def]
        return ",".join(str(int(token_id)) for token_id in token_ids)


class FakeModel:
    def __init__(self) -> None:
        self.generation_kwargs = None

    def generate(self, **kwargs):  # type: ignore[no-untyped-def]
        self.generation_kwargs = kwargs
        prefixes = kwargs["input_ids"]
        outputs = []
        for row_index, prefix in enumerate(prefixes):
            for pass_index in range(kwargs["num_return_sequences"]):
                token = 10 * (row_index + 1) + pass_index
                outputs.append(torch.cat((prefix, torch.tensor([token, token]))))
        return torch.stack(outputs)


class PassKTests(unittest.TestCase):
    def test_pass_at_k_cutoffs_include_standard_intermediate_values(self) -> None:
        self.assertEqual(pass_at_k_cutoffs(1), (1,))
        self.assertEqual(pass_at_k_cutoffs(3), (1, 3))
        self.assertEqual(pass_at_k_cutoffs(10), (1, 5, 10))

    def test_max_similarity_at_k_reuses_the_same_samples(self) -> None:
        scores = [0.2, 0.1, 0.3, 0.4, 0.8, 0.6, 0.5, 0.7, 0.85, 0.9]
        self.assertEqual(
            max_similarity_at_k(scores),
            {1: 0.2, 5: 0.8, 10: 0.9},
        )

    def test_pass_k_requires_sampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires generation.do_sample"):
            GenerationSettings.from_dict({"pass_k": 2})

        settings = GenerationSettings.from_dict({"pass_k": 3, "do_sample": True})
        self.assertEqual(settings.pass_k, 3)

    def test_generation_maps_each_return_sequence_to_its_source_row(self) -> None:
        rows = [
            EvaluationRow(uuid="first", prefix="a", suffix="xy", metadata={}),
            EvaluationRow(uuid="second", prefix="b", suffix="xy", metadata={}),
        ]
        model = FakeModel()
        generated = generate_suffix_batch(
            rows,
            model,  # type: ignore[arg-type]
            FakeTokenizer(),  # type: ignore[arg-type]
            torch.device("cpu"),
            GenerationSettings(pass_k=2, do_sample=True),
        )

        self.assertEqual(
            [(item.row.uuid, item.pass_index, item.generated_suffix) for item in generated],
            [
                ("first", 0, "10,10"),
                ("first", 1, "11,11"),
                ("second", 0, "20,20"),
                ("second", 1, "21,21"),
            ],
        )
        self.assertEqual(model.generation_kwargs["num_return_sequences"], 2)

    def test_result_files_include_pass_metadata(self) -> None:
        config = DatasetEvalConfig(
            dataset_name="dataset",
            prefix_column="prefix",
            suffix_column="suffix",
            uuid_column="uuid",
            mode="secret",
            output_dir=Path("output"),
        )
        generated = GeneratedRow(
            row=EvaluationRow(uuid="id", prefix="p", suffix="s", metadata={}),
            generated_suffix="candidate",
            pass_index=2,
            target_token_count=1,
            max_new_tokens_used=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            rows_path = directory_path / "rows.jsonl"
            with JsonlWriter(rows_path) as writer:
                writer.write_row(
                    generated,
                    ScoreResult(metric="chrf", value=0.8),
                    {"pass_k": 3},
                    is_worst_case=True,
                )
            row = json.loads(rows_path.read_text(encoding="utf-8"))
            self.assertEqual(row["pass_index"], 2)
            self.assertTrue(row["is_worst_case"])

            aggregate_path = directory_path / "aggregate.json"
            write_aggregate_results(
                aggregate_path,
                config,
                metric="chrf",
                num_examples=2,
                average_score=0.7,
                grouped_averages={},
                pass_k=10,
                pass_at_k_results={
                    1: (0.2, {}),
                    5: (0.5, {}),
                    10: (0.7, {}),
                },
            )
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            self.assertEqual(aggregate["num_generated_results"], 20)
            self.assertEqual(aggregate["pass_k"], 10)
            self.assertEqual(aggregate["pass_k_aggregation"], "max_similarity")
            self.assertEqual(
                {
                    key: value["average_similarity_score"]
                    for key, value in aggregate["pass_at_k"].items()
                },
                {"pass@1": 0.2, "pass@5": 0.5, "pass@10": 0.7},
            )


if __name__ == "__main__":
    unittest.main()
