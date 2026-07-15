from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import evaluate_suffix_generation
from suffix_eval.config import DatasetEvalConfig, EvalConfig, GenerationSettings
from suffix_eval.generation import GeneratedRow
from suffix_eval.scoring import ScoreResult


class FakeDataset(list):
    column_names = [
        "uuid",
        "prefix",
        "suffix",
        "secret_location",
        "secret_type",
    ]


class FakeScorer:
    metric_name = "chrf"

    def score(self, generated: str, reference: str) -> ScoreResult:
        return ScoreResult(metric=self.metric_name, value=float(generated))


def fake_generate(rows, model, tokenizer, device, settings):  # type: ignore[no-untyped-def]
    return [
        GeneratedRow(
            row=row,
            generated_suffix=str(pass_index / 10),
            pass_index=pass_index,
            target_token_count=1,
            max_new_tokens_used=1,
        )
        for row in rows
        for pass_index in range(settings.pass_k)
    ]


class FilteredAggregateResultsTests(unittest.TestCase):
    def test_original_and_uuid_filtered_aggregates_are_written_separately(self) -> None:
        dataset = FakeDataset(
            [
                {
                    "uuid": "keep",
                    "prefix": "p1",
                    "suffix": "s1",
                    "secret_location": "code",
                    "secret_type": "password",
                },
                {
                    "uuid": "discard",
                    "prefix": "p2",
                    "suffix": "s2",
                    "secret_location": "documentation",
                    "secret_type": "email",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "matches.csv"
            csv_path.write_text(
                "split,eval_mode,uuid\nforget,secret,keep\n",
                encoding="utf-8",
            )
            dataset_config = DatasetEvalConfig(
                dataset_name="dbaysal/forget",
                prefix_column="prefix",
                suffix_column="suffix",
                uuid_column="uuid",
                mode="secret",
                output_dir=root / "output",
                label="forget",
            )
            suite_config = EvalConfig(
                model_name="model",
                aggregate_filter_csv=csv_path,
                generation=GenerationSettings(
                    pass_k=10,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    batch_size=2,
                ),
                datasets=[dataset_config],
            )
            loaded = SimpleNamespace(model=None, tokenizer=None, device=torch.device("cpu"))

            with (
                patch.object(evaluate_suffix_generation, "load_evaluation_dataset", return_value=dataset),
                patch.object(evaluate_suffix_generation, "build_scorer", return_value=FakeScorer()),
                patch.object(evaluate_suffix_generation, "generate_suffix_batch", side_effect=fake_generate),
            ):
                evaluate_suffix_generation.run_dataset(
                    dataset_config,
                    suite_config,
                    loaded,
                )

            original = json.loads(
                (dataset_config.output_dir / "aggregate_results.json").read_text()
            )
            filtered = json.loads(
                (dataset_config.output_dir / "aggregate_results_filtered.json").read_text()
            )

        self.assertEqual(original["num_evaluated_examples"], 2)
        self.assertNotIn("uuid_filter", original)
        self.assertEqual(filtered["num_evaluated_examples"], 1)
        self.assertEqual(filtered["uuid_filter"]["num_selected_uuids"], 1)
        self.assertEqual(filtered["uuid_filter"]["num_matched_examples"], 1)
        self.assertEqual(
            list(filtered["pass_at_k"]),
            ["pass@1", "pass@5", "pass@10"],
        )


if __name__ == "__main__":
    unittest.main()
