from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from suffix_eval.config import DatasetEvalConfig
from suffix_eval.data import (
    aggregate_filter_model_dir,
    aggregate_filter_selector,
    load_aggregate_uuid_filter,
)


class AggregateUuidFilterTests(unittest.TestCase):
    def make_config(self, label: str, mode: str) -> DatasetEvalConfig:
        return DatasetEvalConfig(
            dataset_name=f"dbaysal/{label}",
            prefix_column="prefix",
            suffix_column="suffix",
            uuid_column="uuid",
            mode=mode,
            output_dir=Path("output"),
            label=label,
        )

    def test_dataset_labels_map_to_corresponding_csv_rows(self) -> None:
        self.assertEqual(
            aggregate_filter_selector(self.make_config("forget", "secret")),
            ("forget", "secret"),
        )
        self.assertEqual(
            aggregate_filter_selector(self.make_config("forget", "code")),
            ("forget", "code-unit"),
        )
        self.assertEqual(
            aggregate_filter_selector(self.make_config("retain", "code")),
            ("retain", "code"),
        )
        self.assertEqual(
            aggregate_filter_selector(self.make_config("approximate", "code")),
            ("held_out_approximate", "code"),
        )

    def test_filter_loads_only_the_matching_split_and_mode(self) -> None:
        csv_text = "\n".join(
            (
                "model_dir,split,eval_mode,uuid",
                "qwen2_5_coder_3b,forget,secret,secret-id",
                "qwen2_5_coder_3b,forget,code-unit,code-id",
                "meta_llama3_2_3b,forget,code-unit,meta-code-id",
                "qwen2_5_coder_3b,retain,code,retain-id",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "matches.csv"
            csv_path.write_text(csv_text, encoding="utf-8")
            uuid_filter = load_aggregate_uuid_filter(
                csv_path,
                self.make_config("forget", "code"),
                "Qwen/Qwen2.5-Coder-3B",
            )

        self.assertEqual(uuid_filter.excluded_uuids, frozenset({"code-id"}))
        self.assertEqual(uuid_filter.model_dir, "qwen2_5_coder_3b")
        self.assertEqual(uuid_filter.split, "forget")
        self.assertEqual(uuid_filter.eval_mode, "code-unit")

    def test_model_names_map_to_csv_model_directories(self) -> None:
        self.assertEqual(
            aggregate_filter_model_dir("Qwen/Qwen2.5-Coder-3B"),
            "qwen2_5_coder_3b",
        )
        self.assertEqual(
            aggregate_filter_model_dir("meta-llama/Llama-3.2-3B"),
            "meta_llama3_2_3b",
        )


if __name__ == "__main__":
    unittest.main()
