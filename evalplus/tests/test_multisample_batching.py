import json
from collections import Counter

from evalplus.codegen import codegen
from evalplus.data.combined import (
    FORGET_UTILITY_EVAL_COMPONENTS,
    FORGET_UTILITY_EVAL_DATASET,
    get_combined_eval_components,
    normalize_combined_eval_dataset_name,
)
from evalplus.evaluate import get_report_k_values


class FakeBatchedDecoder:
    batch_size = 5

    def __init__(self):
        self.calls = []

    def codegen_batch(self, prompts, do_sample=True, num_samples=1):
        self.calls.append((len(prompts), num_samples, do_sample))
        return [
            [f"solution-{prompt_index}-{sample_index}" for sample_index in range(num_samples)]
            for prompt_index, _ in enumerate(prompts)
        ]

    def is_direct_completion(self):
        return False

    def __str__(self):
        return "fake-batched-decoder"


def test_multisample_codegen_batches_problems_times_samples(tmp_path):
    dataset = {
        f"Test/{index}": {"prompt": f"prompt {index}", "entry_point": "solution"}
        for index in range(7)
    }
    output_path = tmp_path / "samples.jsonl"
    model = FakeBatchedDecoder()

    codegen(
        target_path=str(output_path),
        model=model,
        dataset=dataset,
        greedy=False,
        n_samples=5,
        sanitize_output=False,
    )

    assert model.calls == [(5, 5, True), (2, 5, True)]
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 35
    assert Counter(row["task_id"] for row in rows) == {
        task_id: 5 for task_id in dataset
    }


def test_forget_utility_combined_dataset_aliases():
    assert normalize_combined_eval_dataset_name("forget_utility") == (
        FORGET_UTILITY_EVAL_DATASET
    )
    assert get_combined_eval_components("forgeteval-utilityeval") == (
        FORGET_UTILITY_EVAL_COMPONENTS
    )


def test_report_k_values_include_midpoint():
    assert get_report_k_values(1) == [1]
    assert get_report_k_values(5) == [1, 3, 5]
    assert get_report_k_values(10) == [1, 5, 10]
