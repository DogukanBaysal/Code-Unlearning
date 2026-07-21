import csv
import importlib.util
from pathlib import Path


TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "filter_baseline_failed_results.py"
)
SPEC = importlib.util.spec_from_file_location("filter_baseline_failed_results", TOOL_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_records(correct_count: int, total: int = 10):
    return [
        {"base_status": "pass" if index < correct_count else "fail"}
        for index in range(total)
    ]


def test_combined_filter_recomputes_metrics_and_makes_baseline_pass_at_10_one():
    result = {
        "dataset": "forget-utility",
        "eval": {
            "forgeteval": {
                "eval": {
                    "ForgetEval1": make_records(0),
                    "ForgetEval2": make_records(1),
                },
                "pass_at_k": {"base": {"pass@10": 0.5}},
            },
            "utilityeval": {
                "eval": {
                    "FunctionalCorrectness1": make_records(0),
                    "FunctionalCorrectness2": make_records(2),
                },
                "pass_at_k": {"base": {"pass@10": 0.5}},
            },
        },
        "pass_at_k": {},
    }
    excluded = {
        "forgeteval": {"ForgetEval1"},
        "utilityeval": {"FunctionalCorrectness1"},
    }

    filtered = MODULE.filter_result(result, excluded)

    assert set(filtered["eval"]["forgeteval"]["eval"]) == {"ForgetEval2"}
    assert set(filtered["eval"]["utilityeval"]["eval"]) == {
        "FunctionalCorrectness2"
    }
    assert filtered["pass_at_k"]["forgeteval"]["base"]["pass@10"] == 1.0
    assert filtered["pass_at_k"]["utilityeval"]["base"]["pass@10"] == 1.0
    assert filtered["baseline_filter"]["forgeteval"]["excluded_task_count"] == 1
    assert filtered["baseline_filter"]["utilityeval"]["retained_task_count"] == 1


def test_csv_loader_uses_union_of_all_listed_failures(tmp_path):
    csv_path = tmp_path / "failures.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "test_id", "failed_in_both_models"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "forgeteval",
                "test_id": "ForgetEval1",
                "failed_in_both_models": "true",
            }
        )
        writer.writerow(
            {
                "dataset": "utilityeval",
                "test_id": "FunctionalCorrectness1",
                "failed_in_both_models": "false",
            }
        )

    excluded = MODULE.load_excluded_tasks(csv_path)

    assert excluded == {
        "forgeteval": {"ForgetEval1"},
        "utilityeval": {"FunctionalCorrectness1"},
    }
