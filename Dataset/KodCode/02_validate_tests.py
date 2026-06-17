#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm


def get_field(row: Dict[str, Any], *names: str) -> Optional[str]:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def test_needs_pytest(test_code: str) -> bool:
    return (
        "def test_" in test_code
        or "class Test" in test_code
        or "pytest" in test_code
        or "unittest" in test_code
    )


def validate_one(
    index: int,
    row_id: Optional[str],
    solution: Optional[str],
    tests: Optional[str],
    timeout_seconds: int,
) -> Tuple[int, Optional[str], bool, str]:
    if row_id is None:
        return index, row_id, False, "Missing id"

    if solution is None or tests is None:
        return index, row_id, False, "Missing solution or tests"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        solution_path = tmpdir_path / "solution.py"
        test_path = tmpdir_path / "test_solution.py"

        solution_path.write_text(solution, encoding="utf-8")
        test_path.write_text(tests, encoding="utf-8")

        command = (
            [sys.executable, "-m", "pytest", "-q", str(test_path)]
            if test_needs_pytest(tests)
            else [sys.executable, str(test_path)]
        )

        try:
            result = subprocess.run(
                command,
                cwd=tmpdir_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            output = ""

            if exc.stderr:
                output += (
                    exc.stderr
                    if isinstance(exc.stderr, str)
                    else exc.stderr.decode(errors="replace")
                )

            return index, row_id, False, f"TIMEOUT after {timeout_seconds}s\n{output}"

        output = result.stderr or ""
        return index, row_id, result.returncode == 0, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dbaysal/KodCode-filtered")
    parser.add_argument("--split", default="train")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--show-first-failures", type=int, default=20)

    parser.add_argument(
        "--failed-ids-output",
        default="failed_ids.txt",
        help="Path to write UUIDs of failed rows.",
    )
    parser.add_argument(
        "--target-dataset",
        default=None,
        help="Optional Hugging Face dataset repo to push the filtered dataset to.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push target dataset as private.",
    )

    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)

    if "id" not in ds.column_names:
        raise ValueError(
            f"Dataset must contain an 'id' column. Columns: {ds.column_names}"
        )

    total = len(ds)
    if args.max_examples is not None:
        total = min(total, args.max_examples)

    jobs = []

    for i in range(total):
        row = ds[i]
        row_id = get_field(row, "id", "ID", "uuid", "UUID")
        solution = get_field(row, "solution", "Solution")
        tests = get_field(row, "tests", "Tests", "test", "Test")
        jobs.append((i, row_id, solution, tests, args.timeout))

    passed = 0
    failed = 0
    missing = 0
    failed_ids = set()
    first_failures = []

    print(f"Validating {total} examples with {args.workers} workers...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                validate_one,
                index,
                row_id,
                solution,
                tests,
                timeout,
            )
            for index, row_id, solution, tests, timeout in jobs
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating"):
            index, row_id, ok, output = future.result()

            if ok:
                passed += 1
            else:
                failed += 1

                if row_id is not None:
                    failed_ids.add(row_id)

                if output in {"Missing id", "Missing solution or tests"}:
                    missing += 1

                if len(first_failures) < args.show_first_failures:
                    first_failures.append((index, row_id, output.strip()))

    failed_ids_path = Path(args.failed_ids_output)
    failed_ids_path.write_text(
        "\n".join(sorted(failed_ids)) + ("\n" if failed_ids else ""),
        encoding="utf-8",
    )

    print("\n=== Validation Summary ===")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Checked: {total}")
    print(f"Workers: {args.workers}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Missing id/solution/tests: {missing}")
    print(f"Failed UUIDs written to: {failed_ids_path}")

    if first_failures:
        print("\n=== First Failures ===")
        for index, row_id, output in sorted(first_failures, key=lambda x: x[0]):
            print(f"\n--- Example index: {index}, id: {row_id} ---")
            print(output[:4000])

    print("\nFiltering failed rows from dataset...")

    if args.max_examples is None:
        filtered_ds = ds.filter(
            lambda example: example["id"] not in failed_ids,
            desc="Removing failed rows",
        )
    else:
        checked_ids = {jobs[i][1] for i in range(total) if jobs[i][1] is not None}

        filtered_ds = ds.filter(
            lambda example: (
                example["id"] not in checked_ids
                or example["id"] not in failed_ids
            ),
            desc="Removing failed rows from checked subset",
        )

    print(f"Filtered dataset: {len(ds)} -> {len(filtered_ds)}")

    filtered_dataset = DatasetDict({
        args.split: filtered_ds,
    })

    if args.target_dataset:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.target_dataset,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

        print(f"Pushing filtered dataset to: {args.target_dataset}")
        filtered_dataset.push_to_hub(
            args.target_dataset,
            private=args.private,
            token=token,
        )

        print(f"Done: https://huggingface.co/datasets/{args.target_dataset}")
    else:
        print(
            "\nNo --target-dataset provided, so the filtered dataset was not pushed."
        )
        print(
            "To push it, rerun with for example:\n"
            "  --target-dataset dbaysal/KodCode-filtered-valid --private"
        )


if __name__ == "__main__":
    main()