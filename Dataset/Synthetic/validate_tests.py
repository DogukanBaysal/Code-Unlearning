import argparse
import ast
import json
import re
import traceback
from typing import Any, Dict, Iterator, Optional, Tuple


def read_jsonl(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                yield line_no, {
                    "_json_error": f"Invalid JSON on line {line_no}: {e}"
                }


def read_huggingface_dataset(
    dataset_name: str,
    split: str,
    config_name: Optional[str] = None,
    streaming: bool = False,
    token: Optional[str] = None,
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """
    Read rows from a Hugging Face dataset.

    Examples:
        --hf-dataset openai/openai_humaneval --split test
        --hf-dataset bigcode/humanevalpack --hf-config python --split test
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Missing dependency: datasets. Install it with:\n\n"
            "    pip install datasets\n"
        ) from e

    load_kwargs = {
        "path": dataset_name,
        "split": split,
        "streaming": streaming,
    }

    if config_name:
        load_kwargs["name"] = config_name

    if token:
        load_kwargs["token"] = token

    dataset = load_dataset(**load_kwargs)

    for idx, row in enumerate(dataset, start=1):
        # Hugging Face rows are often dict-like but not always plain dicts.
        yield idx, dict(row)


def infer_entry_point_from_code(code: str) -> Optional[str]:
    """
    Infer candidate name from the first top-level class or function.

    Works for:
        class MyClass:
        def my_function(...):
    """
    if not code:
        return None

    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
    except SyntaxError:
        pass

    match = re.search(
        r"^\s*(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)",
        code,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def build_program(row: Dict[str, Any]) -> str:
    """
    Build executable code for one task.

    Supported row formats include:
      - code: full executable canonical code
      - prompt + canonical_solution
      - prompt + solution
      - declaration + canonical_solution
    """
    if row.get("code"):
        return row["code"]

    prompt = (
        row.get("prompt")
        or row.get("declaration")
        or row.get("signature")
        or ""
    )

    canonical_solution = (
        row.get("canonical_solution")
        or row.get("solution")
        or row.get("reference_solution")
    )

    if canonical_solution is None:
        raise KeyError(
            "Missing solution field. Expected one of: "
            "`code`, `canonical_solution`, `solution`, or `reference_solution`."
        )

    stripped = canonical_solution.lstrip()

    # Some datasets store canonical_solution as the full class/function.
    if stripped.startswith("class ") or stripped.startswith("def "):
        return canonical_solution

    # HumanEval-style: prompt contains signature/class header,
    # canonical_solution contains the indented implementation.
    return prompt.rstrip() + "\n" + canonical_solution.lstrip("\n")


def get_test_code(row: Dict[str, Any]) -> str:
    """
    Extract test code from common HumanEval-like dataset formats.
    """
    test = (
        row.get("test")
        or row.get("tests")
        or row.get("test_code")
        or row.get("unit_tests")
    )

    if test is None:
        raise KeyError(
            "Missing test field. Expected one of: "
            "`test`, `tests`, `test_code`, or `unit_tests`."
        )

    return test


def infer_entry_point(row: Dict[str, Any], program: str) -> Optional[str]:
    return (
        row.get("entry_point")
        or row.get("candidate")
        or row.get("name")
        or row.get("function_name")
        or row.get("function")
        or infer_entry_point_from_code(program)
        or infer_entry_point_from_code(row.get("prompt", ""))
    )


def verify_one(row: Dict[str, Any], line_no: int) -> Dict[str, Any]:
    task_id = row.get("task_id", row.get("id", f"<row {line_no}>"))
    namespace: Dict[str, Any] = {}

    try:
        if "_json_error" in row:
            raise ValueError(row["_json_error"])

        program = build_program(row)
        test_code = get_test_code(row)

        entry_point = infer_entry_point(row, program)

        if not entry_point:
            raise NameError("Could not infer entry point from row.")

        exec(program, namespace)

        if entry_point not in namespace:
            raise NameError(f"Entry point `{entry_point}` was not defined.")

        exec(test_code, namespace)

        if "check" not in namespace:
            raise NameError("Test code must define `check(candidate)`.")

        namespace["check"](namespace[entry_point])

        return {
            "task_id": task_id,
            "row": line_no,
            "type": row.get("type"),
            "entry_point": entry_point,
            "passed": True,
            "error": None,
        }

    except BaseException:
        return {
            "task_id": task_id,
            "row": line_no,
            "type": row.get("type"),
            "entry_point": locals().get("entry_point"),
            "passed": False,
            "error": traceback.format_exc(),
            "program": locals().get("program"),
        }


def verify_rows(
    rows: Iterator[Tuple[int, Dict[str, Any]]],
    out_path: str,
    show_errors: bool = False,
    stop_on_fail: bool = False,
):
    results = []

    for line_no, row in rows:
        result = verify_one(row, line_no)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['task_id']}")

        if show_errors and not result["passed"]:
            print("\n--- ERROR ---")
            print(result.get("error") or "<no error>")

            print("--- ENTRY POINT ---")
            print(result.get("entry_point") or "<no entry point>")

            print("--- PROGRAM ---")
            print(result.get("program") or "<no program>")

            print("-------------\n")

        if stop_on_fail and not result["passed"]:
            break

    with open(out_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "output_path": out_path,
    }

    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify canonical solutions in a HumanEval-like JSONL file "
            "or Hugging Face dataset."
        )
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--dataset",
        help="Path to local JSONL dataset.",
    )

    source.add_argument(
        "--hf-dataset",
        help="Hugging Face dataset name, e.g. openai/openai_humaneval.",
    )

    parser.add_argument(
        "--hf-config",
        default=None,
        help="Optional Hugging Face dataset config name.",
    )

    parser.add_argument(
        "--split",
        default="train",
        help="Hugging Face split to load. Default: train.",
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream Hugging Face dataset instead of downloading it fully.",
    )

    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for private or gated datasets.",
    )

    parser.add_argument(
        "--out",
        default="canonical_results.jsonl",
        help="Where to write per-task verification results.",
    )

    parser.add_argument(
        "--show-errors",
        action="store_true",
        help=(
            "Print traceback, inferred entry point, and generated program "
            "for failed tasks."
        ),
    )

    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop after the first failed task.",
    )

    args = parser.parse_args()

    if args.dataset:
        rows = read_jsonl(args.dataset)
    else:
        rows = read_huggingface_dataset(
            dataset_name=args.hf_dataset,
            config_name=args.hf_config,
            split=args.split,
            streaming=args.streaming,
            token=args.hf_token,
        )

    verify_rows(
        rows=rows,
        out_path=args.out,
        show_errors=args.show_errors,
        stop_on_fail=args.stop_on_fail,
    )


if __name__ == "__main__":
    main()