#!/usr/bin/env python3

import argparse
import json
from datasets import load_dataset
from tqdm import tqdm


def check_python_compiles(code: str) -> dict:
    """
    Check whether a string is valid compilable Python code.

    Returns:
        {
            "valid": bool,
            "error_type": str | None,
            "error_message": str | None,
            "lineno": int | None,
            "offset": int | None
        }
    """
    if not isinstance(code, str):
        return {
            "valid": False,
            "error_type": "TypeError",
            "error_message": f"Expected string, got {type(code).__name__}",
            "lineno": None,
            "offset": None,
        }

    try:
        compile(code, "<dataset-content>", "exec")
        return {
            "valid": True,
            "error_type": None,
            "error_message": None,
            "lineno": None,
            "offset": None,
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "error_type": type(e).__name__,
            "error_message": e.msg,
            "lineno": e.lineno,
            "offset": e.offset,
        }
    except Exception as e:
        return {
            "valid": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "lineno": None,
            "offset": None,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Check whether the content column of a Hugging Face dataset contains compilable Python code."
    )
    parser.add_argument(
        "--dataset",
        default="dbaysal/all-content",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to check",
    )
    parser.add_argument(
        "--content-column",
        default="content",
        help="Column containing Python code",
    )
    parser.add_argument(
        "--output",
        default="compile_report.jsonl",
        help="Path to output JSONL report",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset instead of downloading it fully",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to check",
    )

    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
    )

    total = 0
    valid_count = 0
    invalid_count = 0

    with open(args.output, "w", encoding="utf-8") as f:
        iterator = dataset if args.streaming else tqdm(dataset)

        for idx, row in enumerate(iterator):
            if args.max_rows is not None and idx >= args.max_rows:
                break

            code = row.get(args.content_column)
            result = check_python_compiles(code)

            total += 1
            if result["valid"]:
                valid_count += 1
            else:
                invalid_count += 1

            output_row = {
                "row_index": idx,
                "valid": result["valid"],
                "error_type": result["error_type"],
                "error_message": result["error_message"],
                "lineno": result["lineno"],
                "offset": result["offset"],
            }

            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Total checked: {total}")
    print(f"Valid Python files: {valid_count}")
    print(f"Invalid Python files: {invalid_count}")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()