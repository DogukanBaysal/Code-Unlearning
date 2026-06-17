import argparse
import json
from statistics import mean
from collections import Counter

from datasets import load_dataset
from radon.complexity import cc_visit


DIFFICULTY_RANGES = {
    "simple": (1, 10),
    "moderate": (10, 20),
    "complex": (20, 50),
}


def bucket_for_cc(cc: float) -> str:
    if 1 <= cc <= 10:
        return "simple"
    if 10 < cc <= 20:
        return "moderate"
    if 20 < cc <= 50:
        return "complex"
    return "out-of-range"


def format_details(named_complexities):
    """
    Convert [(name, cc), ...] into JSON-friendly detail rows.
    """
    return [
        {
            "name": name,
            "cc": cc,
            "difficulty_bucket": bucket_for_cc(cc),
        }
        for name, cc in named_complexities
    ]


def complexity_for_record(
    record: dict,
    code_column: str = "code",
    type_column: str = "type",
) -> tuple[float | None, list[tuple[str, int]]]:
    code = record.get(code_column) or ""
    item_type = record.get(type_column)

    blocks = cc_visit(code)
    named_complexities = []

    if item_type == "function":
        # For function tasks, use only the raw CC of the main top-level function.
        # Do not include nested functions, helper functions, classes, or methods.
        for block in blocks:
            if block.__class__.__name__ == "Function":
                named_complexities.append((block.name, block.complexity))
                return block.complexity, named_complexities

        return None, []

    elif item_type == "class":
        # For class tasks, average CC of all methods except __init__.
        values = []

        for block in blocks:
            if block.__class__.__name__ == "Class":
                for method in block.methods:
                    if method.name == "__init__":
                        continue

                    values.append(method.complexity)
                    named_complexities.append(
                        (f"{block.name}.{method.name}", method.complexity)
                    )

        if not values:
            return None, []

        return mean(values), named_complexities

    else:
        # Fallback: average all Radon blocks.
        values = [block.complexity for block in blocks]
        named_complexities = [(block.name, block.complexity) for block in blocks]

        if not values:
            return None, []

        return mean(values), named_complexities


def validate_hf_dataset(
    dataset_name: str,
    split: str = "train",
    code_column: str = "code",
    difficulty_column: str = "difficulty",
    type_column: str = "type",
    task_id_column: str = "task_id",
    config_name: str | None = None,
):
    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    results = []

    for index, record in enumerate(dataset):
        task_id = record.get(task_id_column, f"row-{index}")
        declared = record.get(difficulty_column)

        avg_cc, raw_details = complexity_for_record(
            record,
            code_column=code_column,
            type_column=type_column,
        )

        if avg_cc is None:
            results.append({
                "row": index,
                "task_id": task_id,
                "status": "error",
                "reason": "No analyzable function or method blocks found",
                "details": [],
            })
            continue

        expected = bucket_for_cc(avg_cc)
        ok = declared == expected

        results.append({
            "row": index,
            "task_id": task_id,
            "type": record.get(type_column),
            "declared_difficulty": declared,
            "computed_difficulty": expected,
            "average_cc": round(avg_cc, 2),
            "ok": ok,
            "details": format_details(raw_details),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate difficulty labels in a Hugging Face dataset using Radon cyclomatic complexity."
    )

    parser.add_argument(
        "dataset_name",
        help="Hugging Face dataset name, e.g. 'my-org/my-dataset'",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Optional Hugging Face dataset config name.",
    )

    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load. Default: train",
    )

    parser.add_argument(
        "--code-column",
        default="code",
        help="Column containing Python code. Default: code",
    )

    parser.add_argument(
        "--difficulty-column",
        default="difficulty",
        help="Column containing declared difficulty. Default: difficulty",
    )

    parser.add_argument(
        "--type-column",
        default="type",
        help="Column containing item type: function/class/etc. Default: type",
    )

    parser.add_argument(
        "--task-id-column",
        default="task_id",
        help="Column containing task IDs. Default: task_id",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save full results as JSON.",
    )

    args = parser.parse_args()

    results = validate_hf_dataset(
        dataset_name=args.dataset_name,
        config_name=args.config,
        split=args.split,
        code_column=args.code_column,
        difficulty_column=args.difficulty_column,
        type_column=args.type_column,
        task_id_column=args.task_id_column,
    )

    bad = [r for r in results if not r.get("ok")]

    group_counts = Counter(
        r.get("computed_difficulty", "error")
        for r in results
    )

    declared_counts = Counter(
        r.get("declared_difficulty", "missing")
        for r in results
        if "declared_difficulty" in r
    )

    print(f"Checked {len(results)} records")
    print(f"Mismatches/errors: {len(bad)}")

    print("\nComputed difficulty counts:")
    for group in ["simple", "moderate", "complex", "out-of-range", "error"]:
        if group_counts[group]:
            print(f"  {group}: {group_counts[group]}")

    print("\nDeclared difficulty counts:")
    for group in ["simple", "moderate", "complex", "out-of-range", "missing"]:
        if declared_counts[group]:
            print(f"  {group}: {declared_counts[group]}")

    print("\nMismatches/errors:")
    for r in bad:
        print(json.dumps(r, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved full results to {args.output}")


if __name__ == "__main__":
    main()