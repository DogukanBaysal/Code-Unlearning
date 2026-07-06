#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


RUN_RE = re.compile(
    r"^qwen2\.5coder-3b-unlearned-(?P<technique>[^-]+)-lr-(?P<lr>.+)-checkpoint(?P<checkpoint>\d+)$"
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def utility_path(root: Path, technique: str, lr: str, checkpoint: str) -> Path:
    pattern = (
        "Qwen--Qwen2.5-Coder-3B_hf_temp_0.0_peft_"
        f"dbaysal--qwen2.5coder-3b-unlearned-{technique}-lr-{lr}"
        f"_subfolder_checkpoint-{checkpoint}.eval_results.json"
    )
    return root / "utilityeval" / pattern


def sort_key(row: dict) -> tuple:
    return (
        row["technique"],
        float(row["learning_rate"]),
        int(row["checkpoint"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract similarity reduction and utility retention metrics."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("unlearning_similarity_utility_metrics.csv"),
    )
    args = parser.parse_args()

    root = args.root
    baseline_unlearning = read_json(root / "unlearningeval" / "baseline" / "aggregate_results.json")
    baseline_similarity = baseline_unlearning["average_similarity_score"]

    baseline_utility_path = next((root / "utilityeval" / "baseline").glob("*.eval_results.json"))
    baseline_utility = read_json(baseline_utility_path)["pass_at_k"]["base"]["pass@1"]

    rows = []
    missing_utility = []
    for aggregate_path in sorted((root / "unlearningeval").glob("*/aggregate_results.json")):
        run_name = aggregate_path.parent.name
        if run_name == "baseline":
            continue

        match = RUN_RE.match(run_name)
        if not match:
            raise ValueError(f"Unexpected unlearning run folder name: {run_name}")

        technique = match.group("technique")
        lr = match.group("lr")
        checkpoint = match.group("checkpoint")

        unlearning = read_json(aggregate_path)
        similarity = unlearning["average_similarity_score"]
        similarity_reduction = baseline_similarity - similarity

        util_path = utility_path(root, technique, lr, checkpoint)
        if not util_path.exists():
            missing_utility.append(str(util_path))
            utility_score = None
            utility_retention = None
        else:
            utility_score = read_json(util_path)["pass_at_k"]["base"]["pass@1"]
            utility_retention = utility_score / baseline_utility if baseline_utility else None

        rows.append(
            {
                "technique": technique,
                "learning_rate": lr,
                "checkpoint": checkpoint,
                "similarity_score": similarity,
                "baseline_similarity_score": baseline_similarity,
                "average_similarity_score_reduction": similarity_reduction,
                "average_similarity_score_reduction_percent": (
                    similarity_reduction / baseline_similarity * 100
                    if baseline_similarity
                    else None
                ),
                "utility_pass_at_1": utility_score,
                "baseline_utility_pass_at_1": baseline_utility,
                "utility_retention": utility_retention,
                "utility_retention_percent": (
                    utility_retention * 100 if utility_retention is not None else None
                ),
                "unlearningeval_file": str(aggregate_path),
                "utilityeval_file": str(util_path) if util_path.exists() else "",
            }
        )

    fieldnames = [
        "technique",
        "learning_rate",
        "checkpoint",
        "similarity_score",
        "baseline_similarity_score",
        "average_similarity_score_reduction",
        "average_similarity_score_reduction_percent",
        "utility_pass_at_1",
        "baseline_utility_pass_at_1",
        "utility_retention",
        "utility_retention_percent",
        "unlearningeval_file",
        "utilityeval_file",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=sort_key))

    if missing_utility:
        print("Missing utility eval files:")
        for path in missing_utility:
            print(path)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
