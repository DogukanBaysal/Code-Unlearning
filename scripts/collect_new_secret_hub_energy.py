#!/usr/bin/env python3
"""Collect CodeCarbon duration and GPU energy from the 54 new secret runs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


MODELS = ("qwen2_5_coder_3b", "meta_llama3_2_3b")
STANDARD_METHODS = (
    "ga",
    "npo",
    "prod",
    "ga_gd",
    "ga_kl",
    "npo_gd",
    "npo_kl",
    "prod_gd",
    "prod_kl",
)
ORDERED_METHODS = ("ga_gd", "ga_kl", "npo_gd", "npo_kl", "prod_gd", "prod_kl")
ORDERED_VARIANTS = ("retain-first", "forget-first", "random-retain-full")
FIELDNAMES = (
    "repo_id",
    "model",
    "method",
    "variant",
    "duration_seconds",
    "duration_hours",
    "gpu_energy_kwh",
    "total_energy_kwh",
    "gpu_count",
    "gpu_model",
    "timestamp",
    "codecarbon_rows",
    "status",
    "error",
)


def expected_repositories(namespace: str, prefix: str):
    for model in MODELS:
        for method in STANDARD_METHODS:
            name = f"{prefix}secret-unlearning-{model}-{method}"
            yield namespace, name, model, method, "standard"

    for variant in ORDERED_VARIANTS:
        for model in MODELS:
            for method in ORDERED_METHODS:
                name = f"{prefix}secret-unlearning-{model}-{method}-{variant}"
                yield namespace, name, model, method, variant


def optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def collect_row(repo_id: str, model: str, method: str, variant: str, args):
    result = {
        "repo_id": repo_id,
        "model": model,
        "method": method,
        "variant": variant,
        "status": "ok",
        "error": "",
    }
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=args.emissions_path,
            repo_type="model",
            revision=args.revision,
            token=args.token,
        )
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("CodeCarbon CSV contains no data rows")

        latest = rows[-1]
        duration = optional_float(latest.get("duration"))
        gpu_energy = optional_float(latest.get("gpu_energy"))
        total_energy = optional_float(latest.get("energy_consumed"))
        if duration is None or gpu_energy is None:
            raise ValueError("CodeCarbon CSV is missing duration or gpu_energy")

        result.update(
            {
                "duration_seconds": duration,
                "duration_hours": duration / 3600.0,
                "gpu_energy_kwh": gpu_energy,
                "total_energy_kwh": total_energy,
                "gpu_count": latest.get("gpu_count", ""),
                "gpu_model": latest.get("gpu_model", ""),
                "timestamp": latest.get("timestamp", ""),
                "codecarbon_rows": len(rows),
            }
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = " ".join(str(exc).splitlines())
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        default=os.getenv("HUB_NAMESPACE", "dbaysal"),
        help="Hugging Face namespace (default: HUB_NAMESPACE or dbaysal).",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("MODEL_NAME_PREFIX", "new-"),
        help="Repository-name prefix (default: MODEL_NAME_PREFIX or new-).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("new_secret_training_energy.csv"),
        help="Aggregate CSV output path.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub revision to read (default: main).",
    )
    parser.add_argument(
        "--emissions-path",
        default="emissions/emissions.csv",
        help="Path of the CodeCarbon CSV inside each repository.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token; defaults to HF_TOKEN or cached login.",
    )
    parser.add_argument(
        "--list-repos",
        action="store_true",
        help="Print the expected repository IDs without downloading anything.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a nonzero exit status if any repository cannot be collected.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = list(expected_repositories(args.namespace, args.prefix))
    if args.list_repos:
        for namespace, name, _, _, _ in specs:
            print(f"{namespace}/{name}")
        return 0

    results = []
    for index, (namespace, name, model, method, variant) in enumerate(specs, 1):
        repo_id = f"{namespace}/{name}"
        row = collect_row(repo_id, model, method, variant, args)
        results.append(row)
        print(f"[{index:02d}/{len(specs)}] {repo_id}: {row['status']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    failures = sum(row["status"] != "ok" for row in results)
    print(f"Wrote {len(results)} rows to {args.output} ({failures} errors).")
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
