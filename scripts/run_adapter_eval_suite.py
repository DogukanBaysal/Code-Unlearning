#!/usr/bin/env python3
"""Run suffix unlearning and EvalPlus checks for PEFT adapter checkpoints."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
UNLEARNING_EVAL_DIR = REPO_ROOT / "UnlearningEvaluation"
EVALPLUS_DIR = REPO_ROOT / "evalplus"
DEFAULT_CHECKPOINTS = ("checkpoint-4", "checkpoint-8", "checkpoint-12")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate any number of Hugging Face PEFT adapters across checkpoint "
            "subfolders. For each adapter/checkpoint pair this runs forget suffix "
            "evaluation, retain suffix evaluation, and EvalPlus "
            "HumanEval+ForgetEval+UtilityEval."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base Hugging Face model ID or local path.",
    )
    parser.add_argument(
        "--peft-names",
        required=True,
        nargs="+",
        help="One or more PEFT adapter Hugging Face repo IDs or local paths.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
        help="PEFT adapter checkpoint subfolders to evaluate.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "Results" / "adapter_eval_suite"),
        help="Root directory for generated configs and evaluation outputs.",
    )
    parser.add_argument(
        "--forget-dataset",
        default="dbaysal/forget",
        help="Hugging Face dataset used for forget suffix evaluation.",
    )
    parser.add_argument(
        "--forget-prefix-column",
        default="secret_prefix",
        help="Forget dataset prefix column.",
    )
    parser.add_argument(
        "--forget-suffix-column",
        default="secret_suffix",
        help="Forget dataset suffix/reference column.",
    )
    parser.add_argument(
        "--retain-dataset",
        default="dbaysal/retain-half",
        help="Hugging Face dataset used for retain suffix evaluation.",
    )
    parser.add_argument(
        "--retain-prefix-column",
        default="prefix",
        help="Retain dataset prefix column.",
    )
    parser.add_argument(
        "--retain-suffix-column",
        default="suffix",
        help="Retain dataset suffix/reference column.",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split for both suffix evaluations.",
    )
    parser.add_argument(
        "--uuid-column",
        default="uuid",
        help="UUID/id column for suffix evaluations.",
    )
    parser.add_argument(
        "--forget-mode",
        choices=["secret", "code"],
        default="secret",
        help="Scoring mode for forget suffix evaluation.",
    )
    parser.add_argument(
        "--retain-mode",
        choices=["secret", "code"],
        default="code",
        help="Scoring mode for retain suffix evaluation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2056,
        help="Max new tokens for both suffix evaluations.",
    )
    parser.add_argument(
        "--suffix-bs",
        type=int,
        default=32,
        help="Batch size for suffix evaluations.",
    )
    parser.add_argument(
        "--evalplus-bs",
        type=int,
        default=500,
        help="EvalPlus generation batch size.",
    )
    parser.add_argument(
        "--evalplus-dataset",
        default="humaneval-forget-utility",
        help="EvalPlus dataset or alias to run.",
    )
    parser.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "hf_gaudi"],
        help="EvalPlus Hugging Face backend.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="EvalPlus model dtype.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code to both model loaders.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later runs after a failed command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write suffix configs without running evaluations.",
    )
    parser.add_argument(
        "extra_evalplus_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to evalplus.evaluate after a '--' separator.",
    )
    return parser


def slug(value: str) -> str:
    value = value.strip().strip("./")
    value = re.sub(r"[^A-Za-z0-9._-]+", "--", value)
    return value.strip("-") or "adapter"


def write_suffix_config(
    *,
    path: Path,
    model: str,
    peft_name: str,
    checkpoint: str,
    dataset_name: str,
    prefix_column: str,
    suffix_column: str,
    uuid_column: str,
    dataset_split: str,
    mode: str,
    output_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    trust_remote_code: bool,
) -> None:
    config: dict[str, Any] = {
        "model_name": model,
        "peft_name": peft_name,
        "peft_subfolder": checkpoint,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "prefix_column": prefix_column,
        "suffix_column": suffix_column,
        "uuid_column": uuid_column,
        "mode": mode,
        "code_language": "python",
        "trust_remote_code": trust_remote_code,
        "output_dir": str(output_dir),
        "generation": {
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "device": "auto",
            "dtype": "auto",
            "greedy": True,
            "do_sample": False,
            "temperature": 0.2,
            "top_p": 0.8,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def print_command(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ cd {cwd}")
    print("$ " + " ".join(cmd), flush=True)


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> int:
    print_command(cmd, cwd)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def evalplus_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    paths = [str(EVALPLUS_DIR)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def normalize_extra_evalplus_args(raw: list[str]) -> list[str]:
    if raw and raw[0] == "--":
        return raw[1:]
    return raw


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    extra_evalplus_args = normalize_extra_evalplus_args(args.extra_evalplus_args)

    output_root = Path(args.output_root).expanduser().resolve()
    config_root = output_root / "configs"
    suffix_root = output_root / "unlearningeval"
    evalplus_root = output_root / "evalplus"

    failures: list[tuple[str, int]] = []

    for peft_name in args.peft_names:
        adapter_slug = slug(peft_name)
        for checkpoint in args.checkpoints:
            checkpoint_slug = slug(checkpoint)
            run_slug = f"{adapter_slug}_{checkpoint_slug}"

            suffix_runs = (
                {
                    "label": "forget",
                    "dataset": args.forget_dataset,
                    "prefix_column": args.forget_prefix_column,
                    "suffix_column": args.forget_suffix_column,
                    "mode": args.forget_mode,
                },
                {
                    "label": "retain",
                    "dataset": args.retain_dataset,
                    "prefix_column": args.retain_prefix_column,
                    "suffix_column": args.retain_suffix_column,
                    "mode": args.retain_mode,
                },
            )

            for suffix_run in suffix_runs:
                label = suffix_run["label"]
                config_path = config_root / run_slug / f"{label}.yaml"
                output_dir = suffix_root / label / run_slug
                write_suffix_config(
                    path=config_path,
                    model=args.model,
                    peft_name=peft_name,
                    checkpoint=checkpoint,
                    dataset_name=suffix_run["dataset"],
                    prefix_column=suffix_run["prefix_column"],
                    suffix_column=suffix_run["suffix_column"],
                    uuid_column=args.uuid_column,
                    dataset_split=args.dataset_split,
                    mode=suffix_run["mode"],
                    output_dir=output_dir,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.suffix_bs,
                    trust_remote_code=args.trust_remote_code,
                )

                command_name = f"{peft_name} / {checkpoint} / suffix-{label}"
                return_code = run_command(
                    [
                        sys.executable,
                        "evaluate_suffix_generation.py",
                        "--config",
                        str(config_path),
                    ],
                    cwd=UNLEARNING_EVAL_DIR,
                    dry_run=args.dry_run,
                )
                if return_code != 0:
                    failures.append((command_name, return_code))
                    if not args.continue_on_error:
                        return return_code

            evalplus_cmd = [
                sys.executable,
                "-m",
                "evalplus.evaluate",
                "--model",
                args.model,
                "--peft-name",
                peft_name,
                "--peft-subfolder",
                checkpoint,
                "--dataset",
                args.evalplus_dataset,
                "--backend",
                args.backend,
                "--greedy",
                "--defer-sanitize",
                "--bs",
                str(args.evalplus_bs),
                "--force-base-prompt",
                "--root",
                str(evalplus_root),
                "--dtype",
                args.dtype,
            ]
            if args.trust_remote_code:
                evalplus_cmd.append("--trust-remote-code")
            evalplus_cmd.extend(extra_evalplus_args)

            command_name = f"{peft_name} / {checkpoint} / evalplus"
            return_code = run_command(
                evalplus_cmd,
                cwd=REPO_ROOT,
                dry_run=args.dry_run,
                env=evalplus_env(),
            )
            if return_code != 0:
                failures.append((command_name, return_code))
                if not args.continue_on_error:
                    return return_code

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for command_name, return_code in failures:
            print(f"- {command_name}: exit code {return_code}", file=sys.stderr)
        return 1

    print("\nAll evaluation runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
