#!/usr/bin/env python3
"""Run suffix unlearning and EvalPlus checks for PEFT adapter checkpoints."""

from __future__ import annotations

import argparse
import json
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
            "evaluation, retain suffix evaluation, approximate suffix evaluation, and EvalPlus "
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
        default=None,
        help="PEFT adapter checkpoint subfolders to evaluate.",
    )
    parser.add_argument(
        "--discover-checkpoints",
        action="store_true",
        help=(
            "Discover checkpoint-* subfolders from each local adapter directory or "
            "Hugging Face Hub model repo instead of using --checkpoints/defaults."
        ),
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=3,
        help="Number of discovered checkpoint subfolders to evaluate.",
    )
    parser.add_argument(
        "--alias-checkpoints-as-epochs",
        action="store_true",
        help="Use epoch-1, epoch-2, ... in evaluation output paths instead of checkpoint names.",
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
        "--approx-dataset",
        default="dbaysal/approximate",
        help="Hugging Face dataset used for approximate suffix evaluation.",
    )
    parser.add_argument(
        "--approx-prefix-column",
        default="prefix",
        help="Approximate dataset prefix column.",
    )
    parser.add_argument(
        "--approx-suffix-column",
        default="suffix",
        help="Approximate dataset suffix/reference column.",
    )
    parser.add_argument(
        "--approx-mode",
        choices=["secret", "code"],
        default="code",
        help="Scoring mode for approximate suffix evaluation.",
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
        default=64,
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


def checkpoint_sort_key(value: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"checkpoint-(\d+)", value)
    if match:
        return (0, int(match.group(1)))
    return (1, value)


def discover_local_checkpoints(adapter_path: Path) -> list[str]:
    if not adapter_path.exists() or not adapter_path.is_dir():
        return []
    return sorted(
        (
            child.name
            for child in adapter_path.iterdir()
            if child.is_dir() and re.fullmatch(r"checkpoint-\d+", child.name)
        ),
        key=checkpoint_sort_key,
    )


def discover_hf_checkpoints(repo_id: str) -> list[str]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint discovery for Hugging Face repos requires `huggingface_hub`."
        ) from exc

    try:
        repo_files = list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        raise RuntimeError(f"Could not list Hugging Face repo files for {repo_id!r}: {exc}") from exc

    checkpoints = {
        path.split("/", 1)[0]
        for path in repo_files
        if re.match(r"checkpoint-\d+/", path)
    }
    return sorted(checkpoints, key=checkpoint_sort_key)


def resolve_checkpoints(args: argparse.Namespace, peft_name: str) -> list[str]:
    if args.checkpoints:
        return list(args.checkpoints)

    if not args.discover_checkpoints:
        return list(DEFAULT_CHECKPOINTS)

    if args.num_checkpoints <= 0:
        raise ValueError("--num-checkpoints must be greater than 0")

    checkpoints = discover_local_checkpoints(Path(peft_name).expanduser())
    if not checkpoints:
        checkpoints = discover_hf_checkpoints(peft_name)

    if len(checkpoints) < args.num_checkpoints:
        raise RuntimeError(
            f"Found only {len(checkpoints)} checkpoint folder(s) for {peft_name!r}; "
            f"need {args.num_checkpoints}. Pass --checkpoints explicitly if discovery "
            "is unavailable or the repo uses non-standard folder names."
        )
    return checkpoints[: args.num_checkpoints]


def checkpoint_output_name(checkpoint: str, index: int, alias_as_epochs: bool) -> str:
    if alias_as_epochs:
        return f"epoch-{index}"
    return checkpoint


def write_suffix_config(
    *,
    path: Path,
    model: str,
    peft_name: str,
    checkpoint: str,
    uuid_column: str,
    dataset_split: str,
    suffix_runs: tuple[dict[str, Any], ...],
    max_new_tokens: int,
    batch_size: int,
    trust_remote_code: bool,
) -> None:
    config: dict[str, Any] = {
        "model_name": model,
        "peft_name": peft_name,
        "peft_subfolder": checkpoint,
        "trust_remote_code": trust_remote_code,
        "datasets": [
            {
                "label": suffix_run["label"],
                "dataset_name": suffix_run["dataset"],
                "dataset_split": dataset_split,
                "prefix_column": suffix_run["prefix_column"],
                "suffix_column": suffix_run["suffix_column"],
                "uuid_column": uuid_column,
                "mode": suffix_run["mode"],
                "code_language": "python",
                "output_dir": str(suffix_run["output_dir"]),
            }
            for suffix_run in suffix_runs
        ],
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
    for suffix_run in suffix_runs:
        Path(suffix_run["output_dir"]).mkdir(parents=True, exist_ok=True)
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
    checkpoint_manifest: dict[str, dict[str, str]] = {}

    for peft_name in args.peft_names:
        adapter_slug = slug(peft_name)
        try:
            checkpoints = resolve_checkpoints(args, peft_name)
        except Exception as exc:
            print(f"Checkpoint resolution failed for {peft_name!r}: {exc}", file=sys.stderr)
            return 1

        for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
            checkpoint_alias = checkpoint_output_name(
                checkpoint,
                checkpoint_index,
                args.alias_checkpoints_as_epochs,
            )
            checkpoint_manifest.setdefault(peft_name, {})[checkpoint_alias] = checkpoint
            checkpoint_slug = slug(checkpoint_alias)
            run_slug = f"{adapter_slug}_{checkpoint_slug}"

            suffix_runs = (
                {
                    "label": "forget",
                    "dataset": args.forget_dataset,
                    "prefix_column": args.forget_prefix_column,
                    "suffix_column": args.forget_suffix_column,
                    "mode": args.forget_mode,
                    "output_dir": suffix_root / "forget" / run_slug,
                },
                {
                    "label": "retain",
                    "dataset": args.retain_dataset,
                    "prefix_column": args.retain_prefix_column,
                    "suffix_column": args.retain_suffix_column,
                    "mode": args.retain_mode,
                    "output_dir": suffix_root / "retain" / run_slug,
                },
                {
                    "label": "approximate",
                    "dataset": args.approx_dataset,
                    "prefix_column": args.approx_prefix_column,
                    "suffix_column": args.approx_suffix_column,
                    "mode": args.approx_mode,
                    "output_dir": suffix_root / "approximate" / run_slug,
                },
            )

            config_path = config_root / run_slug / "suffix.yaml"
            write_suffix_config(
                path=config_path,
                model=args.model,
                peft_name=peft_name,
                checkpoint=checkpoint,
                uuid_column=args.uuid_column,
                dataset_split=args.dataset_split,
                suffix_runs=suffix_runs,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.suffix_bs,
                trust_remote_code=args.trust_remote_code,
            )

            command_name = f"{peft_name} / {checkpoint} / suffix"
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

            evalplus_result_path = (
                evalplus_root / args.evalplus_dataset / f"{run_slug}.eval_results.json"
            )
            evalplus_result_path.parent.mkdir(parents=True, exist_ok=True)
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
                "--output-file",
                str(evalplus_result_path),
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

    if checkpoint_manifest:
        manifest_path = output_root / "checkpoint_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(checkpoint_manifest, handle, indent=2, sort_keys=True)

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for command_name, return_code in failures:
            print(f"- {command_name}: exit code {return_code}", file=sys.stderr)
        return 1

    print("\nAll evaluation runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
