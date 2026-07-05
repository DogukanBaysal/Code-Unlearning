#!/usr/bin/env python3
"""Run HumanEval for three PEFT adapter subfolders sequentially."""

import argparse
import subprocess
import sys
from typing import List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exactly three Hugging Face PEFT adapter subfolders on "
            "HumanEval one after another."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base Hugging Face model ID or local path.",
    )
    parser.add_argument(
        "--peft-name",
        required=True,
        help="PEFT adapter Hugging Face repo ID or local path.",
    )
    parser.add_argument(
        "--root",
        default="evalplus_results",
        help="Directory where EvalPlus writes generated samples and results.",
    )
    parser.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "hf_gaudi"],
        help="Hugging Face backend to use.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=200,
        help="EvalPlus generation batch size.",
    )
    parser.add_argument(
        "subfolders",
        nargs=3,
        help="The three adapter subfolders to evaluate, in order.",
    )
    return parser


def run_eval(args: argparse.Namespace, subfolder: str, extra_args: List[str]) -> int:
    cmd = [
        sys.executable,
        "-m",
        "evalplus.evaluate",
        "--model",
        args.model,
        "--peft-name",
        args.peft_name,
        "--peft-subfolder",
        subfolder,
        "--dataset",
        "humaneval",
        "--backend",
        args.backend,
        "--greedy",
        "--defer-sanitize",
        "--bs",
        str(args.bs),
        "--force-base-prompt",
        "--root",
        args.root,
        *extra_args,
    ]

    print(f"\n=== HumanEval: {subfolder} ===", flush=True)
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args()

    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    for subfolder in args.subfolders:
        return_code = run_eval(args, subfolder, extra_args)
        if return_code != 0:
            print(
                f"\nStopping after {subfolder!r}: evalplus exited with "
                f"code {return_code}.",
                file=sys.stderr,
            )
            return return_code

    print("\nAll three HumanEval runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
