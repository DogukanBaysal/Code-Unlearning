#!/usr/bin/env python3
"""Run UtilityEval for common PEFT adapter subfolders sequentially."""

import argparse
import subprocess
import sys
from typing import List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate four Hugging Face PEFT adapters on UtilityEval, using "
            "the same checkpoint subfolders for each adapter."
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
        nargs=4,
        help="Four PEFT adapter Hugging Face repo IDs or local paths.",
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
        "checkpoints",
        nargs="+",
        help="Common adapter checkpoint subfolders to evaluate for every PEFT name.",
    )
    return parser


def run_eval(
    args: argparse.Namespace,
    peft_name: str,
    checkpoint: str,
    extra_args: List[str],
) -> int:
    cmd = [
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
        "utilityeval",
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

    print(f"\n=== UtilityEval: {peft_name} / {checkpoint} ===", flush=True)
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args()

    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    for peft_name in args.peft_names:
        for checkpoint in args.checkpoints:
            return_code = run_eval(args, peft_name, checkpoint, extra_args)
            if return_code != 0:
                print(
                    f"\nStopping after {peft_name!r} / {checkpoint!r}: "
                    f"evalplus exited with code {return_code}.",
                    file=sys.stderr,
                )
                return return_code

    print("\nAll UtilityEval runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
