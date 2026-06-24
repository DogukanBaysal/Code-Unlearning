#!/usr/bin/env python3
"""
Mutation testing pipeline for HuggingFace datasets using mutmut.

For each dataset row:
  1. Writes the code to source.py
  2. Adapts the humaneval-style check() test to pytest
  3. Verifies tests pass on the original code
  4. Runs mutmut and collects survived mutants
  5. Emits mutation_report.json + mutation_report.md

Usage:
    pip install mutmut datasets pytest
    python run_mutation_testing.py --dataset your/dataset --split train --limit 10

The output report is designed to be fed to an agent that will write additional
tests to kill the surviving mutants.
"""

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mutmut_cmd(*args: str) -> list[str]:
    """
    Run mutmut through the same Python interpreter that runs this script.

    This avoids PATH problems where `pip install mutmut` succeeds but the
    `mutmut` executable is not visible as a shell command.
    """
    return [sys.executable, "-m", "mutmut", *args]


def extract_entry_point(code: str) -> Optional[str]:
    """Return the name of the first top-level function OR class defined in code.

    Some dataset samples define a class instead of a function, so we must accept both.
    """
    try:
        tree = ast.parse(code)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return node.name
    except SyntaxError:
        pass

    m = re.search(r"^(?:def|class)\s+(\w+)", code, re.MULTILINE)
    return m.group(1) if m else None


def humaneval_to_pytest(test_code: str, entry_point: str, module: str = "source") -> str:
    """
    Wrap a humaneval-style check(candidate) block into a pytest test.
    """
    return (
        f"import pytest\n"
        f"from {module} import {entry_point}\n\n\n"
        f"{test_code.strip()}\n\n\n"
        f"def test_all():\n"
        f"    check({entry_point})\n"
    )


def run_subprocess(cmd: list[str], cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def load_rows(dataset: str, split: str = "train") -> list[dict]:
    """
    Load samples as a list of dicts.

    - A local `.jsonl` / `.json` path is read directly.
    - Anything else is treated as a HuggingFace dataset name.
    """
    path = Path(dataset)
    if path.exists() and path.suffix in (".jsonl", ".json"):
        text = path.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if stripped.startswith("["):
            return json.loads(text)
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not found. Install with:  pip install datasets")
        sys.exit(1)

    return list(load_dataset(dataset, split=split))


# ---------------------------------------------------------------------------
# mutmut result parsing
# ---------------------------------------------------------------------------

def parse_mutmut_run_output(run_log: str) -> dict:
    """
    Parse final counts from mutmut progress output.

    Example:
        31/31  🎉 26 🫥 0  ⏰ 1  🤔 0  🙁 4  🔇 0  🧙 0
    """
    counts = {
        "killed": 0,
        "survived": 0,
        "timeout": 0,
        "suspicious": 0,
        "no_coverage": 0,
        "total": 0,
    }

    progress_lines = [line for line in run_log.splitlines() if re.search(r"\d+/\d+.*🎉", line)]
    if not progress_lines:
        return counts

    last = progress_lines[-1]

    m = re.search(r"(\d+)/(\d+)", last)
    if m:
        counts["total"] = int(m.group(2))

    for emoji, key in [
        ("🎉", "killed"),
        ("🙁", "survived"),
        ("⏰", "timeout"),
        ("🤔", "suspicious"),
        ("🫥", "no_coverage"),
    ]:
        m = re.search(rf"{emoji}\s*(\d+)", last)
        if m:
            counts[key] = int(m.group(1))

    return counts


def parse_mutmut_results_output(results_text: str) -> dict:
    """
    Fallback parser for older mutmut output.

    Example:
        Killed 🎉 (10)
        Survived 🙁 (2)
    """
    counts = {
        "killed": 0,
        "survived": 0,
        "timeout": 0,
        "suspicious": 0,
        "no_coverage": 0,
        "total": 0,
    }

    for key in ("killed", "survived", "timeout", "suspicious", "no_coverage"):
        m = re.search(rf"^{key}[^\n(]*\((\d+)\)", results_text, re.IGNORECASE | re.MULTILINE)
        if m:
            counts[key] = int(m.group(1))

    counts["total"] = (
        counts["killed"]
        + counts["survived"]
        + counts["timeout"]
        + counts["suspicious"]
        + counts["no_coverage"]
    )
    return counts


def parse_results_by_status(results_text: str) -> dict:
    """
    Parse mutmut 3.x `mutmut results` output into {status: [mutant_id, ...]}.

    Example:
        source.x_my_func__mutmut_1: survived
    """
    by_status: dict[str, list[str]] = {}
    line_re = re.compile(r"^\s*(\S*__mutmut_\d+):\s*([a-z][a-z ]*?)\s*$")

    for line in results_text.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        by_status.setdefault(m.group(2).strip(), []).append(m.group(1))

    return by_status


def get_ids_for_status(results_text: str, status: str) -> list[str]:
    """
    Extract mutant IDs for a given status label from `mutmut results`.
    """
    by_status = parse_results_by_status(results_text)

    if by_status:
        aliases = {
            "survived": ["survived"],
            "no coverage": ["no tests", "no coverage"],
        }

        for s in aliases.get(status.strip().lower(), [status.strip().lower()]):
            if by_status.get(s):
                return by_status[s]

        return []

    block2 = re.search(
        rf"^{status}[^\n]*:\s*\n(.*?)(?=^\S|\Z)",
        results_text,
        re.DOTALL | re.MULTILINE,
    )
    if block2:
        ids = re.findall(r"[\w./]+:\d+", block2.group(1))
        if ids:
            return ids

    block1 = re.search(
        rf"^{status}[^\n]*\n(.*?)(?=^(?:Killed|Survived|Timeout|Suspicious|No coverage|To apply)|\Z)",
        results_text,
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    if block1:
        return re.findall(r"\d+", block1.group(1))

    return []


def get_survived_ids(results_text: str) -> list[str]:
    return get_ids_for_status(results_text, "Survived")


def get_no_coverage_ids(results_text: str) -> list[str]:
    return get_ids_for_status(results_text, "No coverage")


def get_mutant_diff(mutant_id: str, work_dir: Path) -> str:
    proc = run_subprocess(mutmut_cmd("show", mutant_id), cwd=work_dir)
    return proc.stdout.strip()


def is_likely_equivalent(diff: str) -> tuple[bool, str]:
    """
    Heuristic detection of equivalent, unkillable mutants.
    """
    changed = [
        line[1:]
        for line in diff.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]

    if not changed:
        return False, ""

    str_assign = re.compile(r'^\s*\w+\s*=\s*(["\']).*\1\s*$')
    if all(str_assign.match(line) for line in changed):
        return True, "string literal assigned to a local variable that tests cannot observe"

    if all(line.strip() == "pass" for line in changed):
        return True, "only a 'pass' statement added/removed — no observable effect"

    return False, ""


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_sample(
    idx: int,
    code: str,
    test: str,
    output_dir: Path,
    tests_dir: Optional[Path] = None,
    entry_point: Optional[str] = None,
) -> dict:
    base = {
        "idx": idx,
        "function_name": None,
        "error": None,
        "killed": 0,
        "survived": 0,
        "timeout": 0,
        "no_coverage": 0,
        "total_mutants": 0,
        "mutation_score_pct": None,
        "survived_mutants": [],
        "no_coverage_mutants": [],
        "work_dir": None,
    }

    func_name = entry_point or extract_entry_point(code)

    if not func_name:
        base["error"] = "Could not extract entry point (function/class) from code"
        return base

    base["function_name"] = func_name

    work_dir = output_dir / f"sample_{idx:04d}"

    if work_dir.exists():
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True)
    base["work_dir"] = str(work_dir)

    # Write source.
    (work_dir / "source.py").write_text(code, encoding="utf-8")

    # Write adapted test.
    custom_test_path = tests_dir / f"sample_{idx:04d}.py" if tests_dir else None

    if custom_test_path and custom_test_path.exists():
        pytest_test = custom_test_path.read_text(encoding="utf-8")
    else:
        try:
            pytest_test = humaneval_to_pytest(test, func_name)
        except Exception as e:
            base["error"] = f"Test adaptation failed: {e}"
            return base

        if tests_dir:
            (tests_dir / f"sample_{idx:04d}.py").write_text(pytest_test, encoding="utf-8")

    (work_dir / "test_source.py").write_text(pytest_test, encoding="utf-8")

    # mutmut config.
    runner_cmd = f"{sys.executable} -m pytest test_source.py -x -q --tb=short"

    (work_dir / "pyproject.toml").write_text(
        "[tool.mutmut]\n"
        'paths_to_mutate = ["source.py"]\n'
        f'runner = "{runner_cmd}"\n',
        encoding="utf-8",
    )

    (work_dir / "setup.cfg").write_text(
        "[mutmut]\n"
        "paths_to_mutate=source.py\n"
        f"runner={runner_cmd}\n"
        "tests_dir=.\n",
        encoding="utf-8",
    )

    # Verify tests pass on original code.
    verify = run_subprocess(
        [sys.executable, "-m", "pytest", "test_source.py", "-q", "--tb=short"],
        cwd=work_dir,
        timeout=30,
    )

    if verify.returncode != 0:
        base["error"] = "Tests fail on original code — skipping mutation"
        base["test_output"] = (verify.stdout + verify.stderr)[:2000]
        return base

    # Run mutmut.
    try:
        run_proc = run_subprocess(
            mutmut_cmd("run"),
            cwd=work_dir,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        base["error"] = "mutmut run timed out (180 s)"
        return base

    raw_run_output = run_proc.stdout + run_proc.stderr
    (work_dir / "mutmut_run.log").write_text(raw_run_output, encoding="utf-8")

    # mutmut exit code 1 can mean survived mutants, which is not a script error.
    # Exit code 2 usually means config / usage error.
    if run_proc.returncode == 2:
        base["error"] = f"mutmut config/usage error: {raw_run_output[:500]}"
        return base

    # Collect results.
    counts = parse_mutmut_run_output(raw_run_output)

    results_proc = run_subprocess(mutmut_cmd("results"), cwd=work_dir)
    results_text = results_proc.stdout
    (work_dir / "mutmut_results.log").write_text(results_text, encoding="utf-8")

    if counts["total"] == 0:
        counts = parse_mutmut_results_output(results_text)

    base.update(counts)
    base["total_mutants"] = counts["total"]

    if counts["total"] > 0:
        base["mutation_score_pct"] = round(counts["killed"] / counts["total"] * 100, 1)

    # Collect survived mutant diffs.
    survived_ids = get_survived_ids(results_text)

    for mid in survived_ids[:50]:
        diff = get_mutant_diff(mid, work_dir)
        equiv, reason = is_likely_equivalent(diff)

        base["survived_mutants"].append(
            {
                "id": mid,
                "diff": diff,
                "equivalent": equiv,
                "equivalent_reason": reason,
            }
        )

    # Collect no-coverage mutant diffs.
    no_cov_ids = get_no_coverage_ids(results_text)

    for mid in no_cov_ids[:50]:
        diff = get_mutant_diff(mid, work_dir)
        base["no_coverage_mutants"].append(
            {
                "id": mid,
                "diff": diff,
            }
        )

    return base


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], output_dir: Path, verbose: bool = True) -> None:
    json_path = output_dir / "mutation_report.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    total_mutants = sum(r["total_mutants"] for r in ok)
    total_killed = sum(r["killed"] for r in ok)
    total_survived = sum(r["survived"] for r in ok)
    overall_score = (total_killed / total_mutants * 100) if total_mutants else 0

    lines = [
        "# Mutation Testing Report",
        "",
        f"**Samples processed:** {len(results)} ({len(ok)} OK, {len(failed)} errored)",
        f"**Total mutants:** {total_mutants}",
        f"**Killed:** {total_killed}",
        f"**Survived:** {total_survived}",
        f"**Overall mutation score:** {overall_score:.1f}%",
        "",
        "## Per-Sample Summary",
        "",
        "| # | Function | Total | Killed | Survived | Score | Error |",
        "|---|----------|-------|--------|----------|-------|-------|",
    ]

    for r in results:
        score = f"{r['mutation_score_pct']}%" if r["mutation_score_pct"] is not None else "—"
        err = r["error"][:60] if r["error"] else ""

        lines.append(
            f"| {r['idx']} | `{r['function_name'] or '?'}` "
            f"| {r['total_mutants']} | {r['killed']} | {r['survived']} "
            f"| {score} | {err} |"
        )

    survived_samples = [r for r in ok if r["survived"] > 0]

    if survived_samples:
        lines += [
            "",
            "---",
            "",
            "## Survived Mutants",
            "",
            "These mutants were NOT killed by the existing tests.",
            "An agent should write additional test cases to kill each one.",
            "",
        ]

        for r in survived_samples:
            lines += [
                f"### Sample {r['idx']} — `{r['function_name']}`",
                "",
                f"Work dir: `{r['work_dir']}`",
                "",
            ]

            for m in r["survived_mutants"]:
                lines += [
                    f"#### Mutant #{m['id']}",
                    "",
                    f"Equivalent heuristic: `{m['equivalent']}`",
                    "",
                ]

                if m["equivalent_reason"]:
                    lines += [
                        f"Reason: {m['equivalent_reason']}",
                        "",
                    ]

                lines += [
                    "```diff",
                    m["diff"],
                    "```",
                    "",
                ]

    no_coverage_samples = [r for r in ok if r.get("no_coverage", 0) > 0]

    if no_coverage_samples:
        lines += [
            "",
            "---",
            "",
            "## No-Coverage Mutants",
            "",
            "These mutants were not reached by the current tests.",
            "",
        ]

        for r in no_coverage_samples:
            lines += [
                f"### Sample {r['idx']} — `{r['function_name']}`",
                "",
                f"Work dir: `{r['work_dir']}`",
                "",
            ]

            for m in r["no_coverage_mutants"]:
                lines += [
                    f"#### Mutant #{m['id']}",
                    "",
                    "```diff",
                    m["diff"],
                    "```",
                    "",
                ]

    if failed:
        lines += [
            "",
            "---",
            "",
            "## Errors",
            "",
        ]

        for r in failed:
            lines.append(
                f"- **Sample {r['idx']}** (`{r['function_name'] or '?'}`): {r['error']}"
            )

    md_path = output_dir / "mutation_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    if verbose:
        print("\nReports written to:")
        print(f"  {json_path}")
        print(f"  {md_path}")


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def generate_comparison_report(after: list[dict], baseline_path: Path, run_dir: Path) -> None:
    """Diff current results against a previous run's mutation_report.json."""
    baseline = {r["idx"]: r for r in json.loads(baseline_path.read_text(encoding="utf-8"))}
    after_map = {r["idx"]: r for r in after}

    all_idx = sorted(set(baseline) | set(after_map))

    lines = [
        "# Mutation Testing — Before vs After",
        "",
        f"Baseline: `{baseline_path}`",
        "",
        "| # | Function | Before score | After score | Survived before | Survived after | Δ survived |",
        "|---|----------|--------------|-------------|-----------------|----------------|------------|",
    ]

    total_before_killed = 0
    total_after_killed = 0
    total_before_mutants = 0
    total_after_mutants = 0
    total_delta_survived = 0

    for idx in all_idx:
        b = baseline.get(idx, {})
        a = after_map.get(idx, {})

        b_score = b.get("mutation_score_pct")
        a_score = a.get("mutation_score_pct")
        b_surv = b.get("survived", "—")
        a_surv = a.get("survived", "—")
        func = a.get("function_name") or b.get("function_name") or "?"

        delta = ""

        if isinstance(b_surv, int) and isinstance(a_surv, int):
            d = a_surv - b_surv
            delta = f"{d:+d}"
            total_delta_survived += d

        b_score_str = f"{b_score}%" if b_score is not None else "—"
        a_score_str = f"{a_score}%" if a_score is not None else "—"

        lines.append(
            f"| {idx} | `{func}` | {b_score_str} | {a_score_str} "
            f"| {b_surv} | {a_surv} | {delta} |"
        )

        if b.get("total_mutants"):
            total_before_killed += b.get("killed", 0)
            total_before_mutants += b["total_mutants"]

        if a.get("total_mutants"):
            total_after_killed += a.get("killed", 0)
            total_after_mutants += a["total_mutants"]

    before_overall = (
        total_before_killed / total_before_mutants * 100
        if total_before_mutants
        else 0
    )

    after_overall = (
        total_after_killed / total_after_mutants * 100
        if total_after_mutants
        else 0
    )

    lines += [
        "",
        f"**Overall score before:** {before_overall:.1f}%  ",
        f"**Overall score after:** {after_overall:.1f}%  ",
        f"**Score improvement:** {after_overall - before_overall:+.1f}pp  ",
        f"**Total survived mutants change:** {total_delta_survived:+d}",
    ]

    report_path = run_dir / "comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nComparison report: {report_path}")
    print(
        f"  Score: {before_overall:.1f}% → "
        f"{after_overall:.1f}% ({after_overall - before_overall:+.1f}pp)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mutmut mutation testing on a HuggingFace dataset."
    )

    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name or local path")
    parser.add_argument("--split", default="train", help="Dataset split, default: train")
    parser.add_argument("--code-col", default="code", help="Column containing code, default: code")
    parser.add_argument("--test-col", default="test", help="Column containing tests, default: test")

    parser.add_argument(
        "--entry-point-col",
        default="entry_point",
        help=(
            "Column with the function/class name under test, default: entry_point. "
            "If absent, the name is extracted from the code via AST."
        ),
    )

    parser.add_argument("--output", default="mutation_results", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process")

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start from this sample index, useful for resuming",
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help="Label for this run. Results saved to <output>/runs/<run-id>/.",
    )

    parser.add_argument(
        "--tests-dir",
        default=None,
        help=(
            "Directory of per-sample test files, sample_NNNN.py. "
            "If a file exists here it overrides the dataset test. "
            "On first run, baseline tests are written here automatically."
        ),
    )

    parser.add_argument(
        "--compare",
        default=None,
        help="Path to a previous run's mutation_report.json to diff against.",
    )

    args = parser.parse_args()

    # Check mutmut is installed using the current Python interpreter.
    check = subprocess.run(
        mutmut_cmd("--version"),
        capture_output=True,
        text=True,
    )

    if check.returncode != 0:
        print("ERROR: mutmut not found for this Python interpreter.")
        print("Install it with:")
        print(f"  {sys.executable} -m pip install mutmut")
        print()
        print("Raw error:")
        print((check.stdout + check.stderr).strip())
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_id:
        run_dir = output_dir / "runs" / args.run_id
    else:
        run_dir = output_dir

    run_dir.mkdir(parents=True, exist_ok=True)

    tests_dir = Path(args.tests_dir) if args.tests_dir else None

    if tests_dir:
        tests_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    rows = load_rows(args.dataset, args.split)

    end = len(rows) if args.limit is None else min(args.start + args.limit, len(rows))
    samples = rows[args.start:end]
    total = len(samples)

    print(f"Processing {total} samples → {run_dir}/\n")

    results = []

    for local_idx, sample in enumerate(samples):
        global_idx = args.start + local_idx

        code = sample[args.code_col]
        test = sample[args.test_col]
        entry_point = sample[args.entry_point_col] if args.entry_point_col in sample else None

        print(f"[{local_idx + 1}/{total}] sample {global_idx} ...", end=" ", flush=True)

        try:
            result = process_sample(
                global_idx,
                code,
                test,
                output_dir,
                tests_dir=tests_dir,
                entry_point=entry_point,
            )
        except Exception as e:
            result = {
                "idx": global_idx,
                "function_name": entry_point,
                "error": f"crashed: {e}",
                "killed": 0,
                "survived": 0,
                "timeout": 0,
                "no_coverage": 0,
                "total_mutants": 0,
                "mutation_score_pct": None,
                "survived_mutants": [],
                "no_coverage_mutants": [],
                "work_dir": None,
            }

        results.append(result)

        if result["error"]:
            print(f"ERROR: {result['error']}")
        else:
            score = result["mutation_score_pct"]
            print(
                f"{result['function_name']}  "
                f"{result['killed']}/{result['total_mutants']} killed  "
                f"score={score}%  survived={result['survived']}"
            )

        generate_report(results, run_dir, verbose=False)

    generate_report(results, run_dir)

    if args.compare:
        generate_comparison_report(results, Path(args.compare), run_dir)

    survived_total = sum(r["survived"] for r in results)

    print(f"\nDone. {survived_total} mutants survived. See {run_dir}/mutation_report.md")


if __name__ == "__main__":
    main()