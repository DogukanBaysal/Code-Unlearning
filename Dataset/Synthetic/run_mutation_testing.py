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

def extract_entry_point(code: str) -> Optional[str]:
    """Return the name of the first top-level function OR class defined in code.

    Some dataset samples define a class instead of a function (the humaneval
    check() then instantiates it as `candidate(...)`), so we must accept both.
    """
    try:
        tree = ast.parse(code)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return node.name
    except SyntaxError:
        pass
    # Fallback: regex (matches both `def name(` and `class name`)
    m = re.search(r"^(?:def|class)\s+(\w+)", code, re.MULTILINE)
    return m.group(1) if m else None


def humaneval_to_pytest(test_code: str, entry_point: str, module: str = "source") -> str:
    """
    Wrap a humaneval-style check(candidate) block into a pytest test.

    `entry_point` is the name of the function OR class under test. For a class,
    check() receives the class object and instantiates it (e.g.
    `shelf = candidate(...)`), so importing the name and passing it to check()
    works identically for functions and classes.

    Input test looks like:
        def check(candidate):
            assert candidate([...]) == [...]

    Output:
        from source import my_func   # or: from source import MyClass
        def check(candidate): ...
        def test_all():
            check(my_func)
    """
    return (
        f"import pytest\n"
        f"from {module} import {entry_point}\n\n\n"
        f"{test_code.strip()}\n\n\n"
        f"def test_all():\n"
        f"    check({entry_point})\n"
    )


def run_subprocess(cmd: list, cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess:
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

    - A local `.jsonl` / `.json` path is read directly (one JSON object per
      line for .jsonl, or a JSON array / lines for .json).
    - Anything else is treated as a HuggingFace dataset name.
    """
    path = Path(dataset)
    if path.exists() and path.suffix in (".jsonl", ".json"):
        text = path.read_text(encoding="utf-8")
        # Try a JSON array first (a plain .json), else fall back to JSON lines.
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
    Parse final counts from the mutmut 2.x run progress spinner output.

    Final line looks like:
        31/31  🎉 26 🫥 0  ⏰ 1  🤔 0  🙁 4  🔇 0  🧙 0

    🫥 = no coverage (code not reached by tests — unkillable by definition).
    """
    counts = {
        "killed": 0, "survived": 0, "timeout": 0,
        "suspicious": 0, "no_coverage": 0, "total": 0,
    }

    progress_lines = [l for l in run_log.splitlines() if re.search(r"\d+/\d+.*🎉", l)]
    if not progress_lines:
        return counts

    last = progress_lines[-1]

    m = re.search(r"(\d+)/(\d+)", last)
    if m:
        counts["total"] = int(m.group(2))

    for emoji, key in [
        ("🎉", "killed"), ("🙁", "survived"), ("⏰", "timeout"),
        ("🤔", "suspicious"), ("🫥", "no_coverage"),
    ]:
        m = re.search(rf"{emoji}\s*(\d+)", last)
        if m:
            counts[key] = int(m.group(1))

    return counts


def parse_mutmut_results_output(results_text: str) -> dict:
    """
    Fallback: parse `mutmut results` text for mutmut 1.x style output.

    mutmut 1.x prints:
        Killed 🎉 (10)
        Survived 🙁 (2)
    """
    counts = {"killed": 0, "survived": 0, "timeout": 0, "suspicious": 0, "total": 0}
    for key in ("killed", "survived", "timeout", "suspicious"):
        m = re.search(rf"^{key}[^\n(]*\((\d+)\)", results_text, re.IGNORECASE | re.MULTILINE)
        if m:
            counts[key] = int(m.group(1))
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


def parse_results_by_status(results_text: str) -> dict:
    """
    Parse mutmut 3.x `mutmut results` output into {status: [mutant_id, ...]}.

    Each line looks like:
        source.x_my_func__mutmut_1: survived

    (Killed mutants are omitted unless `mutmut results --all` is used.)
    Status strings in 3.x: survived, timeout, suspicious, skipped,
    'no tests', 'not checked', killed.
    """
    by_status: dict[str, list[str]] = {}
    # Mutant ids contain `__mutmut_<n>`; anchor on that to avoid matching prose.
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
    Handles mutmut 3.x ("source.x_f__mutmut_1: survived"),
    2.x ("No coverage (N):\nsource.py:3") and 1.x ("---- 3, 7") formats.

    `status` is a human label like "Survived" or "No coverage"; it is mapped
    to the 3.x status strings ("survived" / "no tests").
    """
    # --- mutmut 3.x: "<id>: <status>" lines ---
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

    # --- mutmut 2.x: "Status (N):\nsource.py:3\n..." ---
    block2 = re.search(
        rf"^{status}[^\n]*:\s*\n(.*?)(?=^\S|\Z)",
        results_text,
        re.DOTALL | re.MULTILINE,
    )
    if block2:
        ids = re.findall(r"[\w./]+:\d+", block2.group(1))
        if ids:
            return ids

    # --- mutmut 1.x: "Status ... (N)\n\n---- 3, 7" ---
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
    proc = run_subprocess(["mutmut", "show", mutant_id], cwd=work_dir)
    return proc.stdout.strip()


def is_likely_equivalent(diff: str) -> tuple[bool, str]:
    """
    Heuristic detection of equivalent (unkillable) mutants.

    Returns (True, reason) when the mutation almost certainly cannot be
    caught by any test, False otherwise.

    Heuristics (conservative — false negatives are fine; false positives
    would wrongly tell the agent to skip a killable mutant):

    1. Only string-literal assignments to local variables changed.
       e.g.  -  api_key = "real"
             +  api_key = "XX"
       If the variable is never returned or raised, no test can observe it.

    2. Only a `pass` statement was inserted/removed in an already-empty branch.

    3. Mutation changes only a constant that's immediately overwritten
       (e.g. loop variable initialized before first use).
    """
    changed = [
        l[1:] for l in diff.splitlines()
        if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))
    ]
    if not changed:
        return False, ""

    # Heuristic 1: all changes are bare string-literal assignments to local vars
    # Pattern: `identifier = "..."` with no return/raise/assert/comparison context
    str_assign = re.compile(r'^\s*\w+\s*=\s*(["\']).*\1\s*$')
    if all(str_assign.match(l) for l in changed):
        # Make sure none of the changed lines feed a return/raise/assert
        # (if they did the diff would include more context — simple check)
        return True, "string literal assigned to a local variable that tests cannot observe"

    # Heuristic 2: only `pass` inserted
    if all(l.strip() == "pass" for l in changed):
        return True, "only a 'pass' statement added/removed — no observable effect"

    return False, ""


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_sample(idx: int, code: str, test: str, output_dir: Path,
                   tests_dir: Optional[Path] = None,
                   entry_point: Optional[str] = None) -> dict:
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
        "survived_mutants": [],       # list of {id, diff, equivalent, equivalent_reason}
        "no_coverage_mutants": [],    # list of {id, diff}  — unkillable by definition
        "work_dir": None,
    }

    # Prefer the dataset-provided entry point; fall back to AST extraction
    # (which now handles both functions and classes).
    func_name = entry_point or extract_entry_point(code)
    if not func_name:
        base["error"] = "Could not extract entry point (function/class) from code"
        return base
    base["function_name"] = func_name

    work_dir = output_dir / f"sample_{idx:04d}"
    # Wipe and recreate so stale .mutmut-cache never poisons results
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    base["work_dir"] = str(work_dir)

    # --- Write source ---
    (work_dir / "source.py").write_text(code, encoding="utf-8")

    # --- Write adapted test ---
    # If a custom test file exists in tests_dir, use it (agent-improved version).
    # Otherwise fall back to the dataset's original test.
    custom_test_path = tests_dir / f"sample_{idx:04d}.py" if tests_dir else None
    if custom_test_path and custom_test_path.exists():
        pytest_test = custom_test_path.read_text(encoding="utf-8")
    else:
        try:
            pytest_test = humaneval_to_pytest(test, func_name)
        except Exception as e:
            base["error"] = f"Test adaptation failed: {e}"
            return base
        # Always write the baseline test to tests_dir so agent knows where to edit
        if tests_dir:
            (tests_dir / f"sample_{idx:04d}.py").write_text(pytest_test, encoding="utf-8")
    (work_dir / "test_source.py").write_text(pytest_test, encoding="utf-8")

    # --- mutmut config ---
    # mutmut 2.x reads pyproject.toml ([tool.mutmut]); 1.x reads setup.cfg.
    # We write both so the script works with either version.
    # sys.executable ensures we use the same Python that's running this script.
    runner_cmd = f"{sys.executable} -m pytest test_source.py -x -q --tb=short"

    # mutmut 2.x
    (work_dir / "pyproject.toml").write_text(
        "[tool.mutmut]\n"
        'paths_to_mutate = ["source.py"]\n'
        f'runner = "{runner_cmd}"\n',
        encoding="utf-8",
    )
    # mutmut 1.x fallback
    (work_dir / "setup.cfg").write_text(
        "[mutmut]\n"
        "paths_to_mutate=source.py\n"
        f"runner={runner_cmd}\n"
        "tests_dir=.\n",
        encoding="utf-8",
    )

    # --- Verify tests pass on original code ---
    verify = run_subprocess(
        [sys.executable, "-m", "pytest", "test_source.py", "-q", "--tb=short"],
        cwd=work_dir,
        timeout=30,
    )
    if verify.returncode != 0:
        base["error"] = "Tests fail on original code — skipping mutation"
        base["test_output"] = (verify.stdout + verify.stderr)[:2000]
        return base

    # --- Run mutmut ---
    # mutmut 2.x: `mutmut run` takes no flags — config comes from pyproject.toml.
    # mutmut exits with code 1 when some mutants survive — that's normal, not an error.
    try:
        run_proc = run_subprocess(
            ["mutmut", "run"],
            cwd=work_dir,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        base["error"] = "mutmut run timed out (180 s)"
        return base

    raw_run_output = run_proc.stdout + run_proc.stderr
    # Save raw output for debugging
    (work_dir / "mutmut_run.log").write_text(raw_run_output, encoding="utf-8")

    # Exit code 2 = usage/config error; 0 = all killed; 1 = some survived (OK)
    if run_proc.returncode == 2:
        base["error"] = f"mutmut config/usage error: {raw_run_output[:500]}"
        return base

    # --- Collect results ---
    # Primary: parse counts from the run log (works with mutmut 2.x spinner output).
    counts = parse_mutmut_run_output(raw_run_output)

    # Fallback: if run log parse got nothing, try `mutmut results` (1.x style).
    if counts["total"] == 0:
        results_proc = run_subprocess(["mutmut", "results"], cwd=work_dir)
        results_text = results_proc.stdout
        (work_dir / "mutmut_results.log").write_text(results_text, encoding="utf-8")
        counts = parse_mutmut_results_output(results_text)
    else:
        results_proc = run_subprocess(["mutmut", "results"], cwd=work_dir)
        results_text = results_proc.stdout
        (work_dir / "mutmut_results.log").write_text(results_text, encoding="utf-8")

    base.update(counts)  # merges killed/survived/timeout/total
    base["total_mutants"] = counts["total"]  # normalize key name

    if counts["total"] > 0:
        base["mutation_score_pct"] = round(counts["killed"] / counts["total"] * 100, 1)

    # --- Collect survived mutant diffs (with equivalence hints) ---
    survived_ids = get_survived_ids(results_text)
    for mid in survived_ids[:50]:
        diff = get_mutant_diff(mid, work_dir)
        equiv, reason = is_likely_equivalent(diff)
        base["survived_mutants"].append({
            "id": mid,
            "diff": diff,
            "equivalent": equiv,
            "equivalent_reason": reason,
        })

    # --- Collect no-coverage mutant diffs (unkillable by definition) ---
    no_cov_ids = get_no_coverage_ids(results_text)
    for mid in no_cov_ids[:50]:
        diff = get_mutant_diff(mid, work_dir)
        base["no_coverage_mutants"].append({"id": mid, "diff": diff})

    return base


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], output_dir: Path, verbose: bool = True) -> None:
    json_path = output_dir / "mutation_report.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

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
    ]

    # Per-sample summary table
    lines += [
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

    # Survived mutant diffs — the main payload for the agent
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
                    "```diff",
                    m["diff"],
                    "```",
                    "",
                ]

    if failed:
        lines += [
            "---",
            "",
            "## Errors",
            "",
        ]
        for r in failed:
            lines.append(f"- **Sample {r['idx']}** (`{r['function_name'] or '?'}`): {r['error']}")

    md_path = output_dir / "mutation_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    if verbose:
        print(f"\nReports written to:")
        print(f"  {json_path}")
        print(f"  {md_path}")


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def generate_comparison_report(after: list[dict], baseline_path: Path, run_dir: Path) -> None:
    """Diff current results against a previous run's mutation_report.json."""
    baseline = {r["idx"]: r for r in json.loads(baseline_path.read_text())}
    after_map = {r["idx"]: r for r in after}

    all_idx = sorted(set(baseline) | set(after_map))

    lines = [
        "# Mutation Testing — Before vs After",
        "",
        f"Baseline: `{baseline_path}`",
        "",
        "| # | Function | Before score | After score | Survived before | Survived after | Δ survived |",
        "|---|----------|-------------|------------|----------------|---------------|-----------|",
    ]

    total_before_killed = total_after_killed = 0
    total_before_mutants = total_after_mutants = 0
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

        for r, km, mm in [(b, "killed", "total_mutants"), (a, "killed", "total_mutants")]:
            pass  # aggregate below

        if b.get("total_mutants"):
            total_before_killed += b.get("killed", 0)
            total_before_mutants += b["total_mutants"]
        if a.get("total_mutants"):
            total_after_killed += a.get("killed", 0)
            total_after_mutants += a["total_mutants"]

    before_overall = (total_before_killed / total_before_mutants * 100) if total_before_mutants else 0
    after_overall = (total_after_killed / total_after_mutants * 100) if total_after_mutants else 0

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
    print(f"  Score: {before_overall:.1f}% → {after_overall:.1f}% ({after_overall - before_overall:+.1f}pp)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mutmut mutation testing on a HuggingFace dataset."
    )
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name or local path")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--code-col", default="code", help="Column containing code (default: code)")
    parser.add_argument("--test-col", default="test", help="Column containing tests (default: test)")
    parser.add_argument(
        "--entry-point-col", default="entry_point",
        help="Column with the function/class name under test (default: entry_point). "
             "If the column is absent, the name is extracted from the code via AST."
    )
    parser.add_argument("--output", default="mutation_results", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process")
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start from this sample index (for resuming)"
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Label for this run (e.g. run_001). Results saved to <output>/runs/<run-id>/."
    )
    parser.add_argument(
        "--tests-dir", default=None,
        help="Directory of per-sample test files (sample_NNNN.py). "
             "If a file exists here it overrides the dataset test. "
             "On first run, baseline tests are written here automatically."
    )
    parser.add_argument(
        "--compare", default=None,
        help="Path to a previous run's mutation_report.json to diff against."
    )
    args = parser.parse_args()

    # Check mutmut is installed
    if subprocess.run(["mutmut", "--version"], capture_output=True).returncode != 0:
        print("ERROR: mutmut not found. Install with:  pip install mutmut")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve run-specific results directory
    if args.run_id:
        run_dir = output_dir / "runs" / args.run_id
    else:
        run_dir = output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve shared tests directory
    tests_dir = Path(args.tests_dir) if args.tests_dir else None
    if tests_dir:
        tests_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset. A local .jsonl/.json path is read directly (one JSON
    # object per line); anything else is treated as a HuggingFace dataset name.
    print(f"Loading dataset: {args.dataset}")
    rows = load_rows(args.dataset, args.split)

    end = len(rows) if args.limit is None else min(args.start + args.limit, len(rows))
    samples = rows[args.start:end]
    total = len(samples)

    print(f"Processing {total} samples → {output_dir}/\n")

    results = []
    for local_idx, sample in enumerate(samples):
        global_idx = args.start + local_idx
        code = sample[args.code_col]
        test = sample[args.test_col]
        entry_point = sample[args.entry_point_col] if args.entry_point_col in sample else None

        print(f"[{local_idx + 1}/{total}] sample {global_idx} ...", end=" ", flush=True)

        # Never let one bad sample abort the whole run — otherwise reports
        # (mutation_report.md, etc.) would never be written.
        try:
            result = process_sample(
                global_idx, code, test, output_dir,
                tests_dir=tests_dir, entry_point=entry_point,
            )
        except Exception as e:
            result = {
                "idx": global_idx, "function_name": entry_point, "error": f"crashed: {e}",
                "killed": 0, "survived": 0, "timeout": 0, "no_coverage": 0,
                "total_mutants": 0, "mutation_score_pct": None,
                "survived_mutants": [], "no_coverage_mutants": [], "work_dir": None,
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

        # Rewrite the report after every sample so mutation_report.md/json stay
        # up to date while the run is still in progress (quietly to avoid spam).
        generate_report(results, run_dir, verbose=False)

    generate_report(results, run_dir)

    if args.compare:
        generate_comparison_report(results, Path(args.compare), run_dir)

    survived_total = sum(r["survived"] for r in results)
    print(f"\nDone. {survived_total} mutants survived. See {run_dir}/mutation_report.md")


if __name__ == "__main__":
    main()