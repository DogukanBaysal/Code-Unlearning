import json
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
from termcolor import cprint
from tqdm import tqdm

from evalplus.codegen import run_codegen
from evalplus.config import *
from evalplus.data import (
    COMBINED_EVAL_DATASET,
    get_combined_eval_datasets,
    get_combined_eval_components,
    get_forget_eval,
    get_forget_eval_hash,
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    get_utility_eval,
    get_utility_eval_hash,
    is_combined_eval_dataset,
    is_custom_dataset,
    load_solutions,
    normalize_combined_eval_dataset_name,
    normalize_custom_dataset_name,
    write_jsonl,
)
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.data.utils import CACHE_DIR
from evalplus.eval import (
    PASS,
    compatible_eval_result,
    estimate_pass_at_k,
    untrusted_check,
    untrusted_check_with_tests,
)
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec

# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[bool]]


def get_report_k_values(num_samples: int) -> List[int]:
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero")
    return sorted({1, (num_samples + 1) // 2, num_samples})


def get_groundtruth(problems, hashcode, tasks_only_output_not_none):
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        print(f"Load from ground-truth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Computing expected output...")
    tbegin = time.time()
    expected_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        oracle["base"], oracle["base_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )

        oracle["plus"], oracle["plus_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        expected_output[task_id] = oracle
    print(f"Expected outputs computed in {time.time() - tbegin:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected_output, f)

    return expected_output


def check_correctness(
    dataset: str,
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    expected_output: Dict[str, List],
    base_only=False,
    fast_check=False,
    identifier=None,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> Dict[str, Result]:  # {...}, "base" | "plus" -> (status, details)
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    if expected_output is None and problem.get("test"):
        ret["base"] = untrusted_check_with_tests(
            solution,
            problem["test"],
            problem["entry_point"],
        )
    else:
        ret["base"] = untrusted_check(
            dataset,
            solution,
            problem["base_input"],
            problem["entry_point"],
            expected=expected_output["base"],
            atol=problem["atol"],
            ref_time=expected_output["base_time"],
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )

    if not base_only:
        ret["plus"] = untrusted_check(
            dataset,
            solution,
            problem["plus_input"],
            problem["entry_point"],
            expected=expected_output["plus"],
            atol=problem["atol"],
            ref_time=expected_output["plus_time"],
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )

    return ret


def get_default_result_path(samples: str, output_file: Optional[str] = None) -> str:
    if os.path.isdir(samples):
        result_path = os.path.join(samples, "eval_results.json")
    else:
        assert samples.endswith(".jsonl")
        # legacy compatibility
        if os.path.exists(samples.replace(".jsonl", "_eval_results.json")):
            result_path = samples.replace(".jsonl", "_eval_results.json")
        else:
            result_path = samples.replace(".jsonl", ".eval_results.json")

    if output_file is not None:
        result_path = output_file
    return result_path


def get_combined_parts_dir(result_path: str) -> str:
    if result_path.endswith(".json"):
        return result_path[: -len(".json")] + ".parts"
    return result_path + ".parts"


def split_combined_samples(
    samples: str,
    split_dir: str,
    problems_by_dataset: Dict[str, Dict],
) -> Dict[str, str]:
    os.makedirs(split_dir, exist_ok=True)

    task_owner = {}
    for dataset_name, problems in problems_by_dataset.items():
        for task_id in problems:
            if task_id in task_owner:
                raise ValueError(
                    f"Task id collision for {task_id}: "
                    f"{task_owner[task_id]} and {dataset_name}"
                )
            task_owner[task_id] = dataset_name

    split_samples = {dataset_name: [] for dataset_name in problems_by_dataset}
    for sample in load_solutions(samples):
        dataset_name = task_owner.get(sample["task_id"])
        if dataset_name is None:
            warn(
                f"Task {sample['task_id']} is found in the combined samples but not "
                "found in any combined dataset"
            )
            continue

        split_samples[dataset_name].append(
            {key: value for key, value in sample.items() if not key.startswith("_")}
        )

    split_paths = {}
    for dataset_name, dataset_samples in split_samples.items():
        split_path = os.path.join(split_dir, f"{dataset_name}.jsonl")
        write_jsonl(split_path, dataset_samples)
        split_paths[dataset_name] = split_path

    return split_paths


def backup_existing_result_interactively(result_path: str) -> None:
    decision = ""
    while decision.lower() not in ["y", "n"]:
        print(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
        decision = input()

    if decision.lower() == "y":
        new_path = result_path + ".bak"
        while os.path.isfile(new_path):
            new_path += ".bak"
        os.rename(result_path, new_path)
        print(f"Backup {result_path} to {new_path}")


def evaluate_combined(
    dataset: str = COMBINED_EVAL_DATASET,
    samples: Optional[str] = None,
    output_file: Optional[str] = None,
    parallel: Optional[int] = None,
    i_just_wanna_run: bool = False,
    test_details: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    mini: bool = False,
    noextreme: bool = False,
    version: str = "default",
    gguf_file: Optional[str] = None,
    num_ctx: Optional[int] = None,
    defer_sanitize: bool = False,
    **model_kwargs,
):
    if model_kwargs:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false"
        )

        samples = run_codegen(
            dataset=dataset,
            gguf_file=gguf_file,
            num_ctx=num_ctx,
            defer_sanitize=defer_sanitize,
            **model_kwargs,
        )
    assert samples is not None, "No samples provided"

    result_path = get_default_result_path(samples, output_file)
    if os.path.isfile(result_path) and not i_just_wanna_run:
        print(f"Load from previous combined results from {result_path}")
        return

    if os.path.isfile(result_path) and i_just_wanna_run:
        backup_existing_result_interactively(result_path)

    problems_by_dataset = get_combined_eval_datasets(
        version=version,
        dataset=dataset,
    )
    components = get_combined_eval_components(dataset)
    parts_dir = get_combined_parts_dir(result_path)
    samples_dir = os.path.join(parts_dir, "samples")
    results_dir = os.path.join(parts_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    split_paths = split_combined_samples(samples, samples_dir, problems_by_dataset)
    component_results = {}
    component_result_paths = {}

    for component in components:
        component_result_path = os.path.join(results_dir, f"{component}.json")
        component_result_paths[component] = component_result_path
        evaluate(
            dataset=component,
            samples=split_paths[component],
            output_file=component_result_path,
            parallel=parallel,
            i_just_wanna_run=i_just_wanna_run,
            test_details=test_details,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
            mini=mini,
            noextreme=noextreme,
            version=version,
        )
        with open(component_result_path, "r") as f:
            component_results[component] = json.load(f)

    combined_results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": dataset,
        "hash": {
            component: result.get("hash")
            for component, result in component_results.items()
        },
        "samples": samples,
        "split_samples": split_paths,
        "component_results": component_result_paths,
        "eval": component_results,
        "pass_at_k": {
            component: result.get("pass_at_k", {})
            for component, result in component_results.items()
        },
    }

    with open(result_path, "w") as f:
        json.dump(combined_results, f)

    print(f"Combined results have been saved to {result_path}")


def evaluate(
    dataset: str,
    samples: Optional[str] = None,
    base_only: bool = False,
    parallel: Optional[int] = None,
    i_just_wanna_run: bool = False,
    test_details: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    mini: bool = False,
    noextreme: bool = False,
    version: str = "default",
    output_file: Optional[str] = None,
    gguf_file: Optional[str] = None,
    num_ctx: Optional[int] = None,
    defer_sanitize: bool = False,
    **model_kwargs,
):
    dataset = dataset.lower()
    if is_custom_dataset(dataset):
        dataset = normalize_custom_dataset_name(dataset)
    elif is_combined_eval_dataset(dataset):
        dataset = normalize_combined_eval_dataset_name(dataset)

    if is_combined_eval_dataset(dataset):
        evaluate_combined(
            dataset=dataset,
            samples=samples,
            output_file=output_file,
            parallel=parallel,
            i_just_wanna_run=i_just_wanna_run,
            test_details=test_details,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
            mini=mini,
            noextreme=noextreme,
            version=version,
            gguf_file=gguf_file,
            num_ctx=num_ctx,
            defer_sanitize=defer_sanitize,
            **model_kwargs,
        )
        return

    if model_kwargs:
        # To suppress the warning of tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false"
        )

        samples = run_codegen(
            dataset=dataset,
            gguf_file=gguf_file,
            num_ctx=num_ctx,
            defer_sanitize=defer_sanitize,
            **model_kwargs,
        )
    assert samples is not None, "No samples provided"

    n_workers = parallel or max(1, multiprocessing.cpu_count() // 2)

    result_path = get_default_result_path(samples, output_file)

    if os.path.isfile(result_path) and not i_just_wanna_run:
        print(f"Load from previous results from {result_path}")
        with open(result_path, "r") as f:
            results = json.load(f)

        results = compatible_eval_result(results)
    else:
        if dataset == "humaneval":
            problems = get_human_eval_plus(
                mini=mini, noextreme=noextreme, version=version
            )
            dataset_hash = get_human_eval_plus_hash(
                mini=mini, noextreme=noextreme, version=version
            )
            expected_output = get_groundtruth(problems, dataset_hash, [])
        elif dataset == "mbpp":
            problems = get_mbpp_plus(mini=mini, noextreme=noextreme, version=version)
            dataset_hash = get_mbpp_plus_hash(
                mini=mini, noextreme=noextreme, version=version
            )
            expected_output = get_groundtruth(
                problems,
                dataset_hash,
                MBPP_OUTPUT_NOT_NONE_TASKS,
            )
        elif dataset == "forgeteval":
            problems = get_forget_eval()
            dataset_hash = get_forget_eval_hash()
            base_only = True
            expected_output = {task_id: None for task_id in problems}
        elif dataset == "utilityeval":
            problems = get_utility_eval()
            dataset_hash = get_utility_eval_hash()
            base_only = True
            expected_output = {task_id: None for task_id in problems}
        else:
            raise ValueError(f"Invalid dataset {dataset}")

        results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "hash": dataset_hash,
            "eval": {},
        }

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            eval_results = defaultdict(list)  # task_id ->
            remainings = set()

            print("Reading samples...")
            for sample in tqdm(load_solutions(samples)):
                task_id = sample["task_id"]
                if task_id not in problems:
                    warn(
                        f"Task {task_id} is found in the samples but not found in the dataset"
                    )
                    continue
                solution = (
                    sample["solution"]
                    if "solution" in sample
                    else problems[task_id]["prompt"] + sample["completion"]
                )
                remainings.add(sample["_identifier"])
                args = (
                    dataset,
                    completion_id[task_id],
                    problems[task_id],
                    solution,
                    expected_output[task_id],
                    base_only,
                    not test_details,  # fast_check
                    sample["_identifier"],
                    min_time_limit,
                    gt_time_limit_factor,
                )
                futures.append(executor.submit(check_correctness, *args))
                completion_id[task_id] += 1
                n_samples += 1

            assert n_samples == len(remainings), "Missing problems in unfinished"
            assert len(completion_id) == len(problems), "Missing problems in samples"

            def stucking_checker():
                while remainings:
                    last_size = len(remainings)
                    time.sleep(20)
                    if last_size != len(remainings) or len(remainings) == 0:
                        continue
                    # Potential stucking
                    warn("No samples had finished testing in the last 20s")
                    warn(f"{len(remainings)} samples to be tested: {remainings}")

            # This thread only reports a lack of progress. It must not keep the
            # evaluator process alive if the main evaluation path exits.
            threading.Thread(target=stucking_checker, daemon=True).start()

            for future in tqdm(as_completed(futures), total=n_samples):
                result = future.result()
                remainings.remove(result["_identifier"])
                eval_results[result["task_id"]].append(result)

        # sort the results for each problem by completion_id
        for task_id, task_results in eval_results.items():
            task_results.sort(key=lambda x: x["completion_id"])
            results["eval"][task_id] = []
            for res in task_results:

                def get_failed_tests(stat, details, inputs) -> List[Any]:
                    if stat == PASS or not details:
                        return []

                    if not inputs:
                        return [problems[task_id].get("test", "")]

                    if test_details:
                        return [
                            inputs[i] for i in range(len(details)) if not details[i]
                        ]

                    # else => simply return the only and the last fail test
                    return [inputs[len(details) - 1]]

                base_stat, base_details = res["base"]
                base_fail_tests = get_failed_tests(
                    base_stat, base_details, problems[task_id]["base_input"]
                )

                # initialize plus tests
                plus_stat = None
                plus_fail_tests = []

                # with plus tests
                if not base_only:
                    plus_stat, plus_details = res["plus"]
                    plus_fail_tests = get_failed_tests(
                        plus_stat, plus_details, problems[task_id]["plus_input"]
                    )

                if dataset == "mbpp":
                    base_fail_tests = mbpp_serialize_inputs(task_id, base_fail_tests)
                    plus_fail_tests = mbpp_serialize_inputs(task_id, plus_fail_tests)

                results["eval"][task_id].append(
                    {
                        "task_id": task_id,
                        "solution": res["solution"],
                        "base_status": base_stat,
                        "plus_status": plus_stat,
                        "base_fail_tests": base_fail_tests,
                        "plus_fail_tests": plus_fail_tests,
                    }
                )

    # Calculate pass@k.
    total = np.array([len(r) for r in results["eval"].values()])
    base_correct = []
    new_correct = []

    for res in results["eval"].values():
        bc = sum([r["base_status"] == PASS for r in res])
        base_correct.append(bc)
        if not base_only:
            new_correct.append(
                sum(
                    [
                        res[i]["base_status"] == res[i]["plus_status"] == PASS
                        for i in range(len(res))
                    ]
                )
            )
    base_correct = np.array(base_correct)

    min_samples = int(total.min())
    report_k_values = get_report_k_values(min_samples)
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
        for k in report_k_values
        if total.min() >= k
    }
    cprint(f"{dataset} (base tests)", "red")
    for k, v in pass_at_k.items():
        cprint(f"{k}:\t{v:.3f}", "red")
    results["pass_at_k"] = {"base": pass_at_k}

    if new_correct:
        cprint(f"{dataset}+ (base + extra tests)", "green")
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
            for k in report_k_values
            if (total >= k).all()
        }
        for k, v in pass_at_k.items():
            cprint(f"{k}:\t{v:.3f}", "green")
        results["pass_at_k"]["plus"] = pass_at_k

    # save results
    if os.path.isfile(result_path) and i_just_wanna_run:
        backup_existing_result_interactively(result_path)

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f)


def main():
    from fire import Fire

    Fire(evaluate)


if __name__ == "__main__":
    main()
