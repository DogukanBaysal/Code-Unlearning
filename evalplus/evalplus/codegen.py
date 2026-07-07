import gc
import json
import os
from typing import Dict, List, Optional

from evalplus.data import (
    COMBINED_EVAL_DATASET,
    get_evalperf_data,
    get_combined_eval,
    get_forget_eval,
    get_human_eval_plus,
    get_mbpp_plus,
    get_utility_eval,
    is_combined_eval_dataset,
    is_custom_dataset,
    normalize_combined_eval_dataset_name,
    normalize_custom_dataset_name,
)
from evalplus.provider import DecoderBase, make_model
from evalplus.sanitize import sanitize, sanitize_samples
from evalplus.utils import progress


def get_raw_target_path(target_path: str) -> str:
    if target_path.endswith(".jsonl"):
        return target_path.replace(".jsonl", ".raw.jsonl")
    return target_path + ".raw"


def write_codegen_output(
    target_path: str,
    raw_target_path: Optional[str],
    task_id: str,
    solution: str,
    output_solution: str,
    sample_idx: int,
):
    if target_path.endswith(".jsonl"):
        with open(target_path, "a") as f:
            f.write(
                json.dumps({"task_id": task_id, "solution": output_solution}) + "\n"
            )

        if raw_target_path is not None:
            with open(raw_target_path, "a") as f:
                f.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
        return

    p_name = task_id.replace("/", "_")
    with open(
        os.path.join(target_path, p_name, f"{sample_idx}.py"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(output_solution)

    if raw_target_path is not None:
        with open(
            os.path.join(raw_target_path, p_name, f"{sample_idx}.py"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(solution)


def in_id_range(task_id: str, id_range) -> bool:
    if id_range is None:
        return True

    id_num = int(task_id.split("/")[1])
    low, high = id_range
    return low <= id_num < high


def codegen(
    target_path: str,
    model: DecoderBase,
    dataset: Dict,
    greedy=False,
    n_samples=1,
    id_range=None,
    resume=True,
    num_ctx=None,
    sanitize_output=True,
):
    task2nexist = {}
    if resume and target_path.endswith(".jsonl") and os.path.isfile(target_path):
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                task_id = json.loads(line)["task_id"]
                task2nexist[task_id] = task2nexist.get(task_id, 0) + 1

    raw_target_path = None
    if sanitize_output:
        raw_target_path = get_raw_target_path(target_path)

    if not target_path.endswith(".jsonl"):
        os.makedirs(target_path, exist_ok=True)
        if raw_target_path is not None:
            os.makedirs(raw_target_path, exist_ok=True)

    if sanitize_output:
        print(f"Sanitized code outputs will be saved to {target_path}")
        print(f"Raw outputs will be saved to {raw_target_path}")
    else:
        print(f"Raw code outputs will be saved to {target_path}")

    def prepare_folder_task(task_id: str) -> None:
        if target_path.endswith(".jsonl"):
            return

        p_name = task_id.replace("/", "_")
        os.makedirs(os.path.join(target_path, p_name), exist_ok=True)
        if raw_target_path is not None:
            os.makedirs(os.path.join(raw_target_path, p_name), exist_ok=True)
        task2nexist[task_id] = len(
            [
                f
                for f in os.listdir(os.path.join(target_path, p_name))
                if f.endswith(".py")
            ]
        )

    backend_type: str = type(model).__name__

    if n_samples == 1:
        pending = []

        def flush_pending(p):
            if not pending:
                return

            p.console.print(f"Codegen batch: {len(pending)} tasks @ {model}")
            prompts = [task["prompt"].strip() + "\n" for _, task in pending]
            batch_outputs = model.codegen_batch(
                prompts,
                do_sample=not greedy,
                num_samples=1,
            )
            assert len(batch_outputs) == len(pending)

            for (task_id, task), prompt, outputs in zip(pending, prompts, batch_outputs):
                assert outputs, "No outputs from model!"
                solution = (
                    prompt + outputs[0] if model.is_direct_completion() else outputs[0]
                )
                output_solution = (
                    sanitize(solution, entrypoint=task["entry_point"])
                    if sanitize_output
                    else solution
                )
                write_codegen_output(
                    target_path=target_path,
                    raw_target_path=raw_target_path,
                    task_id=task_id,
                    solution=solution,
                    output_solution=output_solution,
                    sample_idx=0,
                )

            pending.clear()

        with progress(backend_type) as p:
            for task_id, task in p.track(dataset.items()):
                if not in_id_range(task_id, id_range):
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

                prepare_folder_task(task_id)
                if resume and task2nexist.get(task_id, 0) >= n_samples:
                    p.console.print(f"Codegen: {task_id} @ {model} (cached)")
                    continue

                pending.append((task_id, task))
                if len(pending) >= max(1, model.batch_size):
                    flush_pending(p)

            flush_pending(p)
        return

    with progress(backend_type) as p:
        for task_id, task in p.track(dataset.items()):
            if not in_id_range(task_id, id_range):
                p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                continue

            prepare_folder_task(task_id)

            n_more_samples = n_samples
            log = f"Codegen: {task_id} @ {model}"
            if resume and task2nexist.get(task_id, 0) > 0:
                log += f" (resuming from {task2nexist[task_id]})"
                n_more_samples -= task2nexist[task_id]

            p.console.print(log)

            sidx = n_samples - n_more_samples
            while sidx < n_samples:
                prompt = task["prompt"].strip() + "\n"
                outputs = model.codegen(
                    prompt,
                    do_sample=not greedy,
                    num_samples=n_samples - sidx,
                )
                assert outputs, "No outputs from model!"
                for impl in outputs:
                    solution = prompt + impl if model.is_direct_completion() else impl
                    output_solution = (
                        sanitize(solution, entrypoint=task["entry_point"])
                        if sanitize_output
                        else solution
                    )
                    write_codegen_output(
                        target_path=target_path,
                        raw_target_path=raw_target_path,
                        task_id=task_id,
                        solution=solution,
                        output_solution=output_solution,
                        sample_idx=sidx,
                    )
                    sidx += 1


def run_codegen(
    model: str,
    dataset: str,
    root: str = "evalplus_results",
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    num_ctx: Optional[int] = None,
    resume: bool = True,
    greedy: bool = False,
    id_range: List = None,
    version: str = "default",
    backend: str = "vllm",
    force_base_prompt: bool = False,
    base_url: str = None,
    verify_certificate: bool = True,
    tp: int = 1,
    evalperf_type: str = None,  # For EvalPerf
    jsonl_fmt: bool = True,
    attn_implementation: str = "eager",
    device_map: Optional[str] = None,
    peft_name: Optional[str] = None,
    peft_subfolder: Optional[str] = None,
    trust_remote_code: bool = False,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = False,
    dtype: str = "bfloat16",
    gptqmodel_backend: str = "auto",  # For GPTQModel
    gguf_file: Optional[str] = None,
    defer_sanitize: bool = False,
    **kwargs,
):
    dataset = dataset.lower()
    if is_custom_dataset(dataset):
        dataset = normalize_custom_dataset_name(dataset)
    elif is_combined_eval_dataset(dataset):
        dataset = normalize_combined_eval_dataset_name(dataset)

    assert dataset in [
        "humaneval",
        "mbpp",
        "evalperf",
        "forgeteval",
        "utilityeval",
        COMBINED_EVAL_DATASET,
    ], f"Invalid dataset {dataset}"
    assert evalperf_type is None or evalperf_type in [
        "instruct",
        "perf-instruct",
        "perf-CoT",
    ]

    # Make dir for codes generated by each model
    identifier = model.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}"
    if peft_name is not None:
        identifier += f"_peft_{peft_name.strip('./').replace('/', '--')}"
        if peft_subfolder:
            identifier += f"_subfolder_{peft_subfolder.strip('./').replace('/', '--')}"
    if evalperf_type:
        identifier += f"-{evalperf_type}"

    target_path = os.path.join(root, dataset, identifier)
    if jsonl_fmt:
        target_path += ".jsonl"
    else:
        os.makedirs(target_path, exist_ok=True)

    if dataset == "humaneval":
        dataset_dict = get_human_eval_plus(version=version)
    elif dataset == "mbpp":
        dataset_dict = get_mbpp_plus(version=version)
    elif dataset == "evalperf":
        original_dataset = {**get_human_eval_plus(), **get_mbpp_plus()}
        dataset_dict = {k: original_dataset[k] for k in get_evalperf_data()}
        assert id_range is None, "id_range not supported for evalperf"
    elif dataset == "forgeteval":
        dataset_dict = get_forget_eval()
    elif dataset == "utilityeval":
        dataset_dict = get_utility_eval()
    elif dataset == COMBINED_EVAL_DATASET:
        dataset_dict = get_combined_eval(version=version)
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    generation_path = get_raw_target_path(target_path) if defer_sanitize else target_path

    all_tasks_complete = False
    if jsonl_fmt and os.path.isfile(target_path):
        task_counts = {}
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_id = data["task_id"]
                task_counts[task_id] = task_counts.get(task_id, 0) + 1

            all_tasks_complete = all(
                task_counts.get(task_id, 0) >= n_samples
                for task_id in dataset_dict.keys()
            )

    if all_tasks_complete:
        print("All samples are already cached. Skipping codegen.")
        return target_path

    generation_complete = False
    if defer_sanitize and jsonl_fmt and os.path.isfile(generation_path):
        task_counts = {}
        with open(generation_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_id = data["task_id"]
                task_counts[task_id] = task_counts.get(task_id, 0) + 1

            generation_complete = all(
                task_counts.get(task_id, 0) >= n_samples
                for task_id in dataset_dict.keys()
            )

    if generation_complete:
        print("All raw samples are already cached. Skipping codegen.")

    if greedy:
        temperature = 0.0
        n_samples = 1
        bs = bs or 32
        print(
            f"Greedy decoding ON (--greedy): setting bs={bs}, "
            "n_samples=1, temperature=0"
        )

    if id_range is not None:
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    if bs is None:
        bs = 32 if n_samples == 1 else min(n_samples, 32)
        print(f"Setting batch size to {bs}")

    if backend != "ollama" and num_ctx is not None:
        print(
            "Warning --num_ctx can be set on ollama backend only. num_ctx will be ignored."
        )

    # Make project dir
    os.makedirs(root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(root, dataset), exist_ok=True)

    # Model instructions
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

    if evalperf_type == "perf-instruct":
        instruction_prefix = "Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type == "perf-CoT":
        instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type is not None and evalperf_type != "instruct":
        raise ValueError(f"Invalid evalperf_type: {evalperf_type}")

    if not generation_complete:
        # Model creation
        model_runner = make_model(
            model=model,
            backend=backend,
            batch_size=bs,
            temperature=temperature,
            num_ctx=num_ctx,
            force_base_prompt=force_base_prompt,
            dataset=dataset,
            base_url=base_url,
            verify_certificate=verify_certificate,
            tp=tp,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
            device_map=device_map,
            peft_name=peft_name,
            peft_subfolder=peft_subfolder,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=enable_chunked_prefill,
            dtype=dtype,
            gptqmodel_backend=gptqmodel_backend,
            gguf_file=gguf_file,
            **kwargs,
        )

        codegen(
            target_path=generation_path,
            dataset=dataset_dict,
            greedy=greedy,
            model=model_runner,
            n_samples=n_samples,
            resume=resume,
            id_range=id_range,
            sanitize_output=not defer_sanitize,
        )

        # force shutdown the model runner
        del model_runner

        gc.collect()

    if defer_sanitize:
        sanitize_samples(
            samples=generation_path,
            target_path=target_path,
            mbpp_version=version,
            problems=dataset_dict,
        )

    return target_path


def main():
    from fire import Fire

    Fire(run_codegen)


if __name__ == "__main__":
    main()
