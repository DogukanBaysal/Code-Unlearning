#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNLEARNING_EVAL_DIR="${REPO_ROOT}/UnlearningEvaluation"
EVALPLUS_DIR="${REPO_ROOT}/evalplus"

QWEN_MODEL_REPO="${QWEN_MODEL_REPO:-dbaysal/secret-unlearning-qwen-checkpoint282-ga-full}"
LLAMA_MODEL_REPO="${LLAMA_MODEL_REPO:-dbaysal/secret-unlearning-llama-checkpoint282-ga-full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/full_ga_all_evals}"
CHECKPOINTS="${CHECKPOINTS:-}"

SUFFIX_BS="${SUFFIX_BS:-16}"
EVALPLUS_BS="${EVALPLUS_BS:-16}"
PASS_K="${PASS_K:-10}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2056}"
DTYPE="${DTYPE:-bfloat16}"
BACKEND="${BACKEND:-hf}"
EVALPLUS_DATASET="${EVALPLUS_DATASET:-humaneval-forget-utility}"
EVALPLUS_PARALLEL="${EVALPLUS_PARALLEL:-}"
EVALPLUS_TIMEOUT_PER_TASK="${EVALPLUS_TIMEOUT_PER_TASK:-30}"
EVALPLUS_SANITIZE_WORKERS="${EVALPLUS_SANITIZE_WORKERS:-4}"
BASELINE_FILTER_CSV="${BASELINE_FILTER_CSV:-${EVALPLUS_DIR}/evalplus/baseline_failed_test_ids.csv}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"

model_specs=(
    "qwen2_5_coder_3b|${QWEN_MODEL_REPO}"
    "meta_llama3_2_3b|${LLAMA_MODEL_REPO}"
)

detect_gpu_ids() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        if [ "${CUDA_VISIBLE_DEVICES}" = "-1" ]; then
            return 0
        fi
        local old_ifs="${IFS}"
        local visible_id
        IFS=","
        for visible_id in ${CUDA_VISIBLE_DEVICES}; do
            [ -n "${visible_id}" ] && echo "${visible_id}"
        done
        IFS="${old_ifs}"
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits
        return 0
    fi
    echo "0"
}

discover_checkpoints() {
    local repo_id="$1"
    if [ -n "${CHECKPOINTS}" ]; then
        printf '%s\n' ${CHECKPOINTS//,/ }
        return 0
    fi
    python - "${repo_id}" <<'PY'
import re
import sys
from huggingface_hub import list_repo_files

repo_id = sys.argv[1]
repo_files = list_repo_files(repo_id=repo_id, repo_type="model")
checkpoints = {
    path.split("/", 1)[0]
    for path in repo_files
    if re.match(r"checkpoint-\d+/", path)
}
for checkpoint in sorted(checkpoints, key=lambda name: int(name.split("-", 1)[1])):
    print(checkpoint)
PY
}

resolve_checkpoint_path() {
    local repo_id="$1"
    local checkpoint="$2"
    python - "${repo_id}" "${checkpoint}" <<'PY'
import re
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id, checkpoint = sys.argv[1:]
if not re.fullmatch(r"checkpoint-\d+", checkpoint):
    raise ValueError(f"Invalid checkpoint name: {checkpoint}")
snapshot = Path(
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{checkpoint}/*"],
    )
)
checkpoint_path = snapshot / checkpoint
if not (checkpoint_path / "config.json").is_file():
    raise RuntimeError(
        f"{repo_id}/{checkpoint} is not a complete full-model checkpoint: "
        "config.json was not found"
    )
print(checkpoint_path)
PY
}

write_unlearningeval_config() {
    local config_path="$1"
    local model_path="$2"
    local output_dir="$3"
    python - "${config_path}" "${model_path}" "${output_dir}" \
        "${SUFFIX_BS}" "${PASS_K}" "${TEMPERATURE}" "${TOP_P}" \
        "${MAX_NEW_TOKENS}" "${DTYPE}" <<'PY'
import json
import sys

(
    config_path,
    model_path,
    output_dir,
    batch_size,
    pass_k,
    temperature,
    top_p,
    max_new_tokens,
    dtype,
) = sys.argv[1:]

pass_k = int(pass_k)
generation = {
    "max_new_tokens": int(max_new_tokens),
    "pass_k": pass_k,
    "batch_size": int(batch_size),
    "device": "auto",
    "dtype": dtype,
    "greedy": pass_k == 1,
    "do_sample": pass_k > 1,
}
if pass_k > 1:
    generation["temperature"] = float(temperature)
    generation["top_p"] = float(top_p)

config = {
    "model_name": model_path,
    "trust_remote_code": False,
    "datasets": [
        {
            "label": "forget",
            "dataset_name": "dbaysal/forget",
            "dataset_split": "train",
            "prefix_column": "secret_prefix",
            "suffix_column": "secret_suffix",
            "uuid_column": "uuid",
            "mode": "secret",
            "code_language": "python",
            "output_dir": f"{output_dir}/forget",
        },
        {
            "label": "retain",
            "dataset_name": "dbaysal/retain-full",
            "dataset_split": "train",
            "prefix_column": "prefix",
            "suffix_column": "suffix",
            "uuid_column": "uuid",
            "mode": "code",
            "code_language": "python",
            "output_dir": f"{output_dir}/retain",
        },
        {
            "label": "approximate",
            "dataset_name": "dbaysal/approximate",
            "dataset_split": "train",
            "prefix_column": "prefix",
            "suffix_column": "suffix",
            "uuid_column": "uuid",
            "mode": "code",
            "code_language": "python",
            "output_dir": f"{output_dir}/approximate",
        },
    ],
    "generation": generation,
}
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY
}

unlearningeval_complete() {
    local output_dir="$1"
    local dataset_label result_name
    for dataset_label in forget retain approximate; do
        for result_name in row_results.jsonl all_results.jsonl aggregate_results.json; do
            [ -s "${output_dir}/${dataset_label}/${result_name}" ] || return 1
        done
    done
}

run_checkpoint_job() {
    local gpu_id="$1"
    local model_key="$2"
    local repo_id="$3"
    local checkpoint="$4"
    local run_root="${OUTPUT_ROOT}/${model_key}/${checkpoint}"
    local suffix_root="${run_root}/unlearningeval"
    local evalplus_root="${run_root}/evalplus"
    local config_path="${run_root}/configs/unlearningeval.json"
    local result_path="${evalplus_root}/${EVALPLUS_DATASET}.eval_results.json"
    local filtered_result_path="${evalplus_root}/${EVALPLUS_DATASET}.filtered.eval_results.json"
    local job_status=0

    mkdir -p "${run_root}/configs" "${evalplus_root}"

    local model_path
    if [ "${DRY_RUN}" = "1" ]; then
        model_path="HUB_CACHE/${repo_id}/${checkpoint}"
    else
        model_path="$(resolve_checkpoint_path "${repo_id}" "${checkpoint}")"
    fi

    write_unlearningeval_config "${config_path}" "${model_path}" "${suffix_root}"

    echo
    echo "GPU ${gpu_id}: model=${model_key}, repo=${repo_id}, checkpoint=${checkpoint}"
    echo "GPU ${gpu_id}: cached_model=${model_path}"

    if [ "${SKIP_EXISTING}" = "1" ] && unlearningeval_complete "${suffix_root}"; then
        echo "GPU ${gpu_id}: SKIPPED completed UnlearningEvaluation"
    else
        local suffix_cmd=(
            env "CUDA_VISIBLE_DEVICES=${gpu_id}"
            python evaluate_suffix_generation.py
            --config "${config_path}"
        )
        echo "GPU ${gpu_id}: UnlearningEvaluation, batch_size=${SUFFIX_BS}, pass@${PASS_K}"
        printf '$ cd %q &&' "${UNLEARNING_EVAL_DIR}"
        printf ' %q' "${suffix_cmd[@]}"
        echo
        if [ "${DRY_RUN}" != "1" ] && ! (
            cd "${UNLEARNING_EVAL_DIR}"
            "${suffix_cmd[@]}"
        ); then
            echo "GPU ${gpu_id}: UnlearningEvaluation FAILED" >&2
            job_status=1
            if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                return "${job_status}"
            fi
        fi
    fi

    if [ "${SKIP_EXISTING}" = "1" ] && \
        [ -s "${result_path}" ] && \
        { [ ! -f "${BASELINE_FILTER_CSV}" ] || [ -s "${filtered_result_path}" ]; }; then
        echo "GPU ${gpu_id}: SKIPPED completed EvalPlus"
        return "${job_status}"
    fi

    local evalplus_cmd=(
        env
        "CUDA_VISIBLE_DEVICES=${gpu_id}"
        "PYTHONPATH=${EVALPLUS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        "EVALPLUS_TIMEOUT_PER_TASK=${EVALPLUS_TIMEOUT_PER_TASK}"
        "EVALPLUS_SANITIZE_WORKERS=${EVALPLUS_SANITIZE_WORKERS}"
        python -m evalplus.evaluate
        --model "${model_path}"
        --dataset "${EVALPLUS_DATASET}"
        --backend "${BACKEND}"
        --defer-sanitize
        --bs "${EVALPLUS_BS}"
        --force-base-prompt
        --root "${evalplus_root}"
        --output-file "${result_path}"
        --dtype "${DTYPE}"
    )
    if [ "${PASS_K}" -eq 1 ]; then
        evalplus_cmd+=(--greedy)
    else
        evalplus_cmd+=(
            --n-samples "${PASS_K}"
            --temperature "${TEMPERATURE}"
            --top-p "${TOP_P}"
        )
    fi
    if [ -n "${EVALPLUS_PARALLEL}" ]; then
        evalplus_cmd+=(--parallel "${EVALPLUS_PARALLEL}")
    fi

    echo "GPU ${gpu_id}: HumanEval + ForgetEval + UtilityEval, batch_size=${EVALPLUS_BS}, pass@${PASS_K}"
    printf '$'
    printf ' %q' "${evalplus_cmd[@]}"
    echo

    if [ "${DRY_RUN}" != "1" ] && \
        { [ "${SKIP_EXISTING}" != "1" ] || [ ! -s "${result_path}" ]; } && \
        ! "${evalplus_cmd[@]}"; then
        echo "GPU ${gpu_id}: EvalPlus FAILED" >&2
        job_status=1
        if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
            return "${job_status}"
        fi
    fi

    if [ "${DRY_RUN}" != "1" ] && [ -f "${BASELINE_FILTER_CSV}" ] && \
        { [ "${SKIP_EXISTING}" != "1" ] || [ ! -s "${filtered_result_path}" ]; }; then
        if ! python "${EVALPLUS_DIR}/tools/filter_baseline_failed_results.py" \
            "${result_path}" \
            --filter-csv "${BASELINE_FILTER_CSV}" \
            --output "${filtered_result_path}"; then
            echo "GPU ${gpu_id}: baseline-result filtering FAILED" >&2
            job_status=1
        fi
    fi

    return "${job_status}"
}

jobs=()
for model_spec in "${model_specs[@]}"; do
    IFS="|" read -r model_key repo_id <<< "${model_spec}"
    checkpoint_list="$(discover_checkpoints "${repo_id}")"
    while IFS= read -r checkpoint; do
        [ -n "${checkpoint}" ] || continue
        jobs+=("${model_key}|${repo_id}|${checkpoint}")
    done <<< "${checkpoint_list}"
done

if [ "${#jobs[@]}" -eq 0 ]; then
    echo "No checkpoint-* full models were found in the Hub repositories." >&2
    exit 1
fi

gpu_ids=()
while IFS= read -r gpu_id; do
    [ -n "${gpu_id}" ] && gpu_ids+=("${gpu_id}")
done < <(detect_gpu_ids)
if [ "${#gpu_ids[@]}" -eq 0 ]; then
    echo "No GPUs detected." >&2
    exit 1
fi

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    local status=0
    local job_index model_key repo_id checkpoint
    for ((job_index = worker_index; job_index < ${#jobs[@]}; job_index += ${#gpu_ids[@]})); do
        IFS="|" read -r model_key repo_id checkpoint <<< "${jobs[${job_index}]}"
        if ! run_checkpoint_job "${gpu_id}" "${model_key}" "${repo_id}" "${checkpoint}"; then
            status=1
            if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                return "${status}"
            fi
        fi
    done
    return "${status}"
}

echo "Full-model checkpoint evaluation suite"
echo "Qwen: ${QWEN_MODEL_REPO}"
echo "Llama: ${LLAMA_MODEL_REPO}"
echo "Queued checkpoints: ${#jobs[@]}"
echo "GPUs: ${gpu_ids[*]}"
echo "UnlearningEvaluation batch size: ${SUFFIX_BS}"
echo "EvalPlus batch size: ${EVALPLUS_BS}"
echo "EvalPlus dataset: ${EVALPLUS_DATASET}"
echo "Pass@k: ${PASS_K}"
echo "Output root: ${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}/logs"
pids=()
for worker_index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[${worker_index}]}"
    run_worker "${worker_index}" "${gpu_id}" \
        > "${OUTPUT_ROOT}/logs/gpu-${gpu_id}.txt" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
exit "${status}"
