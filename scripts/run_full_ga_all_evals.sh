#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNLEARNING_EVAL_DIR="${REPO_ROOT}/UnlearningEvaluation"
EVALPLUS_DIR="${REPO_ROOT}/evalplus"

QWEN_MODEL_REPO="${QWEN_MODEL_REPO:-dbaysal/secret-unlearning-qwen-checkpoint282-ga-full}"
LLAMA_MODEL_REPO="${LLAMA_MODEL_REPO:-dbaysal/secret-unlearning-llama-checkpoint282-ga-full}"
MODEL_KEYS="${MODEL_KEYS:-qwen2_5_coder_3b meta_llama3_2_3b}"
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
AGGREGATE_FILTER_CSV="${AGGREGATE_FILTER_CSV-${UNLEARNING_EVAL_DIR}/non_exact_matches.csv}"
BASELINE_FILTER_CSV="${BASELINE_FILTER_CSV-${EVALPLUS_DIR}/evalplus/baseline_failed_test_ids.csv}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ -n "${PYTHON_BIN:-}" ]; then
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Configured PYTHON_BIN was not found: ${PYTHON_BIN}" >&2
        exit 127
    fi
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Python was not found. Activate the project environment or set PYTHON_BIN." >&2
    exit 127
fi

for boolean_name in SKIP_EXISTING CONTINUE_ON_ERROR DRY_RUN; do
    boolean_value="${!boolean_name}"
    if [ "${boolean_value}" != "0" ] && [ "${boolean_value}" != "1" ]; then
        echo "${boolean_name} must be 0 or 1: ${boolean_value}" >&2
        exit 2
    fi
done
for filter_spec in \
    "AGGREGATE_FILTER_CSV=${AGGREGATE_FILTER_CSV}" \
    "BASELINE_FILTER_CSV=${BASELINE_FILTER_CSV}"; do
    filter_name="${filter_spec%%=*}"
    filter_path="${filter_spec#*=}"
    if [ -n "${filter_path}" ] && [ ! -f "${filter_path}" ]; then
        echo "${filter_name} was not found: ${filter_path}" >&2
        exit 2
    fi
done
for integer_spec in \
    "SUFFIX_BS=${SUFFIX_BS}" \
    "EVALPLUS_BS=${EVALPLUS_BS}" \
    "PASS_K=${PASS_K}" \
    "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}" \
    "EVALPLUS_TIMEOUT_PER_TASK=${EVALPLUS_TIMEOUT_PER_TASK}" \
    "EVALPLUS_SANITIZE_WORKERS=${EVALPLUS_SANITIZE_WORKERS}"; do
    integer_name="${integer_spec%%=*}"
    integer_value="${integer_spec#*=}"
    if ! [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${integer_name} must be a positive integer: ${integer_value}" >&2
        exit 2
    fi
done
if [ -n "${EVALPLUS_PARALLEL}" ] && \
    ! [[ "${EVALPLUS_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVALPLUS_PARALLEL must be a positive integer: ${EVALPLUS_PARALLEL}" >&2
    exit 2
fi

model_specs=()
normalized_model_keys="${MODEL_KEYS//,/ }"
read -r -a model_keys <<< "${normalized_model_keys}"
for model_key in "${model_keys[@]}"; do
    case "${model_key}" in
        qwen2_5_coder_3b)
            model_specs+=("${model_key}|${QWEN_MODEL_REPO}")
            ;;
        meta_llama3_2_3b)
            model_specs+=("${model_key}|${LLAMA_MODEL_REPO}")
            ;;
        *)
            echo "Unsupported MODEL_KEYS entry: ${model_key}" >&2
            exit 2
            ;;
    esac
done
if [ "${#model_specs[@]}" -eq 0 ]; then
    echo "MODEL_KEYS must contain at least one supported model." >&2
    exit 2
fi

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
    if [ "${DRY_RUN}" = "1" ]; then
        local epoch_index
        for epoch_index in 1 2 3; do
            echo "checkpoint-${epoch_index}"
        done
        return 0
    fi
    "${PYTHON_BIN}" - "${repo_id}" <<'PY'
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
    "${PYTHON_BIN}" - "${repo_id}" "${checkpoint}" <<'PY'
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
weight_files = (
    checkpoint_path / "model.safetensors",
    checkpoint_path / "model.safetensors.index.json",
    checkpoint_path / "pytorch_model.bin",
    checkpoint_path / "pytorch_model.bin.index.json",
)
if not (checkpoint_path / "config.json").is_file() or not any(
    path.is_file() for path in weight_files
):
    raise RuntimeError(
        f"{repo_id}/{checkpoint} is not a complete full-model checkpoint"
    )
if not (checkpoint_path / "tokenizer_config.json").is_file():
    raise RuntimeError(f"{repo_id}/{checkpoint} does not contain the saved tokenizer")
print(checkpoint_path)
PY
}

write_unlearningeval_config() {
    local config_path="$1"
    local model_path="$2"
    local output_dir="$3"
    "${PYTHON_BIN}" - "${config_path}" "${model_path}" "${output_dir}" \
        "${SUFFIX_BS}" "${PASS_K}" "${TEMPERATURE}" "${TOP_P}" \
        "${MAX_NEW_TOKENS}" "${DTYPE}" "${AGGREGATE_FILTER_CSV}" <<'PY'
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
    aggregate_filter_csv,
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
            "dataset_name": "dbaysal/retain-half",
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
if aggregate_filter_csv:
    config["aggregate_filter_csv"] = aggregate_filter_csv
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
        if [ -n "${AGGREGATE_FILTER_CSV}" ]; then
            [ -s "${output_dir}/${dataset_label}/aggregate_results_filtered.json" ] || return 1
        fi
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
        echo "GPU ${gpu_id}: SKIPPED completed filtered UnlearningEvaluation"
    else
        local suffix_cmd=(
            env "CUDA_VISIBLE_DEVICES=${gpu_id}"
            "${PYTHON_BIN}" evaluate_suffix_generation.py
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
        { [ -z "${BASELINE_FILTER_CSV}" ] || [ -s "${filtered_result_path}" ]; }; then
        echo "GPU ${gpu_id}: SKIPPED completed filtered EvalPlus"
        return "${job_status}"
    fi

    local evalplus_cmd=(
        env
        "CUDA_VISIBLE_DEVICES=${gpu_id}"
        "PYTHONPATH=${EVALPLUS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        "EVALPLUS_TIMEOUT_PER_TASK=${EVALPLUS_TIMEOUT_PER_TASK}"
        "EVALPLUS_SANITIZE_WORKERS=${EVALPLUS_SANITIZE_WORKERS}"
        "${PYTHON_BIN}" -m evalplus.evaluate
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

    local filter_cmd=()
    if [ -n "${BASELINE_FILTER_CSV}" ]; then
        filter_cmd=(
            "${PYTHON_BIN}" "${EVALPLUS_DIR}/tools/filter_baseline_failed_results.py"
            "${result_path}"
            --filter-csv "${BASELINE_FILTER_CSV}"
            --output "${filtered_result_path}"
        )
    fi

    echo "GPU ${gpu_id}: HumanEval + ForgetEval + UtilityEval, batch_size=${EVALPLUS_BS}, pass@${PASS_K}"
    printf '$'
    printf ' %q' "${evalplus_cmd[@]}"
    echo
    if [ "${#filter_cmd[@]}" -gt 0 ]; then
        printf '$'
        printf ' %q' "${filter_cmd[@]}"
        echo
    fi

    if [ "${DRY_RUN}" != "1" ]; then
        if [ "${SKIP_EXISTING}" = "1" ] && [ -s "${result_path}" ]; then
            echo "GPU ${gpu_id}: SKIPPED existing raw EvalPlus result"
        elif ! "${evalplus_cmd[@]}"; then
            echo "GPU ${gpu_id}: EvalPlus FAILED" >&2
            job_status=1
            if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                return "${job_status}"
            fi
        fi

        if [ "${#filter_cmd[@]}" -gt 0 ]; then
            if [ "${SKIP_EXISTING}" = "1" ] && [ -s "${filtered_result_path}" ]; then
                echo "GPU ${gpu_id}: SKIPPED existing filtered EvalPlus result"
            elif [ ! -s "${result_path}" ]; then
                echo "GPU ${gpu_id}: cannot filter missing EvalPlus result: ${result_path}" >&2
                job_status=1
            elif ! "${filter_cmd[@]}"; then
                echo "GPU ${gpu_id}: baseline-result filtering FAILED" >&2
                job_status=1
            fi
        fi
    fi

    return "${job_status}"
}

jobs=()
discovery_status=0
for model_spec in "${model_specs[@]}"; do
    IFS="|" read -r model_key repo_id <<< "${model_spec}"
    if ! checkpoint_list="$(discover_checkpoints "${repo_id}")"; then
        echo "Checkpoint discovery failed for ${repo_id}" >&2
        discovery_status=1
        if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
            break
        fi
        continue
    fi
    epoch_index=0
    while IFS= read -r checkpoint; do
        [ -n "${checkpoint}" ] || continue
        if ! [[ "${checkpoint}" =~ ^checkpoint-[0-9]+$ ]]; then
            echo "Invalid checkpoint name from ${repo_id}: ${checkpoint}" >&2
            discovery_status=1
            if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                break
            fi
            continue
        fi
        epoch_index=$((epoch_index + 1))
        jobs+=("${model_key}|${repo_id}|${checkpoint}|${epoch_index}")
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

launch_job() {
    local worker_index="$1"
    local job_index="$2"
    local gpu_id="${gpu_ids[${worker_index}]}"
    local model_key repo_id checkpoint epoch_index
    IFS="|" read -r model_key repo_id checkpoint epoch_index \
        <<< "${jobs[${job_index}]}"
    local log_file="${OUTPUT_ROOT}/logs/gpu-${gpu_id}.txt"

    (
        echo
        echo "Dispatcher: assigned checkpoint job $((job_index + 1))/${#jobs[@]} to GPU ${gpu_id}"
        echo "Dispatcher: model=${model_key}, epoch=${epoch_index}, checkpoint=${checkpoint}"
        if ! run_checkpoint_job "${gpu_id}" "${model_key}" "${repo_id}" "${checkpoint}"; then
            echo "GPU ${gpu_id}: FAILED model=${model_key}, epoch=${epoch_index}, checkpoint=${checkpoint}" >&2
            exit 1
        fi
    ) >> "${log_file}" 2>&1 &
    pids[${worker_index}]="$!"
}

echo "Full-model checkpoint evaluation suite"
for model_spec in "${model_specs[@]}"; do
    IFS="|" read -r model_key repo_id <<< "${model_spec}"
    echo "${model_key}: ${repo_id}"
done
echo "Queued ${#jobs[@]} checkpoint jobs; the next pending epoch goes to the first free GPU."
echo "GPUs: ${gpu_ids[*]}"
echo "UnlearningEvaluation batch size: ${SUFFIX_BS}"
echo "EvalPlus batch size: ${EVALPLUS_BS}"
echo "EvalPlus dataset: ${EVALPLUS_DATASET}"
echo "Pass@k: ${PASS_K}"
if [ -n "${AGGREGATE_FILTER_CSV}" ]; then
    echo "UnlearningEvaluation filter: ${AGGREGATE_FILTER_CSV}"
else
    echo "UnlearningEvaluation filter: disabled"
fi
if [ -n "${BASELINE_FILTER_CSV}" ]; then
    echo "EvalPlus baseline filter: ${BASELINE_FILTER_CSV}"
else
    echo "EvalPlus baseline filter: disabled"
fi
echo "Output root: ${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}/logs"
pids=()
for worker_index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[${worker_index}]}"
    log_file="${OUTPUT_ROOT}/logs/gpu-${gpu_id}.txt"
    : > "${log_file}"
    echo "GPU ${gpu_id} log: ${log_file}"
    pids+=("")
done

status="${discovery_status}"
next_job_index=0
active_jobs=0

for worker_index in "${!gpu_ids[@]}"; do
    if [ "${next_job_index}" -ge "${#jobs[@]}" ]; then
        break
    fi
    launch_job "${worker_index}" "${next_job_index}"
    next_job_index=$((next_job_index + 1))
    active_jobs=$((active_jobs + 1))
done

while [ "${active_jobs}" -gt 0 ]; do
    completion_found=0
    for worker_index in "${!gpu_ids[@]}"; do
        pid="${pids[${worker_index}]:-}"
        if [ -z "${pid}" ] || kill -0 "${pid}" 2>/dev/null; then
            continue
        fi

        job_failed=0
        if ! wait "${pid}"; then
            status=1
            job_failed=1
        fi
        pids[${worker_index}]=""
        active_jobs=$((active_jobs - 1))
        completion_found=1

        if [ "${job_failed}" = "1" ] && [ "${CONTINUE_ON_ERROR}" != "1" ]; then
            next_job_index="${#jobs[@]}"
        fi
        if [ "${next_job_index}" -lt "${#jobs[@]}" ]; then
            launch_job "${worker_index}" "${next_job_index}"
            next_job_index=$((next_job_index + 1))
            active_jobs=$((active_jobs + 1))
        fi
    done

    if [ "${completion_found}" -eq 0 ]; then
        sleep 1
    fi
done

exit "${status}"
