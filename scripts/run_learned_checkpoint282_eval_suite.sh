#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNLEARNING_EVAL_DIR="${REPO_ROOT}/UnlearningEvaluation"
EVALPLUS_DIR="${REPO_ROOT}/evalplus"

cd "${REPO_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/secret_code_unit_eval_suite}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2056}"
SUFFIX_BS="${SUFFIX_BS:-32}"
EVALPLUS_BS="${EVALPLUS_BS:-128}"
EVALPLUS_DATASET="${EVALPLUS_DATASET:-humaneval-forget-utility}"
BACKEND="${BACKEND:-hf}"
DTYPE="${DTYPE:-bfloat16}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"

model_jobs=(
    "qwen2_5_coder_3b|learned_checkpoint-282|Qwen/Qwen2.5-Coder-3B|dbaysal/qwen2.5coder-3b-learned|checkpoint-282"
    "meta_llama3_2_3b|learned_checkpoint-282|meta-llama/Llama-3.2-3B|dbaysal/metallama3.2-3b-learned|checkpoint-282"
)

tasks=(
    "secret"
    "code-unit"
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
            if [ -n "${visible_id}" ]; then
                echo "${visible_id}"
            fi
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

write_suffix_config() {
    local config_path="$1"
    local base_model="$2"
    local peft_name="$3"
    local peft_subfolder="$4"
    local forget_prefix_column="$5"
    local forget_suffix_column="$6"
    local forget_mode="$7"
    local forget_output_dir="$8"
    local retain_output_dir="$9"
    local approximate_output_dir="${10}"

    mkdir -p "$(dirname "${config_path}")"
    python - "${config_path}" \
        "${base_model}" \
        "${peft_name}" \
        "${peft_subfolder}" \
        "${forget_prefix_column}" \
        "${forget_suffix_column}" \
        "${forget_mode}" \
        "${forget_output_dir}" \
        "${retain_output_dir}" \
        "${approximate_output_dir}" \
        "${MAX_NEW_TOKENS}" \
        "${SUFFIX_BS}" <<'PY'
import json
import sys

(
    config_path,
    base_model,
    peft_name,
    peft_subfolder,
    forget_prefix_column,
    forget_suffix_column,
    forget_mode,
    forget_output_dir,
    retain_output_dir,
    approximate_output_dir,
    max_new_tokens,
    suffix_bs,
) = sys.argv[1:]

config = {
    "model_name": base_model,
    "trust_remote_code": False,
    "datasets": [
        {
            "label": "forget",
            "dataset_name": "dbaysal/forget",
            "dataset_split": "train",
            "prefix_column": forget_prefix_column,
            "suffix_column": forget_suffix_column,
            "uuid_column": "uuid",
            "mode": forget_mode,
            "code_language": "python",
            "output_dir": forget_output_dir,
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
            "output_dir": retain_output_dir,
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
            "output_dir": approximate_output_dir,
        },
    ],
    "generation": {
        "max_new_tokens": int(max_new_tokens),
        "batch_size": int(suffix_bs),
        "device": "auto",
        "dtype": "auto",
        "greedy": True,
        "do_sample": False,
        "temperature": 0.2,
        "top_p": 0.8,
    },
}

if peft_name:
    config["peft_name"] = peft_name
if peft_subfolder:
    config["peft_subfolder"] = peft_subfolder

with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY
}

run_cmd() {
    echo
    printf '$'
    printf ' %q' "$@"
    echo
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi
    "$@"
}

run_eval_job() {
    local gpu_id="$1"
    local model_key="$2"
    local variant="$3"
    local base_model="$4"
    local peft_name="$5"
    local peft_subfolder="$6"
    shift 6

    local run_slug="${model_key}_${variant}"
    local evalplus_root="${OUTPUT_ROOT}/learned_checkpoint282/${model_key}/evalplus"
    local evalplus_result_path="${evalplus_root}/${EVALPLUS_DATASET}/${run_slug}.eval_results.json"

    echo
    echo "GPU ${gpu_id}: model=${model_key}, variant=${variant}"
    echo "GPU ${gpu_id}: base_model=${base_model}"
    echo "GPU ${gpu_id}: peft_name=${peft_name}, peft_subfolder=${peft_subfolder}"

    local task
    for task in "${tasks[@]}"; do
        local forget_prefix_column
        local forget_suffix_column
        local forget_mode
        if [ "${task}" = "secret" ]; then
            forget_prefix_column="secret_prefix"
            forget_suffix_column="secret_suffix"
            forget_mode="secret"
        else
            forget_prefix_column="prefix"
            forget_suffix_column="suffix"
            forget_mode="code"
        fi

        local task_output_name="${task/-/_}"
        local run_root="${OUTPUT_ROOT}/${task_output_name}/${model_key}/${variant}"
        local config_path="${run_root}/configs/suffix.json"
        local suffix_root="${run_root}/unlearningeval"

        echo "GPU ${gpu_id}: unlearning task=${task}, run_root=${run_root}"

        write_suffix_config \
            "${config_path}" \
            "${base_model}" \
            "${peft_name}" \
            "${peft_subfolder}" \
            "${forget_prefix_column}" \
            "${forget_suffix_column}" \
            "${forget_mode}" \
            "${suffix_root}/forget/${run_slug}" \
            "${suffix_root}/retain/${run_slug}" \
            "${suffix_root}/approximate/${run_slug}"

        run_cmd env CUDA_VISIBLE_DEVICES="${gpu_id}" \
            bash -c 'cd "$1" && shift && "$@"' _ \
            "${UNLEARNING_EVAL_DIR}" \
            python evaluate_suffix_generation.py --config "${config_path}"
    done

    mkdir -p "$(dirname "${evalplus_result_path}")"

    local evalplus_cmd=(
        env
        "CUDA_VISIBLE_DEVICES=${gpu_id}"
        "PYTHONPATH=${EVALPLUS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        python -m evalplus.evaluate
        --model "${base_model}"
        --dataset "${EVALPLUS_DATASET}"
        --backend "${BACKEND}"
        --greedy
        --defer-sanitize
        --bs "${EVALPLUS_BS}"
        --force-base-prompt
        --root "${evalplus_root}"
        --output-file "${evalplus_result_path}"
        --dtype "${DTYPE}"
        --peft-name "${peft_name}"
        --peft-subfolder "${peft_subfolder}"
    )

    run_cmd "${evalplus_cmd[@]}" "$@"
}

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    shift 2

    local status=0
    local job_index
    for ((job_index = worker_index; job_index < ${#model_jobs[@]}; job_index += ${#gpu_ids[@]})); do
        local job="${model_jobs[${job_index}]}"
        local model_key
        local variant
        local base_model
        local peft_name
        local peft_subfolder
        IFS="|" read -r model_key variant base_model peft_name peft_subfolder <<< "${job}"

        if ! run_eval_job "${gpu_id}" "${model_key}" "${variant}" "${base_model}" "${peft_name}" "${peft_subfolder}" "$@"; then
            status=1
            if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                return "${status}"
            fi
        fi
    done
    return "${status}"
}

gpu_ids=()
while IFS= read -r gpu_id; do
    if [ -n "${gpu_id}" ]; then
        gpu_ids+=("${gpu_id}")
    fi
done < <(detect_gpu_ids)

if [ "${#gpu_ids[@]}" -eq 0 ]; then
    echo "No GPUs detected from CUDA_VISIBLE_DEVICES or nvidia-smi." >&2
    exit 1
fi

echo "Detected ${#gpu_ids[@]} GPU worker(s): ${gpu_ids[*]}"
echo "Queued ${#model_jobs[@]} learned checkpoint-282 model evaluation job(s)."
echo "Output root: ${OUTPUT_ROOT}"

log_dir="${OUTPUT_ROOT}/logs"
mkdir -p "${log_dir}"

pids=()
for worker_index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[${worker_index}]}"
    log_file="${log_dir}/learned-checkpoint282-gpu-${gpu_id}.txt"
    echo "GPU ${gpu_id} log: ${log_file}"
    run_worker "${worker_index}" "${gpu_id}" "$@" > "${log_file}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
