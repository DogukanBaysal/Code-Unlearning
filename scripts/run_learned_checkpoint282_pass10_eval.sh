#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PASS_K="${PASS_K:-10}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2056}"
SUFFIX_BS="${SUFFIX_BS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/learned_checkpoint282_pass10_eval}"

model_specs=(
    "qwen2_5_coder_3b|Qwen/Qwen2.5-Coder-3B|dbaysal/qwen2.5coder-3b-learned"
    "meta_llama3_2_3b|meta-llama/Llama-3.2-3B|dbaysal/metallama3.2-3b-learned"
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

jobs=()
for task in "${tasks[@]}"; do
    for model_spec in "${model_specs[@]}"; do
        jobs+=("${task}|${model_spec}")
    done
done

run_eval_job() {
    local gpu_id="$1"
    local task="$2"
    local model_key="$3"
    local base_model="$4"
    local peft_name="$5"
    shift 5

    local task_output_name="${task/-/_}"
    local output_root="${OUTPUT_ROOT}/${task_output_name}/${model_key}"
    local forget_prefix_column="prefix"
    local forget_suffix_column="suffix"
    local forget_mode="code"

    if [ "${task}" = "secret" ]; then
        forget_prefix_column="secret_prefix"
        forget_suffix_column="secret_suffix"
        forget_mode="secret"
    fi

    echo
    echo "GPU ${gpu_id}: task=${task}, model=${model_key}, checkpoint=checkpoint-282"
    echo "GPU ${gpu_id}: adapter=${peft_name}"
    echo "GPU ${gpu_id}: output=${output_root}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/run_adapter_eval_suite.py \
        --model "${base_model}" \
        --peft-names "${peft_name}" \
        --checkpoints "checkpoint-282" \
        --output-root "${output_root}" \
        --forget-dataset "dbaysal/forget" \
        --forget-prefix-column "${forget_prefix_column}" \
        --forget-suffix-column "${forget_suffix_column}" \
        --forget-mode "${forget_mode}" \
        --retain-dataset "dbaysal/retain-full" \
        --retain-prefix-column "prefix" \
        --retain-suffix-column "suffix" \
        --retain-mode "code" \
        --approx-dataset "dbaysal/approximate" \
        --approx-prefix-column "prefix" \
        --approx-suffix-column "suffix" \
        --approx-mode "code" \
        --pass-k "${PASS_K}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --suffix-bs "${SUFFIX_BS}" \
        --skip-evalplus \
        --continue-on-error \
        "$@"
}

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    shift 2

    local status=0
    local job_index
    for ((job_index = worker_index; job_index < ${#jobs[@]}; job_index += ${#gpu_ids[@]})); do
        local task
        local model_key
        local base_model
        local peft_name
        IFS="|" read -r task model_key base_model peft_name <<< "${jobs[${job_index}]}"
        if ! run_eval_job \
            "${gpu_id}" "${task}" "${model_key}" "${base_model}" "${peft_name}" "$@"; then
            echo "GPU ${gpu_id}: FAILED task=${task}, model=${model_key}" >&2
            status=1
        fi
    done
    return "${status}"
}

echo "Detected ${#gpu_ids[@]} GPU worker(s): ${gpu_ids[*]}"
echo "Queued ${#jobs[@]} learned checkpoint-282 jobs without UUID filtering."
echo "Suffix evaluation: pass@${PASS_K}, temperature=${TEMPERATURE}, top_p=${TOP_P}, batch_size=${SUFFIX_BS}"

log_dir="${OUTPUT_ROOT}/logs"
mkdir -p "${log_dir}"

pids=()
for worker_index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[${worker_index}]}"
    log_file="${log_dir}/gpu-${gpu_id}.txt"
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
