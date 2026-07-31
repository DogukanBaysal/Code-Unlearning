#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVALPLUS_ROOT="${REPO_ROOT}/evalplus"
cd "${REPO_ROOT}"

DATASET="forget-utility"
PASS_K=5
EVALPLUS_BS="${EVALPLUS_BS:-5}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
DTYPE="${DTYPE:-bfloat16}"
BACKEND="${BACKEND:-hf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/learned_checkpoint282_evalplus_pass5}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

model_jobs=(
    "qwen2_5_coder_3b|Qwen/Qwen2.5-Coder-3B|dbaysal/qwen2.5coder-3b-learned"
    "meta_llama3_2_3b|meta-llama/Llama-3.2-3B|dbaysal/metallama3.2-3b-learned"
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

run_eval_job() {
    local gpu_id="$1"
    local model_key="$2"
    local base_model="$3"
    local peft_name="$4"
    shift 4

    local checkpoint="checkpoint-282"
    local adapter_slug="${peft_name//\//--}"
    local run_slug="${adapter_slug}_${checkpoint}"
    local model_output_root="${OUTPUT_ROOT}/${model_key}/evalplus"
    local result_dir="${model_output_root}/${DATASET}/pass-${PASS_K}"
    local result_path="${result_dir}/${run_slug}.eval_results.json"

    if [ "${SKIP_EXISTING}" = "1" ] && [ -s "${result_path}" ]; then
        echo "GPU ${gpu_id}: SKIPPED completed ${model_key}: ${result_path}"
        return 0
    fi

    mkdir -p "${result_dir}"
    echo
    echo "GPU ${gpu_id}: model=${model_key}, checkpoint=${checkpoint}"
    echo "GPU ${gpu_id}: adapter=${peft_name}"
    echo "GPU ${gpu_id}: dataset=${DATASET}, pass@${PASS_K}, batch_size=${EVALPLUS_BS}"
    echo "GPU ${gpu_id}: output=${result_path}"

    local pythonpath="${EVALPLUS_ROOT}"
    if [ -n "${PYTHONPATH:-}" ]; then
        pythonpath="${pythonpath}:${PYTHONPATH}"
    fi

    local cmd=(
        env
        "CUDA_VISIBLE_DEVICES=${gpu_id}"
        "PYTHONPATH=${pythonpath}"
        python -m evalplus.evaluate
        --model "${base_model}"
        --peft-name "${peft_name}"
        --peft-subfolder "${checkpoint}"
        --dataset "${DATASET}"
        --backend "${BACKEND}"
        --defer-sanitize
        --bs "${EVALPLUS_BS}"
        --n-samples "${PASS_K}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --force-base-prompt
        --root "${model_output_root}"
        --output-file "${result_path}"
        --dtype "${DTYPE}"
    )

    printf '$'
    printf ' %q' "${cmd[@]}" "$@"
    echo
    if [ "${DRY_RUN}" != "1" ]; then
        "${cmd[@]}" "$@"
    fi
}

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    shift 2

    local status=0
    local job_index
    for ((job_index = worker_index; job_index < ${#model_jobs[@]}; job_index += ${#gpu_ids[@]})); do
        local model_key
        local base_model
        local peft_name
        IFS="|" read -r model_key base_model peft_name <<< "${model_jobs[${job_index}]}"
        if ! run_eval_job \
            "${gpu_id}" "${model_key}" "${base_model}" "${peft_name}" "$@"; then
            echo "GPU ${gpu_id}: FAILED model=${model_key}, checkpoint=checkpoint-282" >&2
            status=1
        fi
    done
    return "${status}"
}

echo "Detected ${#gpu_ids[@]} GPU worker(s): ${gpu_ids[*]}"
echo "Queued ${#model_jobs[@]} learned checkpoint-282 EvalPlus jobs."
echo "EvalPlus: dataset=${DATASET}, pass@${PASS_K}, batch_size=${EVALPLUS_BS}, temperature=${TEMPERATURE}, top_p=${TOP_P}"
echo "Each full batch generates ${EVALPLUS_BS} x ${PASS_K} sequences."

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
