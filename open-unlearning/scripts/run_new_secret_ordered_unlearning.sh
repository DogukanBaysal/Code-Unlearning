#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_UNLEARNING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${OPEN_UNLEARNING_ROOT}"

ORDERED_RETAIN_DATASET="${ORDERED_RETAIN_DATASET:-dbaysal/retain-half}"
RETAIN_FULL_DATASET="${RETAIN_FULL_DATASET:-dbaysal/retain-full}"
HUB_NAMESPACE="${HUB_NAMESPACE:-dbaysal}"
MODEL_NAME_PREFIX="${MODEL_NAME_PREFIX:-new-}"
HUB_ADAPTER_ENABLED="${HUB_ADAPTER_ENABLED:-true}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-${OPEN_UNLEARNING_ROOT}/outputs/new_secret_ordered_unlearning_logs}"

model_keys=(
    "qwen2_5_coder_3b"
    "meta_llama3_2_3b"
)

methods=(
    "ga_gd"
    "ga_kl"
    "npo_gd"
    "npo_kl"
    "prod_gd"
    "prod_kl"
)

variants=(
    "retain_first|${ORDERED_RETAIN_DATASET}|retain-first"
    "forget_first|${ORDERED_RETAIN_DATASET}|forget-first"
    "random|${RETAIN_FULL_DATASET}|random-retain-full"
)

detect_gpu_ids() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        if [[ "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then
            return 0
        fi
        local old_ifs="${IFS}"
        local visible_id
        IFS=","
        for visible_id in ${CUDA_VISIBLE_DEVICES}; do
            if [[ -n "${visible_id}" ]]; then
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
    if [[ -n "${gpu_id}" ]]; then
        gpu_ids+=("${gpu_id}")
    fi
done < <(detect_gpu_ids)

if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
    echo "No GPUs detected from CUDA_VISIBLE_DEVICES or nvidia-smi." >&2
    exit 1
fi

jobs=()
for variant in "${variants[@]}"; do
    IFS="|" read -r batch_order retain_dataset run_suffix <<< "${variant}"
    for model_key in "${model_keys[@]}"; do
        for method in "${methods[@]}"; do
            jobs+=(
                "${batch_order}|${model_key}|${method}|${retain_dataset}|${run_suffix}"
            )
        done
    done
done

run_cmd() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    "$@"
}

run_training_job() {
    local gpu_id="$1"
    local batch_order="$2"
    local model_key="$3"
    local method="$4"
    local retain_dataset="$5"
    local run_suffix="$6"
    shift 6

    local repo_name
    repo_name="${MODEL_NAME_PREFIX}secret-unlearning-${model_key}-${method}-${run_suffix}"
    local repo_id="${HUB_NAMESPACE}/${repo_name}"
    local task_name_prefix="${MODEL_NAME_PREFIX//[^[:alnum:]_]/_}"
    local task_name
    task_name="custom_hf_${task_name_prefix}secret_${model_key}_${method}_${run_suffix//-/_}"

    echo
    echo "GPU ${gpu_id}: task=secret, order=${batch_order}, model=${model_key}, method=${method}"
    echo "GPU ${gpu_id}: retain_dataset=${retain_dataset}"
    echo "GPU ${gpu_id}: task_name=${task_name}"
    echo "GPU ${gpu_id}: hub_repo=${repo_id}"

    run_cmd env CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python src/train.py \
        experiment=custom_hf_unlearning/secret \
        experiment/custom_hf_unlearning/model="${model_key}" \
        experiment/custom_hf_unlearning/method="${method}" \
        data.batch_mode=unpaired \
        data.batch_order="${batch_order}" \
        retain_dataset_path="${retain_dataset}" \
        task_name="${task_name}" \
        hub_adapter.enabled="${HUB_ADAPTER_ENABLED}" \
        hub_adapter.repo_id="${repo_id}" \
        "$@"
}

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    shift 2

    local status=0
    local job_index
    for ((job_index = worker_index; job_index < ${#jobs[@]}; job_index += ${#gpu_ids[@]})); do
        local batch_order
        local model_key
        local method
        local retain_dataset
        local run_suffix
        IFS="|" read -r batch_order model_key method retain_dataset run_suffix \
            <<< "${jobs[${job_index}]}"
        if ! run_training_job \
            "${gpu_id}" "${batch_order}" "${model_key}" \
            "${method}" "${retain_dataset}" "${run_suffix}" "$@"; then
            echo \
                "GPU ${gpu_id}: FAILED order=${batch_order}, model=${model_key}, method=${method}" \
                >&2
            status=1
        fi
    done
    return "${status}"
}

echo "Detected ${#gpu_ids[@]} GPU worker(s): ${gpu_ids[*]}"
echo "Queued ${#jobs[@]} secret ordered-unlearning jobs."
echo "Retain-first/forget-first dataset: ${ORDERED_RETAIN_DATASET}"
echo "Random retain-full dataset: ${RETAIN_FULL_DATASET}"
echo "Hub namespace: ${HUB_NAMESPACE}"
echo "Hub model-name prefix: ${MODEL_NAME_PREFIX}"

mkdir -p "${LOG_DIR}"

pids=()
for worker_index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[${worker_index}]}"
    log_file="${LOG_DIR}/gpu-${gpu_id}.txt"
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
