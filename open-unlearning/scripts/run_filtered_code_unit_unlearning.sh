#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

models=(
    "qwen2_5_coder_3b"
    "meta_llama3_2_3b"
)

methods=(
    "prod"
    "ga"
    "npo"
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
for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        jobs+=("${model}|${method}")
    done
done

run_train_job() {
    local gpu_id="$1"
    local model="$2"
    local method="$3"

    local repo_id="dbaysal/filtered-code-unit-unlearning-${model}-${method}"

    echo
    echo "GPU ${gpu_id}: filtered code-unit unlearning: model=${model}, method=${method}"
    echo "GPU ${gpu_id}: hub repo=${repo_id}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python src/train.py \
        experiment=custom_hf_unlearning/filtered_code_unit \
        experiment/custom_hf_unlearning/model="${model}" \
        experiment/custom_hf_unlearning/method="${method}" \
        hub_adapter.enabled=true \
        hub_adapter.repo_id="${repo_id}"
}

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"

    local job_index
    for ((job_index = worker_index; job_index < ${#jobs[@]}; job_index += ${#gpu_ids[@]})); do
        local job="${jobs[${job_index}]}"
        local model
        local method
        IFS="|" read -r model method <<< "${job}"
        run_train_job "${gpu_id}" "${model}" "${method}"
    done
}

echo "Detected ${#gpu_ids[@]} GPU worker(s): ${gpu_ids[*]}"
echo "Queued ${#jobs[@]} filtered code-unit unlearning job(s). Each worker runs its assigned jobs sequentially."

pids=()
for worker_index in "${!gpu_ids[@]}"; do
    run_worker "${worker_index}" "${gpu_ids[${worker_index}]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
