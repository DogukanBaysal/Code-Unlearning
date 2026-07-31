#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_UNLEARNING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${OPEN_UNLEARNING_ROOT}"

ORDERED_RETAIN_DATASET="${ORDERED_RETAIN_DATASET:-dbaysal/retain-half}"
RETAIN_FULL_DATASET="${RETAIN_FULL_DATASET:-dbaysal/retain-full}"
HUB_NAMESPACE="${HUB_NAMESPACE:-dbaysal}"
MODEL_NAME_PREFIX="${MODEL_NAME_PREFIX:-new-}"
HUB_ADAPTER_ENABLED="${HUB_ADAPTER_ENABLED:-true}"
DRY_RUN="${DRY_RUN:-0}"

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
    local batch_order="$1"
    local model_key="$2"
    local method="$3"
    local retain_dataset="$4"
    local run_suffix="$5"
    shift 5

    local repo_name
    repo_name="${MODEL_NAME_PREFIX}secret-unlearning-${model_key}-${method}-${run_suffix}"
    local repo_id="${HUB_NAMESPACE}/${repo_name}"
    local task_name_prefix="${MODEL_NAME_PREFIX//[^[:alnum:]_]/_}"
    local task_name
    task_name="custom_hf_${task_name_prefix}secret_${model_key}_${method}_${run_suffix//-/_}"

    echo
    echo "Running secret ordered unlearning: order=${batch_order}, model=${model_key}, method=${method}"
    echo "Retain dataset: ${retain_dataset}"
    echo "Task name: ${task_name}"
    echo "Hub repo: ${repo_id}"

    run_cmd python src/train.py \
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

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Queued 36 secret ordered-unlearning jobs."
echo "Retain-first/forget-first dataset: ${ORDERED_RETAIN_DATASET}"
echo "Random retain-full dataset: ${RETAIN_FULL_DATASET}"
echo "Hub namespace: ${HUB_NAMESPACE}"
echo "Hub model-name prefix: ${MODEL_NAME_PREFIX}"

for variant in "${variants[@]}"; do
    IFS="|" read -r batch_order retain_dataset run_suffix <<< "${variant}"
    for model_key in "${model_keys[@]}"; do
        for method in "${methods[@]}"; do
            run_training_job \
                "${batch_order}" "${model_key}" "${method}" \
                "${retain_dataset}" "${run_suffix}" "$@"
        done
    done
done
