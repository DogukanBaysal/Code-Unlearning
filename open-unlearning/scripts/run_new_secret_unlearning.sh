#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HUB_NAMESPACE="${HUB_NAMESPACE:-dbaysal}"
MODEL_NAME_PREFIX="${MODEL_NAME_PREFIX:-new-}"
HUB_ADAPTER_ENABLED="${HUB_ADAPTER_ENABLED:-true}"
DRY_RUN="${DRY_RUN:-0}"

models=(
    "qwen2_5_coder_3b"
    "meta_llama3_2_3b"
)

methods=(
    "ga"
    "npo"
    "prod"
    "ga_gd"
    "ga_kl"
    "npo_gd"
    "npo_kl"
    "prod_gd"
    "prod_kl"
)

run_command() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Hub namespace: ${HUB_NAMESPACE}"
echo "Hub model-name prefix: ${MODEL_NAME_PREFIX}"

for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        repo_name="${MODEL_NAME_PREFIX}secret-unlearning-${model}-${method}"
        repo_id="${HUB_NAMESPACE}/${repo_name}"
        task_name_prefix="${MODEL_NAME_PREFIX//[^[:alnum:]_]/_}"
        task_name="custom_hf_${task_name_prefix}secret_${model}_${method}"

        echo "Running secret unlearning: model=${model}, method=${method}"
        echo "Task name: ${task_name}"
        echo "Hub repo: ${repo_id}"

        run_command python src/train.py \
            experiment=custom_hf_unlearning/secret \
            experiment/custom_hf_unlearning/model="${model}" \
            experiment/custom_hf_unlearning/method="${method}" \
            task_name="${task_name}" \
            hub_adapter.enabled="${HUB_ADAPTER_ENABLED}" \
            hub_adapter.repo_id="${repo_id}" \
            "$@"
    done
done
