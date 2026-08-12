#!/usr/bin/env bash
# Train and upload new- prefixed Code-unit NPO+KL LoRA adapters for both models.
# Code-unit sequences use the task-specific batch configuration: per-device
# batch size 4 with 8 accumulation steps (effective batch size 32 per GPU).
#
# The output repository IDs are also the Code-unit adapter IDs evaluated by
# run_new_secret_npo_kl_eval_suite.sh:
#   dbaysal/new-code-unit-unlearning-qwen2_5_coder_3b-npo_kl
#   dbaysal/new-code-unit-unlearning-meta_llama3_2_3b-npo_kl
#
# Usage:
#   bash scripts/run_new_code_unit_npo_kl_unlearning.sh
#   DRY_RUN=1 bash scripts/run_new_code_unit_npo_kl_unlearning.sh
#   HUB_NAMESPACE=your-org ADAPTER_PREFIX=experiment- \
#     bash scripts/run_new_code_unit_npo_kl_unlearning.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPEN_UNLEARNING_ROOT="${REPO_ROOT}/open-unlearning"
cd "${OPEN_UNLEARNING_ROOT}"

HUB_NAMESPACE="${HUB_NAMESPACE:-dbaysal}"
ADAPTER_PREFIX="${ADAPTER_PREFIX:-new-}"
HUB_ADAPTER_ENABLED="${HUB_ADAPTER_ENABLED:-true}"
DRY_RUN="${DRY_RUN:-0}"

model_keys=(
    "qwen2_5_coder_3b"
    "meta_llama3_2_3b"
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

task_name_prefix="${ADAPTER_PREFIX//[^[:alnum:]_]/_}"

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Hub namespace: ${HUB_NAMESPACE}"
echo "Hub adapter prefix: ${ADAPTER_PREFIX}"
echo "Hub uploads enabled: ${HUB_ADAPTER_ENABLED}"

for model_key in "${model_keys[@]}"; do
    repo_id="${HUB_NAMESPACE}/${ADAPTER_PREFIX}code-unit-unlearning-${model_key}-npo_kl"
    task_name="custom_hf_${task_name_prefix}code_unit_${model_key}_npo_kl"

    echo
    echo "Training Code-unit NPO+KL: model=${model_key}"
    echo "Task name: ${task_name}"
    echo "Hub repository: ${repo_id}"

    run_command python src/train.py \
        experiment=custom_hf_unlearning/code_unit \
        experiment/custom_hf_unlearning/model="${model_key}" \
        experiment/custom_hf_unlearning/method=npo_kl \
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=16 \
        task_name="${task_name}" \
        hub_adapter.enabled="${HUB_ADAPTER_ENABLED}" \
        hub_adapter.repo_id="${repo_id}" \
        "$@"
done
