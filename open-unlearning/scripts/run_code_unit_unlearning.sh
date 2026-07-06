#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

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

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        repo_id="dbaysal/code-unit-unlearning-${model}-${method}"
        echo "Running code-unit unlearning: model=${model}, method=${method}"
        echo "Hub repo: ${repo_id}"
        python src/train.py \
            experiment=custom_hf_unlearning/code_unit \
            experiment/custom_hf_unlearning/model="${model}" \
            experiment/custom_hf_unlearning/method="${method}" \
            hub_adapter.enabled=true \
            hub_adapter.repo_id="${repo_id}"
    done
done
