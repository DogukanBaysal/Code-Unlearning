#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

runs=(
    "ga_gd.yaml ga-gd 5e-5"
    "npo_gd.yaml npo-gd 1e-4"
    "prod_gd.yaml prod-gd 3e-4"
    "ga_kl.yaml ga-kl 5e-5"
    "npo_kl.yaml npo-kl 1e-4"
    "prod_kl.yaml prod-kl 3e-4"
)

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

for run in "${runs[@]}"; do
    read -r config_name method_name learning_rate <<< "${run}"
    repo_id="dbaysal/qwen2.5coder-3b-unlearned-${method_name}-lr-${learning_rate}"

    echo "Running ${config_name}"
    echo "Hub repo: ${repo_id}"

    python src/train.py \
        --config-name="${config_name}" \
        hub_adapter.enabled=true \
        hub_adapter.repo_id="${repo_id}"
done
