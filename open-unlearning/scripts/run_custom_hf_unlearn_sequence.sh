#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

configs=(
    "ga_gd.yaml"
    "npo_gd.yaml"
    "prod_gd.yaml"
    "ga_kl.yaml"
    "npo_kl.yaml"
    "prod_kl.yaml"
)

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

for config_name in "${configs[@]}"; do
    echo "Running ${config_name}"

    python ../src/train.py \
        --config-name="${config_name}"
done
