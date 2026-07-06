#!/bin/bash
set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate/default_config.yaml}"

configs=(
    "ga_gd.yaml"
    "npo_gd.yaml"
    "prod_gd.yaml"
    "ga_kl.yaml"
    "npo_kl.yaml"
    "prod_kl.yaml"
)

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Master Port: ${MASTER_PORT}"
echo "Accelerate config: ${ACCELERATE_CONFIG}"

for config_name in "${configs[@]}"; do
    echo "Running ${config_name}"

    accelerate launch \
        --config_file "${ACCELERATE_CONFIG}" \
        --main_process_port "${MASTER_PORT}" \
        src/train.py \
        --config-name="${config_name}"
done
