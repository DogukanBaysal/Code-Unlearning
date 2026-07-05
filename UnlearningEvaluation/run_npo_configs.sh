#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-1e-6-checkpoint4/config.yaml"
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-1e-6-checkpoint8/config.yaml"
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-1e-6-checkpoint12/config.yaml"
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-5e-6-checkpoint4/config.yaml"
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-5e-6-checkpoint8/config.yaml"
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-5e-6-checkpoint12/config.yaml"
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-1e-4-checkpoint4/config.yaml"
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-1e-4-checkpoint8/config.yaml"
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-1e-4-checkpoint12/config.yaml"
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-1e-5-checkpoint4/config.yaml"
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-1e-5-checkpoint8/config.yaml"
  "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-1e-5-checkpoint12/config.yaml"
  # "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-5e-5-checkpoint4/config.yaml"
  # "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-5e-5-checkpoint8/config.yaml"
  # "configs/npo/qwen2.5coder-3b-unlearned-npo-lr-5e-5-checkpoint12/config.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python evaluate_suffix_generation.py --config "${config}"
done
