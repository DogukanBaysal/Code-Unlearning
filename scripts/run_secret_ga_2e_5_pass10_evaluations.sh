#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Dedicated pass@10 evaluation of every full-model checkpoint produced by the
# Qwen secret-GA learning-rate run at 2e-5.
exec env \
    MODEL_KEYS=qwen2_5_coder_3b \
    QWEN_MODEL_REPO="${QWEN_MODEL_REPO:-dbaysal/secret-unlearning-qwen2_5_coder_3b-ga-full-lr-2e-5}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/secret_ga_2e_5_pass10}" \
    PASS_K=10 \
    TEMPERATURE="${TEMPERATURE:-0.8}" \
    TOP_P="${TOP_P:-0.95}" \
    EVALPLUS_DATASET=humaneval-forget-utility \
    bash "${SCRIPT_DIR}/run_full_ga_all_evals.sh" "$@"
