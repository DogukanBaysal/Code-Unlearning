#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Full-parameter GA unlearning for Llama at 2e-5. The shared runner saves every
# epoch as a complete model and uploads checkpoint subfolders to the Hub.
exec env \
    MODEL_KEYS=meta_llama3_2_3b \
    LEARNING_RATES="${LEARNING_RATES:-2e-5}" \
    LLAMA_SOURCE_REPO="${LLAMA_SOURCE_REPO:-dbaysal/metallama3.2-3b-learned-checkpoint282-full}" \
    LLAMA_TOKENIZER_REPO="${LLAMA_TOKENIZER_REPO:-meta-llama/Llama-3.2-3B}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/secret_ga_llama_2e_5}" \
    HUB_UPLOAD="${HUB_UPLOAD:-1}" \
    RUN_UNLEARNING=1 \
    RUN_EVALUATIONS=0 \
    bash "${SCRIPT_DIR}/run_secret_ga_full_grid_search.sh" "$@"
