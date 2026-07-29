#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env \
    RUN_UNLEARNING=0 \
    RUN_EVALUATIONS=1 \
    EVAL_MODEL_SOURCE=hub \
    bash "${SCRIPT_DIR}/run_secret_ga_full_grid_search.sh" "$@"
