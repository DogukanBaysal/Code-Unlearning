#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env \
    RUN_UNLEARNING=1 \
    RUN_EVALUATIONS=0 \
    bash "${SCRIPT_DIR}/run_secret_ga_full_grid_search.sh" "$@"
