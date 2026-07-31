#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Phase 1/2: running 18 standard secret-unlearning jobs"
bash "${SCRIPT_DIR}/run_new_secret_unlearning.sh" "$@"

echo
echo "Phase 2/2: running 36 ordered secret-unlearning jobs"
bash "${SCRIPT_DIR}/run_new_secret_ordered_unlearning.sh" "$@"

echo
echo "Completed all 54 secret-unlearning jobs."
