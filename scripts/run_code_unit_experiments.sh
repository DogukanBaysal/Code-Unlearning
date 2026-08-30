#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/thesis_workflow.bash
source "${SCRIPT_DIR}/lib/thesis_workflow.bash"

run_thesis_workflow \
    "code-unit" \
    "run_code_unit_experiments.sh" \
    "Run the complete standard code-unit-unlearning matrix, then evaluate every checkpoint." \
    "$@"
