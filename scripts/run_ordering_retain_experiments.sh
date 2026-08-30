#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/thesis_workflow.bash
source "${SCRIPT_DIR}/lib/thesis_workflow.bash"

run_thesis_workflow \
    "ordering-retain" \
    "run_ordering_retain_experiments.sh" \
    "Run the secret objective-ordering and retain-set-size variants, then evaluate every checkpoint." \
    "$@"
