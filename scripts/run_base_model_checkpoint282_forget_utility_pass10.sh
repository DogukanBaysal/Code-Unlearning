#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_GROUP="base-checkpoint282" exec bash "${SCRIPT_DIR}/_run_forget_utility_pass10_group.sh" "$@"
