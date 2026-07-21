#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_GROUP="ordered" exec bash "${SCRIPT_DIR}/_run_forget_utility_pass10_group.sh" "$@"
