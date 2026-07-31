#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"

if [ "${RUN_IN_BACKGROUND}" = "0" ]; then
    EVAL_GROUP="ordered" exec bash \
        "${SCRIPT_DIR}/_run_forget_utility_pass10_group.sh" "$@"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/ordered_forget_utility_pass10}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${OUTPUT_ROOT}/logs}"
mkdir -p "${LAUNCH_LOG_DIR}"

timestamp="$(date '+%Y%m%d-%H%M%S')"
launch_log="${LAUNCH_LOG_DIR}/dispatcher-${timestamp}.log"
pid_file="${LAUNCH_LOG_DIR}/dispatcher-${timestamp}.pid"

nohup env EVAL_GROUP="ordered" \
    bash "${SCRIPT_DIR}/_run_forget_utility_pass10_group.sh" "$@" \
    > "${launch_log}" 2>&1 < /dev/null &
dispatcher_pid="$!"
printf '%s\n' "${dispatcher_pid}" > "${pid_file}"

echo "Ordering evaluation started in the background."
echo "PID: ${dispatcher_pid}"
echo "PID file: ${pid_file}"
echo "Dispatcher log: ${launch_log}"
