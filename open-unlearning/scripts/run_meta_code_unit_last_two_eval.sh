#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_UNLEARNING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${OPEN_UNLEARNING_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

model_key="meta_llama3_2_3b"
base_model="meta-llama/Llama-3.2-3B"

methods=(
    "ga_gd"
)

detect_gpu_ids() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        if [ "${CUDA_VISIBLE_DEVICES}" = "-1" ]; then
            return 0
        fi
        local old_ifs="${IFS}"
        local visible_id
        IFS=","
        for visible_id in ${CUDA_VISIBLE_DEVICES}; do
            if [ -n "${visible_id}" ]; then
                echo "${visible_id}"
            fi
        done
        IFS="${old_ifs}"
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits
        return 0
    fi

    echo "0"
}

gpu_ids=()
while IFS= read -r gpu_id; do
    if [ -n "${gpu_id}" ]; then
        gpu_ids+=("${gpu_id}")
    fi
done < <(detect_gpu_ids)

if [ "${#gpu_ids[@]}" -eq 0 ]; then
    echo "No GPUs detected from CUDA_VISIBLE_DEVICES or nvidia-smi." >&2
    exit 1
fi

run_eval_job() {
    local gpu_id="$1"
    local method="$2"
    shift 2

    local peft_name="dbaysal/code-unit-unlearning-${model_key}-${method}"
    local output_root="${REPO_ROOT}/Results/secret_code_unit_eval_suite/code_unit/${model_key}/${method}"

    echo
    echo "GPU ${gpu_id}: task=code-unit, model=${model_key}, method=${method}"
    echo "GPU ${gpu_id}: base_model=${base_model}"
    echo "GPU ${gpu_id}: peft_name=${peft_name}"
    echo "GPU ${gpu_id}: output_root=${output_root}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/run_adapter_eval_suite.py \
        --model "${base_model}" \
        --peft-names "${peft_name}" \
        --discover-checkpoints \
        --num-checkpoints 2 \
        --checkpoint-selection last \
        --alias-checkpoints-as-epochs \
        --checkpoint-alias-start 2 \
        --output-root "${output_root}" \
        --forget-dataset "dbaysal/forget" \
        --forget-prefix-column "prefix" \
        --forget-suffix-column "suffix" \
        --forget-mode "code" \
        --retain-dataset "dbaysal/retain-half" \
        --retain-prefix-column "prefix" \
        --retain-suffix-column "suffix" \
        --retain-mode "code" \
        --approx-dataset "dbaysal/approximate" \
        --approx-prefix-column "prefix" \
        --approx-suffix-column "suffix" \
        --approx-mode "code" \
        --max-new-tokens 2056 \
        --evalplus-dataset "humaneval-forget-utility" \
        --evalplus-bs 256 \
        "$@"
}

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    shift 2

    local job_index
    for ((job_index = worker_index; job_index < ${#methods[@]}; job_index += ${#gpu_ids[@]})); do
        run_eval_job "${gpu_id}" "${methods[${job_index}]}" "$@"
    done
}

echo "Detected ${#gpu_ids[@]} GPU worker(s): ${gpu_ids[*]}"
echo "Queued ${#methods[@]} meta code-unit recovery job(s). Each job runs epoch-2 and epoch-3 sequentially."

log_dir="${REPO_ROOT}/Results/secret_code_unit_eval_suite/logs"
mkdir -p "${log_dir}"

pids=()
for worker_index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[${worker_index}]}"
    log_file="${log_dir}/meta-code-unit-last-two-gpu-${gpu_id}.txt"
    echo "GPU ${gpu_id} log: ${log_file}"
    run_worker "${worker_index}" "${gpu_id}" "$@" > "${log_file}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
