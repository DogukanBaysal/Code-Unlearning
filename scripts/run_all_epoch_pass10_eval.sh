#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PASS_K="${PASS_K:-10}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2056}"
SUFFIX_BS="${SUFFIX_BS:-64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/all_epoch_pass10_eval}"
AGGREGATE_FILTER_CSV="${AGGREGATE_FILTER_CSV:-${REPO_ROOT}/UnlearningEvaluation/non_exact_matches.csv}"
# Optional comma- or space-separated override; automatic Hub discovery is used when empty.
CHECKPOINTS="${CHECKPOINTS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

model_specs=(
    "qwen2_5_coder_3b|Qwen/Qwen2.5-Coder-3B"
    "meta_llama3_2_3b|meta-llama/Llama-3.2-3B"
)

methods=(
    "ga"
    "npo"
    "prod"
    "ga_gd"
    "ga_kl"
    "npo_gd"
    "npo_kl"
    "prod_gd"
    "prod_kl"
)

tasks=(
    "secret"
    "code-unit"
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

jobs=()
discovery_status=0
for task in "${tasks[@]}"; do
    for model_spec in "${model_specs[@]}"; do
        IFS="|" read -r model_key base_model <<< "${model_spec}"
        for method in "${methods[@]}"; do
            peft_name="dbaysal/${task}-unlearning-${model_key}-${method}"
            checkpoint_list=""
            if [ -n "${CHECKPOINTS}" ]; then
                normalized_checkpoints="${CHECKPOINTS//,/ }"
                read -r -a override_checkpoints <<< "${normalized_checkpoints}"
                printf -v checkpoint_list '%s\n' "${override_checkpoints[@]}"
            elif ! checkpoint_list="$(
                python scripts/run_adapter_eval_suite.py \
                    --model "${base_model}" \
                    --peft-names "${peft_name}" \
                    --discover-checkpoints \
                    --all-checkpoints \
                    --list-checkpoints-only
            )"; then
                echo "Checkpoint discovery failed: ${peft_name}" >&2
                discovery_status=1
                continue
            fi

            checkpoint_index=0
            while IFS= read -r checkpoint; do
                if [ -z "${checkpoint}" ]; then
                    continue
                fi
                checkpoint_index=$((checkpoint_index + 1))
                jobs+=(
                    "${task}|${model_key}|${base_model}|${method}|${checkpoint}|${checkpoint_index}"
                )
            done <<< "${checkpoint_list}"

            if [ "${checkpoint_index}" -eq 0 ]; then
                echo "No checkpoints found: ${peft_name}" >&2
                discovery_status=1
            fi
        done
    done
done

if [ "${#jobs[@]}" -eq 0 ]; then
    echo "No checkpoint evaluation jobs were created." >&2
    exit 1
fi

evaluation_outputs_complete() {
    local output_root="$1"
    local peft_name="$2"
    local checkpoint="$3"
    local adapter_slug="${peft_name//\//--}"
    local run_slug="${adapter_slug}_${checkpoint}"
    local dataset_label
    local result_name

    for dataset_label in forget retain approximate; do
        local result_dir="${output_root}/unlearningeval/${dataset_label}/${run_slug}"
        for result_name in \
            row_results.jsonl \
            all_results.jsonl \
            aggregate_results.json \
            aggregate_results_filtered.json; do
            if [ ! -s "${result_dir}/${result_name}" ]; then
                return 1
            fi
        done
    done
    return 0
}

run_eval_job() {
    local gpu_id="$1"
    local task="$2"
    local model_key="$3"
    local base_model="$4"
    local method="$5"
    local checkpoint="$6"
    local epoch_index="$7"
    shift 7

    local peft_name="dbaysal/${task}-unlearning-${model_key}-${method}"
    local task_output_name="${task/-/_}"
    local output_root="${OUTPUT_ROOT}/${task_output_name}/${model_key}/${method}/epoch-${epoch_index}"
    local aggregate_filter_csv="${AGGREGATE_FILTER_CSV}"
    local forget_prefix_column="prefix"
    local forget_suffix_column="suffix"
    local forget_mode="code"

    if [ "${task}" = "secret" ]; then
        forget_prefix_column="secret_prefix"
        forget_suffix_column="secret_suffix"
        forget_mode="secret"
    fi

    if [ "${SKIP_EXISTING}" = "1" ] && \
        evaluation_outputs_complete "${output_root}" "${peft_name}" "${checkpoint}"; then
        echo
        echo "GPU ${gpu_id}: SKIPPED complete destination"
        echo "GPU ${gpu_id}: task=${task}, model=${model_key}, method=${method}, epoch=${epoch_index}"
        echo "GPU ${gpu_id}: checkpoint=${checkpoint}"
        echo "GPU ${gpu_id}: output=${output_root}"
        return 0
    fi

    if [ ! -f "${aggregate_filter_csv}" ]; then
        echo "Aggregate filter CSV not found: ${aggregate_filter_csv}" >&2
        return 1
    fi

    echo
    echo "GPU ${gpu_id}: task=${task}, model=${model_key}, method=${method}, epoch=${epoch_index}"
    echo "GPU ${gpu_id}: checkpoint=${checkpoint}"
    echo "GPU ${gpu_id}: adapter=${peft_name}"
    echo "GPU ${gpu_id}: output=${output_root}"
    echo "GPU ${gpu_id}: aggregate_filter_csv=${aggregate_filter_csv}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/run_adapter_eval_suite.py \
        --model "${base_model}" \
        --peft-names "${peft_name}" \
        --checkpoints "${checkpoint}" \
        --output-root "${output_root}" \
        --forget-dataset "dbaysal/forget" \
        --forget-prefix-column "${forget_prefix_column}" \
        --forget-suffix-column "${forget_suffix_column}" \
        --forget-mode "${forget_mode}" \
        --retain-dataset "dbaysal/retain-half" \
        --retain-prefix-column "prefix" \
        --retain-suffix-column "suffix" \
        --retain-mode "code" \
        --approx-dataset "dbaysal/approximate" \
        --approx-prefix-column "prefix" \
        --approx-suffix-column "suffix" \
        --approx-mode "code" \
        --aggregate-filter-csv "${aggregate_filter_csv}" \
        --pass-k "${PASS_K}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --suffix-bs "${SUFFIX_BS}" \
        --skip-evalplus \
        --continue-on-error \
        "$@"
}

launch_job() {
    local worker_index="$1"
    local job_index="$2"
    shift 2

    local gpu_id="${gpu_ids[${worker_index}]}"
    local task
    local model_key
    local base_model
    local method
    local checkpoint
    local epoch_index
    IFS="|" read -r task model_key base_model method checkpoint epoch_index \
        <<< "${jobs[${job_index}]}"

    local log_file="${log_dir}/gpu-${gpu_id}.txt"
    (
        echo
        echo "Dispatcher: assigned job $((job_index + 1))/${#jobs[@]} to GPU ${gpu_id}"
        if ! run_eval_job \
            "${gpu_id}" "${task}" "${model_key}" "${base_model}" "${method}" \
            "${checkpoint}" "${epoch_index}" "$@"; then
            echo \
                "GPU ${gpu_id}: FAILED task=${task}, model=${model_key}, method=${method}, epoch=${epoch_index}" \
                >&2
            exit 1
        fi
    ) >> "${log_file}" 2>&1 &

    pids[${worker_index}]="$!"
}

echo "Detected ${#gpu_ids[@]} GPU worker(s): ${gpu_ids[*]}"
echo "Queued ${#jobs[@]} checkpoint jobs; the next pending epoch goes to the first free GPU."
echo "Suffix evaluation: pass@${PASS_K}, temperature=${TEMPERATURE}, top_p=${TOP_P}, batch_size=${SUFFIX_BS}"
echo "Skip complete destinations: ${SKIP_EXISTING}"

log_dir="${OUTPUT_ROOT}/logs"
mkdir -p "${log_dir}"

pids=()
for worker_index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[${worker_index}]}"
    log_file="${log_dir}/gpu-${gpu_id}.txt"
    : > "${log_file}"
    echo "GPU ${gpu_id} log: ${log_file}"
    pids+=("")
done

status="${discovery_status}"
next_job_index=0
active_jobs=0

# Initially give each GPU at most one job. After that, jobs are assigned only
# when a GPU finishes, so faster GPUs automatically process more epochs.
for worker_index in "${!gpu_ids[@]}"; do
    if [ "${next_job_index}" -ge "${#jobs[@]}" ]; then
        break
    fi
    launch_job "${worker_index}" "${next_job_index}" "$@"
    next_job_index=$((next_job_index + 1))
    active_jobs=$((active_jobs + 1))
done

while [ "${active_jobs}" -gt 0 ]; do
    completion_found=0
    for worker_index in "${!gpu_ids[@]}"; do
        pid="${pids[${worker_index}]:-}"
        if [ -z "${pid}" ] || kill -0 "${pid}" 2>/dev/null; then
            continue
        fi

        if ! wait "${pid}"; then
            status=1
        fi
        pids[${worker_index}]=""
        active_jobs=$((active_jobs - 1))
        completion_found=1

        if [ "${next_job_index}" -lt "${#jobs[@]}" ]; then
            launch_job "${worker_index}" "${next_job_index}" "$@"
            next_job_index=$((next_job_index + 1))
            active_jobs=$((active_jobs + 1))
        fi
    done

    if [ "${completion_found}" -eq 0 ]; then
        sleep 1
    fi
done

exit "${status}"
