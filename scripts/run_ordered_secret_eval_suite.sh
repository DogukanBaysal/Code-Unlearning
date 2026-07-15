#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PASS_K="${PASS_K:-10}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2056}"
SUFFIX_BS="${SUFFIX_BS:-6}"
EVALPLUS_BS="${EVALPLUS_BS:-64}"
EVALPLUS_DATASET="${EVALPLUS_DATASET:-humaneval-forget-utility}"
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/ordered_secret_eval_suite}"
AGGREGATE_FILTER_CSV="${AGGREGATE_FILTER_CSV:-${REPO_ROOT}/UnlearningEvaluation/non_exact_matches.csv}"
CHECKPOINTS="${CHECKPOINTS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

model_specs=(
    "qwen2_5_coder_3b|Qwen/Qwen2.5-Coder-3B"
    "meta_llama3_2_3b|meta-llama/Llama-3.2-3B"
)

methods=(
    "ga_gd"
    "ga_kl"
    "npo_gd"
    "npo_kl"
    "prod_gd"
    "prod_kl"
)

variant_specs=(
    "retain-first|dbaysal/retain-half"
    "forget-first|dbaysal/retain-half"
    "random-retain-full|dbaysal/retain-full"
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
for variant_spec in "${variant_specs[@]}"; do
    IFS="|" read -r variant_suffix retain_dataset <<< "${variant_spec}"
    for model_spec in "${model_specs[@]}"; do
        IFS="|" read -r model_key base_model <<< "${model_spec}"
        for method in "${methods[@]}"; do
            peft_name="dbaysal/secret-unlearning-${model_key}-${method}-${variant_suffix}"
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
                    "${variant_suffix}|${retain_dataset}|${model_key}|${base_model}|${method}|${checkpoint}|${checkpoint_index}"
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

    local evalplus_result
    evalplus_result="${output_root}/evalplus/${EVALPLUS_DATASET}/${run_slug}.eval_results.json"
    [ -s "${evalplus_result}" ]
}

run_eval_job() {
    local gpu_id="$1"
    local variant_suffix="$2"
    local retain_dataset="$3"
    local model_key="$4"
    local base_model="$5"
    local method="$6"
    local checkpoint="$7"
    local epoch_index="$8"
    shift 8

    local peft_name
    peft_name="dbaysal/secret-unlearning-${model_key}-${method}-${variant_suffix}"
    local output_root
    output_root="${OUTPUT_ROOT}/${variant_suffix}/${model_key}/${method}/epoch-${epoch_index}"

    if [ "${SKIP_EXISTING}" = "1" ] && \
        evaluation_outputs_complete "${output_root}" "${peft_name}" "${checkpoint}"; then
        echo
        echo "GPU ${gpu_id}: SKIPPED complete destination"
        echo "GPU ${gpu_id}: variant=${variant_suffix}, model=${model_key}, method=${method}, epoch=${epoch_index}"
        echo "GPU ${gpu_id}: checkpoint=${checkpoint}"
        echo "GPU ${gpu_id}: output=${output_root}"
        return 0
    fi

    if [ ! -f "${AGGREGATE_FILTER_CSV}" ]; then
        echo "Aggregate filter CSV not found: ${AGGREGATE_FILTER_CSV}" >&2
        return 1
    fi

    echo
    echo "GPU ${gpu_id}: variant=${variant_suffix}, model=${model_key}, method=${method}, epoch=${epoch_index}"
    echo "GPU ${gpu_id}: checkpoint=${checkpoint}"
    echo "GPU ${gpu_id}: retain_dataset=${retain_dataset}"
    echo "GPU ${gpu_id}: adapter=${peft_name}"
    echo "GPU ${gpu_id}: output=${output_root}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/run_adapter_eval_suite.py \
        --model "${base_model}" \
        --peft-names "${peft_name}" \
        --checkpoints "${checkpoint}" \
        --output-root "${output_root}" \
        --forget-dataset "dbaysal/forget" \
        --forget-prefix-column "secret_prefix" \
        --forget-suffix-column "secret_suffix" \
        --forget-mode "secret" \
        --retain-dataset "${retain_dataset}" \
        --retain-prefix-column "prefix" \
        --retain-suffix-column "suffix" \
        --retain-mode "code" \
        --approx-dataset "dbaysal/approximate" \
        --approx-prefix-column "prefix" \
        --approx-suffix-column "suffix" \
        --approx-mode "code" \
        --aggregate-filter-csv "${AGGREGATE_FILTER_CSV}" \
        --pass-k "${PASS_K}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --suffix-bs "${SUFFIX_BS}" \
        --evalplus-dataset "${EVALPLUS_DATASET}" \
        --evalplus-bs "${EVALPLUS_BS}" \
        --dtype "${DTYPE}" \
        --continue-on-error \
        "$@"
}

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    shift 2

    local status=0
    local job_index
    for ((job_index = worker_index; job_index < ${#jobs[@]}; job_index += ${#gpu_ids[@]})); do
        local variant_suffix
        local retain_dataset
        local model_key
        local base_model
        local method
        local checkpoint
        local epoch_index
        IFS="|" read -r \
            variant_suffix retain_dataset model_key base_model method checkpoint epoch_index \
            <<< "${jobs[${job_index}]}"
        if ! run_eval_job \
            "${gpu_id}" "${variant_suffix}" "${retain_dataset}" "${model_key}" \
            "${base_model}" "${method}" "${checkpoint}" "${epoch_index}" "$@"; then
            echo \
                "GPU ${gpu_id}: FAILED variant=${variant_suffix}, model=${model_key}, method=${method}, epoch=${epoch_index}" \
                >&2
            status=1
        fi
    done
    return "${status}"
}

echo "Detected ${#gpu_ids[@]} GPU worker(s): ${gpu_ids[*]}"
echo "Queued ${#jobs[@]} checkpoint jobs; epochs are distributed across GPU workers."
echo "UnlearningEvaluation: pass@${PASS_K}, batch_size=${SUFFIX_BS}, temperature=${TEMPERATURE}, top_p=${TOP_P}"
echo "EvalPlus: dataset=${EVALPLUS_DATASET}, batch_size=${EVALPLUS_BS}"
echo "Skip complete destinations: ${SKIP_EXISTING}"

log_dir="${OUTPUT_ROOT}/logs"
mkdir -p "${log_dir}"

pids=()
for worker_index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[${worker_index}]}"
    log_file="${log_dir}/gpu-${gpu_id}.txt"
    echo "GPU ${gpu_id} log: ${log_file}"
    run_worker "${worker_index}" "${gpu_id}" "$@" > "${log_file}" 2>&1 &
    pids+=("$!")
done

status="${discovery_status}"
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
