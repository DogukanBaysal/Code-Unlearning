#!/usr/bin/env bash
# Evaluate the new- prefixed Secret and Code-unit NPO+KL adapters, including
# all available Secret batch-order variants.
#
# Default evaluation settings reproduce the thesis pass@10 setup:
#   suffix pass@10 / EvalPlus pass@10, temperature 0.8, top-p 0.95,
#   three adapter checkpoints (aliased as epochs), 2,056 generated tokens,
#   and EvalPlus batch size 128.
#
# Usage:
#   bash scripts/run_new_secret_npo_kl_eval_suite.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_new_secret_npo_kl_eval_suite.sh
#   bash scripts/run_new_secret_npo_kl_eval_suite.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HUB_NAMESPACE="${HUB_NAMESPACE:-dbaysal}"
ADAPTER_PREFIX="${ADAPTER_PREFIX:-new-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/new_secret_npo_kl_eval_suite}"

model_specs=(
    "qwen2_5_coder_3b|Qwen/Qwen2.5-Coder-3B"
    "meta_llama3_2_3b|meta-llama/Llama-3.2-3B"
)

# The first field is the directory label. The second is appended to the Hub
# Secret adapter name after "-npo_kl"; standard has no suffix.
secret_variants=(
    "standard|"
    "retain-first|-retain-first"
    "forget-first|-forget-first"
    "random-retain-full|-random-retain-full"
)

detect_gpu_ids() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        if [[ "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then
            return 0
        fi
        local visible_id
        local old_ifs="${IFS}"
        IFS=","
        for visible_id in ${CUDA_VISIBLE_DEVICES}; do
            [[ -n "${visible_id}" ]] && echo "${visible_id}"
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
    [[ -n "${gpu_id}" ]] && gpu_ids+=("${gpu_id}")
done < <(detect_gpu_ids)

if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
    echo "No GPUs detected from CUDA_VISIBLE_DEVICES or nvidia-smi." >&2
    exit 1
fi

jobs=()
for model_spec in "${model_specs[@]}"; do
    IFS="|" read -r model_key base_model <<< "${model_spec}"
    for variant in "${secret_variants[@]}"; do
        IFS="|" read -r variant_name adapter_suffix <<< "${variant}"
        jobs+=("secret|${model_key}|${base_model}|${variant_name}|${adapter_suffix}")
    done
    jobs+=("code-unit|${model_key}|${base_model}|standard|")
done

run_eval_job() {
    local gpu_id="$1"
    local task="$2"
    local model_key="$3"
    local base_model="$4"
    local variant_name="$5"
    local adapter_suffix="$6"
    shift 6

    local peft_name
    local output_root
    local forget_prefix_column
    local forget_suffix_column
    local forget_mode
    if [[ "${task}" == "secret" ]]; then
        peft_name="${HUB_NAMESPACE}/${ADAPTER_PREFIX}secret-unlearning-${model_key}-npo_kl${adapter_suffix}"
        output_root="${OUTPUT_ROOT}/secret/${model_key}/npo_kl/${variant_name}"
        forget_prefix_column="secret_prefix"
        forget_suffix_column="secret_suffix"
        forget_mode="secret"
    else
        peft_name="${HUB_NAMESPACE}/${ADAPTER_PREFIX}code-unit-unlearning-${model_key}-npo_kl"
        output_root="${OUTPUT_ROOT}/code_unit/${model_key}/npo_kl/${variant_name}"
        forget_prefix_column="prefix"
        forget_suffix_column="suffix"
        forget_mode="code"
    fi

    echo
    echo "GPU ${gpu_id}: task=${task}, model=${model_key}, method=npo_kl, variant=${variant_name}"
    echo "GPU ${gpu_id}: base_model=${base_model}"
    echo "GPU ${gpu_id}: peft_name=${peft_name}"
    echo "GPU ${gpu_id}: output_root=${output_root}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/run_adapter_eval_suite.py \
        --model "${base_model}" \
        --peft-names "${peft_name}" \
        --discover-checkpoints \
        --num-checkpoints 3 \
        --alias-checkpoints-as-epochs \
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
        --aggregate-filter-csv "UnlearningEvaluation/non_exact_matches.csv" \
        --max-new-tokens 2056 \
        --pass-k 10 \
        --evalplus-bs 128 \
        --evalplus-pass-k 10 \
        --temperature 0.8 \
        --top-p 0.95 \
        "$@"
}

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    shift 2

    local job_index
    for ((job_index = worker_index; job_index < ${#jobs[@]}; job_index += ${#gpu_ids[@]})); do
        local task
        local model_key
        local base_model
        local variant_name
        local adapter_suffix
        IFS="|" read -r task model_key base_model variant_name adapter_suffix <<< "${jobs[${job_index}]}"
        run_eval_job "${gpu_id}" "${task}" "${model_key}" "${base_model}" "${variant_name}" "${adapter_suffix}" "$@"
    done
}

echo "Detected ${#gpu_ids[@]} GPU worker(s): ${gpu_ids[*]}"
echo "Queued ${#jobs[@]} new- NPO+KL adapter evaluation jobs (8 Secret, 2 Code-unit)."

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

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
