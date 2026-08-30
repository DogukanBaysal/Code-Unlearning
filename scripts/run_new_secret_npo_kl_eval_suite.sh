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
NUM_EPOCHS=3
GPU_DISPATCH_POLL_SECONDS="${GPU_DISPATCH_POLL_SECONDS:-1}"

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

adapter_jobs=()
for model_spec in "${model_specs[@]}"; do
    IFS="|" read -r model_key base_model <<< "${model_spec}"
    for variant in "${secret_variants[@]}"; do
        IFS="|" read -r variant_name adapter_suffix <<< "${variant}"
        adapter_jobs+=("secret|${model_key}|${base_model}|${variant_name}|${adapter_suffix}")
    done
    adapter_jobs+=("code-unit|${model_key}|${base_model}|standard|")
done

run_eval_job() {
    local gpu_id="$1"
    local task="$2"
    local model_key="$3"
    local base_model="$4"
    local variant_name="$5"
    local adapter_suffix="$6"
    local epoch_index="$7"
    shift 7

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
    echo "GPU ${gpu_id}: epoch=${epoch_index}, task=${task}, model=${model_key}, method=npo_kl, variant=${variant_name}"
    echo "GPU ${gpu_id}: base_model=${base_model}"
    echo "GPU ${gpu_id}: peft_name=${peft_name}"
    echo "GPU ${gpu_id}: output_root=${output_root}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/run_adapter_eval_suite.py \
        --model "${base_model}" \
        --peft-names "${peft_name}" \
        --discover-checkpoints \
        --num-checkpoints "${NUM_EPOCHS}" \
        --checkpoint-index "${epoch_index}" \
        --alias-checkpoints-as-epochs \
        --checkpoint-alias-start "${epoch_index}" \
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

echo "Detected ${#gpu_ids[@]} GPU worker(s): ${gpu_ids[*]}"
total_jobs=$((${#adapter_jobs[@]} * NUM_EPOCHS))
echo "Queued ${total_jobs} epoch-specific evaluation jobs in one global queue."
echo "Schedule: epoch order is preserved, but there are no barriers between epochs."
echo "GPU assignment: dynamic; each completed GPU receives the next waiting job."

log_dir="${OUTPUT_ROOT}/logs"
mkdir -p "${log_dir}"

for worker_index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[${worker_index}]}"
    log_file="${log_dir}/gpu-${gpu_id}.txt"
    echo "GPU ${gpu_id} log: ${log_file}"
    : > "${log_file}"
done

status=0
gpu_pids=()
gpu_job_indices=()
next_job_index=0
running_jobs=0

while [[ "${running_jobs}" -gt 0 \
    || ("${status}" -eq 0 && "${next_job_index}" -lt "${total_jobs}") ]]; do
    dispatcher_progress=0

    for gpu_slot in "${!gpu_ids[@]}"; do
        gpu_id="${gpu_ids[${gpu_slot}]}"

        if [[ -n "${gpu_pids[${gpu_slot}]:-}" ]] \
            && ! kill -0 "${gpu_pids[${gpu_slot}]}" 2>/dev/null; then
            finished_job_index="${gpu_job_indices[${gpu_slot}]}"
            if ! wait "${gpu_pids[${gpu_slot}]}"; then
                echo "Job $((finished_job_index + 1))/${total_jobs} failed on GPU ${gpu_id}." >&2
                status=1
            fi
            unset 'gpu_pids[gpu_slot]'
            unset 'gpu_job_indices[gpu_slot]'
            running_jobs=$((running_jobs - 1))
            dispatcher_progress=1
        fi

        if [[ -z "${gpu_pids[${gpu_slot}]:-}" \
            && "${next_job_index}" -lt "${total_jobs}" \
            && "${status}" -eq 0 ]]; then
            epoch_index=$((next_job_index / ${#adapter_jobs[@]} + 1))
            adapter_job_index=$((next_job_index % ${#adapter_jobs[@]}))
            IFS="|" read -r task model_key base_model variant_name adapter_suffix \
                <<< "${adapter_jobs[${adapter_job_index}]}"
            log_file="${log_dir}/gpu-${gpu_id}.txt"
            echo "Assigning job $((next_job_index + 1))/${total_jobs} to GPU ${gpu_id} (epoch ${epoch_index}, ${task}, ${model_key}, ${variant_name})."
            run_eval_job \
                "${gpu_id}" \
                "${task}" \
                "${model_key}" \
                "${base_model}" \
                "${variant_name}" \
                "${adapter_suffix}" \
                "${epoch_index}" \
                "$@" >> "${log_file}" 2>&1 &
            gpu_pids[${gpu_slot}]="$!"
            gpu_job_indices[${gpu_slot}]="${next_job_index}"
            next_job_index=$((next_job_index + 1))
            running_jobs=$((running_jobs + 1))
            dispatcher_progress=1
        fi
    done

    if [[ "${running_jobs}" -gt 0 && "${dispatcher_progress}" -eq 0 ]]; then
        sleep "${GPU_DISPATCH_POLL_SECONDS}"
    fi
done

if [[ "${status}" -eq 0 ]]; then
    echo "Completed all ${total_jobs} evaluation jobs."
else
    echo "A job failed; remaining queued jobs were not started." >&2
fi

exit "${status}"
