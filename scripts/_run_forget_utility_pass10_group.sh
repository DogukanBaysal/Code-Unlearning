#!/bin/bash
set -euo pipefail

# Shared implementation for the three public ForgetEval/UtilityEval entry scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVALPLUS_DIR="${REPO_ROOT}/evalplus"
cd "${REPO_ROOT}"

if [ -z "${EVAL_GROUP:-}" ]; then
    echo "EVAL_GROUP must be one of: initial-secret, code-unit, ordered, base-checkpoint282" >&2
    exit 2
fi

PASS_K="${PASS_K:-10}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
EVALPLUS_BS="${EVALPLUS_BS:-25}"
EVALPLUS_DATASET="${EVALPLUS_DATASET:-humaneval-forget-utility}"
EVALPLUS_PARALLEL="${EVALPLUS_PARALLEL:-4}"
EVALPLUS_TIMEOUT_PER_TASK="${EVALPLUS_TIMEOUT_PER_TASK:-30}"
BASELINE_FILTER_CSV="${BASELINE_FILTER_CSV:-${EVALPLUS_DIR}/evalplus/baseline_failed_test_ids.csv}"
BACKEND="${BACKEND:-hf}"
DTYPE="${DTYPE:-bfloat16}"
CHECKPOINTS="${CHECKPOINTS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

case "${EVAL_GROUP}" in
    initial-secret)
        DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/Results/initial_secret_forget_utility_pass10"
        ;;
    code-unit)
        DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/Results/code_unit_forget_utility_pass10"
        ;;
    ordered)
        DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/Results/ordered_forget_utility_pass10"
        ;;
    base-checkpoint282)
        DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/Results/base_checkpoint282_forget_utility_pass10"
        ;;
    *)
        echo "Unknown EVAL_GROUP: ${EVAL_GROUP}" >&2
        exit 2
        ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"

if ! [[ "${PASS_K}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PASS_K must be a positive integer: ${PASS_K}" >&2
    exit 2
fi
if ! [[ "${EVALPLUS_BS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVALPLUS_BS must be a positive integer: ${EVALPLUS_BS}" >&2
    exit 2
fi
if ! [[ "${EVALPLUS_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVALPLUS_PARALLEL must be a positive integer: ${EVALPLUS_PARALLEL}" >&2
    exit 2
fi
if ! [[ "${EVALPLUS_TIMEOUT_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVALPLUS_TIMEOUT_PER_TASK must be a positive integer: ${EVALPLUS_TIMEOUT_PER_TASK}" >&2
    exit 2
fi

model_specs=(
    "qwen2_5_coder_3b|Qwen/Qwen2.5-Coder-3B"
    "meta_llama3_2_3b|meta-llama/Llama-3.2-3B"
)

base_checkpoint282_specs=(
    "qwen2_5_coder_3b|Qwen/Qwen2.5-Coder-3B|dbaysal/qwen2.5coder-3b-learned"
    "meta_llama3_2_3b|meta-llama/Llama-3.2-3B|dbaysal/metallama3.2-3b-learned"
)

all_methods=(
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

ordered_methods=(
    "ga_gd"
    "ga_kl"
    "npo_gd"
    "npo_kl"
    "prod_gd"
    "prod_kl"
)

# setup label | adapter repository suffix
# The full-retain training used random batch ordering and was uploaded with the
# historical random-retain-full suffix.
ordered_setups=(
    "forget_first|forget-first"
    "retain_first|retain-first"
    "full-retain|random-retain-full"
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

    # This keeps dry runs usable on machines without NVIDIA tooling. A real run
    # will fail clearly in the model backend if GPU 0 is unavailable.
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

resolve_checkpoints() {
    local base_model="$1"
    local peft_name="$2"

    if [ -n "${CHECKPOINTS}" ]; then
        local normalized_checkpoints="${CHECKPOINTS//,/ }"
        local override_checkpoints=()
        read -r -a override_checkpoints <<< "${normalized_checkpoints}"
        printf '%s\n' "${override_checkpoints[@]}"
        return 0
    fi

    python scripts/run_adapter_eval_suite.py \
        --model "${base_model}" \
        --peft-names "${peft_name}" \
        --discover-checkpoints \
        --all-checkpoints \
        --list-checkpoints-only
}

# setup | model key | base model | method | adapter | checkpoint | epoch index
jobs=()
discovery_status=0

queue_adapter_checkpoints() {
    local setup="$1"
    local model_key="$2"
    local base_model="$3"
    local method="$4"
    local peft_name="$5"
    local checkpoint_list=""

    if ! checkpoint_list="$(resolve_checkpoints "${base_model}" "${peft_name}")"; then
        echo "Checkpoint discovery failed: ${peft_name}" >&2
        discovery_status=1
        return 0
    fi

    local checkpoint_index=0
    local checkpoint
    while IFS= read -r checkpoint; do
        if [ -z "${checkpoint}" ]; then
            continue
        fi
        checkpoint_index=$((checkpoint_index + 1))
        jobs+=(
            "${setup}|${model_key}|${base_model}|${method}|${peft_name}|${checkpoint}|${checkpoint_index}"
        )
    done <<< "${checkpoint_list}"

    if [ "${checkpoint_index}" -eq 0 ]; then
        echo "No checkpoints found: ${peft_name}" >&2
        discovery_status=1
    fi
}

if [ "${EVAL_GROUP}" = "ordered" ]; then
    for setup_spec in "${ordered_setups[@]}"; do
        IFS="|" read -r setup adapter_suffix <<< "${setup_spec}"
        for model_spec in "${model_specs[@]}"; do
            IFS="|" read -r model_key base_model <<< "${model_spec}"
            for method in "${ordered_methods[@]}"; do
                peft_name="dbaysal/secret-unlearning-${model_key}-${method}-${adapter_suffix}"
                queue_adapter_checkpoints \
                    "${setup}" "${model_key}" "${base_model}" "${method}" "${peft_name}"
            done
        done
    done
elif [ "${EVAL_GROUP}" = "base-checkpoint282" ]; then
    checkpoint_list="${CHECKPOINTS:-checkpoint-282}"
    checkpoint_list="${checkpoint_list//,/ }"
    for model_spec in "${base_checkpoint282_specs[@]}"; do
        IFS="|" read -r model_key base_model peft_name <<< "${model_spec}"
        checkpoint_index=0
        read -r -a fixed_checkpoints <<< "${checkpoint_list}"
        for checkpoint in "${fixed_checkpoints[@]}"; do
            checkpoint_index=$((checkpoint_index + 1))
            jobs+=(
                "base_checkpoint282|${model_key}|${base_model}|learned|${peft_name}|${checkpoint}|${checkpoint_index}"
            )
        done
    done
else
    task_slug="secret"
    setup="initial_secret"
    if [ "${EVAL_GROUP}" = "code-unit" ]; then
        task_slug="code-unit"
        setup="code_unit"
    fi

    for model_spec in "${model_specs[@]}"; do
        IFS="|" read -r model_key base_model <<< "${model_spec}"
        for method in "${all_methods[@]}"; do
            peft_name="dbaysal/${task_slug}-unlearning-${model_key}-${method}"
            queue_adapter_checkpoints \
                "${setup}" "${model_key}" "${base_model}" "${method}" "${peft_name}"
        done
    done
fi

if [ "${#jobs[@]}" -eq 0 ]; then
    echo "No checkpoint evaluation jobs were created." >&2
    exit 1
fi

run_eval_job() {
    local gpu_id="$1"
    local setup="$2"
    local model_key="$3"
    local base_model="$4"
    local method="$5"
    local peft_name="$6"
    local checkpoint="$7"
    local epoch_index="$8"
    shift 8

    local adapter_slug="${peft_name//\//--}"
    local run_slug="${adapter_slug}_${checkpoint}"
    local output_root="${OUTPUT_ROOT}/${setup}/${model_key}/${method}/epoch-${epoch_index}"
    local evalplus_root="${output_root}/evalplus"
    local result_dir="${evalplus_root}/${EVALPLUS_DATASET}"
    if [ "${PASS_K}" -gt 1 ]; then
        result_dir="${result_dir}/pass-${PASS_K}"
    fi
    local result_path="${result_dir}/${run_slug}.eval_results.json"
    local filtered_result_path="${result_dir}/${run_slug}.filtered.eval_results.json"

    if [ "${SKIP_EXISTING}" = "1" ] && \
        [ -s "${result_path}" ] && [ -s "${filtered_result_path}" ]; then
        echo
        echo "GPU ${gpu_id}: SKIPPED complete destination"
        echo "GPU ${gpu_id}: setup=${setup}, model=${model_key}, method=${method}, epoch=${epoch_index}"
        echo "GPU ${gpu_id}: checkpoint=${checkpoint}"
        echo "GPU ${gpu_id}: result=${result_path}"
        echo "GPU ${gpu_id}: filtered_result=${filtered_result_path}"
        return 0
    fi

    echo
    echo "GPU ${gpu_id}: setup=${setup}, model=${model_key}, method=${method}, epoch=${epoch_index}"
    echo "GPU ${gpu_id}: checkpoint=${checkpoint}"
    echo "GPU ${gpu_id}: adapter=${peft_name}"
    echo "GPU ${gpu_id}: result=${result_path}"
    echo "GPU ${gpu_id}: filtered_result=${filtered_result_path}"

    local eval_cmd=(
        env
        "CUDA_VISIBLE_DEVICES=${gpu_id}"
        "PYTHONPATH=${EVALPLUS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        "EVALPLUS_TIMEOUT_PER_TASK=${EVALPLUS_TIMEOUT_PER_TASK}"
        python -m evalplus.evaluate
        --model "${base_model}"
        --peft-name "${peft_name}"
        --peft-subfolder "${checkpoint}"
        --dataset "${EVALPLUS_DATASET}"
        --backend "${BACKEND}"
        --defer-sanitize
        --bs "${EVALPLUS_BS}"
        --parallel "${EVALPLUS_PARALLEL}"
        --force-base-prompt
        --root "${evalplus_root}"
        --output-file "${result_path}"
        --dtype "${DTYPE}"
    )

    if [ "${PASS_K}" -eq 1 ]; then
        eval_cmd+=(--greedy)
    else
        eval_cmd+=(
            --n-samples "${PASS_K}"
            --temperature "${TEMPERATURE}"
            --top-p "${TOP_P}"
        )
    fi

    local filter_cmd=(
        python "${EVALPLUS_DIR}/tools/filter_baseline_failed_results.py"
        "${result_path}"
        --filter-csv "${BASELINE_FILTER_CSV}"
        --output "${filtered_result_path}"
    )

    printf '$'
    printf ' %q' "${eval_cmd[@]}" "$@"
    echo
    printf '$'
    printf ' %q' "${filter_cmd[@]}"
    echo
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi

    mkdir -p "${result_dir}"
    "${eval_cmd[@]}" "$@" || return $?
    "${filter_cmd[@]}"
}

launch_job() {
    local worker_index="$1"
    local job_index="$2"
    shift 2

    local gpu_id="${gpu_ids[${worker_index}]}"
    local setup
    local model_key
    local base_model
    local method
    local peft_name
    local checkpoint
    local epoch_index
    IFS="|" read -r \
        setup model_key base_model method peft_name checkpoint epoch_index \
        <<< "${jobs[${job_index}]}"

    local log_file="${log_dir}/gpu-${gpu_id}.txt"
    (
        echo
        echo "Dispatcher: assigned job $((job_index + 1))/${#jobs[@]} to GPU ${gpu_id}"
        if ! run_eval_job \
            "${gpu_id}" "${setup}" "${model_key}" "${base_model}" "${method}" \
            "${peft_name}" "${checkpoint}" "${epoch_index}" "$@"; then
            echo \
                "GPU ${gpu_id}: FAILED setup=${setup}, model=${model_key}, method=${method}, epoch=${epoch_index}" \
                >&2
            exit 1
        fi
    ) >> "${log_file}" 2>&1 &

    pids[${worker_index}]="$!"
}

echo "Evaluation group: ${EVAL_GROUP}"
echo "Detected ${#gpu_ids[@]} GPU worker(s): ${gpu_ids[*]}"
echo "Queued ${#jobs[@]} checkpoint jobs; the next pending checkpoint goes to the first free GPU."
echo "EvalPlus: HumanEval + ForgetEval + UtilityEval, pass@${PASS_K}, batch_size=${EVALPLUS_BS}"
echo "Testing: parallelism=${EVALPLUS_PARALLEL}, per-solution timeout=${EVALPLUS_TIMEOUT_PER_TASK}s"
echo "Sampling: temperature=${TEMPERATURE}, top_p=${TOP_P}"
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
