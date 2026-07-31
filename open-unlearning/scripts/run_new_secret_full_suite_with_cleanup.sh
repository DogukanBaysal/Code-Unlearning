#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_UNLEARNING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${OPEN_UNLEARNING_ROOT}"

ORDERED_RETAIN_DATASET="${ORDERED_RETAIN_DATASET:-dbaysal/retain-half}"
RETAIN_FULL_DATASET="${RETAIN_FULL_DATASET:-dbaysal/retain-full}"
HUB_NAMESPACE="${HUB_NAMESPACE:-dbaysal}"
MODEL_NAME_PREFIX="${MODEL_NAME_PREFIX:-new-}"
HUB_ADAPTER_ENABLED="${HUB_ADAPTER_ENABLED:-true}"
WAIT_SECONDS="${WAIT_SECONDS:-10}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_ROOT="${OPEN_UNLEARNING_ROOT}/saves/unlearn"

models=(
    "qwen2_5_coder_3b"
    "meta_llama3_2_3b"
)

standard_methods=(
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

ordered_variants=(
    "retain_first|${ORDERED_RETAIN_DATASET}|retain-first"
    "forget_first|${ORDERED_RETAIN_DATASET}|forget-first"
    "random|${RETAIN_FULL_DATASET}|random-retain-full"
)

if [[ ! "${WAIT_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "WAIT_SECONDS must be a non-negative number, got: ${WAIT_SECONDS}" >&2
    exit 2
fi

if [[ "${DRY_RUN}" != "1" && "${HUB_ADAPTER_ENABLED}" != "true" ]]; then
    echo "Refusing cleanup run because HUB_ADAPTER_ENABLED is not true." >&2
    echo "The local saves are deleted after each job, so Hub upload must remain enabled." >&2
    exit 2
fi

total_jobs=$((
    ${#models[@]} * ${#standard_methods[@]}
    + ${#ordered_variants[@]} * ${#models[@]} * ${#ordered_methods[@]}
))
completed_jobs=0

run_command() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

run_directory() {
    local task_name="$1"
    if [[ -z "${task_name}" || "${task_name}" == */* ]]; then
        echo "Refusing unsafe task name for cleanup: ${task_name}" >&2
        return 2
    fi
    printf '%s/%s\n' "${SAVE_ROOT}" "${task_name}"
}

remove_run_directory() {
    local task_name="$1"
    local reason="$2"
    local target
    target="$(run_directory "${task_name}")"

    case "${target}" in
        "${SAVE_ROOT}"/*) ;;
        *)
            echo "Refusing to delete path outside ${SAVE_ROOT}: ${target}" >&2
            return 2
            ;;
    esac

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN: rm -rf -- %q # %s\n' "${target}" "${reason}"
    elif [[ -e "${target}" ]]; then
        echo "Removing ${reason}: ${target}"
        rm -rf -- "${target}"
    fi
}

finish_job() {
    local task_name="$1"
    remove_run_directory "${task_name}" "completed run"
    completed_jobs=$((completed_jobs + 1))

    if (( completed_jobs < total_jobs )); then
        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "DRY RUN: sleep ${WAIT_SECONDS}"
        else
            echo "Waiting ${WAIT_SECONDS}s before the next unlearning job."
            sleep "${WAIT_SECONDS}"
        fi
    fi
}

run_standard_job() {
    local model="$1"
    local method="$2"
    shift 2

    local repo_name="${MODEL_NAME_PREFIX}secret-unlearning-${model}-${method}"
    local repo_id="${HUB_NAMESPACE}/${repo_name}"
    local task_name_prefix="${MODEL_NAME_PREFIX//[^[:alnum:]_]/_}"
    local task_name="custom_hf_${task_name_prefix}secret_${model}_${method}"
    local current_job=$((completed_jobs + 1))

    remove_run_directory "${task_name}" "stale pre-run output"

    echo
    echo "[${current_job}/${total_jobs}] Standard secret unlearning: model=${model}, method=${method}"
    echo "Task name: ${task_name}"
    echo "Hub repo: ${repo_id}"

    run_command python src/train.py \
        experiment=custom_hf_unlearning/secret \
        experiment/custom_hf_unlearning/model="${model}" \
        experiment/custom_hf_unlearning/method="${method}" \
        task_name="${task_name}" \
        hub_adapter.enabled="${HUB_ADAPTER_ENABLED}" \
        hub_adapter.repo_id="${repo_id}" \
        "$@"

    finish_job "${task_name}"
}

run_ordered_job() {
    local batch_order="$1"
    local model="$2"
    local method="$3"
    local retain_dataset="$4"
    local run_suffix="$5"
    shift 5

    local repo_name="${MODEL_NAME_PREFIX}secret-unlearning-${model}-${method}-${run_suffix}"
    local repo_id="${HUB_NAMESPACE}/${repo_name}"
    local task_name_prefix="${MODEL_NAME_PREFIX//[^[:alnum:]_]/_}"
    local task_name
    task_name="custom_hf_${task_name_prefix}secret_${model}_${method}_${run_suffix//-/_}"
    local current_job=$((completed_jobs + 1))

    remove_run_directory "${task_name}" "stale pre-run output"

    echo
    echo "[${current_job}/${total_jobs}] Ordered secret unlearning: order=${batch_order}, model=${model}, method=${method}"
    echo "Retain dataset: ${retain_dataset}"
    echo "Task name: ${task_name}"
    echo "Hub repo: ${repo_id}"

    run_command python src/train.py \
        experiment=custom_hf_unlearning/secret \
        experiment/custom_hf_unlearning/model="${model}" \
        experiment/custom_hf_unlearning/method="${method}" \
        data.batch_mode=unpaired \
        data.batch_order="${batch_order}" \
        retain_dataset_path="${retain_dataset}" \
        task_name="${task_name}" \
        hub_adapter.enabled="${HUB_ADAPTER_ENABLED}" \
        hub_adapter.repo_id="${repo_id}" \
        "$@"

    finish_job "${task_name}"
}

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Queued ${total_jobs} sequential jobs: 18 standard, then 36 ordered."
echo "Completed run directories will be removed from: ${SAVE_ROOT}"
echo "Delay between jobs: ${WAIT_SECONDS}s"

echo
echo "Phase 1/2: standard secret-unlearning jobs"
for model in "${models[@]}"; do
    for method in "${standard_methods[@]}"; do
        run_standard_job "${model}" "${method}" "$@"
    done
done

echo
echo "Phase 2/2: ordered secret-unlearning jobs"
for variant in "${ordered_variants[@]}"; do
    IFS="|" read -r batch_order retain_dataset run_suffix <<< "${variant}"
    for model in "${models[@]}"; do
        for method in "${ordered_methods[@]}"; do
            run_ordered_job \
                "${batch_order}" "${model}" "${method}" \
                "${retain_dataset}" "${run_suffix}" "$@"
        done
    done
done

echo
echo "Completed all ${total_jobs} secret-unlearning jobs."
