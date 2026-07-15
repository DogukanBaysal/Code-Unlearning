#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVALPLUS_ROOT="${REPO_ROOT}/evalplus"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_MODEL="meta-llama/Llama-3.2-3B"
PEFT_NAME="dbaysal/code-unit-unlearning-meta_llama3_2_3b-prod_gd"
DATASET="humaneval-forget-utility"
EVALPLUS_BS="${EVALPLUS_BS:-64}"
DTYPE="${DTYPE:-bfloat16}"
CHECKPOINTS="${CHECKPOINTS:-}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/meta_prod_gd_code_unit_evalplus}"

if [ -n "${CHECKPOINTS}" ]; then
    normalized_checkpoints="${CHECKPOINTS//,/ }"
    read -r -a checkpoints <<< "${normalized_checkpoints}"
else
    checkpoint_list="$(
        python scripts/run_adapter_eval_suite.py \
            --model "${BASE_MODEL}" \
            --peft-names "${PEFT_NAME}" \
            --discover-checkpoints \
            --all-checkpoints \
            --list-checkpoints-only
    )"
    checkpoints=()
    while IFS= read -r checkpoint; do
        if [ -n "${checkpoint}" ]; then
            checkpoints+=("${checkpoint}")
        fi
    done <<< "${checkpoint_list}"
fi

if [ "${#checkpoints[@]}" -eq 0 ]; then
    echo "No checkpoints found for ${PEFT_NAME}." >&2
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Adapter: ${PEFT_NAME}"
echo "Checkpoints: ${checkpoints[*]}"
echo "Dataset: ${DATASET}; batch size: ${EVALPLUS_BS}"

for checkpoint in "${checkpoints[@]}"; do
    result_dir="${OUTPUT_ROOT}/${checkpoint}"
    result_path="${result_dir}/${DATASET}.eval_results.json"
    mkdir -p "${result_dir}"

    if [ -s "${result_path}" ]; then
        echo "Skipping completed ${checkpoint}: ${result_path}"
        continue
    fi

    cmd=(
        env
        "PYTHONPATH=${EVALPLUS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        python -m evalplus.evaluate
        --model "${BASE_MODEL}"
        --peft-name "${PEFT_NAME}"
        --peft-subfolder "${checkpoint}"
        --dataset "${DATASET}"
        --backend hf
        --greedy
        --defer-sanitize
        --bs "${EVALPLUS_BS}"
        --force-base-prompt
        --root "${result_dir}"
        --output-file "${result_path}"
        --dtype "${DTYPE}"
    )

    printf '$'
    printf ' %q' "${cmd[@]}" "$@"
    echo
    if [ "${DRY_RUN}" != "1" ]; then
        "${cmd[@]}" "$@"
    fi
done
