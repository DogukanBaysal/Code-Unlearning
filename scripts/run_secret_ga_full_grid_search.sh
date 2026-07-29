#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPEN_UNLEARNING_DIR="${REPO_ROOT}/open-unlearning"
UNLEARNING_EVAL_DIR="${REPO_ROOT}/UnlearningEvaluation"
EVALPLUS_DIR="${REPO_ROOT}/evalplus"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/secret_ga_full_grid_search}"
LEARNING_RATES="${LEARNING_RATES:-4e-5 3e-5 2e-5}"
MODEL_KEYS="${MODEL_KEYS:-qwen2_5_coder_3b}"
QWEN_SOURCE_REPO="${QWEN_SOURCE_REPO:-dbaysal/qwen2.5coder-3b-learned-checkpoint282-full}"
LLAMA_SOURCE_REPO="${LLAMA_SOURCE_REPO:-dbaysal/metallama3.2-3b-learned-checkpoint282-full}"
QWEN_TOKENIZER_REPO="${QWEN_TOKENIZER_REPO:-Qwen/Qwen2.5-Coder-3B}"
LLAMA_TOKENIZER_REPO="${LLAMA_TOKENIZER_REPO:-meta-llama/Llama-3.2-3B}"
HUB_NAMESPACE="${HUB_NAMESPACE:-dbaysal}"
HUB_UPLOAD="${HUB_UPLOAD:-1}"

# Match the original custom_hf_unlearning/method/ga.yaml defaults exactly.
TRAIN_BS="${TRAIN_BS:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
CHECKPOINTS="${CHECKPOINTS:-}"
SUFFIX_BS="${SUFFIX_BS:-16}"
UTILITY_BS="${UTILITY_BS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2056}"
DTYPE="${DTYPE:-bfloat16}"
BACKEND="${BACKEND:-hf}"
UTILITY_PARALLEL="${UTILITY_PARALLEL:-}"
EVALPLUS_TIMEOUT_PER_TASK="${EVALPLUS_TIMEOUT_PER_TASK:-30}"
EVALPLUS_SANITIZE_WORKERS="${EVALPLUS_SANITIZE_WORKERS:-4}"
AGGREGATE_FILTER_CSV="${AGGREGATE_FILTER_CSV-${UNLEARNING_EVAL_DIR}/non_exact_matches.csv}"
BASELINE_FILTER_CSV="${BASELINE_FILTER_CSV-${EVALPLUS_DIR}/evalplus/baseline_failed_test_ids.csv}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_UNLEARNING="${RUN_UNLEARNING:-1}"
RUN_EVALUATIONS="${RUN_EVALUATIONS:-1}"
EVAL_MODEL_SOURCE="${EVAL_MODEL_SOURCE:-auto}"

if [ -n "${PYTHON_BIN:-}" ]; then
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Configured PYTHON_BIN was not found: ${PYTHON_BIN}" >&2
        exit 127
    fi
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Python was not found. Activate the project environment or set PYTHON_BIN." >&2
    exit 127
fi

for boolean_name in \
    HUB_UPLOAD GRADIENT_CHECKPOINTING SKIP_EXISTING CONTINUE_ON_ERROR DRY_RUN \
    RUN_UNLEARNING RUN_EVALUATIONS; do
    boolean_value="${!boolean_name}"
    if [ "${boolean_value}" != "0" ] && [ "${boolean_value}" != "1" ]; then
        echo "${boolean_name} must be 0 or 1: ${boolean_value}" >&2
        exit 2
    fi
done
if [ "${RUN_UNLEARNING}" = "0" ] && [ "${RUN_EVALUATIONS}" = "0" ]; then
    echo "At least one of RUN_UNLEARNING or RUN_EVALUATIONS must be 1." >&2
    exit 2
fi
if [ "${RUN_EVALUATIONS}" = "1" ]; then
    for filter_spec in \
        "AGGREGATE_FILTER_CSV=${AGGREGATE_FILTER_CSV}" \
        "BASELINE_FILTER_CSV=${BASELINE_FILTER_CSV}"; do
        filter_name="${filter_spec%%=*}"
        filter_path="${filter_spec#*=}"
        if [ -n "${filter_path}" ] && [ ! -f "${filter_path}" ]; then
            echo "${filter_name} was not found: ${filter_path}" >&2
            exit 2
        fi
    done
fi
case "${EVAL_MODEL_SOURCE}" in
    auto|local|hub) ;;
    *)
        echo "EVAL_MODEL_SOURCE must be auto, local, or hub: ${EVAL_MODEL_SOURCE}" >&2
        exit 2
        ;;
esac

for integer_spec in \
    "TRAIN_BS=${TRAIN_BS}" \
    "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}" \
    "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}" \
    "SUFFIX_BS=${SUFFIX_BS}" \
    "UTILITY_BS=${UTILITY_BS}" \
    "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}" \
    "EVALPLUS_TIMEOUT_PER_TASK=${EVALPLUS_TIMEOUT_PER_TASK}" \
    "EVALPLUS_SANITIZE_WORKERS=${EVALPLUS_SANITIZE_WORKERS}"; do
    integer_name="${integer_spec%%=*}"
    integer_value="${integer_spec#*=}"
    if ! [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${integer_name} must be a positive integer: ${integer_value}" >&2
        exit 2
    fi
done
if [ -n "${UTILITY_PARALLEL}" ] && \
    ! [[ "${UTILITY_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "UTILITY_PARALLEL must be a positive integer: ${UTILITY_PARALLEL}" >&2
    exit 2
fi

detect_gpu_ids() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        if [ "${CUDA_VISIBLE_DEVICES}" = "-1" ]; then
            return 0
        fi
        local old_ifs="${IFS}"
        local visible_id
        IFS=","
        for visible_id in ${CUDA_VISIBLE_DEVICES}; do
            [ -n "${visible_id}" ] && echo "${visible_id}"
        done
        IFS="${old_ifs}"
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits
        return 0
    fi
    # Allows command inspection with DRY_RUN=1 on machines without NVIDIA tools.
    echo "0"
}

gpu_ids=()
while IFS= read -r gpu_id; do
    [ -n "${gpu_id}" ] && gpu_ids+=("${gpu_id}")
done < <(detect_gpu_ids)
if [ "${#gpu_ids[@]}" -eq 0 ]; then
    echo "No GPUs detected from CUDA_VISIBLE_DEVICES or nvidia-smi." >&2
    exit 1
fi

learning_rates=()
normalized_learning_rates="${LEARNING_RATES//,/ }"
read -r -a learning_rates <<< "${normalized_learning_rates}"
model_keys=()
normalized_model_keys="${MODEL_KEYS//,/ }"
read -r -a model_keys <<< "${normalized_model_keys}"
if [ "${#learning_rates[@]}" -eq 0 ] || [ "${#model_keys[@]}" -eq 0 ]; then
    echo "LEARNING_RATES and MODEL_KEYS must each contain at least one value." >&2
    exit 2
fi
for learning_rate in "${learning_rates[@]}"; do
    if ! [[ "${learning_rate}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
        echo "Invalid learning rate: ${learning_rate}" >&2
        exit 2
    fi
done

for model_key in "${model_keys[@]}"; do
    case "${model_key}" in
        qwen2_5_coder_3b|meta_llama3_2_3b) ;;
        *)
            echo "Unsupported MODEL_KEYS entry: ${model_key}" >&2
            exit 2
            ;;
    esac
done

source_repo_for_model() {
    local model_key="$1"
    case "${model_key}" in
        qwen2_5_coder_3b) echo "${QWEN_SOURCE_REPO}" ;;
        meta_llama3_2_3b) echo "${LLAMA_SOURCE_REPO}" ;;
    esac
}

tokenizer_repo_for_model() {
    local model_key="$1"
    case "${model_key}" in
        qwen2_5_coder_3b) echo "${QWEN_TOKENIZER_REPO}" ;;
        meta_llama3_2_3b) echo "${LLAMA_TOKENIZER_REPO}" ;;
    esac
}

full_model_complete() {
    local model_dir="$1"
    [ -s "${model_dir}/config.json" ] || return 1
    [ -s "${model_dir}/model.safetensors" ] || \
        [ -s "${model_dir}/model.safetensors.index.json" ] || \
        [ -s "${model_dir}/pytorch_model.bin" ] || \
        [ -s "${model_dir}/pytorch_model.bin.index.json" ]
}

secret_eval_complete() {
    local output_dir="$1"
    local result_name
    for result_name in row_results.jsonl all_results.jsonl aggregate_results.json; do
        [ -s "${output_dir}/${result_name}" ] || return 1
    done
    if [ -n "${AGGREGATE_FILTER_CSV}" ]; then
        [ -s "${output_dir}/aggregate_results_filtered.json" ] || return 1
    fi
}

utility_eval_complete() {
    local output_dir="$1"
    [ -s "${output_dir}/utilityeval.eval_results.json" ] || return 1
    if [ -n "${BASELINE_FILTER_CSV}" ]; then
        [ -s "${output_dir}/utilityeval.filtered.eval_results.json" ] || return 1
    fi
}

run_cmd() {
    printf '$'
    printf ' %q' "$@"
    echo
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi
    "$@"
}

write_secret_eval_config() {
    local config_path="$1"
    local model_path="$2"
    local output_dir="$3"
    "${PYTHON_BIN}" - \
        "${config_path}" "${model_path}" "${output_dir}" \
        "${SUFFIX_BS}" "${MAX_NEW_TOKENS}" "${DTYPE}" \
        "${AGGREGATE_FILTER_CSV}" <<'PY'
import json
import sys

(
    config_path,
    model_path,
    output_dir,
    batch_size,
    max_new_tokens,
    dtype,
    aggregate_filter_csv,
) = sys.argv[1:]
config = {
    "model_name": model_path,
    "trust_remote_code": False,
    "datasets": [
        {
            "label": "forget",
            "dataset_name": "dbaysal/forget",
            "dataset_split": "train",
            "prefix_column": "secret_prefix",
            "suffix_column": "secret_suffix",
            "uuid_column": "uuid",
            "mode": "secret",
            "code_language": "python",
            "output_dir": output_dir,
        }
    ],
    "generation": {
        "max_new_tokens": int(max_new_tokens),
        "pass_k": 1,
        "batch_size": int(batch_size),
        "device": "auto",
        "dtype": dtype,
        "greedy": True,
        "do_sample": False,
    },
}
if aggregate_filter_csv:
    config["aggregate_filter_csv"] = aggregate_filter_csv
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY
}

discover_hub_checkpoints() {
    local repo_id="$1"
    "${PYTHON_BIN}" - "${repo_id}" <<'PY'
import re
import sys

from huggingface_hub import list_repo_files

repo_id = sys.argv[1]
repo_files = list_repo_files(repo_id=repo_id, repo_type="model")
checkpoints = {
    path.split("/", 1)[0]
    for path in repo_files
    if re.match(r"checkpoint-\d+/", path)
}
for checkpoint in sorted(checkpoints, key=lambda name: int(name.split("-", 1)[1])):
    print(checkpoint)
PY
}

resolve_hub_checkpoint_path() {
    local repo_id="$1"
    local checkpoint="$2"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "HF_CACHE/${repo_id}/${checkpoint}"
        return 0
    fi
    "${PYTHON_BIN}" - "${repo_id}" "${checkpoint}" <<'PY'
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id, checkpoint = sys.argv[1:]
if not re.fullmatch(r"checkpoint-\d+", checkpoint):
    raise ValueError(f"Invalid checkpoint name: {checkpoint}")
snapshot = Path(
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{checkpoint}/*"],
    )
)
checkpoint_path = snapshot / checkpoint
weight_files = (
    checkpoint_path / "model.safetensors",
    checkpoint_path / "model.safetensors.index.json",
    checkpoint_path / "pytorch_model.bin",
    checkpoint_path / "pytorch_model.bin.index.json",
)
if not (checkpoint_path / "config.json").is_file() or not any(
    path.is_file() for path in weight_files
):
    raise RuntimeError(
        f"{repo_id}/{checkpoint} is not a complete full-model checkpoint"
    )
if not (checkpoint_path / "tokenizer_config.json").is_file():
    raise RuntimeError(
        f"{repo_id}/{checkpoint} does not contain the saved tokenizer"
    )
print(checkpoint_path)
PY
}

resolve_eval_checkpoints() {
    local model_dir="$1"
    local repo_id="$2"
    if [ -n "${CHECKPOINTS}" ]; then
        local normalized_checkpoints="${CHECKPOINTS//,/ }"
        local override_checkpoints=()
        local checkpoint
        read -r -a override_checkpoints <<< "${normalized_checkpoints}"
        for checkpoint in "${override_checkpoints[@]}"; do
            if ! [[ "${checkpoint}" =~ ^checkpoint-[0-9]+$ ]]; then
                echo "Invalid CHECKPOINTS entry: ${checkpoint}" >&2
                return 2
            fi
            echo "${checkpoint}"
        done
        return 0
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        local epoch_index
        echo "DRY_RUN: using placeholder checkpoint steps 1-${NUM_TRAIN_EPOCHS}" >&2
        for ((epoch_index = 1; epoch_index <= NUM_TRAIN_EPOCHS; epoch_index++)); do
            echo "checkpoint-${epoch_index}"
        done
        return 0
    fi

    local checkpoint_path
    local discovered_checkpoints=()
    for checkpoint_path in "${model_dir}"/checkpoint-*; do
        [ -d "${checkpoint_path}" ] || continue
        if full_model_complete "${checkpoint_path}"; then
            discovered_checkpoints+=("$(basename "${checkpoint_path}")")
        fi
    done
    if [ "${EVAL_MODEL_SOURCE}" != "hub" ] && \
        [ "${#discovered_checkpoints[@]}" -gt 0 ]; then
        printf '%s\n' "${discovered_checkpoints[@]}" | sort -t- -k2,2n
        return 0
    fi

    if [ "${EVAL_MODEL_SOURCE}" = "local" ]; then
        return 0
    fi
    discover_hub_checkpoints "${repo_id}"
}

run_checkpoint_evaluations() {
    local gpu_id="$1"
    local model_key="$2"
    local learning_rate="$3"
    local run_root="$4"
    local checkpoint="$5"
    local epoch_index="$6"
    local repo_id="$7"
    local local_model_path="${run_root}/model/${checkpoint}"
    local model_path=""
    local epoch_label="epoch-${epoch_index}_${checkpoint}"
    local eval_root="${run_root}/evaluations/${epoch_label}"
    local config_dir="${run_root}/configs"
    local secret_output="${eval_root}/secret_forget_pass1"
    local utility_output="${eval_root}/utilityeval_pass1"
    local secret_config="${config_dir}/${epoch_label}_secret_forget_pass1.json"
    local utility_result="${utility_output}/utilityeval.eval_results.json"
    local utility_filtered_result="${utility_output}/utilityeval.filtered.eval_results.json"
    local status=0

    mkdir -p "${config_dir}" "${utility_output}"
    echo
    echo "GPU ${gpu_id}: evaluating model=${model_key}, lr=${learning_rate}, epoch=${epoch_index}, checkpoint=${checkpoint}"

    if [ "${SKIP_EXISTING}" = "1" ] && \
        secret_eval_complete "${secret_output}" && \
        utility_eval_complete "${utility_output}"; then
        echo "GPU ${gpu_id}: SKIPPED completed filtered secret-forget and UtilityEval pass@1"
        return 0
    fi

    if [ "${EVAL_MODEL_SOURCE}" = "hub" ]; then
        if ! model_path="$(resolve_hub_checkpoint_path "${repo_id}" "${checkpoint}")"; then
            echo "GPU ${gpu_id}: failed to download ${repo_id}/${checkpoint}" >&2
            return 1
        fi
        echo "GPU ${gpu_id}: Hub checkpoint cache=${model_path}"
    elif full_model_complete "${local_model_path}"; then
        model_path="${local_model_path}"
        echo "GPU ${gpu_id}: local checkpoint=${model_path}"
    elif [ "${EVAL_MODEL_SOURCE}" = "auto" ]; then
        if ! model_path="$(resolve_hub_checkpoint_path "${repo_id}" "${checkpoint}")"; then
            echo "GPU ${gpu_id}: failed to download ${repo_id}/${checkpoint}" >&2
            return 1
        fi
        echo "GPU ${gpu_id}: local checkpoint missing; Hub checkpoint cache=${model_path}"
    else
        echo "GPU ${gpu_id}: incomplete local full-model checkpoint: ${local_model_path}" >&2
        return 1
    fi

    write_secret_eval_config "${secret_config}" "${model_path}" "${secret_output}"
    if [ "${SKIP_EXISTING}" = "1" ] && secret_eval_complete "${secret_output}"; then
        echo "GPU ${gpu_id}: SKIPPED completed secret-forget pass@1 evaluation"
    else
        local secret_cmd=(
            env "CUDA_VISIBLE_DEVICES=${gpu_id}"
            "${PYTHON_BIN}" evaluate_suffix_generation.py
            --config "${secret_config}"
        )
        echo "GPU ${gpu_id}: secret forget evaluation, pass@1"
        if ! (cd "${UNLEARNING_EVAL_DIR}" && run_cmd "${secret_cmd[@]}"); then
            echo "GPU ${gpu_id}: secret-forget evaluation FAILED for ${model_key} lr=${learning_rate} ${checkpoint}" >&2
            status=1
            if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                return "${status}"
            fi
        fi
    fi

    local utility_cmd=(
        env
        "CUDA_VISIBLE_DEVICES=${gpu_id}"
        "PYTHONPATH=${EVALPLUS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        "EVALPLUS_TIMEOUT_PER_TASK=${EVALPLUS_TIMEOUT_PER_TASK}"
        "EVALPLUS_SANITIZE_WORKERS=${EVALPLUS_SANITIZE_WORKERS}"
        "${PYTHON_BIN}" -m evalplus.evaluate
        --model "${model_path}"
        --dataset utilityeval
        --backend "${BACKEND}"
        --defer-sanitize
        --greedy
        --bs "${UTILITY_BS}"
        --force-base-prompt
        --root "${utility_output}"
        --output-file "${utility_result}"
        --dtype "${DTYPE}"
    )
    if [ -n "${UTILITY_PARALLEL}" ]; then
        utility_cmd+=(--parallel "${UTILITY_PARALLEL}")
    fi

    if [ "${SKIP_EXISTING}" = "1" ] && [ -s "${utility_result}" ]; then
        echo "GPU ${gpu_id}: SKIPPED completed raw UtilityEval pass@1"
    else
        echo "GPU ${gpu_id}: UtilityEval, pass@1"
        if ! run_cmd "${utility_cmd[@]}"; then
            echo "GPU ${gpu_id}: UtilityEval FAILED for ${model_key} lr=${learning_rate} ${checkpoint}" >&2
            status=1
            if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                return "${status}"
            fi
        fi
    fi

    if [ -n "${BASELINE_FILTER_CSV}" ]; then
        if [ "${SKIP_EXISTING}" = "1" ] && [ -s "${utility_filtered_result}" ]; then
            echo "GPU ${gpu_id}: SKIPPED completed filtered UtilityEval pass@1"
        elif [ "${DRY_RUN}" = "1" ] || [ -s "${utility_result}" ]; then
            local utility_filter_cmd=(
                "${PYTHON_BIN}" "${EVALPLUS_DIR}/tools/filter_baseline_failed_results.py"
                "${utility_result}"
                --filter-csv "${BASELINE_FILTER_CSV}"
                --output "${utility_filtered_result}"
                --dataset utilityeval
            )
            echo "GPU ${gpu_id}: filtering UtilityEval baseline-failed tasks"
            if ! run_cmd "${utility_filter_cmd[@]}"; then
                echo "GPU ${gpu_id}: UtilityEval filtering FAILED for ${model_key} lr=${learning_rate} ${checkpoint}" >&2
                status=1
            fi
        else
            echo "GPU ${gpu_id}: cannot filter missing UtilityEval result: ${utility_result}" >&2
            status=1
        fi
    fi
    return "${status}"
}

run_job() {
    local gpu_id="$1"
    local model_key="$2"
    local source_repo="$3"
    local learning_rate="$4"
    local tokenizer_repo
    tokenizer_repo="$(tokenizer_repo_for_model "${model_key}")"
    local lr_slug="${learning_rate//+/_}"
    local run_root="${OUTPUT_ROOT}/${model_key}/lr-${lr_slug}"
    local model_dir="${run_root}/model"
    local repo_id="${HUB_NAMESPACE}/secret-unlearning-${model_key}-ga-full-lr-${lr_slug}"
    local task_name="custom_hf_secret_${model_key}_ga_full_lr_${lr_slug//-/_}"
    local hub_upload_hydra="false"
    local gradient_checkpointing_hydra="false"
    local status=0

    if [ "${HUB_UPLOAD}" = "1" ]; then
        hub_upload_hydra="true"
    fi
    if [ "${GRADIENT_CHECKPOINTING}" = "1" ]; then
        gradient_checkpointing_hydra="true"
    fi

    mkdir -p "${run_root}/configs"

    echo
    echo "GPU ${gpu_id}: model=${model_key}, method=ga, full_model=true, learning_rate=${learning_rate}"
    echo "GPU ${gpu_id}: source_full_model=${source_repo}"
    echo "GPU ${gpu_id}: tokenizer=${tokenizer_repo}"
    echo "GPU ${gpu_id}: output=${run_root}"
    if [ "${HUB_UPLOAD}" = "1" ]; then
        echo "GPU ${gpu_id}: hub_repo=${repo_id}"
    fi

    if [ "${RUN_UNLEARNING}" = "1" ]; then
        if [ "${SKIP_EXISTING}" = "1" ] && full_model_complete "${model_dir}"; then
            echo "GPU ${gpu_id}: SKIPPED completed training"
        else
            local train_cmd=(
                env "CUDA_VISIBLE_DEVICES=${gpu_id}"
                "${PYTHON_BIN}" src/train.py
                experiment=custom_hf_unlearning/secret
                "experiment/custom_hf_unlearning/model=${model_key}"
                experiment/custom_hf_unlearning/method=ga
                "model.model_args.pretrained_model_name_or_path=${source_repo}"
                "model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_repo}"
                '~model.model_args.peft_name'
                '~model.model_args.peft_checkpoint_subfolder'
                "trainer.args.learning_rate=${learning_rate}"
                "trainer.args.per_device_train_batch_size=${TRAIN_BS}"
                "trainer.args.gradient_accumulation_steps=${GRAD_ACCUM_STEPS}"
                "trainer.args.gradient_checkpointing=${gradient_checkpointing_hydra}"
                "trainer.args.num_train_epochs=${NUM_TRAIN_EPOCHS}"
                "paths.output_dir=${model_dir}"
                "task_name=${task_name}"
                "hub_adapter.enabled=${hub_upload_hydra}"
                hub_adapter.artifact_type=full_model
                "hub_adapter.repo_id=${repo_id}"
            )
            echo "GPU ${gpu_id}: training all parameters from the full checkpoint-282 model"
            if ! (cd "${OPEN_UNLEARNING_DIR}" && run_cmd "${train_cmd[@]}"); then
                echo "GPU ${gpu_id}: training FAILED for ${model_key} lr=${learning_rate}" >&2
                return 1
            fi
        fi
    else
        echo "GPU ${gpu_id}: evaluation model source=${EVAL_MODEL_SOURCE}"
        if [ "${EVAL_MODEL_SOURCE}" != "local" ]; then
            echo "GPU ${gpu_id}: evaluation Hub repo=${repo_id}"
        fi
    fi

    if [ "${RUN_EVALUATIONS}" != "1" ]; then
        return 0
    fi

    local checkpoint_list
    if ! checkpoint_list="$(resolve_eval_checkpoints "${model_dir}" "${repo_id}")"; then
        return 1
    fi
    if [ -z "${checkpoint_list}" ]; then
        echo "GPU ${gpu_id}: no checkpoint-* full models found for ${repo_id}" >&2
        return 1
    fi

    local epoch_index=0
    local checkpoint
    while IFS= read -r checkpoint; do
        [ -n "${checkpoint}" ] || continue
        epoch_index=$((epoch_index + 1))
        if ! run_checkpoint_evaluations \
            "${gpu_id}" "${model_key}" "${learning_rate}" "${run_root}" \
            "${checkpoint}" "${epoch_index}" "${repo_id}"; then
            status=1
            if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                return "${status}"
            fi
        fi
    done <<< "${checkpoint_list}"
    return "${status}"
}

jobs=()
for model_key in "${model_keys[@]}"; do
    source_repo="$(source_repo_for_model "${model_key}")"
    for learning_rate in "${learning_rates[@]}"; do
        jobs+=("${model_key}|${source_repo}|${learning_rate}")
    done
done

write_grid_summary() {
    "${PYTHON_BIN}" - \
        "${OUTPUT_ROOT}" \
        "${AGGREGATE_FILTER_CSV}" \
        "${BASELINE_FILTER_CSV}" \
        "${jobs[@]}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
secret_filter_enabled = bool(sys.argv[2])
utility_filter_enabled = bool(sys.argv[3])
jobs = sys.argv[4:]
rows = []
for job in jobs:
    model_key, source_repo, learning_rate = job.split("|", 2)
    lr_slug = learning_rate.replace("+", "_")
    run_root = output_root / model_key / f"lr-{lr_slug}"
    eval_dir_pattern = re.compile(r"epoch-(\d+)_checkpoint-(\d+)$")
    eval_dirs = []
    for eval_dir in (run_root / "evaluations").glob("epoch-*_checkpoint-*"):
        match = eval_dir_pattern.fullmatch(eval_dir.name)
        if match:
            eval_dirs.append((int(match.group(1)), int(match.group(2)), eval_dir))

    for epoch, global_step, eval_dir in sorted(eval_dirs):
        checkpoint = f"checkpoint-{global_step}"
        secret_raw_path = (
            eval_dir / "secret_forget_pass1" / "aggregate_results.json"
        )
        secret_filtered_path = (
            eval_dir
            / "secret_forget_pass1"
            / "aggregate_results_filtered.json"
        )
        utility_raw_path = (
            eval_dir
            / "utilityeval_pass1"
            / "utilityeval.eval_results.json"
        )
        utility_filtered_path = (
            eval_dir
            / "utilityeval_pass1"
            / "utilityeval.filtered.eval_results.json"
        )
        secret_path = (
            secret_filtered_path if secret_filter_enabled else secret_raw_path
        )
        utility_path = (
            utility_filtered_path if utility_filter_enabled else utility_raw_path
        )

        secret_similarity = None
        utility_pass_1 = None
        raw_secret_similarity = None
        raw_utility_pass_1 = None
        secret_excluded_examples = 0
        utility_excluded_tasks = 0
        if secret_raw_path.is_file():
            with secret_raw_path.open(encoding="utf-8") as handle:
                raw_secret_payload = json.load(handle)
            raw_secret_similarity = raw_secret_payload.get(
                "average_similarity_score"
            )
        if utility_raw_path.is_file():
            with utility_raw_path.open(encoding="utf-8") as handle:
                raw_utility_payload = json.load(handle)
            raw_utility_pass_1 = (
                raw_utility_payload.get("pass_at_k", {})
                .get("base", {})
                .get("pass@1")
            )
        if secret_path.is_file():
            with secret_path.open(encoding="utf-8") as handle:
                secret_payload = json.load(handle)
            secret_similarity = secret_payload.get("average_similarity_score")
            secret_excluded_examples = (
                secret_payload.get("uuid_filter", {})
                .get("num_excluded_examples", 0)
            )
        if utility_path.is_file():
            with utility_path.open(encoding="utf-8") as handle:
                utility_payload = json.load(handle)
            utility_pass_1 = (
                utility_payload.get("pass_at_k", {})
                .get("base", {})
                .get("pass@1")
            )
            utility_excluded_tasks = (
                utility_payload.get("baseline_filter", {})
                .get("excluded_task_count", 0)
            )

        rows.append(
            {
                "model": model_key,
                "source_full_model": source_repo,
                "learning_rate": learning_rate,
                "epoch": epoch,
                "checkpoint": checkpoint,
                "global_step": global_step,
                "secret_forget_similarity_pass1_raw": raw_secret_similarity,
                "secret_forget_similarity_pass1": secret_similarity,
                "secret_excluded_examples": secret_excluded_examples,
                "utilityeval_pass1_raw": raw_utility_pass_1,
                "utilityeval_pass1": utility_pass_1,
                "utility_excluded_tasks": utility_excluded_tasks,
                "secret_filter_applied": secret_filter_enabled,
                "utility_filter_applied": utility_filter_enabled,
                "complete": (
                    secret_similarity is not None and utility_pass_1 is not None
                ),
            }
        )

summary_path = output_root / "grid_summary.csv"
summary_path.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "model",
    "source_full_model",
    "learning_rate",
    "epoch",
    "checkpoint",
    "global_step",
    "secret_forget_similarity_pass1_raw",
    "secret_forget_similarity_pass1",
    "secret_excluded_examples",
    "utilityeval_pass1_raw",
    "utilityeval_pass1",
    "utility_excluded_tasks",
    "secret_filter_applied",
    "utility_filter_applied",
    "complete",
]
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Grid summary: {summary_path}")
PY
}

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    local status=0
    local job_index model_key source_repo learning_rate
    for ((job_index = worker_index; job_index < ${#jobs[@]}; job_index += ${#gpu_ids[@]})); do
        IFS="|" read -r model_key source_repo learning_rate <<< "${jobs[${job_index}]}"
        if ! run_job "${gpu_id}" "${model_key}" "${source_repo}" "${learning_rate}"; then
            status=1
            if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                return "${status}"
            fi
        fi
    done
    return "${status}"
}

evaluation_jobs=()

build_evaluation_jobs() {
    local discovery_status=0
    local job model_key source_repo learning_rate lr_slug run_root model_dir repo_id
    local checkpoint_list checkpoint epoch_index

    evaluation_jobs=()
    for job in "${jobs[@]}"; do
        IFS="|" read -r model_key source_repo learning_rate <<< "${job}"
        lr_slug="${learning_rate//+/_}"
        run_root="${OUTPUT_ROOT}/${model_key}/lr-${lr_slug}"
        model_dir="${run_root}/model"
        repo_id="${HUB_NAMESPACE}/secret-unlearning-${model_key}-ga-full-lr-${lr_slug}"

        if ! checkpoint_list="$(resolve_eval_checkpoints "${model_dir}" "${repo_id}")"; then
            echo "Failed to discover evaluation checkpoints for ${repo_id}" >&2
            discovery_status=1
            if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                return "${discovery_status}"
            fi
            continue
        fi
        if [ -z "${checkpoint_list}" ]; then
            echo "No checkpoint-* full models found for ${repo_id}" >&2
            discovery_status=1
            if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                return "${discovery_status}"
            fi
            continue
        fi

        epoch_index=0
        while IFS= read -r checkpoint; do
            [ -n "${checkpoint}" ] || continue
            epoch_index=$((epoch_index + 1))
            evaluation_jobs+=(
                "${model_key}|${learning_rate}|${run_root}|${checkpoint}|${epoch_index}|${repo_id}"
            )
        done <<< "${checkpoint_list}"
    done
    return "${discovery_status}"
}

launch_evaluation_job() {
    local worker_index="$1"
    local job_index="$2"
    local gpu_id="${gpu_ids[${worker_index}]}"
    local model_key learning_rate run_root checkpoint epoch_index repo_id
    IFS="|" read -r \
        model_key learning_rate run_root checkpoint epoch_index repo_id \
        <<< "${evaluation_jobs[${job_index}]}"

    local log_file="${OUTPUT_ROOT}/logs/gpu-${gpu_id}.txt"
    (
        echo
        echo "Dispatcher: assigned checkpoint job $((job_index + 1))/${#evaluation_jobs[@]} to GPU ${gpu_id}"
        echo "Dispatcher: model=${model_key}, lr=${learning_rate}, epoch=${epoch_index}, checkpoint=${checkpoint}"
        if ! run_checkpoint_evaluations \
            "${gpu_id}" "${model_key}" "${learning_rate}" "${run_root}" \
            "${checkpoint}" "${epoch_index}" "${repo_id}"; then
            echo "GPU ${gpu_id}: FAILED model=${model_key}, lr=${learning_rate}, epoch=${epoch_index}, checkpoint=${checkpoint}" >&2
            exit 1
        fi
    ) >> "${log_file}" 2>&1 &
    pids[${worker_index}]="$!"
}

dispatch_evaluation_jobs() {
    local status=0
    local next_job_index=0
    local active_jobs=0
    local completion_found worker_index pid job_failed

    pids=()
    for worker_index in "${!gpu_ids[@]}"; do
        pids+=("")
    done

    # Initially give each GPU one checkpoint. Afterwards, the next pending
    # checkpoint is assigned only when a GPU becomes free.
    for worker_index in "${!gpu_ids[@]}"; do
        if [ "${next_job_index}" -ge "${#evaluation_jobs[@]}" ]; then
            break
        fi
        launch_evaluation_job "${worker_index}" "${next_job_index}"
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

            job_failed=0
            if ! wait "${pid}"; then
                status=1
                job_failed=1
            fi
            pids[${worker_index}]=""
            active_jobs=$((active_jobs - 1))
            completion_found=1

            if [ "${job_failed}" = "1" ] && [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                next_job_index="${#evaluation_jobs[@]}"
            fi
            if [ "${next_job_index}" -lt "${#evaluation_jobs[@]}" ]; then
                launch_evaluation_job "${worker_index}" "${next_job_index}"
                next_job_index=$((next_job_index + 1))
                active_jobs=$((active_jobs + 1))
            fi
        done

        if [ "${completion_found}" -eq 0 ]; then
            sleep 1
        fi
    done
    return "${status}"
}

echo "Secret GA full-model learning-rate grid"
echo "Models: ${model_keys[*]}"
for model_key in "${model_keys[@]}"; do
    echo "${model_key} full source: $(source_repo_for_model "${model_key}")"
done
echo "Learning rates: ${learning_rates[*]}"
echo "GPUs: ${gpu_ids[*]}"
if [ "${RUN_UNLEARNING}" = "1" ]; then
    echo "Queued learning-rate jobs: ${#jobs[@]}"
    echo "Training batch: per_device=${TRAIN_BS}, gradient_accumulation=${GRAD_ACCUM_STEPS}"
    echo "Gradient checkpointing: ${GRADIENT_CHECKPOINTING}"
    echo "Phase: full-model GA unlearning"
fi
if [ "${RUN_EVALUATIONS}" = "1" ]; then
    echo "Evaluation model source: ${EVAL_MODEL_SOURCE}"
    echo "Phase: filtered secret forget pass@1 and filtered UtilityEval pass@1"
    if [ -n "${AGGREGATE_FILTER_CSV}" ]; then
        echo "Secret UUID filter: ${AGGREGATE_FILTER_CSV}"
    else
        echo "Secret UUID filter: disabled"
    fi
    if [ -n "${BASELINE_FILTER_CSV}" ]; then
        echo "UtilityEval baseline filter: ${BASELINE_FILTER_CSV}"
    else
        echo "UtilityEval baseline filter: disabled"
    fi
fi
echo "Output root: ${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}/logs"

# The standalone evaluation wrapper uses the same dynamic checkpoint dispatcher
# as the other evaluation scripts. Training jobs retain learning-rate granularity
# because all checkpoints for a learning rate are created by that training job.
if [ "${RUN_UNLEARNING}" = "0" ] && [ "${RUN_EVALUATIONS}" = "1" ]; then
    status=0
    if ! build_evaluation_jobs; then
        status=1
    fi
    echo "Queued ${#evaluation_jobs[@]} checkpoint jobs; the next pending epoch goes to the first free GPU."

    for gpu_id in "${gpu_ids[@]}"; do
        log_file="${OUTPUT_ROOT}/logs/gpu-${gpu_id}.txt"
        : > "${log_file}"
        echo "GPU ${gpu_id} log: ${log_file}"
    done

    if [ "${#evaluation_jobs[@]}" -eq 0 ]; then
        echo "No checkpoint evaluation jobs were created." >&2
        status=1
    elif ! dispatch_evaluation_jobs; then
        status=1
    fi
    if ! write_grid_summary; then
        echo "Failed to write grid summary." >&2
        status=1
    fi
    exit "${status}"
fi

pids=()
for worker_index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[${worker_index}]}"
    log_file="${OUTPUT_ROOT}/logs/gpu-${gpu_id}.txt"
    echo "GPU ${gpu_id} log: ${log_file}"
    run_worker "${worker_index}" "${gpu_id}" > "${log_file}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
if [ "${RUN_EVALUATIONS}" = "1" ]; then
    if ! write_grid_summary; then
        echo "Failed to write grid summary." >&2
        status=1
    fi
fi
exit "${status}"
