#!/usr/bin/env bash

# Shared implementation for the three thesis workflow entry points. This file is
# sourced by the public scripts; run those scripts rather than invoking it directly.

thesis_workflow_usage() {
    local entry_script="$1"
    local description="$2"
    cat <<EOF
Usage: bash scripts/${entry_script} [options]

${description}

The default run first completes every selected unlearning job and then evaluates
all checkpoints. Options:
  --dry-run      Print every training and evaluation command without running it
  --train-only   Run unlearning, but skip evaluation
  --eval-only    Evaluate existing local or Hub adapters without training
  -h, --help     Show this help

Common environment overrides:
  CUDA_VISIBLE_DEVICES   Comma-separated GPUs; detected automatically when unset
  MODEL_KEYS             Space-separated model config keys
  METHODS                Space-separated method config keys
  HUB_NAMESPACE          Hugging Face namespace (default: dbaysal)
  ADAPTER_PREFIX         Prefix for adapter repository names (default: replication-)
  HUB_ADAPTER_ENABLED    true uploads adapters; false keeps them local
  EVAL_MODEL_SOURCE      auto, hub, or local (default: auto)
  OUTPUT_ROOT            Evaluation output directory
  PASS_K                 Suffix and EvalPlus pass@k (default: 10)
  SUFFIX_BS              Suffix-generation batch size (default: 8)
  EVALPLUS_BS            EvalPlus generation batch size (default: 8)
  EVALPLUS_BASELINE_FILTER_CSV  Baseline-failed functional-test IDs
  CHECKPOINTS            Optional comma- or space-separated checkpoint override
EOF
}

thesis_validate_zero_one() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1, got: ${value}" >&2
        return 2
    fi
}

thesis_detect_gpus() {
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

    # This fallback makes --dry-run useful on login nodes and non-GPU machines.
    echo "0"
}

thesis_base_model() {
    case "$1" in
        qwen2_5_coder_3b)
            echo "Qwen/Qwen2.5-Coder-3B"
            ;;
        meta_llama3_2_3b)
            echo "meta-llama/Llama-3.2-3B"
            ;;
        *)
            echo "Unknown model key: $1" >&2
            return 2
            ;;
    esac
}

thesis_set_adapter_metadata() {
    local model_key="$1"
    local method="$2"
    local variant_suffix="$3"
    local suffix_component=""
    if [[ -n "${variant_suffix}" ]]; then
        suffix_component="-${variant_suffix}"
    fi

    THESIS_REPO_NAME="${ADAPTER_PREFIX}${THESIS_TASK_SLUG}-unlearning-${model_key}-${method}${suffix_component}"
    THESIS_REPO_ID="${HUB_NAMESPACE}/${THESIS_REPO_NAME}"

    local task_prefix="${ADAPTER_PREFIX//[^[:alnum:]_]/_}"
    local task_suffix="${variant_suffix//-/_}"
    if [[ -n "${task_suffix}" ]]; then
        task_suffix="_${task_suffix}"
    fi
    THESIS_TASK_NAME="custom_hf_${task_prefix}${THESIS_EXPERIMENT}_${model_key}_${method}${task_suffix}"
}

thesis_print_command() {
    local command_cwd="$1"
    shift
    printf '\n$ cd %q\n' "${command_cwd}"
    printf '$'
    printf ' %q' "$@"
    printf '\n'
}

thesis_run_command() {
    local command_cwd="$1"
    shift
    thesis_print_command "${command_cwd}" "$@"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    (
        cd "${command_cwd}"
        "$@"
    )
}

thesis_train_job() {
    local gpu_id="$1"
    local job="$2"
    local model_key
    local method
    local batch_order
    local retain_dataset
    local variant_suffix
    IFS="|" read -r model_key method batch_order retain_dataset variant_suffix <<< "${job}"

    thesis_set_adapter_metadata "${model_key}" "${method}" "${variant_suffix}"

    local command=(
        env "CUDA_VISIBLE_DEVICES=${gpu_id}"
        "${PYTHON_BIN}" src/train.py
        "experiment=custom_hf_unlearning/${THESIS_EXPERIMENT}"
        "experiment/custom_hf_unlearning/model=${model_key}"
        "experiment/custom_hf_unlearning/method=${method}"
    )
    if [[ -n "${batch_order}" ]]; then
        command+=(
            "data.batch_mode=unpaired"
            "data.batch_order=${batch_order}"
            "retain_dataset_path=${retain_dataset}"
        )
    fi
    command+=(
        "task_name=${THESIS_TASK_NAME}"
        "hub_adapter.enabled=${HUB_ADAPTER_ENABLED}"
        "hub_adapter.repo_id=${THESIS_REPO_ID}"
    )

    echo "Training: setting=${THESIS_SETTING}, model=${model_key}, method=${method}, variant=${variant_suffix:-standard}"
    echo "Adapter: ${THESIS_REPO_ID}"
    thesis_run_command "${OPEN_UNLEARNING_ROOT}" "${command[@]}"
}

thesis_evaluation_source() {
    local source_mode="${EVAL_MODEL_SOURCE}"
    if [[ "${source_mode}" == "auto" ]]; then
        if [[ "${HUB_ADAPTER_ENABLED}" == "true" ]]; then
            source_mode="hub"
        else
            source_mode="local"
        fi
    fi

    case "${source_mode}" in
        hub)
            echo "${THESIS_REPO_ID}"
            ;;
        local)
            echo "${OPEN_UNLEARNING_ROOT}/saves/unlearn/${THESIS_TASK_NAME}"
            ;;
        *)
            echo "EVAL_MODEL_SOURCE must be auto, hub, or local, got: ${EVAL_MODEL_SOURCE}" >&2
            return 2
            ;;
    esac
}

thesis_eval_job() {
    local gpu_id="$1"
    local job="$2"
    local model_key
    local method
    local batch_order
    local retain_dataset
    local variant_suffix
    IFS="|" read -r model_key method batch_order retain_dataset variant_suffix <<< "${job}"

    thesis_set_adapter_metadata "${model_key}" "${method}" "${variant_suffix}"
    local base_model
    base_model="$(thesis_base_model "${model_key}")"
    local adapter_source
    adapter_source="$(thesis_evaluation_source)"

    local variant_label="${variant_suffix:-standard}"
    local output_root="${OUTPUT_ROOT}/${variant_label}/${model_key}/${method}"
    local forget_prefix_column="secret_prefix"
    local forget_suffix_column="secret_suffix"
    local forget_mode="secret"
    if [[ "${THESIS_SETTING}" == "code-unit" ]]; then
        forget_prefix_column="prefix"
        forget_suffix_column="suffix"
        forget_mode="code"
    fi

    local checkpoint_args=(--discover-checkpoints --all-checkpoints)
    if [[ -n "${CHECKPOINTS}" ]]; then
        local normalized_checkpoints="${CHECKPOINTS//,/ }"
        local selected_checkpoints=()
        read -r -a selected_checkpoints <<< "${normalized_checkpoints}"
        checkpoint_args=(--checkpoints "${selected_checkpoints[@]}")
    fi

    local command=(
        env
        "CUDA_VISIBLE_DEVICES=${gpu_id}"
        "EVALPLUS_TIMEOUT_PER_TASK=${EVALPLUS_TIMEOUT_PER_TASK}"
        "EVALPLUS_SANITIZE_WORKERS=${EVALPLUS_SANITIZE_WORKERS}"
        "${PYTHON_BIN}" scripts/run_adapter_eval_suite.py
        --model "${base_model}"
        --peft-names "${adapter_source}"
        "${checkpoint_args[@]}"
        --alias-checkpoints-as-epochs
        --output-root "${output_root}"
        --forget-dataset "${FORGET_DATASET}"
        --forget-prefix-column "${forget_prefix_column}"
        --forget-suffix-column "${forget_suffix_column}"
        --forget-mode "${forget_mode}"
        --retain-dataset "${retain_dataset}"
        --retain-prefix-column prefix
        --retain-suffix-column suffix
        --retain-mode code
        --approx-dataset "${APPROX_DATASET}"
        --approx-prefix-column prefix
        --approx-suffix-column suffix
        --approx-mode code
        --aggregate-filter-csv "${AGGREGATE_FILTER_CSV}"
        --pass-k "${PASS_K}"
        --evalplus-pass-k "${PASS_K}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --max-new-tokens "${MAX_NEW_TOKENS}"
        --suffix-bs "${SUFFIX_BS}"
        --evalplus-bs "${EVALPLUS_BS}"
        --evalplus-dataset "${EVALPLUS_DATASET}"
        --evalplus-baseline-filter-csv "${EVALPLUS_BASELINE_FILTER_CSV}"
        --backend "${BACKEND}"
        --dtype "${DTYPE}"
    )
    if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
        command+=(--trust-remote-code)
    fi
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        command+=(--continue-on-error)
    fi
    if [[ -n "${EVALPLUS_PARALLEL}" ]]; then
        command+=(-- --parallel "${EVALPLUS_PARALLEL}")
    fi

    echo "Evaluating: setting=${THESIS_SETTING}, model=${model_key}, method=${method}, variant=${variant_label}"
    echo "Adapter source: ${adapter_source}"
    thesis_run_command "${REPO_ROOT}" "${command[@]}"
}

thesis_run_phase() {
    local phase_name="$1"
    local worker_function="$2"
    local phase_log_dir="$3"

    echo
    echo "=== ${phase_name}: ${#THESIS_JOBS[@]} job(s) across ${#THESIS_GPU_IDS[@]} GPU(s) ==="
    mkdir -p "${phase_log_dir}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        local dry_job
        for dry_job in "${THESIS_JOBS[@]}"; do
            "${worker_function}" "${THESIS_GPU_IDS[0]}" "${dry_job}"
        done
        return 0
    fi

    local poll_seconds="${GPU_DISPATCH_POLL_SECONDS}"
    local status=0
    local next_job=0
    local active_jobs=0
    local worker_index
    local gpu_id
    local pid
    local finished_job
    local log_file
    local -a pids=()
    local -a job_indices=()

    for worker_index in "${!THESIS_GPU_IDS[@]}"; do
        gpu_id="${THESIS_GPU_IDS[${worker_index}]}"
        log_file="${phase_log_dir}/gpu-${gpu_id}.log"
        : > "${log_file}"
        echo "GPU ${gpu_id} log: ${log_file}"
        pids+=("")
        job_indices+=("")
    done

    while [[ "${active_jobs}" -gt 0 || ("${status}" -eq 0 && "${next_job}" -lt "${#THESIS_JOBS[@]}") ]]; do
        local progress=0
        for worker_index in "${!THESIS_GPU_IDS[@]}"; do
            gpu_id="${THESIS_GPU_IDS[${worker_index}]}"
            pid="${pids[${worker_index}]:-}"

            if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
                finished_job="${job_indices[${worker_index}]}"
                if ! wait "${pid}"; then
                    echo "${phase_name} job $((finished_job + 1)) failed on GPU ${gpu_id}." >&2
                    status=1
                fi
                pids[${worker_index}]=""
                job_indices[${worker_index}]=""
                active_jobs=$((active_jobs - 1))
                progress=1
            fi

            if [[ -z "${pids[${worker_index}]:-}" && "${status}" -eq 0 && "${next_job}" -lt "${#THESIS_JOBS[@]}" ]]; then
                log_file="${phase_log_dir}/gpu-${gpu_id}.log"
                echo "Assigning ${phase_name} job $((next_job + 1))/${#THESIS_JOBS[@]} to GPU ${gpu_id}."
                "${worker_function}" "${gpu_id}" "${THESIS_JOBS[${next_job}]}" >> "${log_file}" 2>&1 &
                pids[${worker_index}]="$!"
                job_indices[${worker_index}]="${next_job}"
                next_job=$((next_job + 1))
                active_jobs=$((active_jobs + 1))
                progress=1
            fi
        done

        if [[ "${active_jobs}" -gt 0 && "${progress}" -eq 0 ]]; then
            sleep "${poll_seconds}"
        fi
    done

    if [[ "${status}" -ne 0 ]]; then
        echo "${phase_name} failed; no later phase will be started." >&2
        return "${status}"
    fi
    echo "${phase_name} completed successfully."
}

run_thesis_workflow() {
    THESIS_SETTING="$1"
    local entry_script="$2"
    local description="$3"
    shift 3

    RUN_UNLEARNING="${RUN_UNLEARNING:-1}"
    RUN_EVALUATION="${RUN_EVALUATION:-1}"
    DRY_RUN="${DRY_RUN:-0}"

    while (($#)); do
        case "$1" in
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            --train-only)
                RUN_UNLEARNING=1
                RUN_EVALUATION=0
                shift
                ;;
            --eval-only)
                RUN_UNLEARNING=0
                RUN_EVALUATION=1
                shift
                ;;
            -h|--help)
                thesis_workflow_usage "${entry_script}" "${description}"
                return 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                thesis_workflow_usage "${entry_script}" "${description}" >&2
                return 2
                ;;
        esac
    done

    thesis_validate_zero_one RUN_UNLEARNING "${RUN_UNLEARNING}"
    thesis_validate_zero_one RUN_EVALUATION "${RUN_EVALUATION}"
    thesis_validate_zero_one DRY_RUN "${DRY_RUN}"
    if [[ "${RUN_UNLEARNING}" == "0" && "${RUN_EVALUATION}" == "0" ]]; then
        echo "At least one phase must be enabled." >&2
        return 2
    fi

    SCRIPT_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_LIB_DIR}/../.." && pwd)"
    OPEN_UNLEARNING_ROOT="${REPO_ROOT}/open-unlearning"

    if [[ -n "${PYTHON_BIN:-}" ]]; then
        if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
            echo "Configured PYTHON_BIN was not found: ${PYTHON_BIN}" >&2
            return 127
        fi
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Python was not found. Activate .venv or set PYTHON_BIN." >&2
        return 127
    fi

    HUB_NAMESPACE="${HUB_NAMESPACE:-dbaysal}"
    ADAPTER_PREFIX="${ADAPTER_PREFIX-replication-}"
    HUB_ADAPTER_ENABLED="${HUB_ADAPTER_ENABLED:-true}"
    EVAL_MODEL_SOURCE="${EVAL_MODEL_SOURCE:-auto}"
    FORGET_DATASET="${FORGET_DATASET:-dbaysal/forget}"
    RETAIN_HALF_DATASET="${RETAIN_HALF_DATASET:-dbaysal/retain-half}"
    RETAIN_FULL_DATASET="${RETAIN_FULL_DATASET:-dbaysal/retain-full}"
    APPROX_DATASET="${APPROX_DATASET:-dbaysal/approximate}"
    PASS_K="${PASS_K:-10}"
    TEMPERATURE="${TEMPERATURE:-0.8}"
    TOP_P="${TOP_P:-0.95}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2056}"
    SUFFIX_BS="${SUFFIX_BS:-8}"
    EVALPLUS_BS="${EVALPLUS_BS:-8}"
    EVALPLUS_DATASET="${EVALPLUS_DATASET:-humaneval-forget-utility}"
    EVALPLUS_BASELINE_FILTER_CSV="${EVALPLUS_BASELINE_FILTER_CSV:-${REPO_ROOT}/evalplus/evalplus/baseline_failed_test_ids.csv}"
    EVALPLUS_TIMEOUT_PER_TASK="${EVALPLUS_TIMEOUT_PER_TASK:-30}"
    EVALPLUS_SANITIZE_WORKERS="${EVALPLUS_SANITIZE_WORKERS:-4}"
    EVALPLUS_PARALLEL="${EVALPLUS_PARALLEL:-}"
    BACKEND="${BACKEND:-hf}"
    DTYPE="${DTYPE:-bfloat16}"
    CHECKPOINTS="${CHECKPOINTS:-}"
    TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
    CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
    GPU_DISPATCH_POLL_SECONDS="${GPU_DISPATCH_POLL_SECONDS:-1}"
    AGGREGATE_FILTER_CSV="${AGGREGATE_FILTER_CSV:-${REPO_ROOT}/UnlearningEvaluation/non_exact_matches.csv}"

    thesis_validate_zero_one TRUST_REMOTE_CODE "${TRUST_REMOTE_CODE}"
    thesis_validate_zero_one CONTINUE_ON_ERROR "${CONTINUE_ON_ERROR}"
    if [[ "${HUB_ADAPTER_ENABLED}" != "true" && "${HUB_ADAPTER_ENABLED}" != "false" ]]; then
        echo "HUB_ADAPTER_ENABLED must be true or false, got: ${HUB_ADAPTER_ENABLED}" >&2
        return 2
    fi
    case "${EVAL_MODEL_SOURCE}" in
        auto|hub|local) ;;
        *)
            echo "EVAL_MODEL_SOURCE must be auto, hub, or local, got: ${EVAL_MODEL_SOURCE}" >&2
            return 2
            ;;
    esac
    for positive_spec in \
        "PASS_K=${PASS_K}" \
        "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}" \
        "SUFFIX_BS=${SUFFIX_BS}" \
        "EVALPLUS_BS=${EVALPLUS_BS}" \
        "EVALPLUS_TIMEOUT_PER_TASK=${EVALPLUS_TIMEOUT_PER_TASK}" \
        "EVALPLUS_SANITIZE_WORKERS=${EVALPLUS_SANITIZE_WORKERS}"; do
        local positive_name="${positive_spec%%=*}"
        local positive_value="${positive_spec#*=}"
        if ! [[ "${positive_value}" =~ ^[1-9][0-9]*$ ]]; then
            echo "${positive_name} must be a positive integer, got: ${positive_value}" >&2
            return 2
        fi
    done
    if [[ "${RUN_EVALUATION}" == "1" && ! -f "${AGGREGATE_FILTER_CSV}" ]]; then
        echo "Aggregate filter CSV was not found: ${AGGREGATE_FILTER_CSV}" >&2
        return 2
    fi
    if [[ "${RUN_EVALUATION}" == "1" && ! -f "${EVALPLUS_BASELINE_FILTER_CSV}" ]]; then
        echo "EvalPlus baseline filter CSV was not found: ${EVALPLUS_BASELINE_FILTER_CSV}" >&2
        return 2
    fi

    local default_models="qwen2_5_coder_3b meta_llama3_2_3b"
    local default_methods="ga npo prod ga_gd ga_kl npo_gd npo_kl prod_gd prod_kl"
    if [[ "${THESIS_SETTING}" == "ordering-retain" ]]; then
        default_methods="ga_gd ga_kl npo_gd npo_kl prod_gd prod_kl"
    fi
    local selected_models="${MODEL_KEYS:-${default_models}}"
    local selected_methods="${METHODS:-${default_methods}}"
    local -a model_keys=()
    local -a methods=()
    read -r -a model_keys <<< "${selected_models}"
    read -r -a methods <<< "${selected_methods}"
    if [[ "${#model_keys[@]}" -eq 0 || "${#methods[@]}" -eq 0 ]]; then
        echo "MODEL_KEYS and METHODS must each select at least one value." >&2
        return 2
    fi

    THESIS_JOBS=()
    local model_key
    local method
    case "${THESIS_SETTING}" in
        secret)
            THESIS_TASK_SLUG="secret"
            THESIS_EXPERIMENT="secret"
            OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/thesis_secret}"
            for model_key in "${model_keys[@]}"; do
                thesis_base_model "${model_key}" >/dev/null
                for method in "${methods[@]}"; do
                    [[ -f "${OPEN_UNLEARNING_ROOT}/configs/experiment/custom_hf_unlearning/method/${method}.yaml" ]] || {
                        echo "Unknown method config: ${method}" >&2
                        return 2
                    }
                    THESIS_JOBS+=("${model_key}|${method}||${RETAIN_HALF_DATASET}|")
                done
            done
            ;;
        code-unit)
            THESIS_TASK_SLUG="code-unit"
            THESIS_EXPERIMENT="code_unit"
            OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/thesis_code_unit}"
            for model_key in "${model_keys[@]}"; do
                thesis_base_model "${model_key}" >/dev/null
                for method in "${methods[@]}"; do
                    [[ -f "${OPEN_UNLEARNING_ROOT}/configs/experiment/custom_hf_unlearning/method/${method}.yaml" ]] || {
                        echo "Unknown method config: ${method}" >&2
                        return 2
                    }
                    THESIS_JOBS+=("${model_key}|${method}||${RETAIN_HALF_DATASET}|")
                done
            done
            ;;
        ordering-retain)
            THESIS_TASK_SLUG="secret"
            THESIS_EXPERIMENT="secret"
            OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/thesis_ordering_retain}"
            local variant
            local -a variants=(
                "retain_first|${RETAIN_HALF_DATASET}|retain-first"
                "forget_first|${RETAIN_HALF_DATASET}|forget-first"
                "random|${RETAIN_FULL_DATASET}|random-retain-full"
            )
            for variant in "${variants[@]}"; do
                local batch_order
                local retain_dataset
                local variant_suffix
                IFS="|" read -r batch_order retain_dataset variant_suffix <<< "${variant}"
                for model_key in "${model_keys[@]}"; do
                    thesis_base_model "${model_key}" >/dev/null
                    for method in "${methods[@]}"; do
                        case "${method}" in
                            ga_gd|ga_kl|npo_gd|npo_kl|prod_gd|prod_kl) ;;
                            *)
                                echo "Ordering/retain experiments require a GD or KL method, got: ${method}" >&2
                                return 2
                                ;;
                        esac
                        [[ -f "${OPEN_UNLEARNING_ROOT}/configs/experiment/custom_hf_unlearning/method/${method}.yaml" ]] || {
                            echo "Unknown method config: ${method}" >&2
                            return 2
                        }
                        THESIS_JOBS+=("${model_key}|${method}|${batch_order}|${retain_dataset}|${variant_suffix}")
                    done
                done
            done
            ;;
        *)
            echo "Unknown thesis setting: ${THESIS_SETTING}" >&2
            return 2
            ;;
    esac

    THESIS_GPU_IDS=()
    while IFS= read -r gpu_id; do
        [[ -n "${gpu_id}" ]] && THESIS_GPU_IDS+=("${gpu_id}")
    done < <(thesis_detect_gpus)
    if [[ "${#THESIS_GPU_IDS[@]}" -eq 0 ]]; then
        echo "No GPUs were found in CUDA_VISIBLE_DEVICES or nvidia-smi." >&2
        return 1
    fi

    echo "Workflow: ${THESIS_SETTING}"
    echo "Models: ${model_keys[*]}"
    echo "Methods: ${methods[*]}"
    echo "GPUs: ${THESIS_GPU_IDS[*]}"
    echo "Adapter prefix: ${ADAPTER_PREFIX}"
    echo "Jobs per phase: ${#THESIS_JOBS[@]}"
    echo "Evaluation: pass@${PASS_K}, temperature=${TEMPERATURE}, top_p=${TOP_P}, suffix_bs=${SUFFIX_BS}, evalplus_bs=${EVALPLUS_BS}"
    echo "Output root: ${OUTPUT_ROOT}"

    if [[ "${RUN_UNLEARNING}" == "1" ]]; then
        local unlearning_status=0
        thesis_run_phase \
            "Unlearning" thesis_train_job "${OUTPUT_ROOT}/logs/unlearning" \
            || unlearning_status=$?
        if [[ "${unlearning_status}" -ne 0 ]]; then
            return "${unlearning_status}"
        fi
    fi
    if [[ "${RUN_EVALUATION}" == "1" ]]; then
        local evaluation_status=0
        thesis_run_phase \
            "Evaluation" thesis_eval_job "${OUTPUT_ROOT}/logs/evaluation" \
            || evaluation_status=$?
        if [[ "${evaluation_status}" -ne 0 ]]; then
            return "${evaluation_status}"
        fi
    fi

    echo
    echo "Workflow completed: ${THESIS_SETTING}"
}
