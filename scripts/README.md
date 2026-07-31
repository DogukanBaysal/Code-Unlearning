# Experiment and evaluation runners

This directory contains repository-level orchestration for the thesis experiments. The scripts compose three codebases:

- `open-unlearning/` trains or unlearns models.
- `UnlearningEvaluation/` measures prefix/suffix similarity.
- `evalplus/` measures HumanEval, ForgetEval, and UtilityEval functional correctness.

Most shell runners assume access to the Hugging Face resources in the `dbaysal` namespace. Review model, dataset, adapter, output, and upload settings before starting a large job.

## Recommended entry point: one adapter

`run_adapter_eval_suite.py` evaluates one or more PEFT adapters and checkpoint subfolders. For every adapter/checkpoint pair it:

1. Generates and writes a suffix-evaluation YAML config.
2. Evaluates the forget, equal-size retain, and held-out/approximate datasets.
3. Optionally runs the combined EvalPlus functional suite.
4. Writes a checkpoint-to-epoch alias manifest.

Inspect all options:

```bash
python scripts/run_adapter_eval_suite.py --help
```

Run a pass@10 secret evaluation:

```bash
python scripts/run_adapter_eval_suite.py \
  --model Qwen/Qwen2.5-Coder-3B \
  --peft-names YOUR_NAMESPACE/YOUR_ADAPTER \
  --discover-checkpoints \
  --all-checkpoints \
  --alias-checkpoints-as-epochs \
  --output-root Results/my_adapter \
  --aggregate-filter-csv UnlearningEvaluation/non_exact_matches.csv \
  --pass-k 10 \
  --evalplus-pass-k 10 \
  --temperature 0.8 \
  --top-p 0.95
```

Use `--dry-run` to create configs and print commands without loading models. Use `--list-checkpoints-only` with exactly one adapter to test local/Hub checkpoint discovery. Without discovery or explicit `--checkpoints`, the fallback is `checkpoint-4`, `checkpoint-8`, and `checkpoint-12`.

For code-unit forgetting, add:

```bash
--forget-prefix-column prefix \
--forget-suffix-column suffix \
--forget-mode code
```

Useful controls include:

- `--skip-evalplus` for suffix evaluation only.
- `--continue-on-error` to continue after an individual subprocess fails.
- `--suffix-bs` and `--evalplus-bs` for GPU-memory tuning.
- `--checkpoints ...`, `--num-checkpoints`, and `--checkpoint-selection first|last` for checkpoint selection.
- `--` followed by extra arguments to pass through to `evalplus.evaluate`.

Output layout:

```text
OUTPUT_ROOT/
├── checkpoint_manifest.json
├── configs/<adapter>_<checkpoint>/suffix.yaml
├── unlearningeval/
│   ├── forget/<adapter>_<checkpoint>/
│   ├── retain/<adapter>_<checkpoint>/
│   └── approximate/<adapter>_<checkpoint>/
└── evalplus/<dataset>/[pass-K/]
```

The suffix evaluator skips a dataset whose expected output files are already complete. EvalPlus also resumes generation from existing samples. Use a new output root when changing generation settings to avoid mixing configurations.

## Multi-GPU behavior

Most matrix scripts create one worker per ID in `CUDA_VISIBLE_DEVICES`; if it is unset, they query `nvidia-smi`. Jobs are distributed across those workers while each individual model run stays on one GPU.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_all_epoch_pass10_eval.sh
```

Logs normally live under the selected `Results/.../logs/` directory. The original training runs used one A100 80 GB per job. Batch-size environment variables should be reduced before trying smaller GPUs.

## Script groups

### Main LoRA experiment matrices

| Script | Purpose |
| --- | --- |
| `open-unlearning/scripts/run_secret_unlearning.sh` | Train both models with all nine methods on secret targets. Upload is enabled and repository IDs are hard-coded. |
| `open-unlearning/scripts/run_code_unit_unlearning.sh` | Train the same matrix on whole code units. Upload is enabled and repository IDs are hard-coded. |
| `open-unlearning/scripts/run_secret_code_unit_eval_suite.sh` | Evaluate both published matrices across suffix and combined functional suites. |
| `run_all_epoch_pass10_eval.sh` | Run filtered pass@10 suffix evaluations for every discovered epoch of both secret and code-unit matrices. EvalPlus is skipped. |
| `run_initial_secret_forget_utility_pass10.sh` | Run pass@10 ForgetEval/UtilityEval plus separate HumanEval for the original secret matrix. |
| `run_code_unit_forget_utility_pass10.sh` | Equivalent functional evaluation for the code-unit matrix. |

The three small `run_*forget_utility_pass10.sh` launchers delegate to `_run_forget_utility_pass10_group.sh`, which owns checkpoint discovery, multi-GPU dispatch, EvalPlus filtering, and result paths. `_run_forget_utility_pass10_group.sh` is an implementation detail; use the public wrappers.

### Baseline learned-model evaluation

| Script | Purpose |
| --- | --- |
| `run_learned_checkpoint282_eval_suite.sh` | Evaluate the two learned epoch-four adapters on suffix and functional suites. |
| `run_learned_checkpoint282_pass10_eval.sh` | Pass@10 suffix reconstruction for the learned adapters. |
| `run_learned_checkpoint282_evalplus_pass5.sh` | Pass@5 ForgetEval + UtilityEval functional baseline. |
| `run_base_model_checkpoint282_forget_utility_pass10.sh` | Functional pass@10 evaluation for the learned `checkpoint-282` adapters. |

The baseline outputs are used to create `UnlearningEvaluation/non_exact_matches.csv` and `evalplus/evalplus/baseline_failed_test_ids.csv`. The reported thesis aggregates exclude items the learned model did not reproduce or solve at baseline.

### Ordering and retain-size ablations

| Script | Purpose |
| --- | --- |
| `run_ordered_unlearning.sh` | Train retain-first and forget-first jobs on `retain-half`, plus the random-order double-retain job on `retain-full`. |
| `run_ordered_secret_eval_suite.sh` | Evaluate the three published variants with suffix and combined functional pass@k. |
| `run_ordered_forget_utility_pass10.sh` | Evaluate ForgetEval/UtilityEval and HumanEval; starts in the background by default. |

Preview the training matrix without uploads:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
DRY_RUN=1 \
HUB_ADAPTER_ENABLED=false \
bash scripts/run_ordered_unlearning.sh
```

Important environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ORDERED_RETAIN_DATASET` | `dbaysal/retain-half` | Retain dataset for forget-first and retain-first |
| `RANDOM_RETAIN_DATASET` | `dbaysal/retain-full` | Retain dataset for the double-retain random run |
| `HUB_NAMESPACE` | `dbaysal` | Upload namespace |
| `HUB_ADAPTER_ENABLED` | `true` | Whether OpenUnlearning uploads adapters |
| `DRY_RUN` | `0` | Print commands instead of training |
| `LOG_DIR` | `open-unlearning/outputs/ordered_unlearning_logs` | Worker logs |

`run_ordered_forget_utility_pass10.sh` sets `RUN_IN_BACKGROUND=1`. Set `RUN_IN_BACKGROUND=0` for a foreground run whose exit code and output remain attached to the terminal.

### Filtered-code-unit ablation

| Script | Purpose |
| --- | --- |
| `run_filtered_code_unit_unlearning.sh` | Run plain PROD, GA, and NPO on `code_filtered` for both models; uploads to hard-coded repository IDs. |
| `run_filtered_code_unit_eval_suite.sh` | Evaluate the first three discovered checkpoints for those adapters. |

The filtered source field is produced by `Dataset/Synthetic/SyntaxFilter/syntax_filter.py`; read its warning in the [synthetic dataset guide](../Dataset/Synthetic/README.md) before using it.

### Full-parameter GA ablation

The main thesis experiments update LoRA parameters. These runners reproduce the separate full-parameter GA comparison:

| Script | Purpose |
| --- | --- |
| `run_secret_ga_full_grid_search.sh` | Shared training/evaluation driver for full-model GA learning-rate searches. |
| `run_secret_ga_full_grid_unlearning.sh` | Training-only wrapper. |
| `run_secret_ga_full_grid_evaluations.sh` | Evaluation-only wrapper using Hub checkpoints. |
| `run_full_ga_all_evals.sh` | Evaluate all checkpoints from a full-model repository with suffix and functional suites. |
| `run_secret_ga_2e_5_pass10_evaluations.sh` | Qwen `2e-5` pass@10 wrapper. |
| `run_secret_ga_llama_2e_5_full_unlearning.sh` | Llama `2e-5` training wrapper. |
| `run_secret_ga_llama_2e_5_pass10_evaluations.sh` | Llama `2e-5` pass@10 wrapper. |
| `filter_secret_ga_grid_results.py` | Recompute baseline-filtered secret/UtilityEval metrics and rank a completed grid without changing raw files. |

The shared grid driver defaults to Qwen learning rates `4e-5 3e-5 2e-5`, full-model Hub upload, and both training and evaluation. Always start with:

```bash
DRY_RUN=1 HUB_UPLOAD=0 bash scripts/run_secret_ga_full_grid_search.sh
```

Then explicitly choose controls such as:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_KEYS=qwen2_5_coder_3b \
LEARNING_RATES="3e-5 2e-5" \
HUB_UPLOAD=0 \
RUN_UNLEARNING=1 \
RUN_EVALUATIONS=1 \
bash scripts/run_secret_ga_full_grid_search.sh
```

The source model repositories for this workflow are merged full models, not LoRA adapters. Defaults and validation rules are documented at the top of the script.

Post-filter a completed pass@1 grid:

```bash
python scripts/filter_secret_ga_grid_results.py \
  --results-root Results/secret_ga_full_grid_search
```

The post-filter tool never changes raw evaluation files; it writes filtered aggregates and grid summaries alongside them or below `--output-root`.

### Narrow recovery and diagnostic runners

These scripts target specific missing or failed combinations rather than the full study:

- `run_meta_code_unit_last_two_eval.sh`
- `run_meta_prod_gd_code_unit_evalplus.sh`
- `run_secret_ga_2e_5_pass10_evaluations.sh`
- `run_secret_ga_llama_2e_5_pass10_evaluations.sh`

Use them only when their hard-coded adapter naming matches the artifacts being repaired.

## Common environment controls

Not every runner supports every variable, but the large evaluation drivers consistently expose most of the following:

| Variable | Typical default | Purpose |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | detected or `0` | GPU worker IDs |
| `OUTPUT_ROOT` | script-specific `Results/...` | Result destination |
| `CHECKPOINTS` | automatic discovery | Comma- or space-separated override |
| `PASS_K` | `10` in pass@10 wrappers | Samples per task |
| `TEMPERATURE` | `0.8` | Sampling temperature |
| `TOP_P` | `0.95` | Nucleus sampling threshold |
| `SUFFIX_BS` | script-specific | Suffix generation batch size |
| `EVALPLUS_BS` | script-specific | Functional generation batch size |
| `DTYPE` | `bfloat16` | EvalPlus model dtype |
| `SKIP_EXISTING` | `1` | Skip destinations whose expected result files are complete |
| `CONTINUE_ON_ERROR` | `0` | Continue after failed subprocesses where supported |
| `DRY_RUN` | `0` | Print/write setup without running model work |
| `PYTHON_BIN` | active `python` | Interpreter override where supported |

Read a script's first variable block before execution; specialized runners may use a narrower set.

## Filtering and metric semantics

Two independent baseline filters are used:

- `UnlearningEvaluation/non_exact_matches.csv` excludes suffix targets that were not exact matches at the learned baseline. Pass it as `--aggregate-filter-csv`; raw aggregates remain untouched.
- `evalplus/evalplus/baseline_failed_test_ids.csv` excludes functional tasks failed by the learned baseline. Matrix scripts invoke `evalplus/tools/filter_baseline_failed_results.py` and preserve both raw and filtered JSON.

Suffix pass@k and functional pass@k have different meanings:

- Suffix pass@k reports the maximum reference similarity among the first *k* sampled continuations, representing worst-case leakage.
- EvalPlus pass@k estimates the chance that at least one of *k* generated programs passes its tests.

The thesis converts suffix similarity to forget quality with `1 - chrF` for secrets and `1 - BLEU` for code units.

## Safety and artifact management

- EvalPlus executes generated Python. Use its Docker workflow or another isolated executor for untrusted models and datasets.
- Matrix scripts can download hundreds of gigabytes of models and create many Hub repositories. Verify IDs and use dry-run where available.
- Several scripts upload by default. Set their upload controls to false or edit the small hard-coded launchers before use.
- `Results/` is ignored by Git. Back up result roots, configs, manifests, and logs needed for analysis.
- Checkpoint discovery uses local directories first and Hugging Face Hub second; authenticated/private repos require `HF_TOKEN`.

For component-level details, continue with the [OpenUnlearning config guide](../open-unlearning/configs/experiment/custom_hf_unlearning/README.md), [suffix evaluator README](../UnlearningEvaluation/README.md), and [EvalPlus README](../evalplus/README.md).
