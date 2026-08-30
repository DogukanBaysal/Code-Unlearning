# Thesis experiment scripts

These entry points operationalize the three research questions from *Forgetting by
Design* on **MOCHI (Machine Unlearning of Code with Hidden Information)**. Each uses
the Hydra configurations in `open-unlearning`, completes unlearning before evaluation,
evaluates every saved epoch, and distributes independent jobs dynamically across the
visible GPUs.

| Entry point | RQ | Default jobs per phase | Purpose |
| --- | --- | ---: | --- |
| `run_secret_experiments.sh` | RQ1 | 18 | Two models × GA/NPO/PROD and their GD/KL variants on secrets |
| `run_code_unit_experiments.sh` | RQ2 | 18 | Transfer the same model/method matrix from secrets to complete code units |
| `run_ordering_retain_experiments.sh` | RQ3 | 36 | Compare objective ordering and retain-set size for the six forget+retain methods |

`run_adapter_eval_suite.py` and `lib/thesis_workflow.bash` are shared implementation
files, not additional experiment entry points.

## Environment setup

From the top-level repository root:

```bash
git submodule update --init --recursive
bash scripts/setup_environment.sh
source .venv/bin/activate
```

Use `bash scripts/setup_environment.sh --help` for interpreter, environment-path,
and optional FlashAttention controls. See the [top-level README](../README.md) for
hardware prerequisites and the manual installation commands.

## Reproduce all thesis results

First inspect the generated commands. Dry-run mode does not load models, contact the
Hub, or execute evaluation:

```bash
bash scripts/run_secret_experiments.sh --dry-run
bash scripts/run_code_unit_experiments.sh --dry-run
bash scripts/run_ordering_retain_experiments.sh --dry-run
```

Then run all three workflows. Use a Hugging Face namespace where you can create model
repositories; `replication-` prevents collision with the original adapter names.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HUB_NAMESPACE=dbaysal
export ADAPTER_PREFIX=replication-

bash scripts/run_secret_experiments.sh
bash scripts/run_code_unit_experiments.sh
bash scripts/run_ordering_retain_experiments.sh
```

The stage barrier is intentional: if an unlearning worker fails, that script stops
without starting evaluation. Logs are stored under `<OUTPUT_ROOT>/logs/{unlearning,evaluation}/gpu-*.log`.

The ordering/retain script creates these additional secret variants:

| Variant | Batch order | Retain dataset | Thesis comparison |
| --- | --- | --- | --- |
| `retain-first` | All retain batches before forget batches | `dbaysal/retain-half` | Objective order |
| `forget-first` | All forget batches before retain batches | `dbaysal/retain-half` | Objective order |
| `random-retain-full` | Random interleaving | `dbaysal/retain-full` | Retain-set size |

The `standard` output from `run_secret_experiments.sh` is the random-order,
`dbaysal/retain-half` baseline used in both comparisons.

## Reproduction defaults

The launchers preserve the final experiment settings:

- Models: `Qwen/Qwen2.5-Coder-3B` and `meta-llama/Llama-3.2-3B`.
- Methods: `ga`, `npo`, `prod`, `ga_gd`, `ga_kl`, `npo_gd`, `npo_kl`, `prod_gd`, and `prod_kl`.
- Ordering/retain methods: the six `_gd` and `_kl` methods.
- Training: three epochs, seed 42, constant learning rate, epoch checkpoints, bf16, and the method-specific learning rates in the Hydra configs.
- Standard secret batches: 8 per device × 4 gradient-accumulation steps.
- Code-unit batches: 4 per device × 8 gradient-accumulation steps.
- Evaluation: suffix pass@10 and EvalPlus pass@10, temperature 0.8, top-p 0.95, maximum 2056 new tokens, and batches of 8 by default.
- Reconstruction: secret chrF; code-unit, retain, and approximate BLEU.
- Functional correctness: combined `humaneval-forget-utility` evaluation.
- Filtering: raw functional results plus baseline-failed-task-filtered results.

Every setting can be overridden without editing the scripts:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | Auto-detect, then `0` fallback | GPUs used by the dynamic dispatcher |
| `MODEL_KEYS` | Both model keys | Space-separated model subset |
| `METHODS` | Full matrix | Space-separated method subset |
| `HUB_NAMESPACE` | `dbaysal` | Hugging Face account or organization |
| `ADAPTER_PREFIX` | `replication-` | Prefix added to every adapter repository |
| `HUB_ADAPTER_ENABLED` | `true` | Upload training outputs to the Hub |
| `EVAL_MODEL_SOURCE` | `auto` | Evaluate `hub` or `local` adapters |
| `OUTPUT_ROOT` | `Results/thesis_<setting>` | Result and log root |
| `PASS_K` | `10` | Both suffix and EvalPlus sampling count |
| `TEMPERATURE` / `TOP_P` | `0.8` / `0.95` | Sampling parameters |
| `SUFFIX_BS` / `EVALPLUS_BS` | `8` / `8` | Generation batch sizes |
| `CHECKPOINTS` | Discover all | Comma- or space-separated checkpoint names |
| `EVALPLUS_PARALLEL` | EvalPlus default | Parallel functional-test workers |
| `EVALPLUS_BASELINE_FILTER_CSV` | EvalPlus bundled CSV | Baseline-failed ForgetEval/UtilityEval cases |

Use `--train-only` to stop after unlearning and `--eval-only` to resume with existing
adapters. `HUB_ADAPTER_ENABLED=false EVAL_MODEL_SOURCE=local` keeps the complete run
inside `open-unlearning/saves/unlearn/`. Use `ADAPTER_PREFIX=""` only when intentionally
targeting the original unprefixed adapter names.

For a one-model, one-method smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_KEYS=qwen2_5_coder_3b \
METHODS=npo_kl \
ADAPTER_PREFIX=smoke- \
  bash scripts/run_secret_experiments.sh
```

## Evaluate one adapter directly

`run_adapter_eval_suite.py` evaluates one or more PEFT adapters and checkpoint subfolders. For every adapter/checkpoint pair it:

1. Generates and writes a suffix-evaluation YAML config.
2. Evaluates the forget, equal-size retain, and held-out/approximate datasets.
3. Optionally runs the combined EvalPlus functional suite.
4. Optionally removes baseline-failed ForgetEval/UtilityEval cases from a second result.
5. Writes a checkpoint-to-epoch alias manifest.

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
  --evalplus-baseline-filter-csv evalplus/evalplus/baseline_failed_test_ids.csv \
  --pass-k 10 \
  --evalplus-pass-k 10 \
  --temperature 0.8 \
  --top-p 0.95
```

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

With `--evalplus-baseline-filter-csv`, each raw `*.eval_results.json` has a sibling
`*.filtered.eval_results.json` containing recomputed ForgetEval and UtilityEval pass@k.

`collect_new_secret_hub_energy.py` remains available as a result-analysis utility; it
is not part of the three training/evaluation entry points.
