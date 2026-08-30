# Experiment scripts

These are the top-level orchestration scripts for the complete workflow. They expect
`open-unlearning`, `evalplus`, and `UnlearningEvaluation` to be present as initialized
submodules beside this directory.

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

## Other workflow scripts

- `run_new_code_unit_npo_kl_unlearning.sh` runs the code-unit NPO+KL unlearning matrix.
- `run_new_secret_npo_kl_eval_suite.sh` evaluates secret-unlearning checkpoints across GPUs.
- `run_new_npo_kl_eval_4gpu.sbatch` submits the evaluation suite through Slurm.
- `collect_new_secret_hub_energy.py` collects training energy data from Hub-hosted runs.

Each shell script documents its environment-variable overrides and dry-run behavior
near the top of the file. Run the Python entry points with `--help` before launching
a large job.
