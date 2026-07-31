
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

The suffix evaluator skips a dataset whose expected output files are already complete. EvalPlus also resumes generation from existing samples. Use a new output root when changing generation settings to avoid mixing configurations.

