# Thesis-specific OpenUnlearning configs

This Hydra config group adapts the [OpenUnlearning framework](../../../README.md) to the synthetic code-unlearning experiments from the thesis. It composes three independent choices:

```text
task + model + method
```

The training entry point is `open-unlearning/src/train.py`.

## Available choices

### Tasks

| Config | Forget target | Forget target field | Retain field |
| --- | --- | --- | --- |
| `secret` | Injected API key, password, or email | `secret_value` | `code` |
| `code_unit` | Complete function or class | `code` | `code` |
| `filtered_code_unit` | Keyword/decorator-filtered code ablation | `code_filtered` | `code` |

All three use `check` as the completion prefix, `dbaysal/forget` as the default forget dataset, and `dbaysal/retain-half` as the default retain dataset.

### Models

| Config | Base model | Default learned LoRA adapter |
| --- | --- | --- |
| `qwen2_5_coder_3b` | `Qwen/Qwen2.5-Coder-3B` | `dbaysal/qwen2.5coder-3b-learned`, `checkpoint-282` |
| `meta_llama3_2_3b` | `meta-llama/Llama-3.2-3B` | `dbaysal/metallama3.2-3b-learned`, `checkpoint-282` |

Both configs use bf16, Flash Attention 2, base-model prompting without a chat template, and update the loaded LoRA adapter.

### Methods

| Config | Forget objective | Retain objective | Learning rate |
| --- | --- | --- | ---: |
| `ga` | GA | None | `5e-5` |
| `ga_gd` | GA | GD / NLL | `5e-5` |
| `ga_kl` | GA | KL | `5e-5` |
| `npo` | NPO, beta `0.1` | None | `1e-4` |
| `npo_gd` | NPO, beta `0.1` | GD / NLL | `1e-4` |
| `npo_kl` | NPO, beta `0.1` | KL | `1e-4` |
| `prod` | PROD, top-p `0.8`, alpha `0` | None | `3e-4` |
| `prod_gd` | PROD, top-p `0.8`, alpha `0` | GD / NLL | `3e-4` |
| `prod_kl` | PROD, top-p `0.8`, alpha `0` | KL | `3e-4` |

Every method runs for three epochs with AdamW, a constant learning rate, bf16, an effective batch size of 32 on one process, and epoch checkpointing. The retained term has weight `1.0` when enabled. NPO, PROD, and KL use a frozen copy of the learned model; configurations with `reference_model_args.load_in_8bit=true` quantize that oracle with bitsandbytes.

## Run one experiment

From `open-unlearning/`:

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py \
  experiment=custom_hf_unlearning/secret \
  experiment/custom_hf_unlearning/model=qwen2_5_coder_3b \
  experiment/custom_hf_unlearning/method=npo_kl \
  hub_adapter.enabled=false
```

Hydra resolves the output to:

```text
saves/unlearn/custom_hf_<task>_<model>_<method>/
```

The folder includes the final adapter, one checkpoint per epoch, `run_config.yaml`, trainer state, logs, `.hydra/` settings, and `emissions/emissions.csv` when CodeCarbon is enabled.


Preview the fully composed config without starting training:

```bash
python src/train.py \
  experiment=custom_hf_unlearning/secret \
  experiment/custom_hf_unlearning/model=qwen2_5_coder_3b \
  experiment/custom_hf_unlearning/method=npo_kl \
  --cfg job --resolve
```

## Common overrides

Use local or alternative Hub datasets:

```text
forget_dataset_path=/path/or/hub-id
retain_dataset_path=/path/or/hub-id
forget_split=train
retain_split=train
```

Use a locally fine-tuned adapter:

```text
model.model_args.peft_name=/absolute/path/to/adapter
model.model_args.peft_checkpoint_subfolder=checkpoint-N
```

Reduce memory requirements:

```text
trainer.args.per_device_train_batch_size=2
trainer.args.gradient_accumulation_steps=16
trainer.args.gradient_checkpointing=true
model.model_args.attn_implementation=sdpa
```

Disable energy tracking:

```text
codecarbon.enabled=false
```

The forget and retain datasets may be a Hugging Face ID or a local path accepted by `datasets.load_dataset`. Required fields are determined by the selected task table above.

## Objective ordering and retain-set size

Combined methods use homogeneous forget or retain batches with:

```text
data.batch_mode=unpaired
```

Choose their epoch-level batch order with:

```text
data.batch_order=random
data.batch_order=forget_first
data.batch_order=retain_first
```

In `unpaired` mode each split item appears once per epoch. `random` shuffles homogeneous forget and retain batches, while the ordered variants put every batch of the selected split first. Plain forget-only methods default to `paired`, where the forget split anchors dataset length and non-anchor examples are sampled with replacement; their loss ignores retain rows.

Example forget-first GA+KL:

```bash
python src/train.py \
  experiment=custom_hf_unlearning/secret \
  experiment/custom_hf_unlearning/model=qwen2_5_coder_3b \
  experiment/custom_hf_unlearning/method=ga_kl \
  data.batch_mode=unpaired \
  data.batch_order=forget_first \
  retain_dataset_path=dbaysal/retain-half \
  task_name=custom_hf_secret_qwen2_5_coder_3b_ga_kl_forget_first \
  hub_adapter.enabled=false
```

For the double-retain ablation, keep random order and change the dataset:

```text
data.batch_mode=unpaired data.batch_order=random retain_dataset_path=dbaysal/retain-full
```

`scripts/run_ordered_unlearning.sh` automates the study matrix. Its `retain_first` and `forget_first` jobs use `retain-half`; its job named `random-retain-full` is the separate double-retain ablation, not the equal-size random baseline.

## Run a matrix

The scripts below run two models across all nine methods and upload each result to the hard-coded `dbaysal` namespace:

```bash
bash scripts/run_secret_unlearning.sh
bash scripts/run_code_unit_unlearning.sh
```

Review and edit their `repo_id` construction before use. For local-only work, prefer the single command above with `hub_adapter.enabled=false`.

To rerun every secret-unlearning combination under Hub model names prefixed with
`new-`, use:

```bash
bash scripts/run_new_secret_unlearning.sh
```

This produces repositories such as
`dbaysal/new-secret-unlearning-qwen2_5_coder_3b-ga`. Preview all 18 commands or
change the namespace/prefix with environment variables:

```bash
DRY_RUN=1 HUB_NAMESPACE=dbaysal MODEL_NAME_PREFIX=new- \
  bash scripts/run_new_secret_unlearning.sh
```

From the repository root, the more configurable ordering runner supports dry-run and namespace controls:

```bash
DRY_RUN=1 HUB_ADAPTER_ENABLED=false bash scripts/run_ordered_unlearning.sh
```

See the repository-level [`scripts/README.md`](../../../../scripts/README.md) for multi-GPU scheduling, full-parameter GA, evaluation suites, filters, outputs, and resumability.

## Uploading artifacts

Hub upload is off in the base config. To enable it deliberately:

```text
hub_adapter.enabled=true hub_adapter.repo_id=YOUR_NAMESPACE/YOUR_REPOSITORY
```

The default allow-list includes adapter checkpoints, tokenizer files, resolved config, emissions, and Hydra settings. Set `hub_adapter.private=true` if the artifacts must not be public.

## Implementation pointers

- `src/data/pretraining.py`: prefix/target tokenization through `CompletionDataset`.
- `src/data/unlearn.py`: paired, mixed, and ordered/unpaired forget-retain sampling.
- `src/trainer/unlearn/grad_ascent.py`: GA.
- `src/trainer/unlearn/grad_diff.py`: GA+GD and GA+KL, including oracle loading.
- `src/trainer/unlearn/npo.py`: NPO variants.
- `src/trainer/unlearn/prod.py`: PROD variants.
- `src/train.py`: training, CodeCarbon tracking, checkpoint saving, and optional Hub upload.

For generic OpenUnlearning concepts, Hydra usage, distributed training, and upstream benchmarks, continue with the [main OpenUnlearning README](../../../README.md) and `open-unlearning/docs/`.
