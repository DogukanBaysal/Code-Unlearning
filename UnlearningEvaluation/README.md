# Suffix Generation Evaluation

This project evaluates a Hugging Face causal language model on a Hugging Face dataset by:

1. Reading a prefix, target suffix, and UUID from each dataset row.
2. Generating a suffix from the prefix.
3. Capping generated tokens to the tokenized target suffix length.
4. Scoring the generated suffix against the target suffix.
5. Writing inspectable row-level JSONL logs and aggregate JSON results.

## Install

Use a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`peft` is only required when `peft_name` is set.

## Configure

Copy `config.example.yaml` and edit it for your model and dataset:

```bash
cp config.example.yaml config.yaml
```

Required dataset columns are configured with:

- `prefix_column`: input text used as the generation prompt.
- `suffix_column`: target suffix text used as the reference.
- `uuid_column`: stable row identifier.

For `mode: code`, the dataset must also include `difficulty` and `type`.

For `mode: secret`, the dataset must also include `secret_location` and `secret_type`.

Generation settings can be placed under `generation`. If `greedy: true`, sampling is disabled and `temperature` / `top_p` are ignored.

## Run

```bash
python evaluate_suffix_generation.py --config config.yaml
```

The script loads `tokenizer_name` when provided; otherwise it uses `model_name`. If `peft_name` is provided, the base model is loaded first and then the PEFT adapter is applied. Set `peft_subfolder` when the adapter files live in a subfolder of the Hugging Face repository.

## Outputs

The configured `output_dir` receives two files:

- `row_results.jsonl`
- `aggregate_results.json`

Each JSONL row contains:

- `uuid`
- `prefix`
- `real_suffix`
- `generated_suffix`
- `score_type`
- `score_value`
- `metadata`
- `generation`

`aggregate_results.json` contains:

- `mode`
- `num_evaluated_examples`
- `average_similarity_score`
- `score_metric`
- `score_failures`
- `grouped_averages`
- `config`

For `mode: code`, BLEU is used. For `mode: secret`, chrF is used. Scores are
stored on a normalized `0.0` to `1.0` scale.
If scoring fails for an individual row, that row is kept in `row_results.jsonl`,
`score_value` is set to `0.0`, and `score_error` records the failure message.
