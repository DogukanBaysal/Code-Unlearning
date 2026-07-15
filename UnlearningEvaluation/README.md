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

Generation settings can be placed under `generation`. Set `pass_k` to sample multiple
continuations per example. The main result uses the highest similarity across those
attempts, representing the worst case for unlearning. A `pass_k` greater than `1`
requires `do_sample: true` and `greedy: false`. If `greedy: true`, sampling is disabled
and `temperature` / `top_p` are ignored.

The aggregate output also reports cumulative intermediate cutoffs from the same samples.
For example, `pass_k: 10` reports pass@1, pass@5, and pass@10, where each value is the
average of the maximum similarity among the first 1, 5, or 10 attempts per example.

Set `aggregate_filter_csv` to a CSV containing `split`, `eval_mode`, and `uuid` columns
to also write `aggregate_results_filtered.json`. The original aggregate remains
unfiltered. The filtered aggregate selects `forget/secret` or `forget/code-unit`
according to the evaluation mode, `retain/code` for retain, and
`held_out_approximate/code` for approximate.

## Run

```bash
python evaluate_suffix_generation.py --config config.yaml
```

The script loads `tokenizer_name` when provided; otherwise it uses `model_name`. If `peft_name` is provided, the base model is loaded first and then the PEFT adapter is applied. Set `peft_subfolder` when the adapter files live in a subfolder of the Hugging Face repository.

## Outputs

The configured `output_dir` receives three files:

- `row_results.jsonl`
- `all_results.jsonl`
- `aggregate_results.json`
- `aggregate_results_filtered.json` when `aggregate_filter_csv` is configured

`row_results.jsonl` contains one record per example: the highest-similarity attempt
among the `pass_k` generations. `all_results.jsonl` contains every attempt separately.
Both files include `pass_index` (zero-based) and `is_worst_case`.

Each JSONL row contains:

- `uuid`
- `prefix`
- `real_suffix`
- `generated_suffix`
- `score_type`
- `score_value`
- `pass_index`
- `is_worst_case`
- `metadata`
- `generation`

`aggregate_results.json` contains:

- `mode`
- `num_evaluated_examples`
- `num_generated_results`
- `average_similarity_score`
- `score_metric`
- `pass_k`
- `pass_k_aggregation`
- `pass_at_k`, including average similarity and grouped averages for each reported cutoff
- `score_failures`
- `grouped_averages`
- `config`

The filtered aggregate has the same fields plus `uuid_filter`, which records the source
CSV, selected split/mode, number of UUIDs in the filter, and number matched in the
evaluated dataset.

For `mode: code`, BLEU is used. For `mode: secret`, chrF is used. Scores are
stored on a normalized `0.0` to `1.0` scale.
If scoring fails for an individual row, that row is kept in `row_results.jsonl`,
`score_value` is set to `0.0`, and `score_error` records the failure message.
