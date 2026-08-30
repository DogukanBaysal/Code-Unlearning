# MOCHI validation and test hardening

This directory contains the audit and test-hardening stages for **MOCHI (Machine
Unlearning of Code with Hidden Information)**, the final 600-item benchmark introduced
in the thesis. The checks make the secret and whole-code-unit unlearning comparisons
interpretable by validating Python syntax and complexity, limiting cross-split
similarity, and strengthening the ForgetEval and UtilityEval test suites with mutation
testing. It does **not** contain the original code-generation pipeline; the scripts load
the resulting datasets from Hugging Face.

Run the commands below from `Dataset/Synthetic/` unless otherwise noted.

## Tools

| Script | Purpose | Main dependencies |
| --- | --- | --- |
| `verify_code_syntax.py` | Compile every code unit and write a JSONL validity report | `datasets`, `tqdm` |
| `verify_difficulty.py` | Recompute simple/moderate/complex labels using cyclomatic complexity | `datasets`, `radon` |
| `verify_embedding_similarity.py` | Compute raw and mean-centered cosine similarity with two code embedding models | `sentence-transformers`, `torch`, `numpy`, `matplotlib`, `datasets` |
| `verify_token_codebleu_similarity.py` | Compute pairwise direction-averaged CodeBLEU and tokenizer-token Jaccard similarity | `codebleu`, `transformers`, `datasets`, `numpy`, `matplotlib` |
| `validate_tests.py` | Execute HumanEval-style canonical solutions and `check(candidate)` tests | `datasets` for Hub input |
| `run_mutation_testing.py` | Run `mutmut`, preserve per-sample workspaces, and produce JSON/Markdown mutation reports | `mutmut`, `pytest`, `datasets` |
| `SyntaxFilter/syntax_filter.py` | Produce the experimental `code_filtered` forget column | `datasets` |


## 1. Syntax validation

```bash
python verify_code_syntax.py \
  --dataset dbaysal/all-content \
  --split train \
  --content-column content \
  --output reports/compile_report.jsonl
```

Each output row records the source row index, validity, exception type/message, and syntax location. A clean corpus should report 600 valid files and zero invalid files.

## 2. Difficulty validation

```bash
python verify_difficulty.py dbaysal/all-content \
  --split train \
  --code-column content \
  --difficulty-column difficulty \
  --type-column type \
  --task-id-column task_id \
  --output reports/difficulty_report.json
```

Difficulty is derived from Radon cyclomatic complexity:

- Functions use the main top-level function's complexity.
- Classes use the mean complexity of methods other than `__init__`.
- `1–10` is simple, `>10–20` is moderate, and `>20–50` is complex.

The script prints every mismatch and can save the complete per-item analysis.

## 3. Semantic-similarity audit

```bash
python verify_embedding_similarity.py \
  --dataset dbaysal/all-content \
  --column content \
  --review-threshold 0.60 \
  --hard-fail-threshold 0.75 \
  --out-dir reports/embedding_audit_out
```

The default models are:

- `Salesforce/SFR-Embedding-Code-400M_R`
- `Qodo/Qodo-Embed-1-1.5B`

The final score mean-centers embeddings over the comparison pool and normalizes them again before taking cosine similarity. Raw cosine is retained as a diagnostic because code-embedding anisotropy places many unrelated snippets in a narrow positive-cosine band.


## 4. Lexical and structural audit

```bash
python verify_token_codebleu_similarity.py \
  --dataset dbaysal/all-content \
  --split train \
  --content-column content \
  --threshold 0.60 \
  --token-threshold 0.60 \
  --report reports/token_codebleu_reports/codebleu_all_content_report.json \
  --pairs-report reports/token_codebleu_reports/codebleu_pairs_above_threshold.jsonl \
  --plot reports/token_codebleu_reports/codebleu_diversity.pdf
```

For every unordered pair `(i, j)`, the script computes CodeBLEU in both directions and averages the results. It separately computes Jaccard similarity over unique tokenizer token IDs. The final report includes maxima, component scores, threshold counts, row metadata, histograms, and elapsed time.

The checked-in final report contains 179,700 unordered comparisons and no pair at or above `0.60`. Its maxima are approximately:

- Direction-averaged CodeBLEU: `0.55497`
- Token Jaccard similarity: `0.53425`

The default tokenizer is `openai/gpt-oss-120b`; pass `--tokenizer` to reproduce a different tokenization choice.

## 5. Functional-test validation

Validate a Hub dataset:

```bash
python validate_tests.py \
  --hf-dataset dbaysal/UtilityEval \
  --split train \
  --out reports/utilityeval_canonical_results.jsonl \
  --show-errors
```

Or validate a local JSONL file:

```bash
python validate_tests.py \
  --dataset /path/to/tasks.jsonl \
  --out reports/canonical_results.jsonl
```

Supported rows may provide full `code`, or a prompt/declaration plus a canonical solution. Tests must define `check(candidate)`. This verifier executes dataset code directly; use a sandbox for untrusted inputs.

## 6. Mutation hardening

```bash
python run_mutation_testing.py \
  --dataset dbaysal/UtilityEval \
  --split train \
  --code-col code \
  --test-col test \
  --entry-point-col entry_point \
  --output Mutation/UtilityEval \
  --run-id baseline
```

