# KodCode exploratory pipeline

This directory contains an earlier dataset exploration based on `KodCode/KodCode-V1`. It is useful as a record of filtering, validation, embedding, and clustering work, but it is **not** the final synthetic dataset used for the thesis results. See [`../Synthetic/README.md`](../Synthetic/README.md) for the reported study pipeline.

The scripts are numbered in execution order.

## 1. Filter and deduplicate

`01_filter.py`:

- Loads `KodCode/KodCode-V1`.
- Keeps rows with benchmark similarity below `0.75`.
- Requires compilable Python containing exactly one top-level function and only standard-library imports.
- Requires a triple-quoted function docstring and rejects decorated functions or other top-level statements.
- Removes duplicate solutions and adds a UUID `id` column.
- Pushes the result to a private Hugging Face dataset.

The source and destination IDs are module constants. Review `HF_SOURCE_DATASET` and `HF_TARGET_DATASET` before running because the script creates and writes a Hub repository.

```bash
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
python 01_filter.py
```

## 2. Validate tests

`02_validate_tests.py` executes each solution and test pair in a temporary directory, using pytest when the test shape requires it. Failed IDs are written to a text file and may be removed before an optional Hub upload.

```bash
python 02_validate_tests.py \
  --dataset YOUR_NAMESPACE/KodCode-filtered \
  --split train \
  --workers 8 \
  --timeout 10 \
  --failed-ids-output failed_ids.txt
```

To publish the filtered result, additionally pass `--target-dataset NAMESPACE/NAME` and, if needed, `--private`. This stage executes dataset-provided Python; use a sandbox for untrusted code.

## 3. Embed solutions

`03_embed_code.py` embeds the solution column with `Salesforce/SFR-Embedding-Code-400M_R`, normalizes the CLS representation, and writes:

- `embeddings_output/solution_embeddings.npy`
- `embeddings_output/solution_texts.csv`

Dataset, model, field, batch, and output settings are constants at the top of the script. Edit them before running:

```bash
python 03_embed_code.py
```

## 4. Cluster and inspect topics

`04_clustering.py` performs a grid search over UMAP and HDBSCAN. It clusters in solution-embedding space, measures silhouette there, and measures topic coherence over the corresponding natural-language questions. The final BERTopic representation can add POS filtering and maximal marginal relevance.

It expects the files produced by stage 3 and writes its CSV, plots, assignments, and model artifacts below `bertopic_grid_search_outputs/`.

```bash
python 04_clustering.py
```




