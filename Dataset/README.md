# Datasets and curation

This directory contains the validation and curation utilities behind the synthetic dataset used in the thesis. The final dataset rows are hosted on Hugging Face and are not duplicated in this repository.

## Final thesis dataset

The dataset contains 600 synthetic Python functions and classes. Randomized identifiers and multiple similarity checks reduce the chance that the exact code already appeared in model pretraining.

| Thesis component | Size | Hugging Face reference used by the code | Main use |
| --- | ---: | --- | --- |
| Full repeated training corpus | 1,800 | `dbaysal/all-contentx3` | Axolotl fine-tuning; each of 600 units occurs three times |
| Full unique corpus | 600 | `dbaysal/all-content` | Syntax, difficulty, lexical, structural, and semantic audits |
| Forget | 150 | `dbaysal/forget` | Secret and code-unit unlearning |
| Equal-size retain subset | 150 | `dbaysal/retain-half` | Default retain objective |
| Full retain set | 300 | `dbaysal/retain-full` | Double-retain ablation |
| Held-out / approximate | 50 | `dbaysal/approximate` | In-domain suffix retention |
| Forget functional tests | 150 tasks before baseline filtering | `dbaysal/ForgetEval` | Functionality surrounding an injected secret |
| General fine-tuned utility tests | 100 | `dbaysal/UtilityEval` | Functionality acquired during fine-tuning |

Availability and access permissions for these resources are controlled on Hugging Face. Set `HF_TOKEN` in the environment when authentication is required.


## Dataset composition

| Component | Functions | Classes | Total |
| --- | ---: | ---: | ---: |
| Retain | 150 | 150 | 300 |
| Forget | 75 | 75 | 150 |
| Utility | 50 | 50 | 100 |
| Held-out / approximate | 25 | 25 | 50 |
| **Total** | **300** | **300** | **600** |

The forget set is also balanced across three complexity levels—simple, moderate, and complex—and contains 50 API keys, 50 passwords, and 50 email addresses. Secrets occur either in documentation or in executable code.

## Directory layout

- [`Synthetic/`](./Synthetic/README.md) contains the final dataset's syntax and difficulty validation, embedding and CodeBLEU audits, functional-test validation, and mutation testing.
- [`KodCode/`](./KodCode/README.md) contains an earlier exploratory pipeline based on KodCode-V1. It is not the final dataset used for the reported experiments.

The synthetic-code generation and test-generation prompts were executed outside this repository and are not included. The checked-in scripts start at validation and auditing; this is an important boundary when assessing end-to-end reproducibility.

## Curation criteria

The final corpus was checked along four axes:

1. **Validity:** every `content` value must compile as Python.
2. **Difficulty:** Radon cyclomatic complexity must agree with the declared simple/moderate/complex label.
3. **Lexical and structural diversity:** every pair must remain below `0.60` for direction-averaged CodeBLEU and tokenizer-token Jaccard similarity.
4. **Semantic diversity:** mean-centered cosine similarity is computed with `Salesforce/SFR-Embedding-Code-400M_R` and `Qodo/Qodo-Embed-1-1.5B` to reduce embedding-space anisotropy.

The checked-in final reports cover all 179,700 unordered pairs. The maximum direction-averaged CodeBLEU is about `0.555`, and the maximum token Jaccard similarity is about `0.534`, both below the `0.60` threshold. See the [synthetic-data guide](./Synthetic/README.md) for exact commands and report locations.


