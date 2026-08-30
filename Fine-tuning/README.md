# Fine-tuning the learned model

`ft.yaml` is the Axolotl configuration used to teach a base model the 600-item **MOCHI
(Machine Unlearning of Code with Hidden Information)** corpus before unlearning. The
corpus is exposed as `dbaysal/all-contentx3`, where each unique function or class appears
three times, for 1,800 training rows.

The goal is deliberate, measurable memorization. The thesis trains both selected 3B
models for five epochs and selects the learned state after memorization plateaus for
secrets and code units. This controlled acquisition step lets the later evaluation
attribute a reduction in reconstruction or functionality to unlearning rather than to a
failure to learn the target in the first place.

## Requirements

- Python and an Axolotl installation that provides the `axolotl` CLI.
- Hugging Face access to the base model and `dbaysal/all-contentx3`.
- A CUDA GPU. The original study used an NVIDIA A100 80 GB.



```bash
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
axolotl --help
```



## Run fine-tuning


```bash
axolotl train ft.yaml
```
