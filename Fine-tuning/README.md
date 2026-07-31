# Fine-tuning the learned model

`ft.yaml` is the Axolotl configuration used to teach a base model the 600-item synthetic corpus before unlearning. The corpus is exposed as `dbaysal/all-contentx3`, where each unique function or class appears three times, for 1,800 training rows.

The goal is deliberate memorization: the learned checkpoint should reproduce the injected secrets and code units so that a later drop can be attributed to unlearning rather than failure to learn the target.

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
