import hashlib
import json
from typing import Dict, Tuple

from evalplus.data.custom import get_forget_eval, get_utility_eval
from evalplus.data.humaneval import get_human_eval_plus


COMBINED_EVAL_DATASET = "humaneval-forget-utility"
COMBINED_EVAL_COMPONENTS: Tuple[str, ...] = (
    "humaneval",
    "forgeteval",
    "utilityeval",
)
COMBINED_EVAL_ALIASES = {
    COMBINED_EVAL_DATASET,
    "humaneval-forgeteval-utilityeval",
    "humaneval-forget-utility",
    "humaneval_forget_utility",
    "human-forget-utility",
    "all",
}


def normalize_combined_eval_dataset_name(dataset: str) -> str:
    key = dataset.lower().replace("_", "-")
    if key in COMBINED_EVAL_ALIASES:
        return COMBINED_EVAL_DATASET
    raise ValueError(f"Unknown combined dataset: {dataset}")


def is_combined_eval_dataset(dataset: str) -> bool:
    try:
        normalize_combined_eval_dataset_name(dataset)
    except ValueError:
        return False
    return True


def get_combined_eval_datasets(version: str = "default") -> Dict[str, Dict]:
    return {
        "humaneval": get_human_eval_plus(version=version),
        "forgeteval": get_forget_eval(),
        "utilityeval": get_utility_eval(),
    }


def get_combined_eval(version: str = "default") -> Dict:
    datasets = get_combined_eval_datasets(version=version)
    combined = {}
    for dataset_name, tasks in datasets.items():
        overlap = set(combined).intersection(tasks)
        if overlap:
            raise ValueError(
                f"Task id collision while combining {dataset_name}: {sorted(overlap)}"
            )
        combined.update(tasks)
    return combined


def get_combined_eval_hash(version: str = "default") -> str:
    datasets = get_combined_eval_datasets(version=version)
    payload = json.dumps(datasets, sort_keys=True, default=str).encode()
    return hashlib.md5(payload).hexdigest()
