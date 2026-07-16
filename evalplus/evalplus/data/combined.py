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
FORGET_UTILITY_EVAL_DATASET = "forget-utility"
FORGET_UTILITY_EVAL_COMPONENTS: Tuple[str, ...] = (
    "forgeteval",
    "utilityeval",
)
COMBINED_EVAL_DATASETS: Tuple[str, ...] = (
    COMBINED_EVAL_DATASET,
    FORGET_UTILITY_EVAL_DATASET,
)
COMBINED_EVAL_ALIASES = {
    COMBINED_EVAL_DATASET: {
        COMBINED_EVAL_DATASET,
        "humaneval-forgeteval-utilityeval",
        "humaneval-forget-utility",
        "humaneval_forget_utility",
        "human-forget-utility",
        "all",
    },
    FORGET_UTILITY_EVAL_DATASET: {
        FORGET_UTILITY_EVAL_DATASET,
        "forgeteval-utilityeval",
        "forget-utilityeval",
        "forgeteval-utility",
        "forget_utility",
    },
}


def normalize_combined_eval_dataset_name(dataset: str) -> str:
    key = dataset.lower().replace("_", "-")
    for canonical_name, aliases in COMBINED_EVAL_ALIASES.items():
        if key in {alias.replace("_", "-") for alias in aliases}:
            return canonical_name
    raise ValueError(f"Unknown combined dataset: {dataset}")


def is_combined_eval_dataset(dataset: str) -> bool:
    try:
        normalize_combined_eval_dataset_name(dataset)
    except ValueError:
        return False
    return True


def get_combined_eval_components(
    dataset: str = COMBINED_EVAL_DATASET,
) -> Tuple[str, ...]:
    dataset = normalize_combined_eval_dataset_name(dataset)
    if dataset == COMBINED_EVAL_DATASET:
        return COMBINED_EVAL_COMPONENTS
    if dataset == FORGET_UTILITY_EVAL_DATASET:
        return FORGET_UTILITY_EVAL_COMPONENTS
    raise ValueError(f"Unknown combined dataset: {dataset}")


def get_combined_eval_datasets(
    version: str = "default",
    dataset: str = COMBINED_EVAL_DATASET,
) -> Dict[str, Dict]:
    dataset_loaders = {
        "humaneval": lambda: get_human_eval_plus(version=version),
        "forgeteval": get_forget_eval,
        "utilityeval": get_utility_eval,
    }
    return {
        component: dataset_loaders[component]()
        for component in get_combined_eval_components(dataset)
    }


def get_combined_eval(
    version: str = "default",
    dataset: str = COMBINED_EVAL_DATASET,
) -> Dict:
    datasets = get_combined_eval_datasets(version=version, dataset=dataset)
    combined = {}
    for dataset_name, tasks in datasets.items():
        overlap = set(combined).intersection(tasks)
        if overlap:
            raise ValueError(
                f"Task id collision while combining {dataset_name}: {sorted(overlap)}"
            )
        combined.update(tasks)
    return combined


def get_combined_eval_hash(
    version: str = "default",
    dataset: str = COMBINED_EVAL_DATASET,
) -> str:
    datasets = get_combined_eval_datasets(version=version, dataset=dataset)
    payload = json.dumps(datasets, sort_keys=True, default=str).encode()
    return hashlib.md5(payload).hexdigest()
