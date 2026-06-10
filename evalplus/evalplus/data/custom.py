import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

from evalplus.data.utils import completeness_check, stream_jsonl


CUSTOM_DATASETS = {
    "forgeteval": {
        "display": "ForgetEval",
        "env_prefix": "FORGETEVAL",
        "data_file": (
            "https://huggingface.co/datasets/dbaysal/ForgetEval/resolve/main/data/"
            "train-00000-of-00001.parquet"
        ),
    },
    "utilityeval": {
        "display": "UtilityEval",
        "env_prefix": "UTILITYEVAL",
        "data_file": (
            "https://huggingface.co/datasets/dbaysal/UtilityEval/resolve/main/data/"
            "train-00000-of-00001.parquet"
        ),
    },
}

CUSTOM_DATASET_ALIASES = {
    "forgeteval": "forgeteval",
    "forget-eval": "forgeteval",
    "forget_eval": "forgeteval",
    "utilityeval": "utilityeval",
    "utility-eval": "utilityeval",
    "utility_eval": "utilityeval",
}

TASK_ID_KEYS = ("task_id", "id", "name")
PROMPT_KEYS = ("prompt", "instruction", "declaration")
ENTRY_POINT_KEYS = ("entry_point", "function_name", "fn_name", "function")
TEST_KEYS = ("test", "tests")
BASE_INPUT_KEYS = (
    "base_input",
    "base_inputs",
    "inputs",
    "input",
    "test_input",
    "test_inputs",
)
CANONICAL_SOLUTION_KEYS = (
    "canonical_solution",
    "canonical",
    "reference_solution",
    "reference",
    "solution",
)


def normalize_custom_dataset_name(dataset: str) -> str:
    key = dataset.lower()
    if key in CUSTOM_DATASET_ALIASES:
        return CUSTOM_DATASET_ALIASES[key]
    key = key.replace("_", "-")
    if key in CUSTOM_DATASET_ALIASES:
        return CUSTOM_DATASET_ALIASES[key]
    raise ValueError(f"Unknown custom dataset: {dataset}")


def is_custom_dataset(dataset: str) -> bool:
    try:
        normalize_custom_dataset_name(dataset)
    except ValueError:
        return False
    return True


def custom_dataset_display_name(dataset: str) -> str:
    return CUSTOM_DATASETS[normalize_custom_dataset_name(dataset)]["display"]


def _custom_env(dataset: str, key: str) -> Optional[str]:
    prefix = CUSTOM_DATASETS[dataset]["env_prefix"]
    return os.environ.get(f"{prefix}_{key}")


def _first_present(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _json_if_needed(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _normalize_inputs(
    value: Any, field_name: str, task_id: str, required: bool = True
) -> List[Any]:
    value = _json_if_needed(value)
    if value is None:
        if not required:
            return []
        raise KeyError(f"{field_name} not found in custom dataset #{task_id}!")
    if not isinstance(value, list):
        raise TypeError(f"{field_name} for {task_id} must be a list of argument lists")
    return value


def _records_from_json_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        raise TypeError(
            "Custom dataset JSON must be a list, dict of rows, or dict of columns"
        )

    for key in ("data", "rows", "test", "train", "validation"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]

    if payload and all(isinstance(v, dict) for v in payload.values()):
        return list(payload.values())

    if payload and all(isinstance(v, list) for v in payload.values()):
        lengths = {len(v) for v in payload.values()}
        if len(lengths) == 1:
            return [
                {key: values[i] for key, values in payload.items()}
                for i in range(next(iter(lengths)))
            ]

    raise TypeError("Could not infer records from custom dataset JSON")


def _load_local_records(path: str) -> List[Dict[str, Any]]:
    path = os.path.expanduser(path)
    if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
        return list(stream_jsonl(path))
    if path.endswith(".parquet"):
        return load_dataset("parquet", data_files=path, split="train").to_list()

    with open(path, "r") as f:
        return _records_from_json_payload(json.load(f))


def _load_records(dataset: str) -> List[Dict[str, Any]]:
    override_path = _custom_env(dataset, "OVERRIDE_PATH")
    if override_path:
        return _load_local_records(override_path)

    data_file = CUSTOM_DATASETS[dataset]["data_file"]
    return load_dataset("parquet", data_files=data_file, split="train").to_list()


def _normalize_record(dataset: str, record: Dict[str, Any], index: int) -> Dict[str, Any]:
    display = CUSTOM_DATASETS[dataset]["display"]
    task_id = _first_present(record, TASK_ID_KEYS)
    if task_id is None:
        task_id = f"{display}/{index}"
    task_id = str(task_id)

    prompt = _first_present(record, PROMPT_KEYS)
    entry_point = _first_present(record, ENTRY_POINT_KEYS)
    canonical_solution = _first_present(record, CANONICAL_SOLUTION_KEYS)
    test = _first_present(record, TEST_KEYS)
    base_input = _first_present(record, BASE_INPUT_KEYS)

    if test is None and base_input is None:
        raise KeyError(f"test or base_input not found in {display} #{task_id}!")

    normalized = {
        **record,
        "task_id": task_id,
        "prompt": prompt,
        "contract": record.get("contract") or "",
        "canonical_solution": canonical_solution,
        "base_input": _normalize_inputs(
            base_input, "base_input", task_id, required=False
        ),
        "plus_input": [],
        "entry_point": entry_point,
        "test": test,
        "atol": record.get("atol") or 0,
    }

    completeness_check(display, {task_id: normalized})
    if not normalized["entry_point"]:
        raise KeyError(f"entry_point not found in {display} #{task_id}!")
    if normalized["prompt"] is None:
        raise KeyError(f"prompt not found in {display} #{task_id}!")
    if normalized["canonical_solution"] is None:
        raise KeyError(f"canonical_solution not found in {display} #{task_id}!")
    if not normalized["test"] and not normalized["base_input"]:
        raise KeyError(f"test or base_input not found in {display} #{task_id}!")

    return normalized


def get_custom_eval(
    dataset: str, err_incomplete: bool = True, **kwargs
) -> Dict[str, Dict]:
    dataset = normalize_custom_dataset_name(dataset)
    tasks = {
        task["task_id"]: task
        for task in (
            _normalize_record(dataset, record, i)
            for i, record in enumerate(_load_records(dataset))
        )
    }
    if err_incomplete:
        completeness_check(CUSTOM_DATASETS[dataset]["display"], tasks)
    return tasks


def get_custom_eval_hash(dataset: str, **kwargs) -> str:
    tasks = get_custom_eval(dataset, **kwargs)
    payload = json.dumps(list(tasks.values()), sort_keys=True, default=str).encode()
    return hashlib.md5(payload).hexdigest()


def get_forget_eval(**kwargs) -> Dict[str, Dict]:
    return get_custom_eval("forgeteval", **kwargs)


def get_forget_eval_hash(**kwargs) -> str:
    return get_custom_eval_hash("forgeteval", **kwargs)


def get_utility_eval(**kwargs) -> Dict[str, Dict]:
    return get_custom_eval("utilityeval", **kwargs)


def get_utility_eval_hash(**kwargs) -> str:
    return get_custom_eval_hash("utilityeval", **kwargs)
