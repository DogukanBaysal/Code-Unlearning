import json

from datasets import load_dataset

from evalplus.data.custom import (
    custom_dataset_display_name,
    get_custom_eval,
    get_custom_eval_hash,
    get_forget_eval,
    get_forget_eval_hash,
    get_utility_eval,
    get_utility_eval_hash,
    is_custom_dataset,
    normalize_custom_dataset_name,
)
from evalplus.data.combined import (
    COMBINED_EVAL_COMPONENTS,
    COMBINED_EVAL_DATASET,
    get_combined_eval,
    get_combined_eval_datasets,
    get_combined_eval_hash,
    is_combined_eval_dataset,
    normalize_combined_eval_dataset_name,
)
from evalplus.data.humaneval import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.data.mbpp import get_mbpp_plus, get_mbpp_plus_hash
from evalplus.data.utils import load_solutions, write_directory, write_jsonl


def get_evalperf_data():
    dataset = load_dataset("evalplus/evalperf", split="test").to_list()
    for d in dataset:
        d["pe_input"] = json.loads(d["pe_input"])
    return {task["task_id"]: task for task in dataset}
