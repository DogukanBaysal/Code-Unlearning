#!/usr/bin/env python3

import ast
import os
import sys
import uuid
from typing import Any, Dict, Optional

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi


HF_SOURCE_DATASET = "KodCode/KodCode-V1"
HF_TARGET_DATASET = "dbaysal/KodCode-filtered-2"

BENCHMARK_SIMILARITY_FIELD_NAMES = ("benchmark_similarity", "Benchmark_similarity")
MAX_BENCHMARK_SIMILARITY = 0.75


def deduplicate_by_solution(dataset):
    solution_key = next(
        (key for key in ("solution", "Solution") if key in dataset.column_names),
        None,
    )
    if solution_key is None:
        raise ValueError("No solution column found for deduplication.")

    seen = set()
    keep_indices = []
    for index, solution in enumerate(dataset[solution_key]):
        if solution in seen:
            continue
        seen.add(solution)
        keep_indices.append(index)

    return dataset.select(keep_indices)


def is_stdlib_module(module_name: str) -> bool:
    top_level = module_name.split(".", 1)[0]

    if top_level in sys.builtin_module_names:
        return True

    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if stdlib_names is None:
        return False

    return top_level in stdlib_names


def import_is_stdlib(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return all(is_stdlib_module(alias.name) for alias in node.names)

    if isinstance(node, ast.ImportFrom):
        if node.module is None:
            return False
        return is_stdlib_module(node.module)

    return True


def solution_is_single_function_with_imports(solution: str) -> bool:
    """
    Keep only solutions where:
      - the source parses as Python
      - top-level imports are allowed
      - exactly one top-level function is defined
      - no other top-level code elements are allowed
      - all imports must be from the Python standard library
    """
    if not isinstance(solution, str) or not solution.strip():
        return False

    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return False

    top_level_functions = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level_functions.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if not import_is_stdlib(node):
                return False
        else:
            return False

    if len(top_level_functions) != 1:
        return False

    only_function = top_level_functions[0]

    if only_function.decorator_list:
        return False

    # Require the signature to be followed by a docstring.
    docstring = ast.get_docstring(only_function)
    if not docstring or not docstring.strip():
        return False
    
    # ...and require that docstring to be triple-quoted.
    docstring_node = only_function.body[0]
    docstring_src = ast.get_source_segment(solution, docstring_node)
    if docstring_src is None or not docstring_src.lstrip().startswith(('"""', "'''")):
        return False

    for node in ast.walk(only_function):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not import_is_stdlib(node):
                return False

    return True


def get_solution(row: Dict[str, Any]) -> Optional[str]:
    for key in ("solution", "Solution"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def get_benchmark_similarity(row: Dict[str, Any]) -> Optional[float]:
    for key in BENCHMARK_SIMILARITY_FIELD_NAMES:
        value = row.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return None


def filter_split(example: Dict[str, Any]) -> bool:
    benchmark_similarity = get_benchmark_similarity(example)
    if benchmark_similarity is None:
        return False

    if benchmark_similarity >= MAX_BENCHMARK_SIMILARITY:
        return False

    solution = get_solution(example)
    if solution is None:
        return False

    return solution_is_single_function_with_imports(solution)


def add_uuid(example: Dict[str, Any]) -> Dict[str, Any]:
    example["id"] = str(uuid.uuid4())
    return example


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print(f"Loading {HF_SOURCE_DATASET}...")
    dataset = load_dataset(HF_SOURCE_DATASET)

    if "train" not in dataset:
        raise ValueError(f"No train split found. Available splits: {list(dataset.keys())}")

    train = dataset["train"]

    print(
        f"Filtering split: train "
        f"where benchmark_similarity < {MAX_BENCHMARK_SIMILARITY}"
    )

    filtered_train = train.filter(
        filter_split,
        desc="Filtering train",
    )

    print(f"train: {len(train)} -> {len(filtered_train)}")

    print("Removing exact-duplicate solutions...")
    deduped_train = deduplicate_by_solution(filtered_train)
    print(f"train (deduped): {len(filtered_train)} -> {len(deduped_train)}")

    print("Adding UUID id column...")
    filtered_train = deduped_train.map(   
        add_uuid,
        desc="Adding UUID ids",
    )

    filtered_dataset = DatasetDict({
        "train": filtered_train,
    })

    api = HfApi(token=token)
    api.create_repo(
        repo_id=HF_TARGET_DATASET,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    print(f"Pushing to Hugging Face: {HF_TARGET_DATASET}")
    filtered_dataset.push_to_hub(
        HF_TARGET_DATASET,
        private=True,
        token=token,
    )

    print(f"Done: https://huggingface.co/datasets/{HF_TARGET_DATASET}")


if __name__ == "__main__":
    main()