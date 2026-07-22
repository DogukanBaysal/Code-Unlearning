"""Post-processing LLM-generated Python code implemented using tree-sitter."""

from concurrent.futures import ProcessPoolExecutor
import os
import pathlib
import re
from typing import Dict, Generator, List, Optional, Set, Tuple

import tree_sitter_python
from tqdm import tqdm
from tree_sitter import Language, Node, Parser

from evalplus.data import (
    get_human_eval_plus,
    get_mbpp_plus,
    load_solutions,
    write_directory,
    write_jsonl,
)
from evalplus.syncheck import syntax_check

CLASS_TYPE = "class_definition"
FUNCTION_TYPE = "function_definition"
IMPORT_TYPE = ["import_statement", "import_from_statement"]
IDENTIFIER_TYPE = "identifier"
ATTRIBUTE_TYPE = "attribute"
RETURN_TYPE = "return_statement"
EXPRESSION_TYPE = "expression_statement"
ASSIGNMENT_TYPE = "assignment"


def code_extract(text: str) -> str:
    # Remove ANSI escape sequences, which was caused by ollama (small qwen3 models, but may be others as well)
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)

    lines = text.split("\n")

    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:
    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                deps.add(child.text.decode("utf8"))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, str]) -> Set[str]:
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if not (neighbour in visited):
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def get_definition_name(node: Node) -> str:
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            return child.text.decode("utf8")


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def has_return_statement(node: Node) -> bool:
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == RETURN_TYPE:
            return True
    return False


def extract_target_code_or_empty(code: str, entrypoint: Optional[str] = None) -> str:
    code = code_extract(code)
    code_bytes = bytes(code, "utf8")
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)
    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    for child in root_node.children:
        if child.type in IMPORT_TYPE:
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = get_definition_name(child)
            if not (
                name in class_names or name in variable_names or name in function_names
            ):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = get_definition_name(child)
            if not (
                name in function_names or name in variable_names or name in class_names
            ) and has_return_statement(child):
                definition_nodes.append((name, child))
                function_names.add(get_definition_name(child))
        elif (
            child.type == EXPRESSION_TYPE and child.children[0].type == ASSIGNMENT_TYPE
        ):
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (
                name in variable_names or name in function_names or name in class_names
            ):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reacheable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = b""

    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"

    for pair in definition_nodes:
        name, node = pair
        if entrypoint and not (name in reacheable):
            continue
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"
    return sanitized_output[:-1].decode("utf8")


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    sanitized_code = extract_target_code_or_empty(code, entrypoint).strip()
    if not sanitized_code:
        return code_extract(code)
    return sanitized_code


def _sanitize_one(payload: Tuple[str, Optional[str]]) -> str:
    code, entrypoint = payload
    return sanitize(code=code, entrypoint=entrypoint)


def sanitize_samples(
    samples: str,
    target_path: Optional[str] = None,
    inplace: bool = False,
    debug_task: str = None,
    mbpp_version="default",
    problems: Optional[Dict[str, Dict]] = None,
) -> str:
    # task_id -> entry_point
    entry_point = {}
    dataset = problems or {**get_human_eval_plus(), **get_mbpp_plus(version=mbpp_version)}

    for task_id, problem in dataset.items():
        entry_point[task_id] = problem["entry_point"]

    is_folder = os.path.isdir(samples)
    if target_path is None:
        target = pathlib.Path(samples)
        if not inplace:
            if is_folder:
                new_name = target.name + "-sanitized"
            else:
                new_name = target.name.replace(".jsonl", "-sanitized.jsonl")
            target = target.parent / new_name
        target_path = str(target)

    records = []
    for solution in tqdm(load_solutions(samples)):
        task_id = solution["task_id"]
        if task_id not in dataset:
            print(
                f"Skiping {task_id} as it does not existing in the latest EvalPlus dataset."
            )
            continue

        function_name = entry_point[task_id] if task_id in entry_point else None
        if debug_task is not None and task_id != debug_task:
            continue

        if "solution" in solution:
            old_code = solution["solution"]
        else:
            assert "completion" in solution
            old_code = dataset[task_id]["prompt"] + "\n" + solution["completion"]

        records.append((solution, old_code, function_name))

    sanitize_workers = int(os.getenv("EVALPLUS_SANITIZE_WORKERS", "1"))
    if sanitize_workers <= 0:
        raise ValueError("EVALPLUS_SANITIZE_WORKERS must be greater than zero")

    payloads = [(old_code, function_name) for _, old_code, function_name in records]
    if sanitize_workers == 1:
        sanitized_codes = map(_sanitize_one, payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=sanitize_workers)
        sanitized_codes = executor.map(_sanitize_one, payloads, chunksize=1)

    nsan = 0
    new_solutions = []
    try:
        record_outputs = zip(records, sanitized_codes)
        for (solution, old_code, _), new_code in tqdm(
            record_outputs,
            total=len(records),
            desc=f"Sanitizing ({sanitize_workers} worker(s))",
        ):
            task_id = solution["task_id"]

            # if changed, print the message
            if new_code != old_code:
                dbg_identifier = solution["_identifier"]
                msg = "Sanitized: " + dbg_identifier
                if is_folder:
                    msg += " -> " + dbg_identifier.replace(samples, target_path)
                print(msg)
                nsan += 1

            new_solutions.append({"task_id": task_id, "solution": new_code})
    finally:
        if sanitize_workers > 1:
            executor.shutdown()

    if is_folder:
        write_directory(target_path, new_solutions)
    else:
        write_jsonl(target_path, new_solutions)

    if nsan > 0:
        print(f"Sanitized {nsan} out of {len(records)} files.")
    else:
        print(f"All files seems valid -- no files are sanitized.")
    print(f"Check the sanitized files at {target_path}")
    return target_path


def script(
    samples: str, inplace: bool = False, debug_task: str = None, mbpp_version="default"
):
    return sanitize_samples(
        samples=samples,
        inplace=inplace,
        debug_task=debug_task,
        mbpp_version=mbpp_version,
    )


def main():
    from fire import Fire

    Fire(script)


if __name__ == "__main__":
    main()
