import re
import keyword
from datasets import load_dataset

# ── All Python keywords (keyword.kwlist covers the full set for your Python version)
PYTHON_KEYWORDS = set(keyword.kwlist)  # e.g. def, class, for, while, if, else, ...

# ── Dunder pattern: __anything__
DUNDER_RE = re.compile(r'\b__\w+__\b')

# ── Decorator pattern: @identifier (optionally followed by parens — handled by removal)
DECORATOR_RE = re.compile(r'@\w+(\.\w+)*(\(.*?\))?', re.DOTALL)

# ── Keyword pattern: match whole-word keywords only (avoid partial matches)
KEYWORD_RE = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in PYTHON_KEYWORDS) + r')\b'
)

# ── Collapse extra whitespace left behind after removals
MULTI_SPACE_RE = re.compile(r'[ \t]{2,}')
MULTI_NEWLINE_RE = re.compile(r'\n{3,}')


def clean_code(code: str) -> str:
    # 1. Remove decorators first (they can span lines)
    code = DECORATOR_RE.sub('', code)
    # 2. Remove dunder names
    code = DUNDER_RE.sub('', code)
    # 3. Remove Python keywords
    code = KEYWORD_RE.sub('', code)
    # 4. Tidy up leftover whitespace
    code = MULTI_SPACE_RE.sub(' ', code)
    code = MULTI_NEWLINE_RE.sub('\n\n', code)
    return code.strip()


# ── Load your dataset — replace with your actual dataset name/path and split
# Examples:
#   dataset = load_dataset("your-username/your-dataset", split="train")
#   dataset = load_dataset("codeparrot/github-code", split="train")
dataset = load_dataset("dbaysal/forget", split="train")

# ── Add filtered version as a new column
filtered_dataset = dataset.map(
    lambda example: {"code_filtered": clean_code(example["code"])},
    desc="Building code_filtered column",
)

# ── Save result (choose one)
# As a new HuggingFace dataset on disk:
#filtered_dataset.save_to_disk("filtered_dataset")

# As a JSON file:
# filtered_dataset.to_json("filtered_dataset.jsonl")

# Push to Hub (requires huggingface-cli login):
filtered_dataset.push_to_hub("dbaysal/forget")

print(f"Done. {len(filtered_dataset)} rows processed.")
print("\nSample before/after:")
sample_idx = 0
print("BEFORE:", filtered_dataset[sample_idx]["code"][:300])
print("AFTER :", filtered_dataset[sample_idx]["code_filtered"][:300])