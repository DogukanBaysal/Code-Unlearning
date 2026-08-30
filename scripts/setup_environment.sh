#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="python3.11"
VENV_PATH="${REPO_ROOT}/.venv"
WITH_FLASH_ATTN=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_environment.sh [options]

Create one virtual environment for the complete thesis workflow and install all
three subrepositories in it.

Options:
  --python PATH          Python interpreter to use (default: python3.11)
  --venv PATH            Virtual-environment path (default: .venv)
  --with-flash-attn      Also install flash-attn 2.6.3 (Linux/CUDA only)
  -h, --help             Show this help
EOF
}

while (($#)); do
    case "$1" in
        --python)
            [[ $# -ge 2 ]] || { echo "error: --python requires a value" >&2; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --venv)
            [[ $# -ge 2 ]] || { echo "error: --venv requires a value" >&2; exit 2; }
            VENV_PATH="$2"
            shift 2
            ;;
        --with-flash-attn)
            WITH_FLASH_ATTN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

required_files=(
    "open-unlearning/setup.py"
    "evalplus/pyproject.toml"
    "UnlearningEvaluation/requirements.txt"
)

for required_file in "${required_files[@]}"; do
    if [[ ! -f "${REPO_ROOT}/${required_file}" ]]; then
        echo "error: ${required_file} is missing" >&2
        echo "Run 'git submodule update --init --recursive' from ${REPO_ROOT}." >&2
        exit 1
    fi
done

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}/open-unlearning[lm-eval]"
python -m pip install -r "${REPO_ROOT}/UnlearningEvaluation/requirements.txt"
python -m pip install -e "${REPO_ROOT}/evalplus[peft]"

if [[ "${WITH_FLASH_ATTN}" -eq 1 ]]; then
    python -m pip install --no-build-isolation flash-attn==2.6.3
fi

echo
echo "Environment ready. Activate it with:"
echo "  source \"${VENV_PATH}/bin/activate\""
