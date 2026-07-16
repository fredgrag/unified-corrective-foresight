#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
VENV_ROOT="${UCF_VENV_ROOT:-/mnt/workspace/wwl/.venvs/unified-corrective-foresight}"
PYTHON_BIN="${UCF_PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
mkdir -p -- "$(dirname -- "${VENV_ROOT}")"

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv --copies "${VENV_ROOT}"
fi

"${VENV_ROOT}/bin/python" -m pip install \
  pip==26.0.1 setuptools==80.10.2 wheel==0.47.0
"${VENV_ROOT}/bin/python" -m pip install \
  --requirement "${PROJECT_ROOT}/requirements/torch-cu128.txt"
"${VENV_ROOT}/bin/python" -m pip install --editable "${PROJECT_ROOT}[sim]"
"${VENV_ROOT}/bin/python" -m pip check
"${VENV_ROOT}/bin/python" -m pip freeze --all | LC_ALL=C sort \
  > "${PROJECT_ROOT}/requirements/lock-cu128.txt"

echo "Environment ready: ${VENV_ROOT}"

