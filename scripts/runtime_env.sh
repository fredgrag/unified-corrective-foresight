#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "runtime_env.sh must be sourced" >&2
  exit 2
fi

UCF_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
UCF_VENV_ROOT="${UCF_VENV_ROOT:-/mnt/workspace/wwl/.venvs/unified-corrective-foresight}"
if [[ -n "${UCF_PYTHON_BIN:-}" ]]; then
  UCF_PYTHON_BIN_RESOLVED="${UCF_PYTHON_BIN}"
elif [[ -x "${UCF_VENV_ROOT}/bin/python" ]] && "${UCF_VENV_ROOT}/bin/python" -V >/dev/null 2>&1; then
  UCF_PYTHON_BIN_RESOLVED="${UCF_VENV_ROOT}/bin/python"
elif [[ -x /opt/ucf/python-3.12.13/bin/python3.12 ]]; then
  UCF_PYTHON_BIN_RESOLVED="/opt/ucf/python-3.12.13/bin/python3.12"
else
  echo "Missing Python runtime; set UCF_PYTHON_BIN or UCF_VENV_ROOT" >&2
  return 1
fi
if [[ ! -x "${UCF_PYTHON_BIN_RESOLVED}" ]]; then
  echo "Python runtime is not executable: ${UCF_PYTHON_BIN_RESOLVED}" >&2
  return 1
fi

if [[ -d /opt/ucf/python-3.12.13/lib ]]; then
  export LD_LIBRARY_PATH="/opt/ucf/python-3.12.13/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -d "${UCF_VENV_ROOT}/lib/python3.12/site-packages" ]]; then
  export PYTHONPATH="${UCF_VENV_ROOT}/lib/python3.12/site-packages:${UCF_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${UCF_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONNOUSERSITE=1
export UCF_PROJECT_ROOT UCF_VENV_ROOT UCF_PYTHON_BIN_RESOLVED
