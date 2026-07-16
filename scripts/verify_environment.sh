#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
VENV_ROOT="${UCF_VENV_ROOT:-/mnt/workspace/wwl/.venvs/unified-corrective-foresight}"

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  echo "Missing environment: ${VENV_ROOT}" >&2
  exit 1
fi

cd -- "${PROJECT_ROOT}"
"${VENV_ROOT}/bin/python" -m pip check
"${VENV_ROOT}/bin/python" -m unittest \
  tests.regression.test_project_boundary \
  tests.integration.test_environment_contract -v

