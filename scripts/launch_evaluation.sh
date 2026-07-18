#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${PROJECT_ROOT}/scripts/runtime_env.sh"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd -- "${PROJECT_ROOT}"
"${UCF_PYTHON_BIN_RESOLVED}" evaluate.py \
  --config configs/experiments/maniskill_unified.yaml "$@"
