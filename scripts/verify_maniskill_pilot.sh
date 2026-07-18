#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PILOT_CONFIG="${1:-${PROJECT_ROOT}/configs/pilots/maniskill_stable_v1.yaml}"
cd "${PROJECT_ROOT}"
source scripts/runtime_env.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"

"${UCF_PYTHON_BIN_RESOLVED}" -c '
from corrective_foresight.config.pilot import load_pilot_config
import sys
config = load_pilot_config(sys.argv[1])
assert config.effective_global_batch == 64
assert not config.output_root.exists(), config.output_root
print(f"pilot={config.pilot_id}")
print(f"effective_global_batch={config.effective_global_batch}")
print(f"gpu_indices={config.gpu_indices}")
print(f"output_root={config.output_root}")
' "${PILOT_CONFIG}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the four-H20 pilot gate" >&2
  exit 1
fi
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | awk 'BEGIN { count=0 } /NVIDIA H20/ { count += 1 } END { print count }')"
if [[ "${GPU_COUNT}" -lt 4 ]]; then
  echo "four NVIDIA H20 GPUs are required; found ${GPU_COUNT}" >&2
  exit 1
fi

scripts/verify_environment.sh
