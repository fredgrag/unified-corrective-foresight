#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONFLICT_CONFIG="${1:-${PROJECT_ROOT}/configs/pilots/maniskill_conflict_fix_v2.yaml}"

cd -- "${PROJECT_ROOT}"
source scripts/runtime_env.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"

"${UCF_PYTHON_BIN_RESOLVED}" - "${CONFLICT_CONFIG}" <<'PY'
from pathlib import Path
import shutil
import sys

import torch
import wandb

from corrective_foresight.config.conflict_fix import load_conflict_fix_config
from corrective_foresight.runtime import load_production_runtime

config = load_conflict_fix_config(sys.argv[1])
assert config.effective_global_batch == 64
assert not config.output_root.exists() and not config.output_root.is_symlink()
assert torch.__version__.startswith("2.8."), torch.__version__
assert wandb.__version__ == "0.24.2", wandb.__version__
load_production_runtime(config.world_experiment)
load_production_runtime(config.unified_experiment)
probe = config.output_root.parent
probe.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(probe).free
assert free >= 120 * 1024**3, f"requires 120 GiB free, found {free / 1024**3:.1f}"
print(f"pilot={config.pilot_id}")
print(f"output_root={config.output_root}")
print(f"free_gib={free / 1024**3:.1f}")
PY

if [[ -n "$(git status --porcelain)" ]]; then
  echo "conflict-fix launch requires a clean Git worktree" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 1
fi
GPU_COUNT="$(nvidia-smi --id=0,1,2,3 --query-gpu=name --format=csv,noheader | awk '$0 ~ /NVIDIA H20/ { count += 1 } END { print count + 0 }')"
if [[ "${GPU_COUNT}" -ne 4 ]]; then
  echo "four NVIDIA H20 GPUs are required; found ${GPU_COUNT}" >&2
  exit 1
fi
BUSY_GPU_COUNT="$(nvidia-smi --id=0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 2048 { count += 1 } END { print count + 0 }')"
if [[ "${BUSY_GPU_COUNT}" -ne 0 ]]; then
  echo "all four training GPUs must use less than 2 GiB before launch" >&2
  exit 1
fi

scripts/verify_environment.sh
"${UCF_PYTHON_BIN_RESOLVED}" -c 'import wandb; api = wandb.Api(timeout=10); print(api.viewer)'
