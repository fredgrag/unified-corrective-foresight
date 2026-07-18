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

readarray -t PILOT_VALUES < <("${UCF_PYTHON_BIN_RESOLVED}" -c '
from corrective_foresight.config.pilot import load_pilot_config
import sys
config = load_pilot_config(sys.argv[1])
print(config.output_root)
print(config.world_experiment)
print(config.unified_experiment)
' "${PILOT_CONFIG}")
PILOT_OUTPUT_ROOT="${PILOT_VALUES[0]}"
WORLD_CONFIG="${PILOT_VALUES[1]}"
UNIFIED_CONFIG="${PILOT_VALUES[2]}"
EVAL_ROOT="${PILOT_OUTPUT_ROOT}/evaluations"
SEEDS=(0 1 2 3 4 5 6 7 8 9)

"${UCF_PYTHON_BIN_RESOLVED}" evaluate.py \
  --config "${WORLD_CONFIG}" \
  --checkpoint "${PILOT_OUTPUT_ROOT}/world_pretrain/checkpoints/world_pretrain-000000" \
  --seeds "${SEEDS[@]}" \
  --tag untrained \
  --output-root "${EVAL_ROOT}"

"${UCF_PYTHON_BIN_RESOLVED}" evaluate.py \
  --config "${WORLD_CONFIG}" \
  --checkpoint "${PILOT_OUTPUT_ROOT}/world_pretrain/checkpoints/world_pretrain-001000" \
  --seeds "${SEEDS[@]}" \
  --tag world_pretrain_1000 \
  --output-root "${EVAL_ROOT}"

"${UCF_PYTHON_BIN_RESOLVED}" evaluate.py \
  --config "${UNIFIED_CONFIG}" \
  --checkpoint "${PILOT_OUTPUT_ROOT}/unified/checkpoints/unified-002000" \
  --seeds "${SEEDS[@]}" \
  --tag unified_2000 \
  --output-root "${EVAL_ROOT}"
