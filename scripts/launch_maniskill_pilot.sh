#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PILOT_CONFIG="${1:-${PROJECT_ROOT}/configs/pilots/maniskill_stable_v1.yaml}"

cd "${PROJECT_ROOT}"
source scripts/runtime_env.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export UCF_NPROC_PER_NODE=4

PILOT_OUTPUT_ROOT="$("${UCF_PYTHON_BIN_RESOLVED}" -c \
  'from pathlib import Path; from corrective_foresight.config.pilot import load_pilot_config; import sys; print(load_pilot_config(sys.argv[1]).output_root)' \
  "${PILOT_CONFIG}")"
LOCK_PATH="${PILOT_OUTPUT_ROOT}.launch.lock"
if ( set -o noclobber; printf '%s\n' "$$" > "${LOCK_PATH}" ) 2>/dev/null; then
  trap 'rm -f -- "${LOCK_PATH}"' EXIT
else
  echo "pilot launch lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi

TORCHRUN=("${UCF_PYTHON_BIN_RESOLVED}" -m torch.distributed.run
  --standalone --nproc_per_node="${UCF_NPROC_PER_NODE}")

"${TORCHRUN[@]}" train.py \
  --pilot-config "${PILOT_CONFIG}" \
  --pilot-stage world_pretrain

WORLD_CHECKPOINT="${PILOT_OUTPUT_ROOT}/world_pretrain/checkpoints/world_pretrain-001000"
if [[ ! -d "${WORLD_CHECKPOINT}" ]]; then
  echo "world pretrain checkpoint was not published: ${WORLD_CHECKPOINT}" >&2
  exit 1
fi

"${TORCHRUN[@]}" train.py \
  --pilot-config "${PILOT_CONFIG}" \
  --pilot-stage unified \
  --init-checkpoint "${WORLD_CHECKPOINT}"
