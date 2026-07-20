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

readarray -t VALUES < <("${UCF_PYTHON_BIN_RESOLVED}" - "${CONFLICT_CONFIG}" <<'PY'
from corrective_foresight.config.conflict_fix import load_conflict_fix_config
import sys
config = load_conflict_fix_config(sys.argv[1])
print(config.output_root)
print(config.unified_experiment)
PY
)
OUTPUT_ROOT="${VALUES[0]}"
UNIFIED_CONFIG="${VALUES[1]}"
EVALUATION_ROOT="${OUTPUT_ROOT}/evaluations"
BASELINE_CHECKPOINT="${OUTPUT_ROOT}/unified/checkpoints/unified-000000"
GATE_CHECKPOINT="${OUTPUT_ROOT}/unified/checkpoints/unified-005000"
FINAL_CHECKPOINT="${OUTPUT_ROOT}/unified/checkpoints/unified-020000"
SEEDS=(0 1 2 3 4 5 6 7 8 9)

if [[ -d "${FINAL_CHECKPOINT}" ]]; then
  "${UCF_PYTHON_BIN_RESOLVED}" evaluate.py \
    --config "${UNIFIED_CONFIG}" \
    --checkpoint "${FINAL_CHECKPOINT}" \
    --seeds "${SEEDS[@]}" \
    --tag unified_20000 \
    --output-root "${EVALUATION_ROOT}" \
    --conflict-fix-config "${CONFLICT_CONFIG}" \
    --optimizer-step 20000
  "${UCF_PYTHON_BIN_RESOLVED}" - "${CONFLICT_CONFIG}" <<'PY'
from corrective_foresight.config.conflict_fix import load_conflict_fix_config
from corrective_foresight.config.loader import load_action_spec
from corrective_foresight.evaluation.conflict_fix_report import (
    build_paired_report,
    load_episode_summaries,
    write_effect_gate,
)
import sys
config = load_conflict_fix_config(sys.argv[1])
action = load_action_spec(config.unified_config.action_specs[0])
baseline = load_episode_summaries(
    config.output_root / "evaluations",
    "untrained_action_from_world_5000",
    action_spec=action,
)
candidate = load_episode_summaries(
    config.output_root / "evaluations",
    "unified_20000",
    action_spec=action,
)
write_effect_gate(
    config.output_root / "paired-effect-report-20000.json",
    build_paired_report(
        baseline,
        candidate,
        required_success_gain=config.required_success_gain,
    ),
)
PY
  exit 0
fi

for checkpoint in "${BASELINE_CHECKPOINT}" "${GATE_CHECKPOINT}"; do
  if [[ ! -d "${checkpoint}" ]]; then
    echo "required policy checkpoint is unavailable: ${checkpoint}" >&2
    exit 1
  fi
done

"${UCF_PYTHON_BIN_RESOLVED}" evaluate.py \
  --config "${UNIFIED_CONFIG}" \
  --checkpoint "${BASELINE_CHECKPOINT}" \
  --seeds "${SEEDS[@]}" \
  --tag untrained_action_from_world_5000 \
  --output-root "${EVALUATION_ROOT}" \
  --conflict-fix-config "${CONFLICT_CONFIG}" \
  --optimizer-step 0

"${UCF_PYTHON_BIN_RESOLVED}" evaluate.py \
  --config "${UNIFIED_CONFIG}" \
  --checkpoint "${GATE_CHECKPOINT}" \
  --seeds "${SEEDS[@]}" \
  --tag unified_5000 \
  --output-root "${EVALUATION_ROOT}" \
  --conflict-fix-config "${CONFLICT_CONFIG}" \
  --optimizer-step 5000

"${UCF_PYTHON_BIN_RESOLVED}" scripts/build_conflict_fix_gate.py \
  --conflict-fix-config "${CONFLICT_CONFIG}"
