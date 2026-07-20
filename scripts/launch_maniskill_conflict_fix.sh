#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONFLICT_CONFIG="${1:-${PROJECT_ROOT}/configs/pilots/maniskill_conflict_fix_v2.yaml}"

cd -- "${PROJECT_ROOT}"
source scripts/runtime_env.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"

OUTPUT_ROOT="$("${UCF_PYTHON_BIN_RESOLVED}" -c \
  'from corrective_foresight.config.conflict_fix import load_conflict_fix_config; import sys; print(load_conflict_fix_config(sys.argv[1]).output_root)' \
  "${CONFLICT_CONFIG}")"
mkdir -p -- "$(dirname -- "${OUTPUT_ROOT}")"
LOCK_PATH="${OUTPUT_ROOT}.launch.lock"
if ( set -o noclobber; printf '%s\n' "$$" > "${LOCK_PATH}" ) 2>/dev/null; then
  trap 'rm -f -- "${LOCK_PATH}"' EXIT
else
  echo "conflict-fix launch lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi

scripts/verify_maniskill_conflict_fix.sh "${CONFLICT_CONFIG}"

TORCHRUN=("${UCF_PYTHON_BIN_RESOLVED}" -m torch.distributed.run
  --standalone --nproc_per_node=4)
WORLD_CHECKPOINT="${OUTPUT_ROOT}/world_pretrain/checkpoints/world_pretrain-005000"
WORLD_VALIDATION="${OUTPUT_ROOT}/world_pretrain/validation/world_pretrain-005000.json"
WORLD_GATE="${OUTPUT_ROOT}/world-gate-report.json"
AUDIT_METRICS="${OUTPUT_ROOT}/gradient_audit/metrics/rank-0.jsonl"
AUDIT_VALIDATION="${OUTPUT_ROOT}/gradient_audit/validation/unified-000500.json"
AUDIT_DECISION="${OUTPUT_ROOT}/gradient-audit-decision.json"
UNIFIED_CHECKPOINT="${OUTPUT_ROOT}/unified/checkpoints/unified-005000"
UNIFIED_GATE="${OUTPUT_ROOT}/unified-gate-report.json"

"${TORCHRUN[@]}" train.py \
  --conflict-fix-config "${CONFLICT_CONFIG}" \
  --conflict-fix-phase world_pretrain

"${UCF_PYTHON_BIN_RESOLVED}" - "${WORLD_VALIDATION}" "${WORLD_GATE}" <<'PY'
import json
from pathlib import Path
import sys

from corrective_foresight.training.gates import assess_world_gate, write_world_gate_report
from corrective_foresight.training.validation import ValidationRecord

source, destination = map(Path, sys.argv[1:])
value = json.loads(source.read_text(encoding="utf-8"))
record = ValidationRecord(**value)
write_world_gate_report(destination, assess_world_gate(record))
PY

"${UCF_PYTHON_BIN_RESOLVED}" - "${WORLD_GATE}" <<'PY'
from corrective_foresight.training.gates import read_world_gate_report
import sys
result = read_world_gate_report(sys.argv[1])
if not result.accepted:
    raise SystemExit(f"world gate did not pass: {result.outcome}")
PY

"${TORCHRUN[@]}" train.py \
  --conflict-fix-config "${CONFLICT_CONFIG}" \
  --conflict-fix-phase gradient_audit \
  --init-checkpoint "${WORLD_CHECKPOINT}"

"${UCF_PYTHON_BIN_RESOLVED}" - \
  "${AUDIT_METRICS}" "${WORLD_VALIDATION}" "${AUDIT_VALIDATION}" \
  "${AUDIT_DECISION}" <<'PY'
import json
from pathlib import Path
import sys

from corrective_foresight.training.gates import (
    AuditMeasurement,
    decide_gradient_audit,
    write_audit_decision,
)

metrics_path, world_path, audit_path, destination = map(Path, sys.argv[1:])
measurements = []
for line in metrics_path.read_text(encoding="utf-8").splitlines():
    value = json.loads(line)
    step = value["optimizer_step"]
    if 251 <= step <= 500 and step % 10 == 0:
        measurements.append(
            AuditMeasurement(
                optimizer_step=step,
                minimum_dynamics_cosine=value[
                    "gradient_cosine/minimum_dynamics"
                ],
            )
        )
world = json.loads(world_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
decision = decide_gradient_audit(
    measurements,
    world["metrics"]["dynamics_loss"],
    audit["metrics"]["dynamics_loss"],
)
write_audit_decision(destination, decision)
PY

"${TORCHRUN[@]}" train.py \
  --conflict-fix-config "${CONFLICT_CONFIG}" \
  --conflict-fix-phase unified_gate \
  --init-checkpoint "${WORLD_CHECKPOINT}" \
  --audit-decision "${AUDIT_DECISION}"

"${UCF_PYTHON_BIN_RESOLVED}" - "${UNIFIED_CHECKPOINT}" <<'PY'
from pathlib import Path
import hashlib
import sys
from corrective_foresight.training.run_manifest import RunManifest

checkpoint = Path(sys.argv[1])
manifest_path = checkpoint / "manifest.json"
manifest = RunManifest.from_json(manifest_path.read_text(encoding="utf-8"))
for name, metadata in manifest.value["files"].items():
    payload = checkpoint / name
    if payload.stat().st_size != metadata["size"]:
        raise SystemExit(f"checkpoint size mismatch: {name}")
    if hashlib.sha256(payload.read_bytes()).hexdigest() != metadata["sha256"]:
        raise SystemExit(f"checkpoint hash mismatch: {name}")
PY

scripts/evaluate_maniskill_conflict_fix.sh "${CONFLICT_CONFIG}"

"${UCF_PYTHON_BIN_RESOLVED}" - "${UNIFIED_GATE}" <<'PY'
import json
from pathlib import Path
import sys
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("passed") is not True:
    raise SystemExit("unified effect gate did not pass")
PY

if [[ "${UCF_CONTINUE_TO_20000:-0}" != "1" ]]; then
  echo "unified step-5000 gate passed; set UCF_CONTINUE_TO_20000=1 to continue" >&2
  exit 0
fi

"${TORCHRUN[@]}" train.py \
  --conflict-fix-config "${CONFLICT_CONFIG}" \
  --conflict-fix-phase unified_continue \
  --resume "${UNIFIED_CHECKPOINT}" \
  --audit-decision "${AUDIT_DECISION}"
