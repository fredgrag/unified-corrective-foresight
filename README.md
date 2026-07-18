# Unified Corrective Foresight

This is the isolated implementation of the Unified Corrective Foresight model.
It owns its source and dependency contract and does not import the parent
legacy implementation.

The governing design and engineering rule are maintained in the parent
project documentation:

- `docs/research/2026-07-16-unified-corrective-foresight-final-design.md`
- `docs/engineering/unified-model-implementation-rule.md`
- `docs/implementation/2026-07-16-unified-corrective-foresight-core-implementation-plan.md`

## Environment

The project uses an external Python 3.12 environment so runtime packages do
not enter the source tree:

```bash
bash scripts/bootstrap_environment.sh
bash scripts/verify_environment.sh
source /mnt/workspace/wwl/.venvs/unified-corrective-foresight/bin/activate
python -m unittest discover -s tests -v
```

The production CUDA contract is PyTorch 2.8.0 on CUDA 12.8 with at least four
NVIDIA H20 GPUs. Dependencies are exact at the project boundary and the fully
resolved environment is recorded in `requirements/lock-cu128.txt`.

## Frozen DINOv3 Artifact

Download the content-equivalent DINOv3 ViT-B/16 snapshot from its pinned
ModelScope commit before the real-backbone gate or training:

```bash
python scripts/download_dinov3.py --artifact-root artifacts/models
python -m unittest tests.integration.test_real_dinov3 -v
```

`artifacts/` is ignored by Git but remains a physical directory inside this
isolated project. The downloader verifies the immutable tracked size/SHA256
manifest and atomically installs complete files. Runtime loading re-verifies
the snapshot and is local-only; no symlink, mutable branch, ViT-L/16, or hub
fallback is accepted.

## Training Entry Points

The tested operator entry points use only the frozen experiment YAMLs. The
launchers default to four processes; set `UCF_NPROC_PER_NODE` only when
deliberately changing the distributed allocation. A node-local Python runtime
is selected automatically when the shared venv cannot start on a node.

```bash
./scripts/verify_environment.sh
./scripts/launch_world_pretrain.sh
./scripts/launch_unified.sh --init-checkpoint <world-pretrain-checkpoint>
```

To create an explicitly labeled, untrained structural checkpoint for a control
path smoke test without running an optimizer step:

```bash
python train.py \
  --config configs/experiments/maniskill_unified.yaml \
  --initialize-only \
  --output-checkpoint artifacts/checkpoints/untrained_smoke
```

## Closed-Loop Evaluation

The primary protocol is fixed at action horizon 8, execution horizon 1,
temporal ensemble off, midpoint flow integration with 10 intervals and 20
neural function evaluations. Every step executes only the first predicted
action, then re-observes RGB/proprio and replans. Evaluation verifies all
checkpoint payload hashes before restoring policy/EMA weights and writes a
canonical JSON record plus H.264 rollout video.

```bash
./scripts/launch_evaluation.sh \
  --checkpoint artifacts/checkpoints/unified-000200000 \
  --seeds 0 1 2 \
  --tag matched_eval
```

The `untrained_smoke` records validate action bounds, finite outputs, real
camera conversion, re-planning cadence, solver/NFE accounting, and
reproducibility. They are not benchmark or SOTA evidence. A performance claim
requires matched datasets, action/control protocols, at least three training
seeds, confidence intervals, failure analysis, and saved rollout artifacts.
