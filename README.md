# Unified Corrective Foresight

This is the isolated implementation of the Unified Corrective Foresight model.
It owns its source and dependency contract and does not import the parent
legacy implementation.

The governing design and engineering rule are maintained in the parent
project documentation:

- `../docs/research/2026-07-16-unified-corrective-foresight-final-design.md`
- `../docs/engineering/unified-model-implementation-rule.md`
- `../docs/implementation/2026-07-16-unified-corrective-foresight-core-implementation-plan.md`

## Environment

The project uses an external Python 3.12 environment so runtime packages do
not enter the source tree:

```bash
bash scripts/bootstrap_environment.sh
bash scripts/verify_environment.sh
source /mnt/workspace/wwl/.venvs/unified-corrective-foresight/bin/activate
python -m unittest discover -s tests -v
```

The production CUDA contract is PyTorch 2.8.0 on CUDA 12.8 with four NVIDIA
H20 GPUs. Dependencies are exact at the project boundary and the fully
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

Model training and evaluation commands will be added only when their tested
implementations land. Training losses or smoke tests alone are not benchmark
or SOTA evidence.
