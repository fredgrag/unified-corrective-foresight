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

Model training and evaluation commands will be added only when their tested
implementations land. Training losses or smoke tests alone are not benchmark
or SOTA evidence.

