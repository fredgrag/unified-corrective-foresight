# ManiSkill Stable Pilot Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact four-rank checkpoint/resume, deterministic periodic validation, and a guarded two-stage ManiSkill pilot launcher, then run the approved 1,000 + 2,000 step pilot.

**Architecture:** Checkpoint format v2 stores shared model/trainer payloads once and one runtime payload per rank under a coordinated atomic directory. A separate validation runner consumes its own validation mixer and fixed generator, while a pilot controller triggers validation and checkpoint callbacks only after optimizer steps. Existing checkpoint v1, production experiment YAMLs, model architecture, and closed-loop runner remain unchanged.

**Tech Stack:** Python 3.12, PyTorch 2.8 distributed/NCCL, safetensors, LeRobot 0.5.1, ManiSkill 3.0.1, unittest, YAML, H.264/PyAV.

## Global Constraints

- Work only in `unified_corrective_foresight/`; do not import the parent legacy source tree.
- Use LeRobotDataset v3.0 only and keep `lerobot==0.5.1`.
- Use the pinned physical DINOv3 ViT-B/16 ModelScope snapshot and frozen CLIP cache.
- Keep the production model at hidden size 768, 12 layers, 12 heads, H=8, E=1, temporal ensemble off.
- Keep exactly four unified optimized losses and dynamics-only world pretraining.
- Use GPUs 0-3 and effective global batch size 64; do not silently change batch, precision, data, loss, stage, or backbone.
- Every behavior change follows observed RED, minimal GREEN, focused tests, and a commit.
- Generated checkpoints, metrics, records, videos, datasets, and logs remain outside Git.

---

### Task 1: Coordinated Distributed Checkpoint V2

**Files:**
- Create: `corrective_foresight/training/distributed_checkpoint.py`
- Modify: `corrective_foresight/training/checkpoint.py`
- Modify: `corrective_foresight/training/run_manifest.py`
- Test: `tests/unit/test_distributed_checkpoint.py`
- Test: `tests/integration/test_distributed_checkpoint_resume.py`

**Interfaces:**
- Consumes: existing `CheckpointState`, `ExpectedCheckpointContract`, `ResumeState`, `DistributedContext`, `Trainer`, and `StatefulMixer`.
- Produces: `save_distributed_checkpoint_atomic(path: Path, state: CheckpointState, context: DistributedContext) -> None` and `load_distributed_checkpoint_strict(path: Path, expected: ExpectedCheckpointContract, context: DistributedContext) -> ResumeState`.

- [ ] **Step 1: Write the format and ownership RED tests**

Require this exact v2 root file set for world size two:

```python
expected = {
    "online_model.safetensors",
    "ema_target.safetensors",
    "trainer_state.pt",
    "rank-0000-runtime.pt",
    "rank-0001-runtime.pt",
    "manifest.json",
}
self.assertEqual({item.name for item in checkpoint.iterdir()}, expected)
self.assertEqual(manifest.value["format_version"], 2)
self.assertEqual(manifest.value["distributed"], {"world_size": 2})
```

Assert rank payloads contain exactly `version`, `rank`, `world_size`, `mixer_state`, and `rng_state`. A missing rank file, wrong world size, symlink, extra file, bad digest, or rank mismatch must fail before model state is copied.

- [ ] **Step 2: Run test to verify RED**

Run: `python -m unittest tests.unit.test_distributed_checkpoint -v`

Expected: import failure for `corrective_foresight.training.distributed_checkpoint`.

- [ ] **Step 3: Implement v2 local payload helpers**

```python
DISTRIBUTED_FORMAT_VERSION = 2

def rank_runtime_name(rank: int) -> str:
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    return f"rank-{rank:04d}-runtime.pt"

def capture_rank_runtime(state: CheckpointState, context: DistributedContext) -> dict[str, object]:
    return {
        "version": 1,
        "rank": context.rank,
        "world_size": context.world_size,
        "mixer_state": state.mixer.state_dict(),
        "rng_state": capture_rng_state(
            state.flow_generator, local_cuda_device=context.device
        ),
    }
```

Expose or move existing checkpoint tensor/hash/copy helpers without changing v1 behavior. Extend `RunManifest` to accept format 1 or 2 and require `distributed` only for format 2.

- [ ] **Step 4: Implement coordinated atomic save**

The algorithm is exact:

```text
rank 0 validates destination absence and creates one UUID temporary directory
rank 0 broadcasts the absolute temporary path
rank 0 writes shared model/EMA/trainer payloads
every rank writes only rank-NNNN-runtime.pt
all ranks all_gather_object(None or serialized error)
any error prevents publication and rank 0 removes the temporary directory
rank 0 hashes all payloads, writes/fsyncs manifest, fsyncs directory, os.rename publishes
final barrier exposes only the completed destination
```

Use `dist.broadcast_object_list`, `dist.all_gather_object`, and `dist.barrier`. Reject a context that differs from the initialized process group.

- [ ] **Step 5: Implement strict distributed load**

Validate all v2 fields and payload hashes before loading tensors. Every rank loads shared policy/trainer state and only `rank_runtime_name(context.rank)` for mixer/RNG. Reject v1 in this API with an actionable message; preserve `load_checkpoint_strict` v1 behavior.

- [ ] **Step 6: Run unit tests GREEN**

Run:

```bash
python -m unittest tests.unit.test_checkpoint_validation \
  tests.unit.test_distributed_checkpoint -v
```

Expected: all tests pass without warnings.

- [ ] **Step 7: Add fresh-process exact resume integration test**

Spawn two Gloo ranks with deterministic policies and disjoint samplers. Compare uninterrupted four steps against two steps + v2 save + fresh-process load + two steps. Assert exact selected dataset ID, sample indices, flow-generator state, optimized losses, model parameters, optimizer step, and EMA step.

- [ ] **Step 8: Run integration GREEN and commit**

```bash
python -m unittest tests.integration.test_distributed_checkpoint_resume -v
git add corrective_foresight/training tests/unit/test_distributed_checkpoint.py \
  tests/integration/test_distributed_checkpoint_resume.py
git commit -m "feat: add exact distributed checkpoint resume"
```

---

### Task 2: Deterministic No-Update Validation

**Files:**
- Create: `corrective_foresight/training/validation.py`
- Test: `tests/unit/test_validation_runner.py`
- Test: `tests/integration/test_distributed_validation.py`

**Interfaces:**
- Consumes: a dedicated validation `BalancedLeRobotMixer`, `Trainer.policy`, `TrainingStage`, `DistributedContext`, and fixed generator seed.
- Produces: `ValidationConfig`, `ValidationRecord`, and `ValidationRunner.evaluate(stage: TrainingStage | str, global_step: int) -> ValidationRecord`.

- [ ] **Step 1: Write validation mutation RED tests**

```python
config = ValidationConfig(batches_per_rank=8, generator_seed=20261017)
record = runner.evaluate("unified", global_step=100)
self.assertEqual(record.batches_per_rank, 8)
self.assertTrue(all(math.isfinite(value) for value in record.metrics.values()))
```

Before/after evaluation compare policy parameters/buffers, optimizer/scheduler/scaler, EMA step, training mixer state, and global Torch RNG byte-for-byte. Assert the original policy training mode is restored.

- [ ] **Step 2: Run test to verify RED**

Run: `python -m unittest tests.unit.test_validation_runner -v`

Expected: import failure for `corrective_foresight.training.validation`.

- [ ] **Step 3: Implement validation records and runner**

```python
@dataclass(frozen=True, slots=True)
class ValidationConfig:
    batches_per_rank: int
    generator_seed: int

@dataclass(frozen=True, slots=True)
class ValidationRecord:
    global_step: int
    stage: str
    world_size: int
    batches_per_rank: int
    samples: int
    metrics: Mapping[str, float]
```

The runner owns a separate validation mixer and generator. Snapshot/restore global CPU/CUDA RNG around `policy.eval()` and `torch.no_grad()`. Sum detached float64 metrics and sample counts. Use `dist.all_reduce` and return identical records on all ranks. Reject nonfinite and missing required stage metrics.

- [ ] **Step 4: Run unit GREEN**

Run: `python -m unittest tests.unit.test_validation_runner -v`

Expected: deterministic repeated records and no state mutation.

- [ ] **Step 5: Add distributed validation equality test**

Run two ranks over disjoint fixed validation shards. Assert both ranks return identical canonical records and repeated evaluation at the same step is byte-identical.

- [ ] **Step 6: Run integration GREEN and commit**

```bash
python -m unittest tests.integration.test_distributed_validation -v
git add corrective_foresight/training/validation.py \
  tests/unit/test_validation_runner.py tests/integration/test_distributed_validation.py
git commit -m "feat: add deterministic distributed validation"
```

---

### Task 3: Pilot Configuration And Optimizer-Boundary Controller

**Files:**
- Create: `corrective_foresight/config/pilot.py`
- Create: `corrective_foresight/training/pilot.py`
- Create: `configs/pilots/maniskill_stable_v1.yaml`
- Modify: `train.py`
- Create: `scripts/launch_maniskill_pilot.sh`
- Test: `tests/unit/test_pilot_config.py`
- Test: `tests/unit/test_pilot_controller.py`
- Modify: `tests/regression/test_config_contract.py`

**Interfaces:**
- Consumes: existing world/unified experiment YAMLs, Task 1 checkpoint APIs, Task 2 validation runner, and `run_training` optimizer-step results.
- Produces: `PilotConfig`, `load_pilot_config`, `PilotController.on_optimizer_step(step: int)`, and one operator launcher.

- [ ] **Step 1: Write strict pilot YAML RED tests**

Require this exact mapping:

```yaml
schema_version: 1
pilot_id: maniskill.pick_cube.stable.v1
world_experiment: configs/experiments/maniskill_world_pretrain.yaml
unified_experiment: configs/experiments/maniskill_unified.yaml
world_steps: 1000
unified_steps: 2000
checkpoint_interval: 250
validation_interval: 100
validation_batches_per_rank: 8
evaluation_steps: [0, 1000, 2000]
evaluation_seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
gpu_indices: [0, 1, 2, 3]
output_root: /mnt/workspace/wwl/corrective-foresight/unified_corrective_foresight/artifacts/runs/maniskill_pilot_20260718
```

Reject unknown fields, non-absolute output, duplicates, intervals that do not divide stage endpoints, and base experiment/model/spec mismatch.

- [ ] **Step 2: Run config RED**

Run: `python -m unittest tests.unit.test_pilot_config -v`

Expected: import failure for `corrective_foresight.config.pilot`.

- [ ] **Step 3: Implement strict pilot loader and YAML**

Use a frozen `PilotConfig` dataclass with immutable tuples and resolved project paths. Load both base experiments and verify stage, production model, DatasetSpec/ActionSpec hashes, effective global batch 64, and distinct output roots.

- [ ] **Step 4: Write controller RED tests**

Use fake checkpoint/validation callbacks. Feed microsteps with accumulation and assert callbacks occur only at optimizer steps 100/200 for validation and 250/500 for checkpoints; final checkpoint fires once when it coincides with an interval.

- [ ] **Step 5: Implement controller and callback hook**

```python
OptimizerStepCallback = Callable[[int, TrainStepResult], None]

def run_training(
    ...,
    optimizer_step_callback: OptimizerStepCallback | None = None,
) -> int:
    ...
    global_step += 1
    if optimizer_step_callback is not None:
        optimizer_step_callback(global_step, result)
```

`PilotController` validates before checkpoint at coincident steps, writes validation JSON atomically on rank 0, invokes Task 1 save on every rank, and rejects existing output paths.

- [ ] **Step 6: Wire `train.py` and launcher**

Add `--pilot-config` and `--pilot-stage {world_pretrain,unified}`. Build a separate validation adapter/mixer from the validation split. Use checkpoint v2 whenever world size is greater than one. The launcher sets `CUDA_VISIBLE_DEVICES=0,1,2,3`, `UCF_NPROC_PER_NODE=4`, runs world first, strict-loads the final checkpoint, then runs unified. A noclobber lock prevents duplicate launch.

- [ ] **Step 7: Run focused GREEN and commit**

```bash
python -m unittest tests.unit.test_pilot_config \
  tests.unit.test_pilot_controller tests.regression.test_config_contract -v
bash -n scripts/*.sh
git add corrective_foresight/config/pilot.py corrective_foresight/training/pilot.py \
  configs/pilots train.py scripts/launch_maniskill_pilot.sh tests
git commit -m "feat: orchestrate guarded ManiSkill pilot training"
```

---

### Task 4: Four-GPU Gate, Launch, And Effect Report

**Files:**
- Create: `scripts/verify_maniskill_pilot.sh`
- Create: `scripts/evaluate_maniskill_pilot.sh`
- Modify: `README.md`
- Create: `docs/experiments/2026-07-18-maniskill-pilot-run.md`

**Interfaces:**
- Consumes: Tasks 1-3, real artifacts/dataset, `evaluate.py`, and four H20 GPUs.
- Produces: verified implementation commit, live pilot logs, periodic checkpoints/validation records, closed-loop records/videos, and a final comparison report.

- [ ] **Step 1: Implement fail-closed preflight script**

Exit nonzero unless GPUs 0-3 are requested and idle, at least 100 GB shared disk is free, no run directory/lock exists, CUDA/Vulkan and DINO/CLIP hashes pass, LeRobot specs validate, Git is clean, and HEAD exists on its upstream remote.

- [ ] **Step 2: Run complete software gates**

```bash
python -m unittest discover -s tests -v
python -m compileall -q corrective_foresight tests train.py evaluate.py
bash -n scripts/*.sh
find . -path ./.git -prune -o -path ./artifacts -prune -o -type l -print
find . -path ./.git -prune -o -name '._*' -print
git diff --check
```

Expected: all tests pass, no skips/warnings/links/AppleDouble, and no diff errors.

- [ ] **Step 3: Run two-rank and four-rank resume gates**

Run the CPU two-rank integration test, then a real four-H20 gate with two optimizer steps + save + fresh-process resume + two steps. Require exact next-batch IDs, flow seeds, finite losses, and parameters.

- [ ] **Step 4: Commit and push implementation before training**

```bash
git add .
git commit -m "feat: complete stable ManiSkill pilot gates"
git push origin feature/maniskill-pilot-training
git status --short
```

Expected: clean branch tracking the pushed remote commit.

- [ ] **Step 5: Launch pilot as a persistent process**

Run preflight, then launch `scripts/launch_maniskill_pilot.sh` into a new immutable log and record PID, hostname, start time, commit, CUDA devices, and config hash. Do not detach until four ranks initialize, first metrics appear, and output roots are writable.

- [ ] **Step 6: Monitor world stage and transition**

At every validation/checkpoint interval, verify process liveness, GPU memory/utilization, disk, finite metrics, and artifact hashes. Transition to unified only after the world step-1,000 checkpoint strict-load and validation gates pass.

- [ ] **Step 7: Evaluate unified checkpoints**

Run the existing closed-loop evaluator for untrained and unified steps 1,000/2,000 over seeds 0-9 with max 200. Preserve canonical JSON/video artifacts and compare paired outcomes.

- [ ] **Step 8: Write and commit final pilot report**

Record exact commits/config hashes, runtime, checkpoints, validation curves, success counts, paired reward/length changes, consistency, uncertainty, failures, and gate outcomes. State explicitly that the pilot is not benchmark or SOTA evidence.
