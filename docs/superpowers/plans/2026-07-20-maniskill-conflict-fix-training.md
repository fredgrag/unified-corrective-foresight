# ManiSkill Conflict-Fix Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, and launch the approved ManiSkill PickCube conflict-fix experiment with protected optimizer groups, measured gradient conflict, audit-gated shared-only PCGrad, strict stop/effect gates, and rank-0 W&B tracking.

**Architecture:** Keep the causal world-action model and four fixed objectives unchanged. Add explicit training-side modules for parameter ownership, accumulated distributed objective gradients, audit/stop decisions, and W&B telemetry; orchestrate fresh world pretraining, a disposable audit, a 5,000-step unified gate, and exact continuation toward 20,000 steps.

**Tech Stack:** Python 3.12, PyTorch 2.8.0/cu128, torch.distributed/DDP, LeRobot 0.5.1, W&B 0.24.2, PyYAML, unittest, Bash, four NVIDIA H20 GPUs.

## Global Constraints

- Execute all implementation, tests, training, and evaluation on `ssh root@8.130.172.181 -p 1010`.
- Work only in `/mnt/workspace/wwl/corrective-foresight/unified_corrective_foresight`; create no symbolic links and import no parent-project source.
- Preserve DINOv3 ViT-B/16 frozen weights and exact ModelScope delivery provenance.
- Preserve LeRobotDataset v3.0, `lerobot==0.5.1`, DatasetSpec, ActionSpec, causal token views, H=8, E=1, and no temporal ensemble.
- Preserve exactly four optimized objectives with effective weights `1.0`, `1.0`, cycle warmup `[0.0, 0.1]`, and `1.0`; add no optimized loss.
- Apply PCGrad only to globally averaged, full-accumulation shared-Transformer gradients and only when the immutable audit decision enables it.
- Keep local JSONL, validation JSON, evaluation records, and checkpoint v2 authoritative; W&B never receives checkpoint payloads.
- Use RED/GREEN TDD for every behavior change and commit each completed task separately.
- Do not launch long training until the complete suite, four-rank exact-resume gate, W&B auth, environment, disk, and source-boundary checks pass with no skips.

---

## File Map

Create these focused modules:

- `corrective_foresight/config/conflict_fix.py`: strict conflict-fix orchestration config.
- `corrective_foresight/training/parameter_groups.py`: exhaustive parameter ownership.
- `corrective_foresight/training/gradient_conflict.py`: accumulation, all-reduce, diagnostics, and PCGrad.
- `corrective_foresight/training/gates.py`: audit, world, stop, and 5,000-step gate decisions.
- `corrective_foresight/tracking/__init__.py`: tracking package exports.
- `corrective_foresight/tracking/wandb_tracker.py`: rank-0 W&B lifecycle and atomic metadata.
- `corrective_foresight/evaluation/conflict_fix_report.py`: paired ten-seed aggregation and effect gate.
- `configs/experiments/maniskill_world_pretrain_conflict_fix.yaml`: 5,000-step world schedule.
- `configs/experiments/maniskill_unified_conflict_fix.yaml`: 20,000-total unified schedule.
- `configs/pilots/maniskill_conflict_fix_v2.yaml`: approved orchestration and thresholds.
- `scripts/launch_maniskill_conflict_fix.sh`: fail-closed phase launcher.
- `scripts/evaluate_maniskill_conflict_fix.sh`: corrected policy checkpoint evaluation.
- `scripts/verify_maniskill_conflict_fix.sh`: full preflight and exact-resume gate.

Modify these existing integration points:

- `pyproject.toml`: declare `wandb==0.24.2`.
- `corrective_foresight/config/__init__.py`: export conflict-fix config types.
- `corrective_foresight/training/trainer.py`: grouped optimizer and gradient modes.
- `corrective_foresight/training/pilot.py`: validation-driven stop decisions and reports.
- `train.py`: phase selection, composite logging, gate wiring, and graceful stop.
- `evaluate.py`: optional conflict-fix tracking metadata without changing simulator semantics.

## Task 1: Strict Config And Exhaustive Parameter Groups

**Files:**
- Create: `corrective_foresight/config/conflict_fix.py`
- Create: `corrective_foresight/training/parameter_groups.py`
- Create: `tests/unit/test_conflict_fix_config.py`
- Create: `tests/unit/test_optimizer_parameter_groups.py`
- Modify: `corrective_foresight/config/__init__.py`

**Interfaces:**
- Produces: `ConflictFixConfig`, `TrackingConfig`, `load_conflict_fix_config(path) -> ConflictFixConfig`.
- Produces: `OptimizerParameterGroups(protected, action)` and `build_optimizer_parameter_groups(policy) -> OptimizerParameterGroups`.
- Consumes: `UnifiedCorrectiveForesightPolicy` and the existing strict experiment loaders.

- [ ] **Step 1: Write the failing config contract tests**

```python
class ConflictFixConfigTest(unittest.TestCase):
    def test_loads_approved_schedule_and_thresholds(self) -> None:
        config = load_conflict_fix_config(CONFIG_PATH)
        self.assertEqual(config.world_steps, 5000)
        self.assertEqual(config.audit_steps, 500)
        self.assertEqual(config.unified_gate_steps, 5000)
        self.assertEqual(config.unified_total_steps, 20000)
        self.assertEqual(config.validation_interval, 250)
        self.assertEqual(config.checkpoint_interval, 1000)
        self.assertEqual(config.protected_lr_multiplier, 0.1)
        self.assertEqual(config.conflict_cosine_threshold, -0.05)
        self.assertEqual(config.conflict_measurement_minimum, 8)
        self.assertEqual(config.dynamics_degradation_ratio, 1.2)
        self.assertEqual(config.tracking.project, "unified-corrective-foresight")
        self.assertFalse(config.tracking.upload_checkpoints)

    def test_rejects_unknown_field_and_noncanonical_seeds(self) -> None:
        original = CONFIG_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            unknown = Path(directory) / "unknown.yaml"
            unknown.write_text(original + "unknown: true\n", encoding="utf-8")
            bad_seeds = Path(directory) / "bad-seeds.yaml"
            bad_seeds.write_text(
                original.replace(
                    "evaluation_seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]",
                    "evaluation_seeds: [0, 2]",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conflict-fix fields"):
                load_conflict_fix_config(unknown)
            with self.assertRaisesRegex(ValueError, "evaluation_seeds"):
                load_conflict_fix_config(bad_seeds)
```

- [ ] **Step 2: Run the config tests and observe RED**

Run:

```bash
source scripts/runtime_env.sh
"$UCF_PYTHON_BIN_RESOLVED" -m unittest tests.unit.test_conflict_fix_config -v
```

Expected: import failure for `corrective_foresight.config.conflict_fix`.

- [ ] **Step 3: Implement the strict config dataclasses and loader**

```python
@dataclass(frozen=True, slots=True)
class TrackingConfig:
    enabled: bool
    project: str
    group: str
    log_interval: int
    upload_checkpoints: bool


@dataclass(frozen=True, slots=True)
class ConflictFixConfig:
    source_path: Path
    pilot_id: str
    world_experiment: Path
    unified_experiment: Path
    output_root: Path
    world_steps: int
    audit_steps: int
    unified_gate_steps: int
    unified_total_steps: int
    validation_interval: int
    checkpoint_interval: int
    validation_batches_per_rank: int
    protected_lr_multiplier: float
    conflict_log_interval: int
    conflict_cosine_threshold: float
    conflict_measurement_minimum: int
    dynamics_degradation_ratio: float
    world_copy_improvement_target: float
    required_success_gain: int
    evaluation_seeds: tuple[int, ...]
    gpu_indices: tuple[int, ...]
    tracking: TrackingConfig
```

The loader must require an exact key set, physical project-owned experiment files, absolute non-symlink output root, seeds `0..9`, GPUs `0..3`, `5000/500/5000/20000` steps, `250/1000` intervals, effective global batch 64, multiplier `0.1`, threshold `-0.05`, minimum 8, ratio `1.2`, target `0.05`, success gain 2, and W&B checkpoint upload false.

- [ ] **Step 4: Write the failing parameter ownership tests**

```python
class OptimizerParameterGroupsTest(unittest.TestCase):
    def test_groups_are_exhaustive_disjoint_and_semantic(self) -> None:
        policy = make_policy()
        groups = build_optimizer_parameter_groups(policy)
        protected = {id(item) for item in groups.protected}
        action = {id(item) for item in groups.action}
        trainable = {id(item) for item in policy.parameters() if item.requires_grad}
        self.assertFalse(protected & action)
        self.assertEqual(protected | action, trainable)
        self.assertIn(id(policy.world_action_model.delta_projection.weight), protected)
        self.assertIn(id(policy.world_action_model.policy_queries), action)
        adapter = next(iter(policy.world_action_model.action_adapters.adapters.values()))
        self.assertIn(id(adapter.action_input_projection.weight), protected)
        self.assertIn(id(adapter.flow_velocity_head.weight), action)
        self.assertTrue(all(not item.requires_grad for item in policy.online_state_encoder.backbone.parameters()))
```

- [ ] **Step 5: Implement explicit parameter grouping without name fallback**

```python
@dataclass(frozen=True, slots=True)
class OptimizerParameterGroups:
    protected: tuple[nn.Parameter, ...]
    action: tuple[nn.Parameter, ...]

    def validate(self, policy: UnifiedCorrectiveForesightPolicy) -> None:
        protected_ids = {id(item) for item in self.protected}
        action_ids = {id(item) for item in self.action}
        trainable_ids = {id(item) for item in policy.parameters() if item.requires_grad}
        if protected_ids & action_ids:
            raise ValueError("optimizer parameter groups overlap")
        if protected_ids | action_ids != trainable_ids:
            raise ValueError("optimizer parameter groups are not exhaustive")
```

Build tuples by traversing the exact module attributes approved in Section 5 of the design. Put `action_query`, `policy_queries`, and `policy_horizon_embedding` in the action group; put `action_input_projection` in protected; exclude backbone and EMA by identity.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
"$UCF_PYTHON_BIN_RESOLVED" -m unittest \
  tests.unit.test_conflict_fix_config \
  tests.unit.test_optimizer_parameter_groups -v
```

Expected: all tests pass.

Commit:

```bash
git add corrective_foresight/config corrective_foresight/training/parameter_groups.py tests/unit/test_conflict_fix_config.py tests/unit/test_optimizer_parameter_groups.py
git commit -m "feat: add conflict-fix config and optimizer groups"
```

## Task 2: Accumulated Distributed Gradient Diagnostics And PCGrad

**Files:**
- Create: `corrective_foresight/training/gradient_conflict.py`
- Create: `tests/unit/test_gradient_conflict.py`
- Create: `tests/integration/test_distributed_gradient_conflict.py`
- Modify: `corrective_foresight/training/numerics.py`

**Interfaces:**
- Produces: `ObjectiveGradientAccumulator`, `GlobalObjectiveGradients`, `GradientDiagnostics`, `project_pcgrad`.
- Consumes: ordered shared-Transformer parameters, raw objective tensors, current effective weights, accumulation factor, `DistributedContext`, global step, seed.

- [ ] **Step 1: Write pure-math RED tests**

```python
class GradientConflictTest(unittest.TestCase):
    def test_cosines_cover_aligned_orthogonal_and_opposed(self) -> None:
        gradients = GlobalObjectiveGradients.from_flattened(
            raw={
                "dynamics_loss": torch.tensor([1.0, 0.0]),
                "inverse_action_loss": torch.tensor([-1.0, 0.0]),
                "action_cycle_loss": torch.tensor([0.0, 1.0]),
                "policy_flow_loss": torch.tensor([1.0, 0.0]),
            },
            effective_weights={
                "dynamics_loss": 1.0,
                "inverse_action_loss": 1.0,
                "action_cycle_loss": 0.1,
                "policy_flow_loss": 1.0,
            },
        )
        diagnostics = gradients.diagnostics()
        self.assertEqual(diagnostics.cosines["dynamics_loss/inverse_action_loss"], -1.0)
        self.assertEqual(diagnostics.cosines["dynamics_loss/action_cycle_loss"], 0.0)
        self.assertEqual(diagnostics.cosines["dynamics_loss/policy_flow_loss"], 1.0)

    def test_zero_effective_cycle_is_excluded_not_missing(self) -> None:
        gradients = GlobalObjectiveGradients.from_flattened(
            raw={
                "dynamics_loss": torch.tensor([1.0, 0.0]),
                "inverse_action_loss": torch.tensor([0.0, 1.0]),
                "action_cycle_loss": torch.tensor([1.0, 1.0]),
                "policy_flow_loss": torch.tensor([-1.0, 0.0]),
            },
            effective_weights={
                "dynamics_loss": 1.0,
                "inverse_action_loss": 1.0,
                "action_cycle_loss": 0.0,
                "policy_flow_loss": 1.0,
            },
        )
        projected = project_pcgrad(gradients, seed=17, global_step=0)
        self.assertNotIn("action_cycle_loss", projected.active_objectives)
        self.assertIn("action_cycle_loss", projected.raw_objectives)
```

- [ ] **Step 2: Run pure tests and observe RED**

Run `"$UCF_PYTHON_BIN_RESOLVED" -m unittest tests.unit.test_gradient_conflict -v`.

Expected: module import failure.

- [ ] **Step 3: Implement immutable gradient values and diagnostics**

```python
@dataclass(frozen=True, slots=True)
class GradientDiagnostics:
    raw_norms: Mapping[str, float]
    effective_norms: Mapping[str, float]
    cosines: Mapping[str, float]
    minimum_dynamics_cosine: float
    ordinary_effective_norm: float


@dataclass(frozen=True, slots=True)
class GlobalObjectiveGradients:
    names: tuple[str, ...]
    raw: Mapping[str, tuple[Tensor | None, ...]]
    effective_weights: Mapping[str, float]

    @classmethod
    def from_flattened(
        cls,
        *,
        raw: Mapping[str, Tensor],
        effective_weights: Mapping[str, float],
    ) -> GlobalObjectiveGradients:
        names = tuple(raw)
        return cls(
            names=names,
            raw={name: (raw[name],) for name in names},
            effective_weights=dict(effective_weights),
        )

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.names if self.effective_weights[name] > 0.0)

    def diagnostics(self) -> GradientDiagnostics:
        return compute_gradient_diagnostics(self)


@dataclass(frozen=True, slots=True)
class ProjectedGradientResult:
    gradient: tuple[Tensor | None, ...]
    active_objectives: tuple[str, ...]
    raw_objectives: tuple[str, ...]
    removed_fraction: float


def project_pcgrad(
    gradients: GlobalObjectiveGradients,
    *,
    seed: int,
    global_step: int,
) -> ProjectedGradientResult:
    return project_effective_gradients(
        gradients,
        projection_order(
            gradients.active_names,
            seed=seed,
            global_step=global_step,
        ),
    )
```

Use float32 dot products, `1e-12` norm floor only for division, sorted metric names, and a fatal `NumericalGuardError` for missing raw shared gradients or non-finite tensors.

- [ ] **Step 4: Implement deterministic effective-weight PCGrad**

```python
def projection_order(names: tuple[str, ...], *, seed: int, global_step: int) -> tuple[str, ...]:
    material = f"ucf-pcgrad-v1:{seed}:{global_step}:" + ",".join(names)
    order_seed = int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")
    generator = random.Random(order_seed)
    result = list(names)
    generator.shuffle(result)
    return tuple(result)


def project_pair(current: Tensor, other: Tensor) -> Tensor:
    dot = torch.dot(current, other)
    if dot >= 0:
        return current
    return current - dot * other / other.square().sum().clamp_min(1e-12)
```

Project only active effective gradients, restore tensor shapes without flattening model state, sum projected objectives, and return projection-removal metrics. Do not mutate raw buffers.

- [ ] **Step 5: Write and run distributed accumulation RED tests**

The two-rank test must accumulate two micro-steps per rank, assert all-reduced per-objective gradients equal the manually computed four-microbatch mean, and prove that projecting rank-local gradients produces a different value and is rejected by the public API.

Run:

```bash
"$UCF_PYTHON_BIN_RESOLVED" -m unittest tests.integration.test_distributed_gradient_conflict -v
```

Expected: failure because `ObjectiveGradientAccumulator.finalize` is absent.

- [ ] **Step 6: Implement accumulation and all-reduce**

```python
class ObjectiveGradientAccumulator:
    def __init__(self, parameters: tuple[nn.Parameter, ...], names: tuple[str, ...]) -> None:
        self.parameters = parameters
        self.names = names
        self._buffers = {name: [None for _ in parameters] for name in names}
        self._micro_steps = 0

    def add(self, losses: Mapping[str, Tensor], *, accumulation_steps: int) -> None:
        gradients = capture_raw_gradients(losses, self.parameters)
        add_scaled_gradients(self._buffers, gradients, scale=1.0 / accumulation_steps)
        self._micro_steps += 1

    def finalize(self, context: DistributedContext, effective_weights: Mapping[str, float]) -> GlobalObjectiveGradients:
        if self._micro_steps == 0:
            raise NumericalGuardError("objective gradient accumulator is empty")
        reduced = all_reduce_objective_gradients(self._buffers, context)
        self.reset()
        return GlobalObjectiveGradients(self.names, reduced, effective_weights)
```

All-reduce every non-None tensor with SUM and divide by world size. Require identical None/non-None layouts on every rank via `all_gather_object` before tensor collectives.

- [ ] **Step 7: Run focused tests and commit**

Run both new tests plus `tests.unit.test_numerical_guards`. Expected: all pass.

Commit:

```bash
git add corrective_foresight/training/gradient_conflict.py corrective_foresight/training/numerics.py tests/unit/test_gradient_conflict.py tests/integration/test_distributed_gradient_conflict.py
git commit -m "feat: add distributed gradient conflict engine"
```

## Task 3: Trainer Integration, Grouped Scheduler, And Exact Resume

**Files:**
- Modify: `corrective_foresight/training/trainer.py`
- Modify: `train.py`
- Create: `tests/unit/test_trainer_gradient_modes.py`
- Modify: `tests/integration/test_distributed_checkpoint_resume.py`

**Interfaces:**
- Extends `TrainerConfig` with protected multiplier, gradient mode, diagnostic interval, and PCGrad seed.
- Extends `TrainStepResult` with `learning_rates` and `gradient_metrics` while retaining `learning_rate` as the action-group compatibility value.
- Consumes Task 1 parameter groups and Task 2 gradient engine.

- [ ] **Step 1: Write grouped-LR and mode RED tests**

```python
def make_mode_trainer(
    *,
    gradient_mode: str,
    diagnostic_interval: int = 10,
    learning_rate: float = 5e-5,
    protected_lr_multiplier: float = 0.1,
    warmup_steps: int = 500,
    total_steps: int = 20000,
) -> Trainer:
    policy = make_policy()
    return Trainer(
        policy,
        TrainerConfig(
            learning_rate=learning_rate,
            weight_decay=0.05,
            accumulation_steps=1,
            max_grad_norm=1.0,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            bf16=False,
            ddp=False,
            stage="unified",
            protected_lr_multiplier=protected_lr_multiplier,
            gradient_mode=gradient_mode,
            gradient_diagnostic_interval=diagnostic_interval,
            pcgrad_seed=20260720,
        ),
        rank=0,
    )


def run_optimizer_steps(trainer: Trainer, count: int) -> list[TrainStepResult]:
    batch = make_batch(trainer.policy)
    generator = torch.Generator().manual_seed(1901)
    return [
        trainer.train_step(batch, "unified", step, generator)
        for step in range(count)
    ]


def test_unified_optimizer_uses_approved_group_lrs(self) -> None:
    trainer = make_mode_trainer(
        gradient_mode="audit",
        learning_rate=5e-5,
        protected_lr_multiplier=0.1,
        warmup_steps=500,
        total_steps=20000,
    )
    self.assertEqual(tuple(group["name"] for group in trainer.optimizer.param_groups), ("protected", "action"))
    self.assertEqual(trainer.optimizer.param_groups[0]["initial_lr"], 5e-6)
    self.assertEqual(trainer.optimizer.param_groups[1]["initial_lr"], 5e-5)

def test_audit_logs_only_on_tenth_optimizer_step(self) -> None:
    trainer = make_mode_trainer(gradient_mode="audit", diagnostic_interval=10)
    results = run_optimizer_steps(trainer, 10)
    self.assertEqual(results[8].gradient_metrics, {})
    self.assertIn("gradient_cosine/dynamics_loss/inverse_action_loss", results[9].gradient_metrics)
```

- [ ] **Step 2: Run tests and observe RED**

Expected: `TrainerConfig` rejects the new keywords.

- [ ] **Step 3: Add explicit grouped optimizer and scheduler state**

```python
@dataclass(frozen=True, slots=True)
class TrainerConfig:
    learning_rate: float
    weight_decay: float
    accumulation_steps: int
    max_grad_norm: float
    warmup_steps: int
    total_steps: int
    bf16: bool
    ddp: bool
    stage: str = "unified"
    protected_lr_multiplier: float = 1.0
    gradient_mode: str = "ordinary"
    gradient_diagnostic_interval: int = 10
    pcgrad_seed: int = 0
```

Validate `gradient_mode in {"ordinary", "audit", "pcgrad"}`. World pretraining requires multiplier `1.0` and ordinary mode. Unified requires multiplier `0.1` for conflict-fix runs. Create named param groups with base LRs `learning_rate * multiplier` and `learning_rate`; use one common LambdaLR multiplier so their ratio remains exact.

- [ ] **Step 4: Integrate full-accumulation diagnostics and shared replacement**

During each micro-step, capture objective gradients only when mode is PCGrad or when `(global_step + 1) % diagnostic_interval == 0` in audit mode. After the ordinary DDP backward reaches the optimizer boundary, finalize globally averaged per-objective gradients. In PCGrad mode overwrite only `world_action_model.transformer` synchronized gradients before `clip_and_validate_gradients`; in audit mode leave ordinary gradients unchanged. Return sorted scalar metrics.

- [ ] **Step 5: Bump and test trainer checkpoint state**

Set trainer state version to 2 and include config, optimizer, scheduler, scaler, micro-step count, last EMA step, and a required-empty gradient accumulator marker at checkpoint boundaries. `load_state_dict` must reject a checkpoint created with a different parameter multiplier, gradient mode, diagnostic interval, or seed.

- [ ] **Step 6: Extend exact-resume integration coverage**

Run the existing two-rank baseline/save/load comparison twice, once with `gradient_mode="ordinary"` and once with `gradient_mode="pcgrad"`, `accumulation_steps=2`, and nonzero cycle weight. Compare model, EMA, optimizer, scheduler, flow RNG, mixer, next-step metrics, projection order, and gradient metrics.

- [ ] **Step 7: Run focused tests and commit**

```bash
"$UCF_PYTHON_BIN_RESOLVED" -m unittest \
  tests.unit.test_trainer_gradient_modes \
  tests.integration.test_distributed_checkpoint_resume -v
```

Expected: all pass.

Commit:

```bash
git add corrective_foresight/training/trainer.py train.py tests/unit/test_trainer_gradient_modes.py tests/integration/test_distributed_checkpoint_resume.py
git commit -m "feat: integrate protected and projected optimizer steps"
```

## Task 4: Audit, Stop, World, And Effect Gate State Machines

**Files:**
- Create: `corrective_foresight/training/gates.py`
- Create: `tests/unit/test_conflict_fix_gates.py`
- Modify: `corrective_foresight/training/pilot.py`
- Modify: `tests/unit/test_pilot_controller.py`

**Interfaces:**
- Produces: `AuditDecision`, `decide_gradient_audit`, `ValidationStopMonitor`, `WorldGateResult`, `UnifiedGateResult`.
- Produces: atomic `gradient-audit-decision.json`, `stop-report.json`, and gate report writers.
- Changes `PilotController.on_optimizer_step` to return `PilotStepDecision(should_stop, reason)`.

- [ ] **Step 1: Write audit threshold RED tests**

```python
def test_audit_requires_eight_conflicts_and_dynamics_degradation(self) -> None:
    seven = [
        AuditMeasurement(
            optimizer_step=260 + 10 * index,
            minimum_dynamics_cosine=-0.06 if index < 7 else 0.0,
        )
        for index in range(25)
    ]
    eight = [
        AuditMeasurement(
            optimizer_step=260 + 10 * index,
            minimum_dynamics_cosine=-0.06 if index < 8 else 0.0,
        )
        for index in range(25)
    ]
    self.assertFalse(decide_gradient_audit(seven, 0.004, 0.00481).enable_pcgrad)
    self.assertFalse(decide_gradient_audit(eight, 0.004, 0.00480).enable_pcgrad)
    self.assertTrue(decide_gradient_audit(eight, 0.004, 0.00481).enable_pcgrad)
```

The ratio comparison is strict `> 1.20`, so exactly `0.00480` does not enable PCGrad.

- [ ] **Step 2: Write three-window stop RED tests**

Test copy-last non-improvement, dynamics ratio, two-loss deterioration with a negative inverse NLL best, reset after a healthy window, and no stop before unified step 500.

- [ ] **Step 3: Implement pure decisions and atomic JSON writers**

```python
@dataclass(frozen=True, slots=True)
class AuditMeasurement:
    optimizer_step: int
    minimum_dynamics_cosine: float


@dataclass(frozen=True, slots=True)
class PilotStepDecision:
    should_stop: bool
    reason: str | None


def negative_nll_deteriorated(current: float, best: float) -> bool:
    return current > best + 0.20 * max(abs(best), 1e-8)
```

Writers must use exclusive temporary files, flush, fsync, rename, and fsync the parent directory. Existing output files and symlinks are fatal.

- [ ] **Step 4: Wire validation decisions into `PilotController`**

After validation is written, feed the record to the stop monitor. If it stops, save the boundary checkpoint exactly once, write the stop report on rank 0, broadcast the decision, and return `PilotStepDecision(True, reason)`. Modify `run_training` to break only on this explicit return and return the actual final step.

- [ ] **Step 5: Run tests and commit**

Run `tests.unit.test_conflict_fix_gates` and `tests.unit.test_pilot_controller`. Expected: all pass.

Commit:

```bash
git add corrective_foresight/training/gates.py corrective_foresight/training/pilot.py tests/unit/test_conflict_fix_gates.py tests/unit/test_pilot_controller.py train.py
git commit -m "feat: add fail-closed conflict-fix gates"
```

## Task 5: Rank-Zero W&B Tracking With Atomic Resume Metadata

**Files:**
- Create: `corrective_foresight/tracking/__init__.py`
- Create: `corrective_foresight/tracking/wandb_tracker.py`
- Create: `tests/unit/test_wandb_tracker.py`
- Modify: `pyproject.toml`
- Modify: `train.py`
- Modify: `evaluate.py`

**Interfaces:**
- Produces: `WandbTracker`, `TrackingMetadata`, `reduce_scalar_metrics`.
- Consumes: `TrackingConfig`, rank/world size, output root, resolved config, explicit optimizer step, and an injectable W&B module for tests.

- [ ] **Step 1: Add `wandb==0.24.2` to project dependencies and write RED tests**

```python
class FakeRun:
    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.logged: list[tuple[dict[str, float], int]] = []

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self.logged.append((metrics, step))

    def finish(self) -> None:
        return None


class FakeWandb:
    def __init__(self) -> None:
        self.init_calls = 0
        self.last_resume: str | None = None

    def init(self, **kwargs: object) -> FakeRun:
        self.init_calls += 1
        self.last_resume = kwargs.get("resume") if isinstance(kwargs.get("resume"), str) else None
        run_id = kwargs.get("id") if isinstance(kwargs.get("id"), str) else "run-fixed"
        return FakeRun(run_id)


def tracking_config() -> TrackingConfig:
    return TrackingConfig(
        enabled=True,
        project="unified-corrective-foresight",
        group="maniskill-pickcube-conflict-fix-v2",
        log_interval=10,
        upload_checkpoints=False,
    )


class WandbTrackerTest(unittest.TestCase):
    def test_rank_zero_initializes_and_nonzero_rank_is_noop(self) -> None:
        backend = FakeWandb()
        rank_zero = WandbTracker.start(config=tracking_config(), rank=0, backend=backend, output_root=self.root)
        rank_one = WandbTracker.start(config=tracking_config(), rank=1, backend=backend, output_root=self.root)
        self.assertEqual(backend.init_calls, 1)
        self.assertTrue(rank_zero.enabled)
        self.assertFalse(rank_one.enabled)

    def test_resume_must_uses_atomic_run_id(self) -> None:
        first = WandbTracker.start(config=tracking_config(), rank=0, backend=self.backend, output_root=self.root)
        run_id = first.run_id
        first.finish(sync_complete=True)
        resumed = WandbTracker.resume(config=tracking_config(), rank=0, backend=self.backend, output_root=self.root)
        self.assertEqual(resumed.run_id, run_id)
        self.assertEqual(self.backend.last_resume, "must")
```

- [ ] **Step 2: Implement tracking metadata and lifecycle**

```python
@dataclass(frozen=True, slots=True)
class TrackingMetadata:
    format_version: int
    entity: str
    project: str
    group: str
    run_id: str
    mode: str
    last_optimizer_step: int
    sync_complete: bool


class WandbTracker:
    def log(self, metrics: Mapping[str, float], *, optimizer_step: int) -> None:
        if optimizer_step <= self.last_optimizer_step:
            raise ValueError("W&B optimizer steps must be strictly increasing")
        self.run.log(dict(metrics), step=optimizer_step)
        self._write_metadata(optimizer_step, sync_complete=False)
```

Call the injected backend exactly as follows on resume:

```python
run = backend.init(
    project=config.project,
    group=config.group,
    job_type=job_type,
    id=metadata.run_id,
    resume="must",
    config=sanitized_run_config,
)
```

Define `optimizer_step` as the step metric. Never serialize environment values or API keys. Refuse checkpoint artifact logging in the public interface.

- [ ] **Step 3: Implement four-rank scalar reduction at the 10-step logging boundary**

Require identical sorted metric names across ranks, all-reduce float64 values with SUM, divide by world size, and return a plain sorted mapping. Include raw/effective gradient norms, six cosines, conflict fractions, group LRs, pre-clip norm, clip coefficient, throughput, step time, and peak allocated memory.

- [ ] **Step 4: Integrate composite local/W&B logging**

Keep rank-0 JSONL on every optimizer step. At steps divisible by 10, all ranks participate in reduction and rank 0 logs W&B. Validation callbacks log globally reduced metrics with the same optimizer-step axis. An init/auth failure aborts preflight. A later network exception sets local metadata `sync_complete=false`, retains the SDK queue, and does not alter gradients.

- [ ] **Step 5: Run tests and commit**

Run `tests.unit.test_wandb_tracker`, `tests.unit.test_train_entrypoint`, and `tests.unit.test_evaluate_entrypoint`. Expected: all pass without contacting W&B.

Commit:

```bash
git add pyproject.toml corrective_foresight/tracking train.py evaluate.py tests/unit/test_wandb_tracker.py
git commit -m "feat: add rank-zero wandb experiment tracking"
```

## Task 6: Approved Configs And Fail-Closed Phase Launcher

**Files:**
- Create: `configs/experiments/maniskill_world_pretrain_conflict_fix.yaml`
- Create: `configs/experiments/maniskill_unified_conflict_fix.yaml`
- Create: `configs/pilots/maniskill_conflict_fix_v2.yaml`
- Create: `scripts/launch_maniskill_conflict_fix.sh`
- Create: `scripts/verify_maniskill_conflict_fix.sh`
- Modify: `train.py`
- Modify: `tests/unit/test_train_entrypoint.py`
- Create: `tests/integration/test_conflict_fix_orchestration.py`

**Interfaces:**
- Adds CLI `--conflict-fix-config` and `--conflict-fix-phase {world_pretrain,gradient_audit,unified_gate,unified_continue}`.
- `gradient_audit` and both unified phases remain `TrainingStage.UNIFIED` internally.
- Launcher consumes gate JSONs and never derives a decision from terminal text.

- [ ] **Step 1: Write exact YAML configs**

World config uses LR `0.0001`, warmup 500, total 5,000, checkpoint 1,000, validation 250. Unified config uses LR `0.00005`, warmup 500, total 20,000, checkpoint 1,000, validation 250. Both retain existing batch/accumulation values and effective global batch 64.

Conflict-fix config must contain exactly:

```yaml
schema_version: 1
pilot_id: maniskill.pick_cube.conflict_fix.v2
world_experiment: configs/experiments/maniskill_world_pretrain_conflict_fix.yaml
unified_experiment: configs/experiments/maniskill_unified_conflict_fix.yaml
world_steps: 5000
audit_steps: 500
unified_gate_steps: 5000
unified_total_steps: 20000
validation_interval: 250
checkpoint_interval: 1000
validation_batches_per_rank: 8
protected_lr_multiplier: 0.1
conflict_log_interval: 10
conflict_cosine_threshold: -0.05
conflict_measurement_minimum: 8
dynamics_degradation_ratio: 1.2
world_copy_improvement_target: 0.05
required_success_gain: 2
evaluation_seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
gpu_indices: [0, 1, 2, 3]
output_root: /mnt/workspace/wwl/corrective-foresight/unified_corrective_foresight/artifacts/runs/maniskill_conflict_fix_v2
tracking:
  enabled: true
  project: unified-corrective-foresight
  group: maniskill-pickcube-conflict-fix-v2
  log_interval: 10
  upload_checkpoints: false
```

- [ ] **Step 2: Write CLI and orchestration RED tests**

Test mutual exclusion with legacy config modes, required init checkpoints, required audit decision for unified phases, exact resume requirement for continuation, and refusal to launch if output roots or gate files already exist unexpectedly.

- [ ] **Step 3: Implement phase resolution in `train.py`**

Map phases to runtime stage, target step, output root, optimizer mode, W&B job type, and checkpoint loading:

```text
world_pretrain  -> fresh, ordinary, world/0..5000
gradient_audit -> warmstart world-005000, audit, audit/0..500
unified_gate   -> warmstart world-005000, decision ordinary|pcgrad, unified/0..5000
unified_continue -> exact resume unified-005000, same decision, unified/5000..20000
```

- [ ] **Step 4: Implement the shell launcher**

The launcher uses a noclobber lock, offline HF/Transformers, `CUDA_VISIBLE_DEVICES=0,1,2,3`, and four-process torchrun. It runs verification, world training, world gate, audit, immutable decision, unified gate training, checkpoint verification, closed-loop evaluation, and unified effect gate in that order. It exits before continuation; a separate explicit invocation with `UCF_CONTINUE_TO_20000=1` is required after the gate report authorizes continuation.

- [ ] **Step 5: Run orchestration tests and commit**

Run config, entrypoint, and orchestration tests plus `bash -n` for both scripts. Expected: all pass.

Commit:

```bash
git add configs corrective_foresight/config/conflict_fix.py train.py scripts/launch_maniskill_conflict_fix.sh scripts/verify_maniskill_conflict_fix.sh tests/unit/test_train_entrypoint.py tests/integration/test_conflict_fix_orchestration.py
git commit -m "feat: orchestrate conflict-fix training phases"
```

## Task 7: Corrected Closed-Loop Report, Effect Gate, And Retention

**Files:**
- Create: `corrective_foresight/evaluation/conflict_fix_report.py`
- Create: `tests/unit/test_conflict_fix_report.py`
- Create: `scripts/evaluate_maniskill_conflict_fix.sh`
- Modify: `evaluate.py`
- Modify: `scripts/evaluate_maniskill_pilot.sh`
- Create: `tests/regression/test_conflict_fix_evaluation_selection.py`

**Interfaces:**
- Produces: `EpisodeSummary`, `PairedEvaluationReport`, `build_paired_report`, `write_effect_gate`.
- Accepts only `untrained_action_from_world_5000`, `unified_5000`, and optionally `unified_20000` policy checkpoints.
- Consumes structured episode JSON produced by the existing runner; never parses console output.

- [ ] **Step 1: Write paired-report RED tests**

```python
def summaries(*, successes: set[int], rewards: list[float]) -> tuple[EpisodeSummary, ...]:
    return tuple(
        EpisodeSummary(
            seed=seed,
            success=seed in successes,
            total_reward=rewards[seed],
            episode_length=200,
            action_bounds_ok=True,
            protocol_hash="a" * 64,
        )
        for seed in range(10)
    )


def test_gate_requires_two_more_successes_and_positive_median_reward(self) -> None:
    baseline = summaries(successes={1}, rewards=[1.0] * 10)
    candidate = summaries(successes={1, 2, 3}, rewards=[1.2] * 10)
    report = build_paired_report(baseline, candidate, required_success_gain=2)
    self.assertEqual(report.success_gain, 2)
    self.assertGreater(report.median_reward_change, 0.0)
    self.assertTrue(report.effect_gate_passed)

def test_world_checkpoint_is_rejected_as_policy_result(self) -> None:
    with self.assertRaisesRegex(ValueError, "world-pretrain is not a policy result"):
        validate_policy_label("world_pretrain_5000")
```

- [ ] **Step 2: Implement strict record loading and aggregation**

Require exactly seeds 0 through 9, matching environment/protocol/spec hashes, unique records, finite rewards, action-bound compliance, valid video hashes, matched flow protocol, and distinct checkpoint manifest hashes. Compute success count/rate, paired reward changes, median change, episode lengths, consistency, inverse uncertainty, and NFE.

```python
@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    seed: int
    success: bool
    total_reward: float
    episode_length: int
    action_bounds_ok: bool
    protocol_hash: str


@dataclass(frozen=True, slots=True)
class PairedEvaluationReport:
    baseline_success_count: int
    candidate_success_count: int
    success_gain: int
    median_reward_change: float
    effect_gate_passed: bool
```

- [ ] **Step 3: Implement corrected evaluation script**

At the 5,000 gate evaluate only:

```text
unified/checkpoints/unified-000000 -> untrained_action_from_world_5000
unified/checkpoints/unified-005000 -> unified_5000
```

After authorized continuation add `unified-020000 -> unified_20000`. Use the unified experiment config for every policy result. Upload per-seed W&B tables and videos through Task 5 tracking; upload no checkpoint.

- [ ] **Step 4: Implement retention after verified reports**

Select best checkpoints from validation records using positive copy-last first and lowest dynamics loss second. Refuse deletion until final manifest hashes, gate JSON, W&B sync state, and exact-resume checkpoint are verified. Retain initial/best/final as specified; audit large checkpoints are deleted only after decision hash verification.

- [ ] **Step 5: Run tests and commit**

Run unit and regression tests plus the existing evaluation protocol and entrypoint tests. Expected: all pass.

Commit:

```bash
git add corrective_foresight/evaluation/conflict_fix_report.py evaluate.py scripts/evaluate_maniskill_conflict_fix.sh scripts/evaluate_maniskill_pilot.sh tests/unit/test_conflict_fix_report.py tests/regression/test_conflict_fix_evaluation_selection.py
git commit -m "feat: close conflict-fix evaluation and retention gates"
```

## Task 8: Complete Verification, Push, And Start World Pretraining

**Files:**
- Modify only if a verification command exposes a defect covered by Tasks 1-7.
- Generated runtime outputs remain under ignored `artifacts/`.

**Interfaces:**
- Consumes all prior tasks.
- Produces a clean pushed implementation commit and a monitored four-H20 world-pretraining process on port 1010.

- [ ] **Step 1: Run focused conflict-fix suite**

```bash
source scripts/runtime_env.sh
"$UCF_PYTHON_BIN_RESOLVED" -m unittest \
  tests.unit.test_conflict_fix_config \
  tests.unit.test_optimizer_parameter_groups \
  tests.unit.test_gradient_conflict \
  tests.integration.test_distributed_gradient_conflict \
  tests.unit.test_trainer_gradient_modes \
  tests.unit.test_conflict_fix_gates \
  tests.unit.test_wandb_tracker \
  tests.integration.test_conflict_fix_orchestration \
  tests.unit.test_conflict_fix_report \
  tests.regression.test_conflict_fix_evaluation_selection -v
```

Expected: all pass, no skips or warnings.

- [ ] **Step 2: Run complete source and runtime verification**

```bash
"$UCF_PYTHON_BIN_RESOLVED" -m unittest discover -s tests -p 'test_*.py' -v
"$UCF_PYTHON_BIN_RESOLVED" -m compileall -q corrective_foresight train.py evaluate.py scripts
for file in scripts/*.sh; do bash -n "$file"; done
scripts/verify_environment.sh
scripts/verify_maniskill_conflict_fix.sh configs/pilots/maniskill_conflict_fix_v2.yaml
```

Expected: zero exit status, no skips, no tolerated warnings, no generated tracked files, and a successful four-rank ordinary/PCGrad save-resume comparison.

- [ ] **Step 3: Verify W&B and storage without exposing secrets**

```bash
"$UCF_PYTHON_BIN_RESOLVED" -c 'import wandb; api = wandb.Api(timeout=10); print(api.viewer)'
df -h /mnt/workspace
test "$(find artifacts/runs artifacts/checkpoints artifacts/evaluations -mindepth 1 -maxdepth 1 | wc -l)" -eq 0
```

Expected: authenticated viewer, at least 100 GB free, and clean conflict-fix output roots before launch.

- [ ] **Step 4: Review, commit any verification-only fix, and push**

```bash
git status --short
git diff --check
git log --oneline --decorate -10
git push origin feature/maniskill-pilot-training
git status -sb
```

Expected: clean branch synchronized with origin. Never launch from a dirty or unpushed commit.

- [ ] **Step 5: Start only the world-pretraining phase**

```bash
nohup env UCF_CONFLICT_FIX_PHASE=world_pretrain \
  scripts/launch_maniskill_conflict_fix.sh configs/pilots/maniskill_conflict_fix_v2.yaml \
  > artifacts/runs/maniskill_conflict_fix_v2-world-launch.log 2>&1 &
echo $! > artifacts/runs/maniskill_conflict_fix_v2-world-launch.pid
```

Expected within the initial observation window:

```text
four torchrun workers on GPUs 0,1,2,3
W&B world_pretrain run initialized once by rank 0
world_pretrain-000000 checkpoint published atomically
finite optimizer-step metrics advancing
```

- [ ] **Step 6: Monitor through the first validation and checkpoint boundary**

Check process liveness, W&B/local step agreement, GPU utilization/memory, finite metrics, validation at steps 250/500/750/1000, and the step-1,000 checkpoint manifest hashes. Do not start audit until world step 5,000 passes the `+0.05` copy-last target or the user explicitly approves a marginal positive result.

- [ ] **Step 7: Record launch state**

Report the W&B run URL, PID, output root, current optimizer step, latest validation metrics, disk free space, and next automatic boundary. Do not claim the full experiment complete while world pretraining is running.
