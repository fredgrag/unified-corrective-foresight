# Protected World Freeze Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze every world-sensitive parameter and its EMA target during unified training while continuing to train action-only parameters and collect PCGrad diagnostics.

**Architecture:** Reuse the existing protected/action optimizer ownership boundary. Permit an explicit zero protected multiplier for unified stages, skip EMA updates under that contract, and keep all objective and PCGrad telemetry unchanged.

**Tech Stack:** Python 3.12, PyTorch 2.9, unittest, four-rank torchrun, W&B 0.24.2.

## Global Constraints

- Never modify or overwrite `world_pretrain-005000`.
- Keep the four optimized objective definitions and weights unchanged.
- Keep the immutable audit decision and its SHA-256 binding.
- Accept step 250 only when the fixed dynamics ratio is at most 1.20.
- Do not continue a failed disposable unified attempt.

---

### Task 1: Freeze Contract Tests

**Files:**
- Modify: `tests/unit/test_trainer_gradient_modes.py`
- Modify: `tests/unit/test_conflict_fix_config.py`

**Interfaces:**
- Consumes: `TrainerConfig.protected_lr_multiplier`, `Trainer.train_step`.
- Produces: regression coverage for zero protected LR and frozen EMA behavior.

- [ ] **Step 1: Write the failing tests**

Add a unified trainer with `protected_lr_multiplier=0.0`, snapshot every
protected and action parameter plus EMA state, run one optimizer step, and
assert protected/EMA tensors are identical while at least one action tensor
changes. Update the approved conflict-fix config assertion to expect `0.0`.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.unit.test_trainer_gradient_modes tests.unit.test_conflict_fix_config -v
```

Expected: failure because zero protected multiplier is rejected and the
approved configuration still resolves to `0.1`.

### Task 2: Minimal Freeze Implementation

**Files:**
- Modify: `corrective_foresight/training/trainer.py`
- Modify: `corrective_foresight/config/conflict_fix.py`
- Modify: `configs/pilots/maniskill_conflict_fix_v2.yaml`

**Interfaces:**
- Consumes: the existing protected optimizer group.
- Produces: zero protected AdamW LR and skipped EMA update for unified stages.

- [ ] **Step 1: Permit zero for unified only**

Change validation to require `0.0 <= protected_lr_multiplier <= 1.0`, retaining
the exact world-pretrain requirement of `1.0`.

- [ ] **Step 2: Freeze EMA with the protected group**

Guard the EMA update:

```python
if self.config.protected_lr_multiplier > 0.0:
    self.policy.update_ema(global_step)
```

- [ ] **Step 3: Freeze the approved conflict-fix config**

Set both the YAML value and exact parser contract to `0.0`.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 command and expect all tests to pass.

- [ ] **Step 5: Run focused verification**

Run:

```bash
python -m unittest \
  tests.unit.test_optimizer_parameter_groups \
  tests.unit.test_trainer_gradient_modes \
  tests.unit.test_conflict_fix_config \
  tests.integration.test_distributed_checkpoint_resume \
  tests.integration.test_conflict_fix_orchestration -v
```

Expected: all tests pass without skips.

### Task 3: Integrate and Retry

**Files:**
- Runtime only: `artifacts/runs/maniskill_conflict_fix_v2/failed_attempts/`

**Interfaces:**
- Consumes: `world_pretrain-005000`, `gradient-audit-decision.json`.
- Produces: a fresh canonical unified run with frozen protected parameters.

- [ ] **Step 1: Run the complete suite**

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 2: Commit, merge, and push**

Commit the tested change, fast-forward the main feature branch, and verify the
local and GitHub commit hashes match.

- [ ] **Step 3: Archive the failed disposable attempt**

Move the existing `unified/` directory to a timestamped physical directory
under `failed_attempts/`; refuse to proceed if the canonical path still exists.

- [ ] **Step 4: Start a fresh four-rank unified gate**

Warm-start only from `world_pretrain-005000` and bind the canonical audit
decision. Never resume the failed attempt.

- [ ] **Step 5: Enforce the step-250 gate**

Verify `learning_rate/protected == 0.0`, `pcgrad/enabled == 1.0`, and fixed
dynamics ratio `<= 1.20`. Stop and retain evidence on failure; continue the
5,000-step gate on success.

