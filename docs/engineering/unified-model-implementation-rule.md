# Unified Model Implementation Rule

Last updated: 2026-07-17

## Purpose

This document is the reviewable engineering checklist for the new Unified
Corrective Foresight project. `AGENTS.md` is the enforceable root rule; this
document explains the acceptance criteria behind it. The task-by-task
execution contract is
`docs/implementation/2026-07-16-unified-corrective-foresight-core-implementation-plan.md`.

## 1. Repository Boundary

The new implementation lives at:

```text
/mnt/workspace/wwl/corrective-foresight/unified_corrective_foresight/
```

The existing source tree is a retained baseline. New main-method code must not
be added to it. No symbolic link may connect the new subproject to the old
source tree. Reused local source must be copied, reviewed, and owned inside the
new subproject.

The copy operation must exclude runtime data and generated artifacts:

```text
.venv
data
datasets
outputs
checkpoints
logs
wandb
eval_results
artifacts
__pycache__
._*
```

## 2. Required Project Layout

```text
unified_corrective_foresight/
├── README.md
├── pyproject.toml
├── train.py
├── evaluate.py
├── scripts/
├── configs/
│   ├── datasets/
│   ├── actions/
│   ├── model/
│   └── experiments/
├── corrective_foresight/
│   ├── data/
│   ├── conditioning/
│   ├── model/
│   ├── policy/
│   ├── training/
│   └── evaluation/
└── tests/
    ├── unit/
    ├── integration/
    └── regression/
```

Files are split by responsibility. Dataset conversion does not belong in the
model. Causal masks do not belong in the policy wrapper. Individual losses do
not belong in the training loop.

## 3. Persistent Data Standard

LeRobotDataset v3.0 is the only on-disk and Hub data standard. The initial
dependency pin is `lerobot==0.5.1`. A version change requires:

1. An explicit dependency update.
2. A migration note.
3. Dataset metadata compatibility tests.
4. Window, padding, and video alignment tests against the new version.

The project must not depend on the current official multi-dataset training
factory. Use a project-owned `BalancedLeRobotMixer` that wraps independent
`LeRobotDataset` instances.

Each dataset has a versioned `DatasetSpec` that defines:

- Hub repository or local root and revision.
- Camera feature mapping and roles.
- Proprio feature mapping and units.
- Task and language mapping.
- FPS and supported timestamp alignment.
- `ActionSpec` identifier.
- Train/validation/evaluation episode filters.
- Per-dataset sampling weight.

The runtime adapter must produce masks instead of silently deleting or
inventing observations. Boundary-clamped LeRobot windows are marked invalid by
the corresponding `*_is_pad` values.

## 4. Heterogeneous Action Spaces

An `ActionSpec` defines:

```text
dimension
names
units
joint or end-effector space
absolute, relative, or delta mode
rotation representation
arm count
gripper indices
control mode
normalization statistics
```

The shared Transformer operates in hidden space, but action I/O remains
grounded in each physical action schema:

```text
normalized raw action
  -> per-ActionSpec input projection
  -> shared world-action Transformer
  -> per-ActionSpec flow or inverse head
  -> normalized raw action
  -> per-ActionSpec denormalization
```

This avoids padding unrelated action semantics into one vector and avoids an
untrained universal action decoder. One training batch contains one
`ActionSpec`. In distributed training all ranks use the same selected dataset
and action schema for a step.

## 5. Causality And Leakage

The canonical semantic sequence is:

```text
[condition, state, action, state-change, state, action, ...]
```

It is implemented with label-safe query views:

```text
forward: [condition, state, demonstrated-action, DELTA_QUERY]
inverse: [condition, state, supplied-delta, ACTION_QUERY]
cycle:   [condition, state, predicted-delta, ACTION_QUERY]
policy:  [condition, observed-state-history, POLICY_QUERY...]
```

The implementation must prove all of these properties with a Transformer of
at least two layers:

- Future state changes cannot alter past forward predictions.
- Target deltas cannot alter their own forward prediction.
- Demonstrated actions cannot alter the same-step inverse prediction because
  they are absent from that view.
- Future observations or actions cannot alter policy outputs.
- Condition tokens cannot become an indirect label channel.
- Padded or missing tokens cannot contribute as attention keys.

## 6. State Representation

The frozen DINOv3 patch grid is adaptively pooled to a deterministic 2-by-4
anchor grid. A trainable spatial resampler predicts residual updates to those
eight anchor tokens. One projected proprio token, or an explicit null token,
completes the state block.

The target state block uses an EMA copy of the resampler and proprio
projection. The DINO backbone remains frozen in both paths. All target tensors
are stop-gradient. The deterministic anchor, LayerNorm, EMA target, and target
stop-gradient provide state stability without a fifth state-alignment loss.

The backbone remains the canonical
`facebook/dinov3-vitb16-pretrain-lvd1689m` Hugging Face revision
`5931719e67bbdb9737e363e781fb0c67687896bc`. Its exact bytes are delivered by
ModelScope commit `23d0280ae6ee4ced592a3459674ad027d3c18906` and stored physically under the
isolated project's ignored `artifacts/models/` tree. The downloader verifies
the tracked byte sizes and SHA256 values before atomic promotion. Runtime
loading re-verifies the snapshot and uses `local_files_only=True`.

Changing the source to ViT-L/16, using `master`, accepting a corrupt or
symlinked artifact, or falling back to a different hub violates the state
contract. The real CUDA integration test cannot skip this local artifact gate.

## 7. Default Loss Contract

Only four losses are optimized in the main method:

```text
L = L_dynamics + L_inverse + 0.1 L_action_cycle + L_policy_flow
```

### Dynamics

`L_dynamics` recursively predicts target state blocks at horizons 1, 2, 4,
and 8 with demonstrated actions. It uses masked Smooth L1 loss. Horizon
weights are `[1.0, 0.8, 0.64, 0.512]` and are normalized by their sum.

### Inverse

`L_inverse` uses target state change and predicts the mean and log variance of
the normalized real action. Log variance is clamped to `[-10, 2]`. The loss is
masked diagonal Gaussian NLL averaged over valid action dimensions and valid
transitions.

### Action Cycle

`L_action_cycle` applies masked Smooth L1 between the demonstrated action and
the inverse mean recovered from the predicted state change. Before inverse
decoding, normalized predicted deltas receive training-only Gaussian noise
with standard deviation `0.01` and feature dropout probability `0.05`. This
reduces hidden-channel shortcuts. The cycle weight warms from 0 to 0.1 over
5,000 optimizer steps.

### Policy Flow

`L_policy_flow` uses rectified flow in each `ActionSpec`'s normalized real
action space:

```text
epsilon ~ Normal(0, I)
x_t = (1 - t) epsilon + t action
target_velocity = action - epsilon
```

The masked velocity MSE is averaged only across valid action steps and action
dimensions.

### Metrics And Ablations

The following do not enter the default optimized loss:

```text
delta cycle
policy-forward loss
deterministic action loss
cosine loss
variance loss
state-alignment loss
copy-last baseline
token and delta standard deviations
```

They remain metrics or named ablations. A nonzero default weight for any of
them violates the main-method contract.

### Required Diagnostics

The maintained research design requires the following logging surface:

```text
train_loss
val_loss
dynamics_loss
visual_loss
visual_mse_loss
visual_delta_loss
visual_cosine_loss
policy_flow_loss
action_loss
action_horizon_loss
inverse_action_loss
inverse_action_mse
self_correction_cycle_loss
copy_last_mse
improvement_vs_copy_last
pred_token_std
target_token_std
pred_delta_std
target_delta_std
```

`visual_loss` is a compatibility alias for the optimized dynamics loss.
`action_loss` is a compatibility alias for the optimized policy flow loss.
`inverse_action_loss` is the optimized inverse NLL. The corresponding MSE is
logged separately as `inverse_action_mse`.

`action_horizon_loss` is a diagnostic masked MSE from the flow clean-action
estimate:

```text
predicted_clean_action = x_t + (1 - t) predicted_velocity
```

It is not added to the objective. No compatibility alias may cause a loss to
be optimized twice.

## 8. Training Stages

`world_pretrain` uses demonstrated actions and optimizes `L_dynamics` only.
It initializes the same token layout and Transformer used by the main method.

`unified` optimizes all four default losses. It may initialize matching
weights from a new-project `world_pretrain` checkpoint. Legacy checkpoints may
load only explicitly matched visual weights and must report every missing and
unexpected key.

Closed-loop evaluation is not represented as another training stage.

## 9. Control Protocol

The main evaluation protocol is:

```text
observe
-> produce an eight-action flow chunk
-> execute one action
-> receive a new observation
-> replan
```

Temporal ensemble is off in the primary result and may be enabled only as a
reported ablation. Evaluation records include flow solver, integration grid,
neural function evaluations, environment seed, flow-noise seed, and action
execution horizon.

Forward-inverse consistency and inverse uncertainty are diagnostics. They do
not rank or change actions without a separately specified task-value model.

## 10. Failure Policy

Raise an error rather than silently continue when:

- A dataset revision or feature schema does not match its `DatasetSpec`.
- An action dimension or name does not match its `ActionSpec`.
- Timestamp offsets are not compatible with the dataset FPS.
- Required camera, state, task, mask, or normalization metadata is absent.
- A loss has zero valid elements without the caller explicitly allowing an
  empty batch.
- A tensor, gradient, or logged optimized loss is non-finite.
- A checkpoint silently drops model parameters.
- Distributed ranks select different dataset adapters for one optimizer step.

## 11. Verification Gates

Implementation proceeds through test-driven changes. Before a completion
claim, the project must pass:

```bash
python -m unittest discover -s tests -v
python -m compileall -q corrective_foresight tests
bash -n scripts/*.sh
```

Additional required evidence:

- RED/GREEN causality and gradient-routing tests.
- Dataset adapter tests on real LeRobot metadata and a small local fixture.
- Tiny-batch overfit checks for world pretraining and unified training.
- Checkpoint save/resume equivalence including mixer state.
- Closed-loop smoke rollouts before benchmark-scale runs.
- At least three training seeds and confidence intervals for paper claims.

## 12. Benchmark Scope

PushT is not part of the new project. The intended progression is:

```text
ManiSkill3 development and contact-rich smoke tasks
LIBERO compatibility and comparison
RoboCasa365 primary multi-task/generalization benchmark
RoboTwin 2.0 later bimanual and embodiment stress test
```

Scores from different suites, dataset sizes, action representations, initial
state distributions, or execution horizons must not be compared as one shared
leaderboard.
