# Unified Corrective Foresight Final Design

Last updated: 2026-07-18

Status: design approved and implementation verified; the core implementation plan is maintained at
`docs/implementation/2026-07-16-unified-corrective-foresight-core-implementation-plan.md`.
Tasks 1 through 15 are implemented. Benchmark training and comparative claims
remain future experimental work. The approved
ModelScope delivery amendment is maintained at
`docs/research/2026-07-17-modelscope-dinov3-source-amendment.md`.

## 1. Goal

Build a causal world-action model for robot control that jointly learns:

```text
action-conditioned future state prediction
inverse action recovery from state change
forward-inverse action consistency
distributional action-chunk control
```

The central claim is limited and testable:

> Predicted future state changes are grounded by real transition targets and
> constrained to remain decodable into demonstrated executable actions.

This is forward-inverse self-correction during training. It is not a claim that
the policy performs task-value planning or automatically repairs an action at
runtime.

## 2. Non-Goals

The first implementation does not claim:

- A general-purpose foundation model.
- Cross-embodiment joint training before heterogeneous adapters are evaluated.
- RGB video generation from DINO state tokens.
- Online task-value planning or candidate-action ranking.
- SOTA based on a single seed or unmatched benchmark protocol.

## 3. Isolated Project

All new executable files live under:

```text
/mnt/workspace/wwl/corrective-foresight/unified_corrective_foresight/
```

The project is self-contained at the source level. It does not import local
source files from the parent legacy tree and uses no symbolic links. External
libraries are pinned dependencies. Datasets remain outside source control and
are addressed by explicit local roots or Hub revisions.

The intended layout is:

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

## 4. Data Architecture

### 4.1 Persistent Format

LeRobotDataset v3.0 is the only persistent format. The initial implementation
pins `lerobot==0.5.1` rather than following the upstream main branch.

LeRobot provides:

- Parquet state, action, timestamp, task, and episode data.
- Per-camera MP4 streams.
- Feature schemas, statistics, FPS, and episode metadata.
- Timestamp-relative observation/action windows.
- Episode-boundary padding indicators.
- Local, Hub, and streaming access.

### 4.2 DatasetSpec

Every dataset has a versioned `DatasetSpec` that maps its LeRobot features into
the runtime contract. It records repository/revision, camera roles, proprio
fields, task language, FPS, action schema, sampling weight, and episode split.

The initial target set is:

```text
ManiSkill3: development and fast contact-rich simulation
LIBERO: compatibility and standard comparison
RoboCasa365: primary multi-task and generalization benchmark
RoboTwin 2.0: later bimanual and embodiment extension
```

PushT is intentionally absent from the new project.

### 4.3 Runtime Batch

The in-memory contract is not a second storage format. It is the collated,
validated model input:

```text
TrajectoryBatch
├── rgb                    [B, T, V, C, H, W]
├── camera_mask            [B, T, V]
├── proprio                [B, T, Dp]
├── proprio_mask           [B, T, Dp]
├── action                 [B, A, Da]
├── observation_valid_mask [B, T]
├── action_valid_mask      [B, A]
├── transition_valid_mask  [B, T-1]
├── delta_time             [B, T-1]
├── task_text
└── condition identifiers
```

The adapter derives validity from LeRobot `*_is_pad` outputs and feature
presence. Missing optional modalities use explicit null tokens and false
masks. Missing required modalities fail closed.

### 4.4 Multi-Dataset Mixing

The official current `MultiLeRobotDataset` is not used because it keeps only
intersection features, assumes compatible FPS, and aggregates statistics
across robots. The project creates one independent LeRobot dataset and
processor per repository.

`BalancedLeRobotMixer` selects one dataset for an entire batch. Distributed
ranks share the same selected dataset ID for each optimizer step. The mixer
supports dataset/task weights and persists its RNG and sampler state.

## 5. Heterogeneous Action Architecture

Each `ActionSpec` fixes the physical action semantics:

```text
dimension and named dimensions
units
joint-space or end-effector-space
absolute, relative, or delta mode
rotation representation
arm count and gripper indices
control mode and frequency
normalization statistics
```

The model shares the world-action Transformer, not raw action dimensions.
Each action schema owns:

- An action input projection from normalized real actions to hidden tokens.
- A flow noisy-action projection and velocity output head.
- An inverse mean/log-variance output head.
- Normalization and executable-action conversion.

All three adapters are directly grounded by the four optimized objectives.
There is no universal action autoencoder and no adapter reconstruction loss.

## 6. Condition Representation

The condition prefix contains separate tokens for:

```text
dataset
task
embodiment
ActionSpec
control mode
language or null-language
```

Discrete IDs use an explicit vocabulary stored in the checkpoint. Unknown IDs
map to a declared `UNK` entry rather than a hash collision. LIBERO language is
encoded once per task with a frozen CLIP text encoder and cached. Datasets
without language use a learned null-language token.

Camera/view embeddings identify visual sources. Continuous `delta_time`
embeddings condition dynamics on the physical time interval. Fixed-FPS
LeRobot data uses the exact dataset FPS. Truly asynchronous source data must be
explicitly resampled during conversion and cannot masquerade as synchronized
data.

## 7. World-State Encoder

The frozen backbone is the canonical DINOv3 ViT-B/16 revision with hidden
width 768, patch size 16, and 4 register tokens. Content-equivalent bytes are
downloaded from ModelScope commit
`23d0280ae6ee4ced592a3459674ad027d3c18906`, verified against the tracked
size/SHA256 manifest, stored as physical project artifacts, and loaded locally
with `local_files_only=True`. The original Hugging Face revision remains in
checkpoint provenance. No ViT-L/16 or fallback weight path is allowed.

For every camera and timestep:

1. Frozen DINOv3 produces a patch grid.
2. The grid is adaptively pooled to a deterministic 2-by-4 anchor grid.
3. A trainable cross-attention resampler produces residual updates to the
   eight anchors.
4. Camera/view embeddings are added before cross-view fusion.
5. A projected proprio token, or learned null-proprio token, is appended.
6. LayerNorm produces the state block.

The online encoder produces model inputs. An EMA copy of the trainable
resampler and proprio projection produces state targets. Both paths share the
same frozen DINO backbone and deterministic anchors. Target tensors are always
stop-gradient.

The world state at timestep `t` is therefore a block rather than one pooled
vector:

```text
z_t = [spatial_1, ..., spatial_8, proprio]
```

The state-change block is:

```text
delta_z_{t+1} = z_target_{t+1} - stop_grad(z_target_t)
```

## 8. Shared Token Model

The model uses one causal Transformer with:

- Hidden dimension 768.
- Token-type embeddings.
- Modality embeddings.
- Timestep embeddings.
- Camera/view embeddings in the state encoder.
- Block-causal attention supporting bidirectional same-frame state fusion and
  causal temporal prediction.

The canonical semantics are:

```text
[condition, z_0, a_0, delta_z_1, z_1, a_1, delta_z_2, ...]
```

Target-safe query views implement those semantics.

### 8.1 Forward View

```text
[condition, z_t, demonstrated a_t, DELTA_QUERY]
    -> predicted delta_z_{t+1}
```

The delta query can read conditions, history, current state, and current
action. Target delta and future state are absent.

### 8.2 Direct Inverse View

```text
[condition, z_t, target delta_z_{t+1}, ACTION_QUERY]
    -> inverse action mean and log variance
```

The demonstrated same-step action is absent, preventing label copying.

### 8.3 Action-Cycle View

```text
[condition, z_t, predicted delta_z_{t+1}, ACTION_QUERY]
    -> recovered action
```

Training-only normalized delta noise and feature dropout reduce the capacity
for forward/inverse hidden communication.

### 8.4 Policy View

```text
[condition, observed state history, POLICY_QUERY_0, ..., POLICY_QUERY_7]
    -> flow velocity for an eight-action chunk
```

Future state, future action, and demonstrated target actions are absent from
the policy context.

All views share the Transformer, state encoder, embeddings, and relevant
heads. Compatible view batches may be concatenated along the batch dimension
for efficient execution.

## 9. Default Objective

The main method optimizes exactly four losses:

```text
L = 1.0 L_dynamics
  + 1.0 L_inverse
  + 0.1 L_action_cycle
  + 1.0 L_policy_flow
```

### 9.1 Dynamics Loss

Using demonstrated future actions, the model recursively predicts state blocks
at horizons 1, 2, 4, and 8. Masked Smooth L1 errors use normalized weights:

```text
raw horizon weights = [1.0, 0.8, 0.64, 0.512]
normalized weight_h = raw_h / sum(raw weights)
```

Horizon 1 is not optimized anywhere else. Delta MSE, next-state MSE, cosine,
copy-last improvement, and state/delta statistics are reported as metrics.

### 9.2 Inverse Loss

The direct inverse view predicts diagonal Gaussian parameters in normalized
real action space. Log variance is clamped to `[-10, 2]`. Masked Gaussian NLL
is averaged over valid transitions and valid action dimensions. Inverse action
MSE is a metric, not a second optimized loss.

### 9.3 Action-Cycle Loss

The recovered inverse mean from predicted delta is compared with the
demonstrated normalized action using masked Smooth L1. Its weight increases
linearly from 0 to 0.1 during the first 5,000 optimizer steps of unified
training.

### 9.4 Policy Flow Loss

For normalized action chunk `a`:

```text
epsilon ~ Normal(0, I)
t ~ Uniform(0, 1)
x_t = (1 - t) epsilon + t a
v_target = a - epsilon
L_policy_flow = masked MSE(v_theta(x_t, t, context), v_target)
```

Padded action positions and invalid dimensions do not contribute.

### 9.5 Non-Default Objectives

Delta cycle, policy-forward, deterministic action, state alignment, cosine,
and variance objectives have default optimized weight zero. They may be added
only as named ablations. This keeps the main method attributable and avoids
redundant or conflicting gradients.

### 9.6 Required Metrics

The final implementation preserves the original research diagnostics:

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

Compatibility names do not add optimized terms:

- `visual_loss` aliases `dynamics_loss`.
- `action_loss` aliases `policy_flow_loss`.
- `inverse_action_loss` is inverse Gaussian NLL.
- `self_correction_cycle_loss` is the optimized action-cycle loss.
- `action_horizon_loss` is masked MSE of the flow clean-action estimate
  `x_t + (1 - t) predicted_velocity` against the normalized target chunk.
- Visual MSE/delta/cosine, copy-last improvement, and token/delta statistics
  are metrics computed from the same predictions and targets.

The aliases are logged once and never summed back into the objective.

## 10. Training

### 10.1 World Pretraining

`world_pretrain` uses real demonstrated actions and optimizes only
`L_dynamics`. It uses the final state encoder, action projections, token layout,
Transformer, and delta head, so its checkpoint is structurally compatible
with unified training.

### 10.2 Unified Training

`unified` optimizes all four default losses. It may warm-start from a
new-project world-pretraining checkpoint. Legacy Stage 1 checkpoints may load
only explicitly matching visual parameters. Missing and unexpected keys are
logged and thresholded; silent broad `strict=False` loading is forbidden.

The default objective uses fixed weights. Per-loss gradient norms on shared
parameters are logged. PCGrad or adaptive weighting is an ablation activated
only after measured conflict.

### 10.3 Numerical Rules

- Every optimized reduction is mask-aware and divides by valid element count.
- An unexpectedly empty optimized reduction raises an error.
- Non-finite tensors, losses, or gradients stop training and identify the
  originating objective and dataset.
- AMP reductions use float32 accumulation.
- Gradient clipping and its measured pre-clip norm are logged.

## 11. Inference And Closed-Loop Control

The primary policy uses action horizon 8 and execution horizon 1:

```text
observation
-> online state encoder and conditions
-> rectified-flow action chunk
-> execute first action
-> receive new observation
-> replan
```

The default solver is selected in configuration and reported with its neural
function evaluations. Solver intervals and function evaluations are never
conflated. Flow randomness is reproducible from an evaluation seed stream.

Temporal ensemble is disabled in the primary result and tested separately.
Forward-inverse consistency and inverse uncertainty are logged but do not
change actions because the method has no task-value model.

## 12. Checkpoint Contract

Every resumable checkpoint stores:

- Online model and EMA state target weights.
- Optimizer, scheduler, scaler, epoch, global step, and cycle warmup state.
- Dataset mixer RNG and sampler state.
- Condition vocabulary.
- DatasetSpec and ActionSpec content plus hashes.
- Per-dataset/per-robot normalization statistics.
- LeRobot version and dataset revisions.
- Model and loss configuration.

Evaluation records additionally store environment seeds, flow-noise seeds,
solver, integration grid, function evaluations, execution horizon, and
temporal-ensemble setting.

## 13. Error Handling

The implementation fails closed on schema mismatches, unknown action spaces,
unsupported timestamps, missing required metadata, inconsistent distributed
dataset choices, empty masks, non-finite values, and unsafe checkpoint loads.

Optional language or proprio is represented by explicit null tokens. This is
not treated as a failure when the `DatasetSpec` declares the modality absent.

## 14. Testing Strategy

Every production behavior follows RED/GREEN TDD.

### 14.1 Unit Tests

- State-block shapes, deterministic anchors, EMA updates, and stop-gradient.
- Condition vocabulary, unknown IDs, language/null-language, and camera masks.
- ActionSpec normalization and per-schema forward/inverse/flow heads.
- LeRobot window-to-mask conversion.
- Balanced dataset sampling and deterministic resume.
- Masked loss reductions, empty masks, and finite log-variance bounds.
- Rectified-flow interpolation, velocity targets, and padding masks.
- Cycle gradient reaches forward and inverse paths but not EMA targets.

### 14.2 Causality Tests

With at least two Transformer layers:

- Perturb future states and confirm earlier forward predictions are unchanged.
- Perturb target deltas and confirm their forward predictions are unchanged.
- Perturb demonstrated actions outside the inverse view and confirm inverse
  output is unchanged.
- Perturb future observations/actions and confirm policy outputs are unchanged.
- Confirm condition tokens cannot relay trajectory labels across layers.

### 14.3 Integration Tests

- Load a small real LeRobot v3 fixture with video, task, state, action, and
  padding metadata.
- Run one world-pretraining optimizer step and one unified optimizer step.
- Overfit a tiny transition fixture and demonstrate improvement over copy-last.
- Save and resume a checkpoint, reproducing the next dataset selection and
  loss within numerical tolerance.
- Run a closed-loop simulator smoke episode before benchmark training.

### 14.4 Verification Commands

```bash
python -m unittest discover -s tests -v
python -m compileall -q corrective_foresight tests
bash -n scripts/*.sh
```

Benchmark claims additionally require matched protocols, at least three
training seeds, confidence intervals, failure analysis, and saved rollout
artifacts.

## 15. Evaluation Plan

The implementation order is:

```text
ManiSkill3 adapter and closed-loop development
LIBERO adapter and protocol comparison
RoboCasa365 multi-task/generalization experiments
RoboTwin 2.0 bimanual ActionSpec extension
```

Required ablations include:

- Policy flow without dynamics/inverse/cycle.
- Dynamics plus policy without inverse/cycle.
- Direct inverse without action cycle.
- Full four-loss method.
- Single-step versus multi-horizon dynamics.
- Flow neural function evaluations.
- Execution horizon 1, 2, 4, and 8.
- Temporal ensemble off versus on.
- Optional delta-cycle and policy-forward losses, each isolated from the main
  method.

## 16. Claim Boundary

A lower forward, inverse, cycle, or policy loss does not establish better
control. The primary evidence is closed-loop task success under matched data,
environment, action, and execution protocols.

The phrase `action-executable future` means that a grounded predicted state
change is recoverable through inverse dynamics and is supported by rollout
evidence. It does not mean that cycle consistency alone proves physical
executability.

## 17. Final Design Audit

The final design resolves the known structural risks:

- Query views remove direct and multi-layer label leakage paths.
- EMA anchored state blocks reduce target drift without an extra optimized
  alignment loss.
- Real target delta inverse training, cycle noise/dropout, and a strong
  dynamics target reduce hidden-channel cycle shortcuts.
- Per-ActionSpec heads ground heterogeneous actions without an extra action
  autoencoder loss.
- LeRobot padding indicators become attention and loss masks.
- The four-loss objective removes duplicate state/delta errors and optional
  cycles from the default method.
- Receding-horizon control closes the environment feedback loop without
  overstating runtime self-correction.

The remaining risks are empirical optimization, simulator integration,
dataset quality, compute scale, and benchmark variance. They require tests and
rollouts; they cannot be eliminated by adding more architectural objectives.
