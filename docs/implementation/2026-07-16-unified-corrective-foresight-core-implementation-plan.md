# Unified Corrective Foresight Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` only when delegation is explicitly
> authorized; otherwise use `superpowers:executing-plans`. Every production
> behavior follows RED/GREEN TDD. Steps use checkbox (`- [ ]`) syntax for
> tracking.

Last updated: 2026-07-18

Status: approved design implemented and verified; Tasks 1 through 15 are
complete. Benchmark training and SOTA claims remain intentionally pending.

**Goal:** Build the full token-level Unified Corrective Foresight model, its
LeRobot v3 data system, resumable training, and ManiSkill3 closed-loop path in
an isolated new project.

**Architecture:** Frozen DINOv3 patch features become an anchored 8+1 state
block with EMA targets. One 12-layer causal Transformer serves label-safe
forward, inverse, cycle, and rectified-flow policy views through per-ActionSpec
adapters and exactly four optimized objectives.

**Tech Stack:** Python 3.12, PyTorch 2.8/cu128, LeRobot 0.5.1, Transformers
5.3.0, DINOv3 ViT-B/16, ManiSkill 3.0.1, safetensors, unittest, four NVIDIA
H20 GPUs.

## Global Constraints

- All executable source is below `unified_corrective_foresight/`.
- No source symlinks, parent-source imports, copied runtime data, or legacy
  environment reuse.
- LeRobotDataset v3.0 is the sole persistent dataset format.
- Official `MultiLeRobotDataset` is forbidden; one same-ActionSpec dataset is
  selected per batch and synchronized across DDP ranks.
- State/action/delta/condition/query tokens remain distinct and share one
  Transformer under the same attention contract at every layer.
- The main method optimizes only dynamics, inverse NLL, action cycle, and
  policy flow with the approved weights/schedule.
- Unit-scale injected components do not satisfy real DINOv3, real video,
  full-size CUDA, four-rank DDP, or RGB simulator acceptance gates.
- Training/evaluation artifacts live outside source and never enter Git.
- No SOTA/executability claim is allowed without matched closed-loop,
  multi-seed evidence.

---

## Deliverable

Create a source-isolated project at:

```text
/mnt/workspace/wwl/corrective-foresight/unified_corrective_foresight/
```

The deliverable is the complete core method, not a reduced proxy:

- LeRobotDataset v3.0 input and a same-schema balanced dataset mixer.
- Frozen DINOv3 patch features, deterministic 2-by-4 anchors, trainable
  residual resampling, and EMA state targets.
- Distinct condition, state, action, delta, and query tokens processed by one
  shared causal Transformer.
- Forward, direct-inverse, action-cycle, and rectified-flow policy views.
- Exactly four default optimized losses.
- Resumable four-GPU distributed training.
- ManiSkill3-to-LeRobot conversion and RGB closed-loop evaluation.

Benchmark-scale training and claims are separate experiments. Core completion
requires real data, real DINOv3 weights, and simulator rollouts, but does not
authorize a SOTA claim.

## Sources Of Truth

Use this precedence order:

1. The user's latest explicit instruction.
2. `/mnt/workspace/wwl/corrective-foresight/AGENTS.md`.
3. `docs/research/2026-07-16-unified-corrective-foresight-final-design.md`.
4. `docs/research/2026-07-17-modelscope-dinov3-source-amendment.md`.
5. `docs/research/2026-07-16-unified-corrective-foresight-model-design.md`.
6. `docs/engineering/unified-model-implementation-rule.md`.
7. This plan.

If this plan disagrees with a higher-priority source, update the plan before
changing production code.

## Frozen Production Defaults

```text
Python                         3.12
PyTorch                        2.8.0, CUDA 12.8 wheel
torchvision                    0.23.0
torchcodec                     0.7.0
LeRobot                        0.5.1
ManiSkill                      3.0.1
DINOv3 model                   facebook/dinov3-vitb16-pretrain-lvd1689m
DINOv3 canonical revision      5931719e67bbdb9737e363e781fb0c67687896bc
DINOv3 ModelScope revision     23d0280ae6ee4ced592a3459674ad027d3c18906
DINOv3 weight SHA256           9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b
CLIP model                     openai/clip-vit-base-patch32
CLIP revision                  3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
state block                    8 spatial + 1 proprio/null token
shared Transformer             hidden 768, 12 layers, 12 heads, MLP ratio 4
action horizon                 8
execution horizon              1
flow solver                    midpoint, 10 intervals, 20 NFE
temporal ensemble              false
```

DINOv3 retains its canonical Hugging Face identity, but production bytes are
delivered from the content-equivalent pinned ModelScope commit. A tracked
manifest verifies exact sizes and SHA256 values before local-only Transformers
loading. Unit tests inject a deterministic patch backbone only to isolate
project logic; they never replace the real ModelScope snapshot CUDA gate.

The Vulkan loader and pinned NVIDIA ICD are installed without replacing the
driver. `vulkaninfo` enumerates all four H20 GPUs, and the real ManiSkill RGB
gate passes with base and wrist camera observations. State-only simulation
remains an unacceptable substitute.

## Project Layout

```text
unified_corrective_foresight/
├── README.md
├── pyproject.toml
├── train.py
├── evaluate.py
├── requirements/
├── scripts/
├── configs/
│   ├── datasets/
│   ├── actions/
│   ├── model/
│   └── experiments/
├── corrective_foresight/
│   ├── config/
│   ├── data/
│   ├── conditioning/
│   ├── model/
│   ├── policy/
│   ├── training/
│   └── evaluation/
└── tests/
    ├── unit/
    ├── integration/
    ├── regression/
    └── fixtures/
```

The new project owns a nested Git repository. It does not import parent Python
source, contain symlinks, or copy parent runtime artifacts. Its environment is
outside source at `/mnt/workspace/wwl/.venvs/unified-corrective-foresight` and
is created with `venv --copies`.

## Stable Runtime Contracts

Implement immutable, versioned `ActionSpec` and `DatasetSpec` dataclasses.
`ActionSpec` contains dimension, names, units, joint/end-effector space,
absolute/relative/delta mode, rotation representation, arm/gripper structure,
control mode, frequency, and mean/std normalization. `DatasetSpec` contains
Hub/local location, immutable revision, FPS, camera roles, proprio fields,
task/language fields, embodiment ID, action feature/spec ID, splits, and
sampling weight.
Both serialize to sorted canonical JSON and expose SHA-256.

`TrajectoryBatch` is:

```python
@dataclass
class TrajectoryBatch:
    rgb: Tensor                    # [B,T,V,3,H,W]
    camera_mask: Tensor            # [B,T,V], bool
    proprio: Tensor                # [B,T,Dp]
    proprio_mask: Tensor           # [B,T,Dp], bool
    action: Tensor                 # [B,T-1,Da]
    action_dimension_mask: Tensor  # [B,T-1,Da], bool
    observation_valid_mask: Tensor # [B,T], bool
    action_valid_mask: Tensor      # [B,T-1], bool
    transition_valid_mask: Tensor  # [B,T-1], bool
    delta_time: Tensor             # [B,T-1]
    context_index: int
    task_text: tuple[str | None, ...]
    condition_ids: Mapping[str, Tensor]
    dataset_id: str
    action_spec_id: str
```

The window contains observed history plus at least eight future transitions.
Policy context ends at `context_index`; its target is the next eight actions.
Dynamics/inverse use all valid origins. Horizon `h` is valid only when every
transition in its recursive chain is valid.

State is `[B,T,9,768]`: eight fused spatial tokens plus one proprio/null token.
A `DELTA_QUERY` is a nine-token block. Each inverse transition has one
`ACTION_QUERY`; policy has eight `POLICY_QUERY` tokens.

## Tasks

All paths in task file lists are relative to
`/mnt/workspace/wwl/corrective-foresight/unified_corrective_foresight/`.
For every task, run its focused command immediately after the RED checkbox and
before production edits. Expected RED is the named missing symbol or behavioral
assertion, never a syntax/import typo. Rerun the same command after the GREEN
implementation and require `OK` with no skip or warning before refactoring and
committing.

### Task 1: Isolated Repository And Environment

**Files:** `.gitignore`, `README.md`, `pyproject.toml`,
`requirements/torch-cu128.txt`, `requirements/lock-cu128.txt`,
`scripts/bootstrap_environment.sh`, `scripts/verify_environment.sh`, package
`__init__.py` files, `tests/regression/test_project_boundary.py`, and
`tests/integration/test_environment_contract.py`.

**Interfaces:** Consumes system Python 3.12, four H20 GPUs, FFmpeg, and the
parent rules. Produces the importable `corrective_foresight` package, external
venv path, exact dependency lock, and nested Git repository used by every task.

- [x] RED: boundary test walks without following links and rejects symlinks,
   `._*`, runtime data/artifact directories, parent-source imports, and import
   origins outside the new project.
- [x] Create regular-file project tree and run
   `git init -b implementation/unified-core`; implementation never starts on
   `main`.
- [x] Pin `torch==2.8.0`, `torchvision==0.23.0`, and `torchcodec==0.7.0` from the
   cu128 index. Pin Python `>=3.12,<3.13`, `lerobot==0.5.1`,
   `transformers==5.3.0`, `mani_skill==3.0.1` in the sim extra,
   `numpy==2.2.6`, `safetensors==0.7.0`, and `PyYAML==6.0.3`. The resolved
   lock pins every transitive dependency.
- [x] Bootstrap with `python3 -m venv --copies` at the external path, install the
   CUDA wheel set first, install editable project, and record the resolved
   lock. Never touch the legacy `.venv`.
- [x] GREEN: assert exact versions, four CUDA devices, FFmpeg, LeRobot v3 API,
   and project import origin.

```bash
bash scripts/bootstrap_environment.sh
bash scripts/verify_environment.sh
python -m unittest tests.regression.test_project_boundary \
  tests.integration.test_environment_contract -v
git add .
git commit -m "build: initialize isolated unified foresight project"
```

### Task 2: Strict Schemas And Runtime Batch

**Files:** `corrective_foresight/config/schema.py`, `config/loader.py`,
`data/batch.py`, and unit tests for ActionSpec, DatasetSpec, and
TrajectoryBatch.

**Interfaces:** Consumes no production API. Produces `ActionSpec`,
`DatasetSpec`, `TrajectoryBatch`, `load_action_spec(path)`, and
`load_dataset_spec(path)` with canonical `content_hash` properties.

- [x] RED: test tuple lengths, positive dimensions/FPS/std, unique names, gripper
   indices, mutually exclusive Hub/local location, required revision, camera
   roles, disjoint splits, unknown YAML fields, and tensor shape/dtype rules.
- [x] Implement frozen dataclasses, canonical hashes, strict YAML loader,
   normalize/denormalize, and reject std below `1e-6` rather than clamping.
- [x] Validate `T >= context_index + 9`, action length `T-1`, one dataset/spec per
   batch, finite masked values, and that a valid transition implies both
   endpoint observations plus its action are valid.
- [x] Add `.to()`, `.pin_memory()`, and deterministic time-window selection.

```bash
python -m unittest tests.unit.test_action_spec tests.unit.test_dataset_spec \
  tests.unit.test_trajectory_batch -v
git add corrective_foresight/config corrective_foresight/data tests/unit
git commit -m "feat: define strict dataset action and batch contracts"
```

### Task 3: LeRobot v3 Adapter And Real Video Fixture

**Files:** `data/lerobot_adapter.py`, `data/collate.py`,
`tests/fixtures/create_lerobot_v3_fixture.py`, `test_lerobot_masks.py`, and
`tests/integration/test_lerobot_v3_fixture.py`.

**Interfaces:** Consumes `DatasetSpec`, `ActionSpec`, and `TrajectoryBatch`.
Produces `LeRobotTrajectoryAdapter(spec, action_spec)`,
`validate_lerobot_metadata(metadata)`, and
`collate_trajectory_samples(samples) -> TrajectoryBatch`.

- [x] RED: table-test start/end padding, missing optional proprio, missing
   required camera, unsupported timestamp offsets, and action mismatch.
- [x] Generate two temporary LeRobot v3 episodes through public 0.5.1 APIs with
   two real MP4 streams, proprio, 7-D action, timestamps, task, Parquet, stats,
   and metadata. Commit no generated video.
- [x] Instantiate one LeRobotDataset per DatasetSpec. Derive exact
   `delta_timestamps` from FPS. Validate metadata before sampling.
- [x] Map every `*_is_pad` field to masks. Clamped values stay in tensors but have
   false masks; validity is never inferred from repeated values.
- [x] Convert uint8/HWC to float32/CHW without losing camera identity. Report all
   consumed keys and fail on a silently unused mapped field.

```bash
python -m unittest tests.unit.test_lerobot_masks \
  tests.integration.test_lerobot_v3_fixture -v
git add corrective_foresight/data tests
git commit -m "feat: adapt LeRobot v3 trajectories with explicit masks"
```

### Task 4: Deterministic Same-Schema Dataset Mixer

**Files:** `data/stateful_sampler.py`, `data/mixer.py`, sampler/mixer unit tests,
and `tests/integration/test_distributed_mixer.py`.

**Interfaces:** Consumes independent adapted LeRobot datasets. Produces
`StatefulDistributedBatchSampler.state_dict/load_state_dict` and
`BalancedLeRobotMixer.next_batch() -> TrajectoryBatch` plus complete mixer
state for Task 12.

- [x] RED: test weighted choice over 20,000 draws, seeded sequence, same-schema
   batches, epoch rollover, exact state resume, and invalid weights.
- [x] Give every dataset a stateful rank-sharded permutation with seed, epoch,
   and cursor. Keep dataset-choice RNG separate.
- [x] In DDP, rank zero selects and broadcasts dataset ID before any iterator
   advances. Verify dataset, ActionSpec, and batch size across ranks.
- [x] Never instantiate official `MultiLeRobotDataset`.

```bash
python -m unittest tests.unit.test_stateful_sampler \
  tests.unit.test_balanced_mixer -v
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m unittest tests.integration.test_distributed_mixer -v
git add corrective_foresight/data tests
git commit -m "feat: add resumable distributed LeRobot mixer"
```

### Task 5: Explicit Conditions And Cached Language

**Files:** `conditioning/vocabulary.py`, `language_cache.py`, `encoder.py`,
`scripts/precompute_text_embeddings.py`, and condition unit tests.

**Interfaces:** Consumes batch condition strings/IDs. Produces
`ConditionVocabulary`, `LanguageEmbeddingCache`, and
`ConditionEncoder.forward(...) -> Tensor[B,6,768]`, all with content hashes.

- [x] RED: require reserved `PAD`, `UNK`, and null-language entries. IDs come
   from sorted declarations and never Python hashes. Unknown IDs map to `UNK`
   and increment a metric.
- [x] Precompute normalized frozen CLIP embeddings at the exact revision into
   safetensors plus JSON text/model/revision/dimension/hash metadata. Training
   reads only the cache; a missing declared-language embedding fails closed.
- [x] Produce six distinct 768-D prefix tokens in fixed order: dataset, task,
   embodiment, ActionSpec, control mode, language/null-language. Do not sum
   them into one token.

```bash
python -m unittest tests.unit.test_condition_vocabulary \
  tests.unit.test_condition_encoder -v
git add corrective_foresight/conditioning scripts tests/unit
git commit -m "feat: encode explicit revisioned condition tokens"
```

### Task 6: Frozen DINOv3 State Blocks And EMA Targets

**Files:** `model/dinov3_artifact.py`, the immutable DINOv3 source manifest,
`scripts/download_dinov3.py`, `model/dinov3_backbone.py`,
`spatial_resampler.py`, `state_encoder.py`, `ema.py`, `tests/unit/fakes.py`,
artifact/state/EMA unit tests, and `tests/integration/test_real_dinov3.py`.

**Interfaces:** Consumes RGB/camera masks/proprio masks from `TrajectoryBatch`.
Produces `DinoV3PatchBackbone.forward(rgb) -> PatchGrid`,
`OnlineStateEncoder.forward(...) -> Tensor[B,T,9,768]`, and
`EMAStateTarget.encode_target(...) -> detached Tensor[B,T,9,768]`.

- [x] RED with an injected deterministic patch backbone: prove stable row-major
   2-by-4 anchor order, mask-invariance, failure when all cameras are masked,
   `[B,T,9,768]` output, online-adapter gradients, frozen-backbone no-grad, EMA
   stop-gradient, and `target = tau*old + (1-tau)*online` updates.
- [x] Pin the content-equivalent ModelScope commit and exact sizes/SHA256 values
   for config, processor, license, and weights. Download through full commit
   URLs to `.partial` files, verify, `fsync`, atomically promote, and reject
   corruption, truncation, extra/missing files, unsafe names, and symlinks.
- [x] Re-verify the physical local snapshot before Transformers construction.
   Load processor/model with `local_files_only=True` and require the exact
   `DINOv3ViTModel`, hidden 768, patch 16, 4 registers, 12 layers, and 12 heads.
   Use patch metadata to separate CLS/register tokens and assert the exact grid.
   Override training mode so the backbone always stays eval and frozen.
- [x] Per camera, adaptively pool patches to eight anchors and add camera ID.
   Mask-average valid-camera anchors for the deterministic base. Eight learned
   queries cross-attend to all valid per-camera patches and add a zero-init
   projected residual. Append projected proprio or learned null-proprio, then
   LayerNorm.
- [x] EMA-copy every trainable state-adapter parameter, including camera and null
   tokens. Share only frozen DINO and deterministic pooling. Detach targets at
   the public boundary.
- [x] Define the only dynamics target as
   `delta_target[t+1] = detach(z_ema[t+1] - z_ema[t])`. Both endpoints come
   from the EMA adapter; no online tensor enters a target.
- [x] Real gate: install the exact ModelScope snapshot, independently verify
   its weight SHA256, load it locally on CUDA, run two RGB frames, assert
   nonconstant patches and `[1,2,9,768]`, then backward and prove DINO still
   has no gradients. This test cannot skip.

```bash
python scripts/download_dinov3.py --artifact-root artifacts/models
python -m unittest tests.unit.test_spatial_resampler \
  tests.unit.test_state_encoder tests.unit.test_ema_state_target \
  tests.integration.test_real_dinov3 -v
git add corrective_foresight/model tests
git commit -m "feat: add anchored DINOv3 state encoder and EMA targets"
```

### Task 7: Label-Safe Token Views And Attention Compiler

**Files:** `model/token_types.py`, `token_views.py`,
`attention_contract.py`, `transformer.py`, `test_token_views.py`, and
`tests/regression/test_multilayer_causality.py`.

**Interfaces:** Consumes condition/state/action/delta/flow tensors. Produces
`TokenView(tokens, metadata, key_padding_mask)`, `build_forward_view`,
`build_inverse_view`, `build_cycle_view`, `build_policy_view`, and
`compile_attention_mask(metadata) -> Tensor[L,L]`.

- [x] RED: assert exact view membership:

```text
forward = condition + state + demonstrated action + 9 DELTA_QUERY;
          target delta/future state absent
inverse = condition + state + supplied target delta + ACTION_QUERY;
          demonstrated same-step action absent
cycle   = condition + state + noisy/dropout predicted delta + ACTION_QUERY
policy  = condition + observed state history + 8 POLICY_QUERY;
          future observations and demonstrated target actions absent
```

- [x] Each policy query includes its per-ActionSpec noisy flow-state projection,
   scalar flow-time embedding, and horizon position. This is a generative
   state, not a demonstrated label.
- [x] Represent each token by role, semantic time, block ID, validity, and
   condition flag. Compile one allow-matrix per view plus per-sample key
   padding. Condition queries see condition keys only; trajectory/query tokens
   see only declared prior/peer blocks; same-frame state fusion is
   bidirectional; invalid keys are invisible.
- [x] Apply the same contract in all Transformer layers.
- [x] With dropout off and at least two layers, perturb future state, target
   delta, demonstrated inverse action, future policy observation/action,
   padding, and a condition relay attempt. Prohibited perturbations leave the
   checked prediction equal within `1e-6`; allowed perturbations change it.

```bash
python -m unittest tests.unit.test_token_views \
  tests.regression.test_multilayer_causality -v
git add corrective_foresight/model tests
git commit -m "feat: enforce multilayer causal token contracts"
```

### Task 8: Per-ActionSpec Heads And Rectified Flow

**Files:** `model/action_adapters.py`, `model/flow.py`, and action/flow/solver
unit tests.

**Interfaces:** Consumes `ActionSpec`. Produces `ActionAdapterRegistry`,
`FlowTrainingSample(epsilon,t,x_t,target_velocity)`,
`sample_rectified_flow(...) -> (actions, IntegrationReport)`, and per-schema
inverse/flow/action projections.

- [x] RED: create two incompatible ActionSpecs and prove each owns a separate
   action input projection, flow-state projection/velocity head, inverse
   mean/logvar head, and normalization. Unknown or mixed schemas fail before
   Transformer execution.
- [x] Implement seeded rectified-flow training exactly:

```python
epsilon = torch.randn(..., generator=generator)
t = torch.rand(B, 1, 1, generator=generator)
x_t = (1.0 - t) * epsilon + t * action
target_velocity = action - epsilon
clean_estimate = x_t + (1.0 - t) * predicted_velocity
```

- [x] Masks affect reductions, not random draw order, so resume is exact. AMP
   reductions use float32.
- [x] Implement Euler and midpoint behind one interface. Return solver, time
   grid, intervals, NFE, and noise seed. Production midpoint has 10 intervals
   and exactly 20 NFE. Test constant/linear fields and reject inconsistent NFE.

```bash
python -m unittest tests.unit.test_action_adapters \
  tests.unit.test_rectified_flow tests.unit.test_flow_solver -v
git add corrective_foresight/model tests/unit
git commit -m "feat: add grounded action adapters and rectified flow"
```

### Task 9: Shared World-Action Transformer

**Files:** `model/world_action_transformer.py`, `model/outputs.py`,
`configs/model/unified_base.yaml`, model unit tests, and
`tests/regression/test_single_shared_transformer.py`.

**Interfaces:** Consumes Task 5-8 encodings/views/adapters. Produces
`WorldActionTransformer.predict_delta`, `.predict_inverse`, `.predict_cycle`,
and `.predict_policy_velocity` with the shapes listed below.

- [x] RED with a two-layer test config: `predict_delta` returns `[B,N,9,H]`,
   inverse mean/logvar `[B,N,Da]`, cycle does not detach predicted delta,
   policy velocity `[B,8,Da]`, logvar is in `[-10,2]`, and all four methods
   reference the same Transformer parameters.
- [x] Own one shared causal Transformer, embeddings, delta projection, and
   per-schema adapter registry. Compatible views may concatenate for speed but
   cannot create model copies.
- [x] Predict nine delta tokens and recurse as
   `z_next = z_current + predicted_delta`; never collapse state to one token.
- [x] Add distinct token-type, modality, timestep, query-position, and continuous
   `delta_time` embeddings. Encode `delta_time` with a deterministic Fourier
   basis plus MLP; tests perturb it and require dynamics output to change.
   Normalize supplied target/predicted delta blocks through the same declared
   delta LayerNorm before direct-inverse/cycle conditioning; apply cycle noise
   and dropout after this normalization.
- [x] Freeze production YAML at hidden 768, 12 layers, 12 heads, MLP ratio 4,
   dropout 0.1, state block 9, action horizon 8, gradient checkpointing, and
   bf16. Reject all configs below two layers. Small dimensions exist only in
   constructor-level unit tests.

```bash
python -m unittest tests.unit.test_world_action_transformer \
  tests.regression.test_single_shared_transformer -v
git add corrective_foresight/model configs/model tests
git commit -m "feat: assemble shared causal world action transformer"
```

### Task 10: Exactly Four Objectives And All Diagnostics

**Files:** `model/masked_reductions.py`, `objectives.py`, `metrics.py`, focused
objective tests, `tests/regression/test_default_objective_contract.py`, and
`tests/regression/test_gradient_routing.py`.

**Interfaces:** Consumes `WorldActionTransformer`, online/EMA state blocks,
normalized actions, and batch masks. Produces
`ObjectiveResult(total_loss, optimized_terms, losses, metrics)` and the exact
metric surface below.

- [x] RED: masked means divide by true elements in float32 and identify an empty
   or nonfinite objective. For dynamics, use known errors and assert normalized
   weights from `[1.0,0.8,0.64,0.512]`.
- [x] Recursively roll every valid origin with demonstrated intermediate actions
   and optimize masked Smooth L1 at horizons 1, 2, 4, 8. A horizon is valid
   only if its whole chain is valid. Horizon 1 appears once in the graph.
- [x] Implement exactly:

```text
L_dynamics     masked recursive Smooth L1
L_inverse      masked diagonal Gaussian NLL, logvar [-10,2]
L_action_cycle masked Smooth L1 of inverse mean from predicted delta
L_policy_flow  masked rectified-flow velocity MSE
```

- [x] Cycle predicted delta remains differentiable and receives training-only
   noise std 0.01 and feature dropout 0.05. Effective cycle weight is
   `0.1 * min(max(global_step,0)/5000, 1)`.
- [x] Return every required diagnostic. `visual_loss` aliases dynamics,
   `action_loss` aliases policy flow, `inverse_action_loss` is NLL, and
   `self_correction_cycle_loss` aliases action cycle. `action_horizon_loss`
   uses `clean_estimate` and never enters total loss. Visual MSE/delta/cosine,
   inverse MSE, copy-last improvement, and token/delta stds are metrics only.
   The exact logging surface is:

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
- [x] Regression whitelist must equal exactly:

```python
{"dynamics_loss", "inverse_action_loss", "action_cycle_loss",
 "policy_flow_loss"}
```

   Delta-cycle, policy-forward, deterministic action, cosine, variance, and
   state alignment are absent or zero by default; nonzero default config fails.
- [x] Prove gradients: dynamics reaches online state/action/Transformer/delta;
   inverse reaches inverse/Transformer; cycle reaches forward and inverse;
   flow reaches flow/Transformer; nothing reaches DINO or EMA targets.

```bash
python -m unittest tests.unit.test_masked_reductions \
  tests.unit.test_dynamics_objective \
  tests.unit.test_inverse_cycle_objectives \
  tests.unit.test_policy_flow_objective \
  tests.regression.test_default_objective_contract \
  tests.regression.test_gradient_routing -v
git add corrective_foresight/model tests
git commit -m "feat: implement four-loss corrective foresight objective"
```

### Task 11: Policy, Training Stages, And Numerical Guards

**Files:** `policy/unified_policy.py`, `training/stages.py`, `trainer.py`,
`distributed.py`, `numerics.py`, `train.py`, stage/numerics unit tests,
`tests/integration/test_optimizer_steps.py`, and `test_tiny_overfit.py`.

**Interfaces:** Consumes mixer batches and model/objective APIs. Produces
`UnifiedCorrectiveForesightPolicy`, `TrainingStage`, `Trainer.train_step`, and
the public training/inference methods listed below.

- [x] RED: `world_pretrain` optimizes dynamics only while using the final encoder,
   action projection, token layout, Transformer, and delta head. `unified`
   optimizes all four terms with current cycle warmup. Reject every other stage;
   closed-loop evaluation is not a stage.
- [x] Policy public methods are
   `compute_training_objective(batch, stage, global_step, generator)`,
   `predict_action_chunk(observation, action_spec_id, flow_seed, solver)`, and
   `update_ema(global_step)`. Data adaptation and loop logic stay outside.
- [x] Inference returns normalized/denormalized actions, consistency, inverse
   uncertainty, and solver report. Diagnostics never rerank actions.
- [x] Trainer uses DDP, bf16 autocast, float32 loss reduction, AdamW, configured
   scheduler, accumulation, measured pre-clip norm, and clipping. Before step,
   identify nonfinite objective/gradient with dataset/rank/step. Log per-loss
   shared-parameter gradient norms but do not enable PCGrad.
- [x] Update EMA only after a successful optimizer step. Log aliases once and
   never sum them.
- [x] Run one optimizer step for both stages using the real LeRobot fixture plus
   injected backbone, then a production 768/12-layer synthetic CUDA smoke to
   catch memory/layout failures.
- [x] Tiny-overfit deterministic transitions; require finite/decreasing dynamics
   and flow losses and next-state MSE below copy-last. This checks mechanics,
   not policy quality.

```bash
python -m unittest tests.unit.test_training_stages \
  tests.unit.test_numerical_guards \
  tests.integration.test_optimizer_steps \
  tests.integration.test_tiny_overfit -v
git add corrective_foresight/policy corrective_foresight/training train.py tests
git commit -m "feat: train unified policy with strict stages and numerics"
```

### Task 12: Atomic Checkpoint And Exact Resume

**Files:** `training/checkpoint.py`, `training/run_manifest.py`, checkpoint unit
tests, and `tests/integration/test_exact_resume.py`.

**Interfaces:** Consumes policy/trainer/mixer/spec/vocabulary/RNG state.
Produces `save_checkpoint_atomic(path, state)`,
`load_checkpoint_strict(path, expected_contract)`, and canonical `RunManifest`.

- [x] RED: require online/EMA weights, optimizer, scheduler, scaler, epoch,
   global/optimizer step, warmup, Python/NumPy/torch/CUDA/generator RNG, mixer
   and sampler state, vocabulary, spec contents/hashes, normalization,
   LeRobot/revisions, model/loss config, and Git commit/dirty state.
- [x] Removing or corrupting each field fails actionably. Legacy partial load uses
   an explicit key allowlist and thresholds; broad silent `strict=False` is
   forbidden.
- [x] Write model/EMA safetensors, trainer state, and canonical JSON manifest into
   a temporary directory; fsync, hash, then atomic rename. Never publish a
   partial checkpoint.
- [x] Compare two continuous steps against one-step-save-fresh-process-resume-one
   step. Match next dataset/sample indices, flow random draws, cycle
   noise/dropout, losses/metrics, online/EMA parameters, optimizer/scheduler,
   global step, and warmup within declared tolerance.

```bash
python -m unittest tests.unit.test_checkpoint_validation \
  tests.integration.test_exact_resume -v
git add corrective_foresight/training tests
git commit -m "feat: save complete atomic and reproducible checkpoints"
```

### Task 13: ManiSkill3 Conversion And RGB Runtime Gate

**Files:** `data/maniskill_conversion.py`,
`scripts/convert_maniskill_to_lerobot.py`, initial ActionSpec/DatasetSpec YAML,
`test_maniskill_feature_mapping.py`, and
`tests/integration/test_maniskill_lerobot_conversion.py`.

**Interfaces:** Consumes pinned ManiSkill trajectories and initial specs.
Produces `ManiSkillEpisodeReader`, `convert_to_lerobot_v3(...)`, the validated
PickCube LeRobot dataset, and its ActionSpec/DatasetSpec YAML.

- [x] Clear Vulkan first, following official ManiSkill Ubuntu instructions. Do not
   replace the installed NVIDIA driver. Create the NVIDIA ICD JSON pointing to
   the already present `libGLX_nvidia.so.0` only if the manifest is absent.

```bash
apt-get update
apt-get install -y libvulkan1 vulkan-tools libglvnd-dev
vulkaninfo --summary
python -m mani_skill.examples.demo_random_action -e PickCube-v1
```

   `vulkaninfo` must enumerate an H20 and the demo must use an RGB-capable
   renderer. CPU-only/state-only execution, segfault, or missing ICD blocks.
- [x] RED: freeze Panda `pd_ee_delta_pose` names/translation/axis-angle/gripper
   units, control frequency, and training-split normalization. Query the real
   environment action space and reject mismatch.
- [x] Record a deterministic RGB episode, write LeRobot v3, reopen through Task 3,
   and compare cameras, proprio, action, timestamps, success metadata, masks.
- [x] Download pinned official demonstrations:

```bash
python -m mani_skill.utils.download_demo PickCube-v1
```

- [x] Replay HDF5/JSON using configured RGB mode and control mode. Capture declared
   base/wrist cameras and robot proprio. Write only via LeRobot 0.5.1 APIs.
   Metadata records source hashes, versions, task, robot, modes, cameras,
   seeds, FPS, success, and converter Git commit.
- [x] Convert into a temporary root and atomically publish after validating episode
   counts, valid transitions, success distribution, action statistics, video
   decode, FPS, and split disjointness. No custom persistent format.

```bash
python scripts/convert_maniskill_to_lerobot.py \
  --config configs/datasets/maniskill_pick_cube_v1.yaml --validate-only
python -m unittest tests.unit.test_maniskill_feature_mapping \
  tests.integration.test_maniskill_lerobot_conversion -v
git add corrective_foresight/data scripts configs tests
git commit -m "feat: convert ManiSkill trajectories to LeRobot v3"
```

Commit source/config only, never datasets.

Completion evidence on 2026-07-17: strict validation reports 1,013 episodes,
50,360 frames, 49,347 valid transitions, 1,013 successes, and 80 MP4 files.
The published dataset has disjoint 912/51/50 train/validation/evaluation
episode splits. The immutable DatasetSpec hash is
`3f6383665d9dc7a3476cea649d02c8be2ae735d616f7ca7aa4d4065e9ed91a9d`, and
the ActionSpec hash is
`13c05454f557d69af8f4ceab9a40318bda1203b513d6b25ee7e1c0c10a1c1000`.

### Task 14: Receding-Horizon ManiSkill Evaluation

**Files:** `evaluation/maniskill_runner.py`, `records.py`, `video.py`,
`evaluate.py`, protocol unit tests, and
`tests/integration/test_maniskill_closed_loop.py`.

**Interfaces:** Consumes strict checkpoint, policy, specs, and ManiSkill env.
Produces `ManiSkillClosedLoopRunner.run_episode(seed, flow_seed_stream)` and
canonical `EvaluationRecord` JSON plus rollout video.

- [x] RED with fake env/policy: observe, predict 8, execute action 0 only,
   re-observe/replan each step; temporal ensemble remains off; consistency and
   uncertainty never rerank; denormalized actions obey explicit bounds; env and
   flow seeds are separate; solver intervals and NFE are distinct.
- [x] Reuse the conversion DatasetSpec for online RGB/proprio mapping. Each record
   includes checkpoint/spec hashes, task/robot/modes, environment and per-step
   flow seeds, solver/grid/NFE, E, ensemble flag, success/reward/length/reason,
   diagnostics, and rollout video/hash.
- [x] Run fixed-seed RGB episodes from a fresh checkpoint labeled
   `untrained_smoke`. Task success is not required, but complete valid-action
   episodes, replanning, no NaNs, and reproducibility are required.
- [x] Overfit the tiny converted fixture and run the identical closed-loop path to
   prove actions are model-produced. Do not report this as a benchmark result.

```bash
python evaluate.py --config configs/experiments/maniskill_unified.yaml \
  --checkpoint <checkpoint-directory> --seeds 0 1 2 --tag untrained_smoke
python -m unittest tests.unit.test_evaluation_protocol \
  tests.integration.test_maniskill_closed_loop -v
git add corrective_foresight/evaluation evaluate.py tests
git commit -m "feat: add closed loop ManiSkill policy evaluation"
```

Completion evidence on 2026-07-18: the production 768/12/12 policy was
assembled from the pinned DINOv3 and CLIP artifacts, saved as an atomic
`untrained_smoke` checkpoint, and evaluated on RGB `PickCube-v1` seeds 0, 1,
and 2 with two valid H=8/E=1 replanning steps per episode. Every record reports
midpoint/10 with 20 NFE per step, finite bounded actions, separate flow seeds,
and physical H.264 video. A second run with the same checkpoint and seeds
matched all actions, diagnostics, results, and video SHA256 values. The tiny
converted fixture also produced model-generated actions through the identical
runner path.

### Task 15: Launchers, Documentation, And Final Gates

**Files:** experiment YAML, world/unified/evaluation shell launchers, README,
`tests/regression/test_config_contract.py`, and
`tests/regression/test_no_legacy_dependency.py`.

**Interfaces:** Consumes every prior task. Produces operator-facing launchers,
validated production configs, final documentation, and the completion evidence
matrix artifacts.

- [x] RED: load every YAML and assert frozen defaults, hashes, objective whitelist,
   H=8/E=1, ensemble off, solver/NFE, dataset revisions, and external data/output
   roots. Scan imports/resolved paths for parent source; reject symlinks,
   AppleDouble, credentials, placeholders, unsupported dataset configs, and
   generated artifacts.
- [x] Launchers use the external venv and fail unless environment, real DINOv3,
   dataset, CUDA, and Vulkan gates pass. Four-GPU commands are:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
  --config configs/experiments/maniskill_world_pretrain.yaml
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
  --config configs/experiments/maniskill_unified.yaml \
  --init-checkpoint <world-pretrain-checkpoint>
```

- [x] README documents bootstrap, model access, conversion/validation, both stages,
   exact resume, evaluation, aliases, ablations, failures, artifact roots, and
   claim limits. No implicit fallback dataset/backbone/schema/stage/checkpoint.
- [x] Run from a clean shell:

```bash
python -m unittest discover -s tests -v
python -m compileall -q corrective_foresight tests train.py evaluate.py
bash -n scripts/*.sh
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m unittest tests.integration.test_distributed_mixer -v
python -m unittest tests.integration.test_real_dinov3 \
  tests.integration.test_lerobot_v3_fixture \
  tests.integration.test_optimizer_steps \
  tests.integration.test_exact_resume \
  tests.integration.test_maniskill_closed_loop -v
find . -path ./.git -prune -o -type l -print
find . -name '._*' -print
git diff --check
git status --short
```

Expected: all executed commands exit 0, no skips, no tolerated warnings/NaNs, no links or
AppleDouble, and a clean repository.
- [x] Manually inspect a decoded camera sequence, state/action trace, checkpoint
   manifest, resume comparison, and rollout video. Confirm action bounds and
   replan cadence from records.
- [x] Final commit:

```bash
git add .
git commit -m "docs: complete unified foresight workflow"
git status --short
```

Completion evidence on 2026-07-18: `196` tests passed on the replacement
`1005` node; the launcher path completed real one-step world-pretrain and
unified warm-start optimizer runs; compileall, all shell syntax checks,
Vulkan/NVIDIA H20, real DINO CUDA, source-boundary, no-link, no-AppleDouble,
and `git diff --check` gates passed. The default launchers remain four-process
commands; single-GPU validation explicitly used `UCF_NPROC_PER_NODE=1` and
`CUDA_VISIBLE_DEVICES=0` to avoid consuming unrelated GPUs.

## Completion Evidence Matrix

| Contract | Required evidence |
|---|---|
| Source isolation | boundary test, no symlinks, import-origin scan |
| LeRobot v3 only | real MP4/Parquet fixture and ManiSkill round trip |
| Heterogeneous actions | two incompatible ActionSpec tests |
| Same-schema DDP mixing | two-rank test and four-rank final smoke |
| Frozen DINOv3 | pinned ModelScope snapshot, canonical provenance, CUDA test, and no-grad proof |
| 8+1 state block | anchor-order, camera-mask, EMA tests |
| One shared Transformer | parameter-identity regression |
| No label leakage | multi-layer prohibited-source perturbations |
| Exactly four losses | whitelist and graph accounting |
| Cycle grounding | real-target inverse and noisy-cycle gradients |
| Rectified flow | interpolation, mask, clean estimate, solver/NFE tests |
| Numerical safety | empty-mask, nonfinite, clipping tests |
| Exact resume | dataset/random/loss/parameter equivalence |
| Closed control loop | H=8/E=1 RGB rollout record and video |
| Claim discipline | no benchmark claim from losses/smoke tests |

## Deferred Plans

Only after this plan is green, write separate reviewed plans for:

1. Matched multi-seed ManiSkill3 training and ablations.
2. LIBERO conversion/protocol compatibility.
3. RoboCasa365 multi-task/generalization training.
4. RoboTwin 2.0 bimanual ActionSpec extension.

They may add adapters and experiments, but cannot change the four-loss main
method, causal contract, or checkpoint schema without a new design review.
