# ManiSkill Conflict-Fix Training Design

Date: 2026-07-20

Status: user-approved design

## 1. Purpose

The next experiment must determine whether Unified Corrective Foresight can
retain a useful learned world model while improving closed-loop PickCube
control. It is not a scale experiment and it is not a multi-dataset
experiment. It keeps the maintained architecture, data contract, backbone,
and four-objective method fixed while addressing the optimization
interference measured in the first pilot.

The execution host for implementation, tests, training, and evaluation is:

```text
ssh root@8.130.172.181 -p 1010
```

The project remains at:

```text
/mnt/workspace/wwl/corrective-foresight/unified_corrective_foresight
```

## 2. Evidence From The First Pilot

The first four-GPU run completed 1,000 world-pretraining optimizer steps and
2,000 unified optimizer steps without a non-finite value, DDP failure, or
checkpoint-integrity failure. Both final format-v2 checkpoints passed full
manifest size and SHA-256 verification.

The result was operationally stable but scientifically insufficient:

- world validation dynamics loss improved from `0.39076946` at step 100 to
  `0.00440251` at step 1,000;
- world `improvement_vs_copy_last` reached only `+0.01701777`;
- unified policy-flow loss fell from `1.5500058` to `0.49370078`;
- unified inverse-action MSE fell from `0.59149123` to `0.24655337`;
- unified action-cycle loss fell from `0.20845025` to `0.09941996`;
- unified dynamics loss ended at `0.00775961`, while its best validation value
  was `0.00435444` at step 200;
- final unified `improvement_vs_copy_last` was `-0.70836073`, so the state
  prediction MSE was about 70.8 percent worse than copying the current state;
- the final 100-step mean pre-clip gradient norm was about `67.996`, against a
  configured clip limit of `1.0`;
- the final 100-step shared-gradient means were approximately `0.0713` for
  dynamics, `21.216` for inverse, `1.118` for action cycle, and `6.799` for
  policy flow;
- the unified learning rate reached only 40 percent of its configured peak,
  and the action-cycle weight reached only 40 percent of its configured final
  weight;
- no formal trained-policy closed-loop evaluation was produced.

These observations prove a severe gradient-scale imbalance and downstream
world-model degradation. They do not by themselves prove negative gradient
direction conflict, because pairwise gradient cosine was not recorded. The
new design therefore measures direction before enabling PCGrad.

## 3. Goals And Non-Goals

### 3.1 Goals

The experiment must:

1. regenerate a stronger world-pretraining checkpoint;
2. measure pairwise gradient direction on the shared Transformer;
3. protect world/shared parameters during unified optimization;
4. enable PCGrad only when a predeclared audit rule proves it is warranted;
5. complete a 5,000-step unified pilot with a scheduler that can resume
   exactly to 20,000 steps;
6. retain positive improvement over copy-last in unified validation;
7. demonstrate a meaningful success-count increase on ten matched closed-loop
   seeds before continuing to 20,000 steps;
8. record the complete run in Weights & Biases without making W&B the source
   of truth for recovery.

### 3.2 Non-Goals

This iteration does not:

- add, remove, or replace an optimized objective;
- change the fixed default objective weights;
- change the token-level causal architecture;
- change DINOv3 ViT-B/16 or unfreeze the DINOv3 backbone;
- add another dataset, task, robot, action schema, or camera contract;
- enlarge the Transformer;
- introduce temporal ensembling, action reranking, or a different primary
  closed-loop protocol;
- upload model checkpoints to W&B;
- claim SOTA from loss curves or from this single-task pilot.

## 4. Preserved Model And Objective Contract

The main objective remains exactly:

```text
L = 1.0 L_dynamics
  + 1.0 L_inverse
  + 0.1 L_action_cycle
  + 1.0 L_policy_flow
```

The model continues to use the maintained query-based forward, inverse,
cycle, and policy views over one shared causal Transformer. State, action,
delta, condition, and query tokens remain distinct. The causal visibility
rules, LeRobotDataset v3.0 storage contract, DatasetSpec and ActionSpec
contracts, action horizon 8, execution horizon 1, and no-temporal-ensemble
primary protocol remain unchanged.

DINOv3 remains frozen. The EMA state target is updated only by its existing
EMA rule and never enters the optimizer.

## 5. Parameter Ownership And Optimizer Groups

Every trainable parameter must belong to exactly one of the following groups.
Initialization fails if a parameter is missing, duplicated, or assigned by an
unreviewed name fallback.

### 5.1 Protected World/Shared Group

This group contains:

- `online_state_encoder.adapter`, including the spatial resampler, proprio
  projection, null-proprio token, and output norm;
- the complete `condition_encoder`;
- the shared causal Transformer;
- role, modality, timestep, query-position, and continuous-time embeddings;
- delta queries;
- `delta_projection` and `delta_norm`;
- each ActionAdapter `action_input_projection`, because it is part of the
  demonstrated-action-to-dynamics path.

During unified training this group uses a learning-rate multiplier of `0.1`.
With a unified action-group peak learning rate of `5e-5`, the protected peak
learning rate is therefore `5e-6`.

### 5.2 Action-Side Group

This group contains the global action-side query parameters:

- the action query used by inverse and cycle views;
- policy queries and the policy-horizon embedding.

It also contains, for each ActionSpec:

- `flow_state_projection`;
- `flow_velocity_head`;
- `inverse_mean_head`;
- `inverse_logvar_head`.

During unified training this group uses the full scheduled learning rate.

### 5.3 Frozen Or Non-Optimized State

The DINOv3 backbone, EMA target copy, cached language embeddings, dataset
state, and normalization statistics do not enter either optimizer group.

## 6. Gradient Direction Audit

The audit is an orchestration mode over the existing `unified` training stage;
it is not a third TrainingStage and does not alter the model objective.

The audit starts from the accepted world-pretrain step-5,000 checkpoint and
runs 500 optimizer steps with protected parameter groups and no PCGrad. It
validates at steps 0, 250, and 500 and records shared-Transformer gradient
diagnostics every 10 optimizer steps.

For every measurement, the implementation computes raw gradients of each of
the four objective terms with respect to the same ordered tuple of trainable
shared-Transformer parameters. Directional cosine uses these raw gradients so
the cycle warmup does not erase the direction being audited. It also derives
the effective gradients after applying the current declared objective weights,
including the current action-cycle warmup weight. It records:

- each raw and effective objective gradient norm;
- all six pairwise cosine similarities;
- the minimum cosine involving `L_dynamics`;
- whether the measurement is a conflicting step;
- the norm of the ordinary summed shared gradient.

A measurement is a conflicting step when at least one of these values is less
than `-0.05`:

```text
cos(L_dynamics, L_inverse)
cos(L_dynamics, L_action_cycle)
cos(L_dynamics, L_policy_flow)
```

The formal 5,000-step pilot enables PCGrad only when both conditions hold:

1. at least 30 percent of measurements from audit steps 251 through 500 are
   conflicting steps; and
2. audit validation dynamics loss at step 500 is more than 1.20 times the
   fixed-window world-pretrain step-5,000 validation dynamics loss.

There are 25 measurements in the decision window, so the first condition
requires at least eight conflicting measurements. The decision and all source
statistics are written to an atomic `gradient-audit-decision.json` file and
the W&B run summary.

The formal pilot always restarts from the original world-pretrain step-5,000
checkpoint. It never continues from audit-updated weights.

## 7. PCGrad Boundary And Determinism

When the audit rule is false, unified training uses the ordinary fixed-weight
sum with protected optimizer groups.

When the audit rule is true, PCGrad applies only to shared causal Transformer
parameters. It projects the effective per-objective gradients after applying
the declared weights `1.0`, `1.0`, the current cycle-warmup weight in
`[0.0, 0.1]`, and `1.0`. An objective whose effective weight is exactly zero
is excluded from projection for that optimizer step; its raw gradient must
still exist and remain finite for diagnostics. Action-side heads and all other
protected parameters receive the ordinary gradient of the fixed-weight total
objective. This restriction prevents gradient surgery from silently changing
head supervision or the declared scalar objective.

PCGrad processes the active effective objective gradients in an order derived
deterministically from the experiment seed, global optimizer step, and ordered
objective names. Rank is deliberately excluded because all ranks project the
same globally averaged gradients.
The derivation is stateless and recorded in the resolved run configuration.
Fresh-process resume at a given global step must reproduce the same order and
the same next update. Non-finite projected gradients, a missing raw shared
gradient, or a mismatch between declared and available objectives is fatal.

The implementation records ordinary and projected shared-gradient norms and
the fraction removed by projection. PCGrad is an explicit run mode selected
by the audit decision; it is never enabled or disabled silently during the
formal pilot.

### 7.1 Distributed Accumulation Semantics

Gradient direction and projection operate on the same effective batch as the
optimizer update. For every objective, shared-Transformer gradients are scaled
by the accumulation factor and accumulated across every micro-step belonging
to the optimizer step. At the optimizer boundary, each objective-gradient
tensor is all-reduced and divided by world size before cosine calculation or
projection. All ranks therefore project the same global per-objective
gradients; rank-local projection followed by averaging is forbidden.

The ordinary fixed-objective backward path remains responsible for gradients
outside the shared Transformer and for DDP synchronization. When PCGrad is
active, the trainer replaces only the synchronized shared-Transformer
gradients with the projected global gradients before global norm validation
and clipping. During audit mode, the same globally averaged accumulated
gradients feed diagnostics but do not replace the ordinary gradients.

## 8. Training Schedule

### 8.1 World Pretraining

World pretraining runs 5,000 optimizer steps from a fresh initialization:

```text
peak learning rate:       1e-4
warmup:                    500 optimizer steps
schedule:                  cosine decay through step 5,000
validation interval:      250 optimizer steps
checkpoint interval:      1,000 optimizer steps
optimized objective:      L_dynamics only
```

The fixed validation windows and full horizon-chain sampling rules from the
first pilot remain mandatory. World pretraining is accepted only when all
values are finite, the final `improvement_vs_copy_last` is positive, and the
target result is at least `+0.05`. A positive value below `+0.05` is reported
as marginal and requires explicit user approval before the audit starts.

### 8.2 Unified Audit

The 500-step audit uses:

```text
action-group peak learning rate:       5e-5
protected-group peak learning rate:    5e-6
warmup:                                500 optimizer steps
PCGrad:                                disabled
validation:                            steps 0, 250, 500
gradient direction logging:           every 10 optimizer steps
large checkpoint retention:           none after the decision is verified
```

### 8.3 Unified 5,000-Step Pilot

The formal pilot starts again from the world-pretrain step-5,000 checkpoint:

```text
action-group peak learning rate:       5e-5
protected-group peak learning rate:    5e-6
LR warmup:                             500 optimizer steps
scheduler total:                       20,000 optimizer steps
action-cycle warmup:                   5,000 optimizer steps to weight 0.1
validation interval:                   250 optimizer steps
checkpoint interval:                   1,000 optimizer steps
PCGrad:                                fixed from the audit decision
```

The scheduler, optimizer parameter groups, EMA, flow generator, mixer state,
rank-local RNG state, PCGrad seed derivation, and W&B run identifier must be
recoverable without changing the next optimizer step.

### 8.4 Continuation To 20,000 Steps

Training pauses after the verified step-5,000 checkpoint and closed-loop gate.
It continues only if every gate in Section 10 passes. Continuation uses exact
distributed resume from step 5,000, preserves the original total-step-20,000
scheduler, and resumes the same W&B run with `resume="must"`.

## 9. Automatic Stop Rules

All existing fatal conditions remain in force: non-finite values, empty
optimized masks, OOM, DDP contract mismatch, checkpoint/hash/resume failure,
dataset or spec drift, and physical action-bound violations.

Beginning with the validation at unified step 500, the pilot stops at a
validation boundary when either condition is true for three consecutive
validation windows:

1. `improvement_vs_copy_last <= 0`; or
2. unified validation dynamics loss is more than 1.20 times the accepted
   world-pretrain step-5,000 validation dynamics loss.

The maintained rule also remains active: stop after three consecutive
validation windows in which at least two optimized terms are more than 20
percent above their best prior finite validation value. For a best value that
is zero or negative, deterioration is measured as
`current > best + 0.20 * max(abs(best), 1e-8)` so the rule remains meaningful
for Gaussian NLL values below zero.

On a validation-triggered stop, all ranks coordinate an atomic final-safe
checkpoint at that boundary. Rank 0 writes `stop-report.json` containing the
reason, triggering windows, resolved configuration, Git commit, data/spec
hashes, and checkpoint manifest hash. No parameter, batch, precision, loss,
or schedule is modified automatically.

## 10. Five-Thousand-Step Gate

Continuation to 20,000 steps requires all of the following:

1. every optimized loss, diagnostic, gradient, and pre-clip norm is finite;
2. no existing or new automatic stop rule fired;
3. final unified `improvement_vs_copy_last > 0`;
4. final unified validation dynamics loss is no more than 1.20 times the
   accepted world-pretrain baseline;
5. policy-flow, inverse-action MSE, and action-cycle validation metrics do not
   show sustained divergence;
6. the step-5,000 format-v2 checkpoint passes complete size and SHA-256
   verification and a fresh four-rank next-step resume check;
7. on the same ten environment and flow-noise seeds, unified step 5,000 has at
   least two more successes than the untrained action policy initialized from
   the world-pretrain step-5,000 checkpoint;
8. the median paired per-seed reward change is positive;
9. every executed physical action is within its ActionSpec bounds;
10. W&B and local records agree on final step, checkpoint manifest hash, and
    gate decision.

Failure or an inconclusive result stops the workflow before the long run. A
reward increase without the required success-count increase is not sufficient.

## 11. Closed-Loop Evaluation

World-pretraining checkpoints are not policy results. The corrected formal
comparison is:

```text
untrained_action_from_world_5000
unified_5000
unified_20000
```

`untrained_action_from_world_5000` is the unified step-0 checkpoint created
from the accepted world checkpoint, before the action-specific policy and
inverse heads receive unified updates.

Each checkpoint uses the same ten environment seeds and deterministic derived
flow-noise streams, action horizon 8, execution horizon 1, temporal ensemble
off, midpoint integration with 10 intervals and 20 NFE, and at most 200
environment steps. Records contain success, total reward, episode length,
termination reason, consistency, inverse uncertainty, physical and normalized
actions, bounds, NFE, checkpoint manifest hash, record hash, and video hash.

The evaluator uploads a per-seed W&B table and evaluation videos. It does not
upload checkpoint payloads.

## 12. Weights & Biases Contract

The server already provides authenticated `wandb==0.24.2`. The run hierarchy
is:

```text
project: unified-corrective-foresight
group:   maniskill-pickcube-conflict-fix-v2
jobs:    world_pretrain, gradient_audit, unified_pilot, closed_loop_eval
```

Only rank 0 initializes and writes W&B. Every 10 optimizer steps, scalar
training metrics are reduced across all ranks before logging. Losses,
diagnostics, and gradient norms log the global mean. Gradient-cosine logging
includes the global mean and the fraction of ranks below the conflict
threshold. Validation metrics are already globally reduced by the validation
runner.

The W&B optimizer-step axis is explicit and shared by train, validation, and
system metrics. The run logs:

- all four losses and maintained diagnostics;
- learning rate for each optimizer group;
- per-objective raw and effective shared-gradient norms;
- all six pairwise gradient cosines and conflict rates;
- ordinary/projected combined gradient norms when PCGrad is active;
- pre-clip norm, effective clip coefficient, and clipping frequency;
- optimizer-step time, samples per second, and peak GPU memory;
- resolved experiment, pilot, model, dataset, action, and evaluation config;
- Git commit and dirty state;
- DINOv3 identity, revisions, and SHA-256;
- DatasetSpec, ActionSpec, dataset revision, normalization, condition
  vocabulary, checkpoint manifest, and audit-decision hashes;
- training, validation, environment, and flow seeds;
- GPU, driver, CUDA, PyTorch, LeRobot, and W&B versions;
- closed-loop tables, aggregate metrics, and videos.

Secrets, API keys, proxy credentials, and environment-variable values are
never copied into config or logs.

Local JSONL, validation JSON, evaluation records, and format-v2 checkpoints
remain authoritative. W&B initialization or authentication failure is a
preflight failure and prevents launch. A transient post-launch network outage
does not modify or stop optimization: rank 0 retains the local W&B queue,
marks tracking incomplete in local run metadata, and runs `wandb sync` after
training. Completion requires the local metadata to record a successful sync.

An atomic tracking metadata file under the run output root stores W&B entity,
project, group, run ID, mode, last logged optimizer step, and sync status. The
step-5,000 continuation uses this ID with `resume="must"`; it never creates a
replacement run silently.

## 13. Checkpoint And Disk Retention

Training checkpoints are published atomically with the existing format-v2
distributed contract. During a stage, each 1,000-step recovery checkpoint is
retained. Deletion occurs only after final hash verification and gate output
publication.

Long-term retention is:

- world pretraining: initial, best accepted validation, and step 5,000;
- unified audit: decision JSON and local/W&B metrics, no large checkpoint;
- unified pilot: initial, best accepted validation, step 5,000, and the newest
  recovery checkpoint until the gate completes;
- unified long run: step 5,000, best accepted validation, step 20,000, and the
  newest recovery checkpoint until final verification;
- evaluation: all ten structured records and videos for each formal policy
  checkpoint.

Checkpoint retention never deletes the only exact-resume point. W&B does not
receive model, optimizer, EMA, or rank-runtime payloads.

## 14. Components And Boundaries

Implementation keeps these responsibilities separate:

- parameter-group construction owns exhaustive, explicit parameter
  classification and learning-rate multipliers;
- gradient diagnostics owns finite per-loss gradients, norms, dot products,
  cosines, and distributed summaries;
- PCGrad owns projection of shared-Transformer gradients only;
- the trainer owns optimizer steps, scheduler steps, clipping, and exact state;
- the pilot gate owns validation history, stop decisions, and gate reports;
- W&B tracking owns rank-0 telemetry and atomic tracking metadata, but does not
  own training recovery;
- the existing checkpoint coordinator remains the sole owner of model and
  optimizer checkpoint publication;
- the existing evaluator remains the sole owner of simulator execution,
  structured episode records, and videos.

Core model modules must not import W&B. W&B integration belongs in a training
tracking module and the entrypoint/orchestrator boundary.

## 15. Test And Verification Requirements

All behavior changes follow RED/GREEN TDD. Required tests include:

1. exhaustive parameter classification with no missing or duplicated trainable
   parameters, and explicit exclusion of DINOv3 and EMA state;
2. exact action/protected learning rates through warmup and cosine scheduling;
3. exact optimizer/scheduler/EMA/RNG/mixer continuation from step 5,000 toward
   the original 20,000-step schedule;
4. gradient norm and cosine math for aligned, orthogonal, opposing, zero,
   unused, and non-finite gradients;
5. distributed reduction of gradient diagnostics;
6. exact audit threshold behavior at seven versus eight conflicting
   measurements and at the 1.20 dynamics boundary;
7. PCGrad projection only on shared Transformer parameters, current effective
   objective-weight preservation, zero-effective-weight exclusion, unchanged
   ordinary gradients on all other parameters, finite guards, and
   deterministic order;
8. full-microbatch accumulation and per-objective all-reduce before projection,
   rejection of rank-local projection, and fresh-process two-rank and
   four-rank exact-resume equivalence with PCGrad enabled and disabled;
9. rank-0-only W&B initialization, globally reduced logging, explicit step
   semantics, secret exclusion, atomic run metadata, incomplete-sync state,
   and `resume="must"` behavior using a mocked W&B boundary;
10. three-window stop rules, negative-NLL deterioration math, coordinated
    final-safe checkpointing, and stop-report content;
11. evaluation selection that accepts the untrained-action and unified policy
    checkpoints and rejects world-pretrain as a policy result;
12. checkpoint-retention behavior that cannot delete the only exact-resume
    checkpoint;
13. the existing causal, dataset, action, environment, numerical, checkpoint,
    evaluation, and project-boundary suites.

Before the long process starts, the complete test suite, compile checks, shell
syntax checks, source-boundary checks, CUDA/DINO/Vulkan/environment preflight,
W&B authentication check, disk-space check, and four-rank smoke/resume gate
must pass with no skips or tolerated warnings. The implementation commit must
be clean and pushed before training begins.

## 16. Deliverables

The work produces:

- reviewed implementation and tests for protected optimizer groups, gradient
  audit, optional shared-only PCGrad, stop/gate rules, and W&B tracking;
- resolved world, audit, unified-pilot, and evaluation configurations;
- an accepted or rejected world step-5,000 checkpoint;
- an immutable gradient-audit decision record;
- an accepted, rejected, or stopped unified step-5,000 run;
- matched ten-seed closed-loop results against the untrained action policy;
- a gate report that either authorizes exact continuation to 20,000 or records
  why continuation is forbidden;
- if authorized, an exact-resumed unified step-20,000 checkpoint and final
  matched closed-loop report.

No training or evaluation begins until the implementation plan derived from
this design is separately reviewed.
