# ManiSkill Stable Pilot Training Design

Last updated: 2026-07-18

Status: approved for implementation

## 1. Goal

Run a controlled four-GPU ManiSkill `PickCube-v1` pilot that proves both:

1. numerical and operational stability, including exact distributed resume;
2. initial closed-loop control improvement over the untrained policy.

The pilot uses the existing production 768-wide, 12-layer Unified Corrective
Foresight model, pinned DINOv3 and CLIP artifacts, and the converted LeRobot
v3 dataset. It does not change the model architecture, objective whitelist,
backbone, action schema, or primary H=8/E=1 control protocol.

## 2. Non-Goals

- No SOTA or benchmark claim.
- No PushT, fallback dataset, alternative backbone, or silent batch reduction.
- No tuning on the evaluation split.
- No automatic deletion of checkpoints.
- No eight-GPU scaling change during this pilot.

## 3. Fixed Data And Resources

Dataset: `maniskill.pick_cube.panda_wristcam.v1` in LeRobot v3 format.

- 1,013 successful episodes and 50,360 frames total.
- 912 train, 51 validation, and 50 evaluation episodes.
- Base and wrist RGB cameras, 25-dimensional proprioception, and 7-dimensional
  physical Panda `pd_ee_delta_pose` actions at 20 Hz.
- One task string: `Pick the red cube and place it at the green goal.`

The pilot uses GPUs 0-3 on the replacement `1005` node. The effective global
batch size remains 64 in both stages:

- world pretraining: 4 samples/rank x 4 ranks x accumulation 4;
- unified training: 2 samples/rank x 4 ranks x accumulation 8.

The shared filesystem currently has approximately 568 GB free. A trained
checkpoint is approximately 1.1 GB before distributed runtime payloads, so the
pilot checkpoint schedule is expected to consume less than 20 GB.

## 4. Pilot Schedule

The pilot is a strict two-stage run:

```text
world_pretrain: 1,000 optimizer steps
        -> strict policy/EMA warm-start
unified:        2,000 optimizer steps
```

Both stages save an atomic checkpoint every 250 optimizer steps and run a
fixed validation window every 100 optimizer steps. Pilot-specific experiment
YAMLs own these values; the existing 100k/200k production YAMLs remain
unchanged.

All outputs live under:

```text
artifacts/runs/maniskill_pilot_20260718/
  world_pretrain/
  unified/
  checkpoints/
  validation/
  evaluations/
  metrics/
```

No pilot path may overwrite an existing checkpoint, record, video, metric log,
or production run directory.

## 5. Distributed Checkpoint Contract

The current v1 checkpoint remains readable for single-process smoke artifacts.
Four-GPU training introduces checkpoint format v2 with coordinated atomic
publication.

Shared payloads are written once by rank 0:

- online policy and EMA target weights;
- optimizer, scheduler, scaler, and trainer state;
- immutable model, loss, dataset, action, vocabulary, and provenance metadata.

Every rank writes one rank-specific runtime payload containing:

- rank/world-size identity;
- dataset mixer generator and sampler state;
- Python, NumPy, Torch CPU, local CUDA, and flow-generator RNG state.

Rank 0 creates the temporary checkpoint directory and broadcasts its physical
path. All ranks write unique files, synchronize, and report success. Rank 0
hashes every shared and rank-specific file, writes the canonical manifest,
fsyncs the directory, and atomically renames it into place. Any rank failure
prevents publication and removes the unpublished temporary directory.

Resume requires the same world size and exact runtime contract. Every rank
loads the shared state plus its own rank payload. A mandatory integration gate
compares the next dataset choice, sample indices, flow noise, optimized losses,
and parameters against an uninterrupted four-rank run.

## 6. Validation Contract

Validation uses only the declared validation split. At each interval, every
rank evaluates eight deterministic validation batches with:

- `policy.eval()` and `torch.no_grad()`;
- fixed validation sampler state and flow seed stream;
- no optimizer, scheduler, scaler, EMA, training mixer, or training RNG update;
- distributed metric reduction after all ranks finish.

The fixed-window result is a reproducible trend estimator rather than a full
dataset benchmark. It records each optimized term and all required diagnostic
aliases. The world stage emphasizes dynamics, copy-last improvement, and state
statistics. The unified stage reports dynamics, inverse NLL, action cycle,
policy flow, gradient norms, and the complete diagnostic surface separately;
the total loss is not used alone because cycle warmup changes during the run.

Validation writes canonical JSONL records containing checkpoint step, dataset
and spec hashes, number of ranks/batches/samples, fixed seeds, and finite
metrics. Existing files are never appended by a different run identity.

## 7. Closed-Loop Effect Evaluation

Closed-loop comparison uses the identical real RGB ManiSkill runner at:

```text
untrained baseline
unified step 1,000
unified step 2,000
```

Each checkpoint runs the same ten environment seeds and derived per-step flow
seed streams, with H=8, E=1, temporal ensemble off, midpoint/10 integration,
and at most 200 environment steps. The runner continues to execute chunk index
0 only and never reranks actions using consistency or uncertainty.

The report includes success count/rate, paired reward changes, episode length,
consistency, inverse uncertainty, action bounds, solver intervals, NFE, record
hashes, and video hashes. World-pretrain checkpoints are not treated as policy
results because the policy-flow head is not optimized in that stage.

## 8. Pass And Stop Conditions

The stability gate passes only when:

- every optimized loss, metric, gradient, and pre-clip norm is finite;
- no required reduction has an empty mask;
- checkpoint v2 reproduces the exact next four-rank step in a fresh process;
- world validation dynamics improves from its initial fixed-window value;
- unified per-loss validation trends do not show sustained divergence;
- every executed physical action remains within its ActionSpec bounds.

The effect gate passes only when unified step 2,000 has a strictly larger
success count than the untrained policy on the same ten seeds. A reward
increase with equal success count is reported as stable but insufficient
control evidence.

Training stops immediately on a nonfinite value, OOM, DDP contract mismatch,
checkpoint/hash/resume failure, dataset/spec drift, action-bound violation, or
three consecutive validation windows in which at least two optimized terms
are more than 20 percent above their best prior finite validation value.
There is no silent retry with different data, batch size, precision, stage,
backbone, or objective.

## 9. Implementation Boundaries

The implementation keeps responsibilities separate:

- a distributed checkpoint coordinator owns format v2 publication and resume;
- a validation runner owns deterministic no-update validation;
- pilot experiment YAMLs own schedule and output paths;
- the training controller triggers intervals only at optimizer boundaries;
- the existing closed-loop evaluator owns simulator records and videos.

The normal single-process checkpoint API and existing evaluation runner remain
compatible. Generated datasets, checkpoints, metrics, and videos remain
ignored by Git.

## 10. Verification Sequence

Implementation follows RED/GREEN TDD:

1. Unit tests freeze format v2 fields, hashes, rank ownership, and failure
   cleanup.
2. Validation tests prove deterministic batches and no mutation of training
   state.
3. Controller tests prove exact 100/250 optimizer-boundary triggers.
4. A two-rank test exercises coordinated save and resume cheaply.
5. A four-rank gate runs two steps, saves, resumes in a fresh process, and
   matches two uninterrupted steps.
6. The complete test, compile, shell, source-boundary, CUDA, DINO, Vulkan, and
   disk preflight gates run before pilot launch.
7. The implementation commit is pushed to GitHub before the long-running pilot
   starts.

The pilot launches only after every gate exits zero with no skips, tolerated
warnings, NaNs, symbolic links, or generated source artifacts.
