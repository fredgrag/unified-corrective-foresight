# Robotics A-Conference Experiment Plan

## Target Venues

The project direction is best aligned with robotics and embodied AI venues such as CoRL, RSS, ICRA, and IROS.

## Core Claim

Causal temporal video-action prediction plus low-latency flow matching and self-correction improves closed-loop robot control while preserving fast inference.

## Required Comparisons

| Variant | Temporal Policy | Decoder | Self-Correction |
| --- | --- | --- | --- |
| Legacy diffusion baseline | bidirectional masked | diffusion | no |
| Bidirectional flow baseline | bidirectional masked | flow matching | no |
| Causal flow baseline | causal | flow matching | no |
| Corrective Foresight | causal | flow matching | yes |

## Metrics

- PushT closed-loop reward or success.
- LIBERO-10 success rate.
- Future visual quality with FVD or similar rollout metric.
- Action L2 as a diagnostic metric.
- Inference latency for 1, 2, 4, and 8 flow steps.

## Minimum Ablations

1. Flow steps: `1`, `2`, `4`, `8`.
2. Self-correction weights: off, `lambda_v2a` only, `lambda_a2v` only, both.
3. Warmup schedule: no warmup vs configured warmup.
4. Temporal policy: bidirectional masked vs causal.
5. PushT replanning horizon: execute `1`, `2`, `4`, or all predicted actions
   before taking a new observation.

## PushT Closed-Loop Replanning Sweep

The policy can predict an `n_action_steps` chunk while the runner executes a
shorter prefix before replanning. This is controlled by
`task.env_runner.execution_horizon`.

- `execution_horizon: null` keeps the original behavior and executes the full
  predicted chunk.
- `execution_horizon: 1`, `2`, or `4` executes a shorter prefix and replans more
  often.

This sweep is an evaluation-time control experiment. It does not change the
trained checkpoint, the action prediction target, or the training loss. For
PushT, shorter execution horizons may improve final T-block alignment because
the policy receives more frequent visual feedback near contact and rotation
correction.

## Evidence Standard

Only report numbers that come from saved logs and reproducible scripts. Avoid README-level claims that are not backed by experiment outputs.

## Next Engineering Milestones

1. Wire raw-image/VAE datasets into the token-first causal policy.
2. Add latency benchmark script.
3. Run PushT debug training and rollout.
4. Run full PushT ablation.
5. Run LIBERO-10 small-scale validation.
