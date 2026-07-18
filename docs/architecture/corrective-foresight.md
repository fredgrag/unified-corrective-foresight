# Corrective Foresight Architecture

## Core Idea

Corrective Foresight reframes the current video-action prototype as a robot-control world model:

```text
Past observations/actions
        ↓
Causal temporal video-action model
        ↓
Shared future latent
        ↓
Flow matching decoders
   ├─ future action prediction
   └─ future visual prediction
        ↓
Self-correction consistency
```

## Attention Policy

The recommended main method uses:

- **Causal temporal prediction**: future steps cannot attend to future ground-truth tokens.
- **Bidirectional spatial fusion**: patches from the same frame can attend to each other.

This is the compromise that keeps the model aligned with causal robot control while preserving strong visual representation inside each frame.

## Current Code State

Implemented foundations:

- `causal_mask.py` builds causal temporal masks.
- `causal_video_action.py` implements a tested causal temporal video-action backbone.
- `corrective_foresight_policy.py` wraps the backbone in a token-first trainable policy.
- `self_correction.py` exposes a shape-safe `SelfCorrectionModule`.
- `mar_con_unified.py` now receives self-correction config correctly and uses the new module for consistency losses.

Still next-stage:

- Raw-image/VAE dataset integration for the causal policy path.
- Benchmark-scale training and ablation results.

## Phase-2 Causal Backbone

`CausalVideoActionModel` consumes one feature vector per timestep:

```text
visual_tokens: [B, T, Dv]
action_tokens: [B, T, Da] optional
proprio_tokens: [B, T, Dp] optional
```

It outputs:

```text
temporal_features: [B, T, H]
action_pred: [B, T, Da]
visual_pred: [B, T, Dv]
```

The leakage-prevention test changes the final timestep input and checks that earlier timestep features stay unchanged. That gives the backbone a concrete causality contract before it is wired into training.

## Phase-3 Token-First Policy

`CorrectiveForesightPolicy` consumes precomputed tokens:

```text
visual_tokens: [B, T, Dv]
target_visual_tokens: [B, T, Dv]
action: [B, T, Da]
```

It computes:

```text
action_loss = MSE(action_pred, action)
visual_loss = MSE(visual_pred, target_visual_tokens)
optional self-correction losses
```

This gives the project a trainable causal policy layer without requiring VAE weights or real datasets during unit testing.

## Losses

Base losses:

```text
L_video = flow matching loss for future visual latents
L_action = flow matching loss for future actions
```

Self-correction:

```text
L_v2a = inverse consistency: predicted visual future -> future action
L_a2v = forward consistency: predicted future action -> future visual features
```

Total joint loss:

```text
L = L_video + L_action + lambda_v2a * L_v2a + lambda_a2v * L_a2v
```

Self-correction is controlled by:

```yaml
enable_self_correction: true
lambda_v2a: 0.1
lambda_a2v: 0.1
correction_warmup_steps: 5000
```
