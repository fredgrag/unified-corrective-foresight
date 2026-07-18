# Unified Corrective Foresight Implementation Rules

## Scope

These rules apply to the entire `corrective-foresight` project. They are
mandatory for all work on the new unified model.

## Sources Of Truth

Use this precedence order:

1. The user's latest explicit instruction.
2. This `AGENTS.md` rule.
3. `docs/research/2026-07-16-unified-corrective-foresight-final-design.md`.
4. `docs/research/2026-07-17-modelscope-dinov3-source-amendment.md`.
5. `docs/research/2026-07-16-unified-corrective-foresight-model-design.md`.
6. `docs/implementation/2026-07-16-unified-corrective-foresight-core-implementation-plan.md`.
7. `docs/README.md` and the remaining maintained documentation.

The final design records user-approved refinements to the original research
design. Do not silently fall back to the original minimal-upgrade route.

## Project Isolation

- Put all new executable project files under `unified_corrective_foresight/`.
- Do not modify the existing model implementation to host the new main method.
- Do not create symbolic links.
- When local source code must be reused, copy it into the new subproject and
  maintain the copy there. Do not import source from the parent legacy tree.
- External packages are dependencies, not copied project source. Pin their
  versions.
- Do not copy caches, datasets, checkpoints, logs, `.venv`, `wandb`,
  `__pycache__`, or AppleDouble `._*` files into the new subproject.
- PushT is excluded from the new project. The parent project remains the
  historical PushT implementation.

## Model Contract

- Implement a real token-level causal world-action model. State, action,
  delta, condition, and query tokens must remain distinct.
- Use query-based forward, inverse, cycle, and policy views that share one
  Transformer. Never place a target label where its prediction query can
  attend to it.
- Condition prefix tokens must never attend to trajectory tokens.
- State tokens must never attend to same-step or future action/delta labels.
- Forward delta queries may attend to current state and demonstrated action,
  but not target delta or future state.
- Inverse action queries may attend to current state and supplied delta, but
  the same-step demonstrated action must not exist in that view.
- Policy queries may attend to conditions and observed state history only.
- Apply the same causal contract at every Transformer layer. Prove it with
  multi-layer perturbation tests.

## State And Action Contracts

- Use frozen DINOv3 patch features, eight spatial state tokens from an
  anchored resampler, and one proprio or null-proprio token per timestep.
- Keep the canonical DINOv3 ViT-B/16 Hugging Face identity, but deliver the
  exact equivalent bytes from ModelScope commit
  `23d0280ae6ee4ced592a3459674ad027d3c18906`. Verify the tracked size/SHA256
  manifest and load locally with `local_files_only=True`; do not use ViT-L/16,
  an unpinned branch, a symlink, or a fallback hub.
- Generate state targets with an EMA target resampler and stop gradients at
  every target.
- Use one `ActionSpec` per physical action representation.
- Use per-`ActionSpec` action input projections, flow heads, and inverse heads.
  Do not use an ungrounded universal action autoencoder.
- Keep action normalization, dimensions, units, control mode, rotation
  representation, gripper indices, embodiment, and frequency explicit.

## Data Contract

- LeRobotDataset v3.0 is the only persistent dataset format for the new
  project. Pin `lerobot==0.5.1` until a reviewed migration changes the pin.
- Do not use the current official `MultiLeRobotDataset` as the training mixer.
  It drops non-common features and does not correctly handle heterogeneous
  robots, statistics, or FPS.
- Instantiate one LeRobot dataset per repository, adapt it with a versioned
  `DatasetSpec`, and mix complete same-schema batches with the project mixer.
- Never silently discard cameras, proprio fields, actions, tasks, or masks.
- Every runtime batch must include observation, action, transition, camera,
  and proprio validity masks as applicable.
- Any unsupported action schema, timestamp alignment, or missing required
  metadata must fail closed with an actionable error.

## Default Objective

The main method has exactly four optimized objectives:

```text
L = 1.0 L_dynamics
  + 1.0 L_inverse
  + 0.1 L_action_cycle
  + 1.0 L_policy_flow
```

- `L_dynamics` combines masked horizons 1, 2, 4, and 8. Do not double-count
  horizon 1 in another optimized loss.
- `L_inverse` is masked Gaussian NLL on real normalized actions, with bounded
  log variance. Report inverse MSE as a metric only.
- `L_action_cycle` is the differentiable
  `action -> predicted delta -> recovered action` constraint. Warm its weight
  linearly from 0 to 0.1 over 5,000 optimizer steps.
- `L_policy_flow` is masked rectified-flow velocity matching over action
  chunks.
- Delta cycle, policy-forward, deterministic action, cosine, variance,
  copy-last, and state-alignment quantities are metrics or explicit ablations.
  Their default optimized weight is zero.

Keep the original research diagnostics even when they are not optimized:

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

Metric compatibility aliases must be documented and must not create duplicate
optimized terms.

## Training And Evaluation

- `world_pretrain` uses demonstrated actions and optimizes only
  `L_dynamics`.
- `unified` is the main training stage and optimizes the four default losses.
- Closed-loop evaluation is not a training stage.
- The primary control protocol uses action horizon 8, execution horizon 1,
  and no temporal ensemble. Temporal ensemble is an explicit ablation.
- Report flow solver and neural function evaluations. Do not report solver
  intervals as if they were function evaluations.
- Save environment seeds, flow-noise seeds, dataset revisions, `DatasetSpec`
  hashes, `ActionSpec` hashes, normalization statistics, condition vocabulary,
  and mixer state with checkpoints or evaluation records.
- Do not claim SOTA, executability, generality, or improvement from training
  losses alone. Closed-loop rollouts, multiple seeds, uncertainty, and a
  matched protocol are required.

## Engineering Process

- Use test-driven development for every behavior change: failing test first,
  observe the expected failure, minimal implementation, then refactor while
  green.
- Do not write production model code before its failing test exists.
- Keep units small and interfaces explicit. Avoid a single policy file that
  owns data adaptation, token layout, model layers, all losses, and inference.
- Run focused tests after each change and the complete verification suite
  before any completion claim.
- Treat skipped tests, warnings, NaNs, missing metrics, non-finite gradients,
  silent checkpoint mismatches, and unused dataset fields as failures unless
  explicitly documented and approved.
