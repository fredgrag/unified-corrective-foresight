# Optimization Summary

Date: 2026-07-13

## What Changed

The project was refocused from a broad video-action prototype into a Corrective Foresight research codebase.

### Added

- `unified_video_action/model/autoregressive/causal_mask.py`
  - Temporal causal mask helpers for future transformer integration.
- `unified_video_action/model/autoregressive/causal_video_action.py`
  - Tested causal temporal video-action backbone.
- `unified_video_action/policy/corrective_foresight_policy.py`
  - Token-first trainable Corrective Foresight policy wrapper.
- `SelfCorrectionModule` in `self_correction.py`
  - Shape-safe bidirectional consistency losses.
- `unified_video_action/eval/checkpoint_discovery.py`
  - `.ckpt` discovery for Hydra output directories.
- `unified_video_action/config/model/corrective_foresight_backbone.yaml`
  - Phase-2 causal backbone model config.
- `unified_video_action/config/model/corrective_foresight_policy.yaml`
  - Workspace-compatible phase-3 policy config.
- `unified_video_action/config/corrective_foresight_pusht.yaml`
  - PushT entry config for the phase-2 causal backbone.
- `tests/`
  - Unit tests for causal masks, causal backbone leakage prevention, token-first policy behavior, self-correction shapes, config propagation, MAR action grouping, and checkpoint discovery.
- Organized docs under `docs/`.

### Fixed

- Self-correction config now actually reaches MAR.
- The old inline self-correction block no longer assumes `target` is 2D.
- Action horizon is explicitly grouped to visual timesteps before correction losses.
- Evaluation script no longer defaults to stale `.pth` checkpoint paths.
- W&B prediction video path now matches the generated `.mp4` path.

### Rewritten

- `README.md`
  - Removed unverified speed and success-rate claims.
  - Added honest implementation status and recommended experiment protocol.

### Removed In Cleanup

- `local_delivery/corrective-foresight-2026-07-13/`
  - Removed duplicate source/document snapshot package.
- `docs/superpowers/`
  - Removed process-only specs and implementation plans from the maintained project docs.
- `Corrective_Foresight_Architecture.md`
  - Removed duplicate architecture draft in favor of `docs/architecture/corrective-foresight.md`.
- `.DS_Store`
  - Removed macOS metadata files.

## Key Modified Files

| File | Purpose |
| --- | --- |
| `README.md` | Research-facing project overview |
| `eval_corrective_foresight.sh` | Checkpoint discovery path update |
| `unified_video_action/model/autoregressive/causal_mask.py` | New causal mask helpers |
| `unified_video_action/model/autoregressive/causal_video_action.py` | New causal temporal backbone |
| `unified_video_action/model/autoregressive/self_correction.py` | Shape-safe correction module |
| `unified_video_action/policy/corrective_foresight_policy.py` | Token-first causal policy |
| `unified_video_action/model/autoregressive/mar_con_unified.py` | Config + self-correction integration |
| `unified_video_action/policy/unified_video_action_policy.py` | Hydra config propagation |
| `unified_video_action/eval/checkpoint_discovery.py` | New checkpoint discovery utility |
| `unified_video_action/eval/eval.py` | Generated video path fix |
| `unified_video_action/config/model/corrective_foresight_backbone.yaml` | Causal backbone config |
| `unified_video_action/config/model/corrective_foresight_policy.yaml` | Causal policy config |
| `unified_video_action/config/corrective_foresight_pusht.yaml` | PushT causal backbone entry config |
| `tests/*.py` | Lightweight verification tests |

## Verification

Latest local checks:

```text
python -m unittest discover -s tests -v
Ran 22 tests
OK
```

```text
python -m py_compile ...
exit code 0
```

```text
bash -n train_corrective_foresight.sh eval_corrective_foresight.sh check_env.sh
exit code 0
```

## Known Limits

- This pass implements the standalone causal transformer backbone and token-first policy; raw-image/VAE dataset integration is still next-stage.
- No benchmark-scale PushT/LIBERO result has been generated in this pass.
- The project folder is not a git repository, so no commit was created.
