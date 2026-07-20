# Protected World Freeze Retry Design

## Context

The 500-step ordinary-gradient audit measured 25 conflicts out of 25 and a
final dynamics ratio of 41.2241 relative to the accepted world checkpoint.
Shared-transformer PCGrad reduced the step-250 fixed-validation dynamics ratio
only from 4.906 to 4.673. This is insufficient because world-sensitive modules
outside the shared transformer still receive action-objective updates, and the
action gradients exceed the dynamics gradients by several orders of magnitude.

## Goal

Preserve the accepted `world_pretrain-005000` dynamics exactly enough to keep
every fixed-validation dynamics ratio at or below 1.20 while training the
action-only parameters with the existing four objective definitions.

## Design

The conflict-fix configuration will set `protected_lr_multiplier` to `0.0`.
Zero is valid only for unified training; world pretraining continues to require
ordinary gradients and multiplier `1.0`.

The existing protected group remains the ownership boundary. It includes the
online state adapter, condition encoder, shared transformer, embeddings, delta
prediction path, and action input projections. AdamW receives a zero learning
rate for this group, so neither gradients nor weight decay can move it. The
action group retains the approved base learning rate and includes the policy
queries, flow projection/head, and inverse heads.

PCGrad remains enabled and measured because its diagnostics are still required
by the audit contract. Its projected protected gradient is computed but cannot
update frozen protected parameters. Action-only gradients continue through the
ordinary optimizer path.

EMA target updates are skipped while the protected multiplier is zero. This
keeps both sides of the world validation contract fixed instead of allowing the
target representation to drift toward a frozen online encoder.

No loss is removed or reweighted. The dynamics, inverse, action-cycle, and
policy-flow objectives remain present in metrics and W&B. The protected
learning rate of zero is the explicit evidence that dynamics is diagnostic
during this gate attempt.

## Runtime Protocol

The failed unified attempt is retained under `failed_attempts/` with its
validation and metrics. The canonical `unified/` output is then recreated from
the immutable world checkpoint and immutable audit decision. It must not resume
from the failed attempt.

At step 250, the run is accepted for continuation only when:

- `pcgrad/enabled` is `1.0`;
- `learning_rate/protected` is exactly `0.0`;
- fixed-validation `dynamics_loss / world_dynamics_loss <= 1.20`;
- metrics and logs contain no non-finite value, OOM, NCCL failure, or traceback.

Failure at step 250 stops the disposable retry. Success allows the existing
three-window stop monitor and 5,000-step unified effect gate to continue.

## Tests

Unit tests must prove that zero protected multiplier is accepted for unified
training and rejected for world pretraining, protected parameters do not change
after an optimizer step, action parameters do change, and EMA parameters and
EMA step do not change. Existing checkpoint configuration mismatch tests remain
strict. The focused conflict-fix suite and complete suite must pass before the
retry starts.

