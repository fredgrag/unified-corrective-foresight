from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class DeltaPrediction:
    delta: Tensor
    states: Tensor

    def __post_init__(self) -> None:
        if self.delta.ndim != 4 or self.states.ndim != 4:
            raise ValueError("delta and states must have shapes [B,N,S,H] and [B,N+1,S,H]")
        if self.states.shape[0] != self.delta.shape[0] or self.states.shape[1] != self.delta.shape[1] + 1:
            raise ValueError("state rollout length must equal delta length + 1")
        if self.states.shape[2:] != self.delta.shape[2:]:
            raise ValueError("delta and state token dimensions must match")
        _require_finite("delta", self.delta)
        _require_finite("states", self.states)


@dataclass(frozen=True, slots=True)
class InversePrediction:
    mean: Tensor
    log_variance: Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 3 or self.log_variance.shape != self.mean.shape:
            raise ValueError("inverse outputs must share shape [B,N,Da]")
        _require_finite("inverse mean", self.mean)
        _require_finite("inverse log variance", self.log_variance)


@dataclass(frozen=True, slots=True)
class CyclePrediction:
    mean: Tensor
    log_variance: Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 3 or self.log_variance.shape != self.mean.shape:
            raise ValueError("cycle outputs must share shape [B,N,Da]")
        _require_finite("cycle mean", self.mean)
        _require_finite("cycle log variance", self.log_variance)


@dataclass(frozen=True, slots=True)
class PolicyVelocityPrediction:
    velocity: Tensor

    def __post_init__(self) -> None:
        if self.velocity.ndim != 3:
            raise ValueError("policy velocity must have shape [B,8,Da]")
        _require_finite("policy velocity", self.velocity)


def _require_finite(name: str, value: Tensor) -> None:
    if not value.is_floating_point() or not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite floating point")
