from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class TrainingStage(str, Enum):
    WORLD_PRETRAIN = "world_pretrain"
    UNIFIED = "unified"

    @classmethod
    def parse(cls, value: TrainingStage | str) -> TrainingStage:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError("training stage must be world_pretrain or unified")
        try:
            return cls(value)
        except ValueError as error:
            raise ValueError(
                f"unsupported training stage {value!r}; "
                "expected world_pretrain or unified"
            ) from error


@dataclass(frozen=True, slots=True)
class TrainingOutput:
    stage: TrainingStage
    loss: Tensor
    optimized_terms: frozenset[str]
    losses: Mapping[str, Tensor]
    metrics: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if (
            self.loss.ndim != 0
            or not self.loss.is_floating_point()
            or not torch.isfinite(self.loss).item()
        ):
            raise ValueError("training loss must be a finite floating-point scalar")
        if set(self.losses) != set(self.optimized_terms):
            raise ValueError("training losses and optimized term whitelist must match")
        for name, value in (*self.losses.items(), *self.metrics.items()):
            if (
                not isinstance(value, Tensor)
                or value.ndim != 0
                or not value.is_floating_point()
                or not torch.isfinite(value).item()
            ):
                raise ValueError(f"{name} must be a finite floating-point scalar")
