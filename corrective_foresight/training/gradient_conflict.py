from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import random
from types import MappingProxyType

import torch
import torch.distributed as dist
from torch import Tensor, nn

from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.numerics import (
    NumericalGuardError,
    require_finite_objective,
)


@dataclass(frozen=True, slots=True)
class GradientDiagnostics:
    raw_norms: Mapping[str, float]
    effective_norms: Mapping[str, float]
    cosines: Mapping[str, float]
    minimum_dynamics_cosine: float
    ordinary_effective_norm: float


@dataclass(frozen=True, slots=True)
class GlobalObjectiveGradients:
    names: tuple[str, ...]
    raw: Mapping[str, tuple[Tensor | None, ...]]
    effective_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if (
            not self.names
            or len(set(self.names)) != len(self.names)
            or set(self.raw) != set(self.names)
            or set(self.effective_weights) != set(self.names)
        ):
            raise ValueError("objective gradient names must be nonempty and exact")
        lengths = {len(self.raw[name]) for name in self.names}
        if len(lengths) != 1 or next(iter(lengths)) <= 0:
            raise ValueError("objective gradient layouts must match and be nonempty")
        copied_raw: dict[str, tuple[Tensor | None, ...]] = {}
        for name in self.names:
            values = tuple(self.raw[name])
            if any(value is None for value in values):
                raise ValueError(
                    "global objective gradients must be fully materialized"
                )
            for value in values:
                if value is None:
                    continue
                if not value.is_floating_point() or not torch.isfinite(value).all().item():
                    raise NumericalGuardError(
                        f"objective {name} has non-finite shared gradient"
                    )
            copied_raw[name] = values
            weight = self.effective_weights[name]
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) < 0.0
            ):
                raise ValueError(f"effective objective weight {name} is invalid")
        object.__setattr__(self, "raw", MappingProxyType(copied_raw))
        object.__setattr__(
            self,
            "effective_weights",
            MappingProxyType(
                {name: float(self.effective_weights[name]) for name in self.names}
            ),
        )

    @classmethod
    def from_flattened(
        cls,
        *,
        raw: Mapping[str, Tensor],
        effective_weights: Mapping[str, float],
    ) -> GlobalObjectiveGradients:
        names = tuple(raw)
        return cls(
            names=names,
            raw={name: (raw[name],) for name in names},
            effective_weights=effective_weights,
        )

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.names if self.effective_weights[name] > 0.0
        )

    def diagnostics(self) -> GradientDiagnostics:
        raw_norms = {
            name: float(_norm(self.raw[name]).item()) for name in self.names
        }
        effective = {
            name: _scale(self.raw[name], self.effective_weights[name])
            for name in self.names
        }
        effective_norms = {
            name: float(_norm(effective[name]).item()) for name in self.names
        }
        cosines: dict[str, float] = {}
        for index, name in enumerate(self.names):
            for other in self.names[index + 1 :]:
                cosines[f"{name}/{other}"] = float(
                    _cosine(self.raw[name], self.raw[other]).item()
                )
        dynamic_cosines = [
            value
            for pair, value in cosines.items()
            if "dynamics_loss" in pair
        ]
        minimum = min(dynamic_cosines) if dynamic_cosines else 0.0
        ordinary = _sum_gradients(
            tuple(effective[name] for name in self.active_names)
        )
        return GradientDiagnostics(
            raw_norms=MappingProxyType(raw_norms),
            effective_norms=MappingProxyType(effective_norms),
            cosines=MappingProxyType(cosines),
            minimum_dynamics_cosine=minimum,
            ordinary_effective_norm=float(_norm(ordinary).item()),
        )


@dataclass(frozen=True, slots=True)
class ProjectedGradientResult:
    gradient: tuple[Tensor, ...]
    active_objectives: tuple[str, ...]
    raw_objectives: tuple[str, ...]
    removed_fraction: float
    projected_norm: float


class ObjectiveGradientAccumulator:
    def __init__(
        self,
        parameters: Sequence[nn.Parameter],
        names: tuple[str, ...],
    ) -> None:
        self.parameters = tuple(
            parameter for parameter in parameters if parameter.requires_grad
        )
        if not self.parameters:
            raise ValueError("objective gradient parameters cannot be empty")
        if not names or len(set(names)) != len(names):
            raise ValueError("objective gradient names must be unique and nonempty")
        self.names = names
        self._buffers: dict[str, list[Tensor | None]] = {}
        self._micro_steps = 0
        self._expected_micro_steps: int | None = None
        self.reset()

    def add(
        self,
        losses: Mapping[str, Tensor],
        *,
        accumulation_steps: int,
    ) -> None:
        if set(losses) != set(self.names):
            raise ValueError("objective losses do not match accumulator names")
        if type(accumulation_steps) is not int or accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be a positive integer")
        if self._expected_micro_steps is None:
            self._expected_micro_steps = accumulation_steps
        elif self._expected_micro_steps != accumulation_steps:
            raise ValueError("accumulation_steps changed within optimizer step")
        if self._micro_steps >= accumulation_steps:
            raise ValueError("too many objective-gradient micro-steps")
        scale = 1.0 / accumulation_steps
        captured: dict[str, tuple[Tensor | None, ...]] = {}
        for name in self.names:
            loss = losses[name]
            require_finite_objective(
                loss,
                objective_name=name,
                dataset_id="gradient-accumulator",
                rank=0,
                global_step=self._micro_steps,
            )
            values = torch.autograd.grad(
                loss.float(),
                self.parameters,
                retain_graph=True,
                allow_unused=True,
            )
            if all(value is None for value in values):
                raise NumericalGuardError(
                    f"objective {name} has no shared gradient"
                )
            captured[name] = values
        for name, values in captured.items():
            for index, value in enumerate(values):
                if value is None:
                    continue
                detached = value.detach().float()
                if not torch.isfinite(detached).all().item():
                    raise NumericalGuardError(
                        f"objective {name} has non-finite shared gradient"
                    )
                scaled = detached * scale
                current = self._buffers[name][index]
                self._buffers[name][index] = (
                    scaled.clone() if current is None else current + scaled
                )
        self._micro_steps += 1

    def finalize(
        self,
        context: DistributedContext,
        effective_weights: Mapping[str, float],
    ) -> GlobalObjectiveGradients:
        if not isinstance(context, DistributedContext):
            raise ValueError("gradient finalize requires DistributedContext")
        if (
            self._expected_micro_steps is None
            or self._micro_steps != self._expected_micro_steps
        ):
            raise NumericalGuardError(
                "objective gradients do not cover the full accumulation batch"
            )
        layout = {
            name: tuple(value is not None for value in self._buffers[name])
            for name in self.names
        }
        if context.world_size > 1:
            gathered: list[dict[str, tuple[bool, ...]] | None] = [
                None
            ] * context.world_size
            dist.all_gather_object(gathered, layout)
            if any(item != layout for item in gathered):
                raise NumericalGuardError(
                    "objective gradient layouts differ across ranks"
                )
        reduced: dict[str, tuple[Tensor, ...]] = {}
        for name in self.names:
            tensors: list[Tensor] = []
            for parameter, value in zip(
                self.parameters,
                self._buffers[name],
                strict=True,
            ):
                tensor = (
                    torch.zeros_like(parameter, dtype=torch.float32)
                    if value is None
                    else value.clone()
                )
                if context.world_size > 1:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    tensor.div_(context.world_size)
                tensors.append(tensor)
            reduced[name] = tuple(tensors)
        self.reset()
        return GlobalObjectiveGradients(
            names=self.names,
            raw=reduced,
            effective_weights=effective_weights,
        )

    def reset(self) -> None:
        self._buffers = {
            name: [None for _ in self.parameters] for name in self.names
        }
        self._micro_steps = 0
        self._expected_micro_steps = None


def projection_order(
    names: tuple[str, ...],
    *,
    seed: int,
    global_step: int,
) -> tuple[str, ...]:
    if not names or len(set(names)) != len(names):
        raise ValueError("projection names must be unique and nonempty")
    if type(seed) is not int or seed < 0:
        raise ValueError("projection seed must be nonnegative")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("projection global_step must be nonnegative")
    material = f"ucf-pcgrad-v1:{seed}:{global_step}:" + ",".join(names)
    order_seed = int.from_bytes(
        hashlib.sha256(material.encode("ascii")).digest()[:8],
        "big",
    )
    generator = random.Random(order_seed)
    result = list(names)
    generator.shuffle(result)
    return tuple(result)


def project_pcgrad(
    gradients: GlobalObjectiveGradients,
    *,
    seed: int,
    global_step: int,
) -> ProjectedGradientResult:
    active = gradients.active_names
    if not active:
        raise NumericalGuardError("PCGrad has no active objectives")
    effective = {
        name: _scale(gradients.raw[name], gradients.effective_weights[name])
        for name in active
    }
    order = projection_order(active, seed=seed, global_step=global_step)
    projected: dict[str, tuple[Tensor, ...]] = {}
    for name in active:
        current = tuple(value.clone() for value in effective[name])
        for other_name in order:
            if other_name == name:
                continue
            other = effective[other_name]
            dot = _dot(current, other)
            denominator = _dot(other, other)
            if dot.item() < 0.0 and denominator.item() > 0.0:
                coefficient = dot / denominator
                current = tuple(
                    value - coefficient * other_value
                    for value, other_value in zip(current, other, strict=True)
                )
        projected[name] = current
    ordinary = _sum_gradients(tuple(effective[name] for name in active))
    combined = _sum_gradients(tuple(projected[name] for name in active))
    removed = tuple(
        before - after for before, after in zip(ordinary, combined, strict=True)
    )
    ordinary_norm = _norm(ordinary)
    removed_fraction = float(
        (_norm(removed) / ordinary_norm.clamp_min(1e-12)).item()
    )
    if any(not torch.isfinite(value).all().item() for value in combined):
        raise NumericalGuardError("PCGrad produced non-finite gradient")
    return ProjectedGradientResult(
        gradient=combined,
        active_objectives=active,
        raw_objectives=gradients.names,
        removed_fraction=removed_fraction,
        projected_norm=float(_norm(combined).item()),
    )


def _scale(
    values: tuple[Tensor | None, ...],
    weight: float,
) -> tuple[Tensor, ...]:
    result: list[Tensor] = []
    template = next((value for value in values if value is not None), None)
    if template is None:
        raise NumericalGuardError("objective has no shared gradient")
    for value in values:
        result.append(
            torch.zeros_like(template) if value is None else value * weight
        )
    return tuple(result)


def _dot(
    first: tuple[Tensor | None, ...],
    second: tuple[Tensor | None, ...],
) -> Tensor:
    if len(first) != len(second) or not first:
        raise ValueError("gradient tuple layouts do not match")
    total: Tensor | None = None
    for left, right in zip(first, second, strict=True):
        if left is None or right is None:
            continue
        value = (left.float() * right.float()).sum()
        total = value if total is None else total + value
    if total is None:
        template = next(
            (value for value in (*first, *second) if value is not None),
            None,
        )
        if template is None:
            raise NumericalGuardError("gradient dot product has no tensors")
        return template.new_zeros((), dtype=torch.float32)
    return total


def _norm(values: tuple[Tensor | None, ...]) -> Tensor:
    return _dot(values, values).clamp_min(0.0).sqrt()


def _cosine(
    first: tuple[Tensor | None, ...],
    second: tuple[Tensor | None, ...],
) -> Tensor:
    denominator = _norm(first) * _norm(second)
    if denominator.item() == 0.0:
        return denominator.new_zeros(())
    return (_dot(first, second) / denominator).clamp(-1.0, 1.0)


def _sum_gradients(
    gradients: tuple[tuple[Tensor, ...], ...],
) -> tuple[Tensor, ...]:
    if not gradients:
        raise NumericalGuardError("cannot sum an empty gradient collection")
    width = len(gradients[0])
    if width == 0 or any(len(values) != width for values in gradients):
        raise ValueError("gradient collection layouts do not match")
    return tuple(
        sum((values[index] for values in gradients), start=torch.zeros_like(gradients[0][index]))
        for index in range(width)
    )
