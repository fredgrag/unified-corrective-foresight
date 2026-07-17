from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
from torch import Tensor


VelocityField = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class FlowTrainingSample:
    epsilon: Tensor
    t: Tensor
    x_t: Tensor
    target_velocity: Tensor

    def __post_init__(self) -> None:
        if self.epsilon.ndim != 3 or min(self.epsilon.shape) <= 0:
            raise ValueError("epsilon must have shape [B,H,Da]")
        if self.t.shape != (self.epsilon.shape[0], 1, 1):
            raise ValueError("flow time must have shape [B,1,1]")
        if self.x_t.shape != self.epsilon.shape or self.target_velocity.shape != self.epsilon.shape:
            raise ValueError("flow tensors must share shape [B,H,Da]")
        for name, value in (
            ("epsilon", self.epsilon),
            ("t", self.t),
            ("x_t", self.x_t),
            ("target_velocity", self.target_velocity),
        ):
            if not value.is_floating_point() or not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite floating point")
        if (self.t < 0).any().item() or (self.t >= 1).any().item():
            raise ValueError("flow time must satisfy 0 <= t < 1")

    def clean_estimate(self, predicted_velocity: Tensor) -> Tensor:
        if predicted_velocity.shape != self.x_t.shape:
            raise ValueError("predicted velocity shape must match x_t")
        if not predicted_velocity.is_floating_point() or not torch.isfinite(
            predicted_velocity
        ).all().item():
            raise ValueError("predicted velocity must be finite floating point")
        return self.x_t + (1.0 - self.t) * predicted_velocity


@dataclass(frozen=True, slots=True)
class IntegrationReport:
    solver: str
    time_grid: tuple[float, ...]
    intervals: int
    nfe: int
    noise_seed: int

    def __post_init__(self) -> None:
        if self.solver not in {"euler", "midpoint"}:
            raise ValueError("solver must be euler or midpoint")
        if type(self.intervals) is not int or self.intervals <= 0:
            raise ValueError("intervals must be a positive integer")
        if type(self.noise_seed) is not int or self.noise_seed < 0:
            raise ValueError("noise_seed must be a nonnegative integer")
        expected_nfe = self.intervals * (1 if self.solver == "euler" else 2)
        if self.nfe != expected_nfe:
            raise ValueError(
                f"NFE must be {expected_nfe} for {self.solver}, got {self.nfe}"
            )
        if len(self.time_grid) != self.intervals + 1:
            raise ValueError("time grid length must equal intervals + 1")
        if not all(math.isfinite(value) for value in self.time_grid):
            raise ValueError("time grid must be finite")
        if self.time_grid[0] != 0.0 or self.time_grid[-1] != 1.0:
            raise ValueError("time grid must start at 0 and end at 1")
        if any(
            right <= left for left, right in zip(self.time_grid, self.time_grid[1:])
        ):
            raise ValueError("time grid must be strictly increasing")


def sample_flow_training(
    action: Tensor,
    *,
    generator: torch.Generator,
    valid_mask: Tensor | None = None,
) -> FlowTrainingSample:
    if action.ndim != 3 or min(action.shape) <= 0 or not action.is_floating_point():
        raise ValueError("action must be floating point with shape [B,H,Da]")
    if valid_mask is None:
        valid_mask = torch.ones_like(action, dtype=torch.bool)
    if valid_mask.dtype is not torch.bool or valid_mask.shape != action.shape:
        raise ValueError("valid_mask must be bool and match action")
    if valid_mask.device != action.device:
        raise ValueError("valid_mask must be on the action device")
    if not torch.isfinite(action[valid_mask]).all().item():
        raise ValueError("valid action values must be finite")
    generator_device = torch.device(generator.device)
    if generator_device.type != action.device.type:
        raise ValueError("generator and action must use the same device type")
    safe_action = torch.where(valid_mask, action, 0.0)
    epsilon = torch.randn(
        action.shape,
        dtype=action.dtype,
        device=action.device,
        generator=generator,
    )
    t = torch.rand(
        (action.shape[0], 1, 1),
        dtype=action.dtype,
        device=action.device,
        generator=generator,
    )
    return FlowTrainingSample(
        epsilon=epsilon,
        t=t,
        x_t=(1.0 - t) * epsilon + t * safe_action,
        target_velocity=safe_action - epsilon,
    )


def masked_velocity_mse(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target velocity must share shape [B,H,Da]")
    if not predicted.is_floating_point() or not target.is_floating_point():
        raise ValueError("velocity tensors must be floating point")
    if mask.dtype is not torch.bool or mask.shape != predicted.shape:
        raise ValueError("velocity mask must be bool and match velocity shape")
    if mask.device != predicted.device or target.device != predicted.device:
        raise ValueError("velocity tensors and mask must share a device")
    if not mask.any().item():
        raise ValueError("no valid velocity elements")
    if not torch.isfinite(predicted[mask]).all().item() or not torch.isfinite(
        target[mask]
    ).all().item():
        raise ValueError("valid velocity values must be finite")
    squared_error = (predicted.float() - target.float()).square()
    return squared_error[mask].mean(dtype=torch.float32)


def sample_rectified_flow(
    velocity_field: VelocityField,
    action_shape: tuple[int, int, int] | None = None,
    *,
    initial_noise: Tensor | None = None,
    solver: str = "midpoint",
    intervals: int = 10,
    noise_seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, IntegrationReport]:
    if solver not in {"euler", "midpoint"}:
        raise ValueError("solver must be euler or midpoint")
    if type(intervals) is not int or intervals <= 0:
        raise ValueError("intervals must be a positive integer")
    if type(noise_seed) is not int or noise_seed < 0:
        raise ValueError("noise_seed must be a nonnegative integer")
    if initial_noise is not None and action_shape is not None:
        raise ValueError("provide action_shape or initial_noise, not both")
    if initial_noise is None:
        if (
            action_shape is None
            or len(action_shape) != 3
            or any(type(value) is not int or value <= 0 for value in action_shape)
        ):
            raise ValueError("action_shape must contain three positive integers")
        if not dtype.is_floating_point:
            raise ValueError("flow integration dtype must be floating point")
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        actions = torch.randn(
            action_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        if (
            initial_noise.ndim != 3
            or min(initial_noise.shape) <= 0
            or not initial_noise.is_floating_point()
            or not torch.isfinite(initial_noise).all().item()
        ):
            raise ValueError("initial_noise must be finite floating point [B,H,Da]")
        actions = initial_noise.clone()

    time_grid = tuple(index / intervals for index in range(intervals + 1))
    nfe = 0
    for index in range(intervals):
        start = time_grid[index]
        step = time_grid[index + 1] - start
        time = actions.new_full((actions.shape[0], 1, 1), start)
        first_velocity = _evaluate_velocity(velocity_field, actions, time)
        nfe += 1
        if solver == "euler":
            actions = actions + step * first_velocity
        else:
            midpoint = actions + 0.5 * step * first_velocity
            midpoint_time = actions.new_full(
                (actions.shape[0], 1, 1), start + 0.5 * step
            )
            midpoint_velocity = _evaluate_velocity(
                velocity_field, midpoint, midpoint_time
            )
            nfe += 1
            actions = actions + step * midpoint_velocity
    report = IntegrationReport(
        solver=solver,
        time_grid=time_grid,
        intervals=intervals,
        nfe=nfe,
        noise_seed=noise_seed,
    )
    return actions, report


def _evaluate_velocity(
    velocity_field: VelocityField, actions: Tensor, time: Tensor
) -> Tensor:
    velocity = velocity_field(actions, time)
    if velocity.shape != actions.shape:
        raise ValueError(
            f"velocity field shape must be {tuple(actions.shape)}, got {tuple(velocity.shape)}"
        )
    if not velocity.is_floating_point() or not torch.isfinite(velocity).all().item():
        raise ValueError("velocity field output must be finite floating point")
    return velocity
