from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Mapping

import torch
from torch import Tensor


@dataclass(slots=True)
class TrajectoryBatch:
    rgb: Tensor
    camera_mask: Tensor
    proprio: Tensor
    proprio_mask: Tensor
    action: Tensor
    action_dimension_mask: Tensor
    observation_valid_mask: Tensor
    action_valid_mask: Tensor
    transition_valid_mask: Tensor
    delta_time: Tensor
    context_index: int
    task_text: tuple[str | None, ...]
    condition_ids: Mapping[str, Tensor]
    dataset_id: str
    action_spec_id: str

    def validate(self, expected_action_spec_id: str | None = None) -> None:
        if self.rgb.ndim != 6 or self.rgb.shape[3] != 3:
            raise ValueError("rgb must have shape [B,T,V,3,H,W]")
        if not self.rgb.is_floating_point():
            raise ValueError("rgb must be floating point")
        batch_size, time_steps, views = self.rgb.shape[:3]
        if batch_size <= 0 or views <= 0:
            raise ValueError("rgb batch and view dimensions must be positive")
        if time_steps < self.context_index + 9:
            raise ValueError("trajectory must contain eight actions after context_index")
        if self.context_index < 0 or self.context_index >= time_steps - 1:
            raise ValueError("context_index is outside the trajectory")

        self._require_bool_shape(
            "camera_mask", self.camera_mask, (batch_size, time_steps, views)
        )
        self._require_bool_shape(
            "observation_valid_mask",
            self.observation_valid_mask,
            (batch_size, time_steps),
        )
        if self.proprio.ndim != 3 or self.proprio.shape[:2] != (batch_size, time_steps):
            raise ValueError("proprio must have shape [B,T,Dp]")
        if not self.proprio.is_floating_point():
            raise ValueError("proprio must be floating point")
        self._require_bool_shape("proprio_mask", self.proprio_mask, self.proprio.shape)

        transition_steps = time_steps - 1
        if self.action.ndim != 3 or self.action.shape[:2] != (
            batch_size,
            transition_steps,
        ):
            raise ValueError("action must have shape [B,T-1,Da]")
        if self.action.shape[-1] <= 0 or not self.action.is_floating_point():
            raise ValueError("action dimension must be positive and floating point")
        self._require_bool_shape(
            "action_dimension_mask", self.action_dimension_mask, self.action.shape
        )
        self._require_bool_shape(
            "action_valid_mask",
            self.action_valid_mask,
            (batch_size, transition_steps),
        )
        self._require_bool_shape(
            "transition_valid_mask",
            self.transition_valid_mask,
            (batch_size, transition_steps),
        )
        if self.delta_time.shape != (batch_size, transition_steps) or not (
            self.delta_time.is_floating_point()
        ):
            raise ValueError("delta_time must be floating point with shape [B,T-1]")

        if len(self.task_text) != batch_size:
            raise ValueError("task_text length must equal batch size")
        if not self.dataset_id:
            raise ValueError("dataset_id must be nonempty")
        if not self.action_spec_id:
            raise ValueError("action_spec_id must be nonempty")
        if (
            expected_action_spec_id is not None
            and self.action_spec_id != expected_action_spec_id
        ):
            raise ValueError(
                "every batch must use the same ActionSpec; "
                f"expected {expected_action_spec_id}, got {self.action_spec_id}"
            )

        for name, identifiers in self.condition_ids.items():
            if not name:
                raise ValueError("condition identifier names must be nonempty")
            if identifiers.shape != (batch_size,) or identifiers.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError(f"condition_ids[{name}] must be integer shape [B]")
            if (identifiers < 0).any().item():
                raise ValueError(f"condition_ids[{name}] must be nonnegative")

        camera_without_observation = self.camera_mask & ~self.observation_valid_mask[
            :, :, None
        ]
        if camera_without_observation.any().item():
            raise ValueError("camera_mask cannot be true for an invalid observation")
        proprio_without_observation = self.proprio_mask & ~self.observation_valid_mask[
            :, :, None
        ]
        if proprio_without_observation.any().item():
            raise ValueError("proprio_mask cannot be true for an invalid observation")
        dimensions_without_action = self.action_dimension_mask & ~self.action_valid_mask[
            :, :, None
        ]
        if dimensions_without_action.any().item():
            raise ValueError("action_dimension_mask cannot be true for an invalid action")

        valid_transition_inputs = (
            self.observation_valid_mask[:, :-1]
            & self.observation_valid_mask[:, 1:]
            & self.action_valid_mask
        )
        if (self.transition_valid_mask & ~valid_transition_inputs).any().item():
            raise ValueError(
                "transition_valid_mask requires both observations and action to be valid"
            )
        if not torch.isfinite(self.delta_time[self.transition_valid_mask]).all().item():
            raise ValueError("delta_time must be finite on valid transitions")
        if (self.delta_time[self.transition_valid_mask] <= 0).any().item():
            raise ValueError("delta_time must be positive on valid transitions")

        rgb_mask = self.camera_mask[:, :, :, None, None, None].expand_as(self.rgb)
        self._require_finite("rgb", self.rgb, rgb_mask)
        self._require_finite("proprio", self.proprio, self.proprio_mask)
        valid_action_dimensions = self.action_dimension_mask & self.action_valid_mask[
            :, :, None
        ]
        self._require_finite("action", self.action, valid_action_dimensions)

    def to(self, device: torch.device | str, non_blocking: bool = False) -> TrajectoryBatch:
        updates = {
            field.name: getattr(self, field.name).to(device, non_blocking=non_blocking)
            for field in fields(self)
            if isinstance(getattr(self, field.name), Tensor)
        }
        updates["condition_ids"] = {
            name: value.to(device, non_blocking=non_blocking)
            for name, value in self.condition_ids.items()
        }
        return replace(self, **updates)

    def pin_memory(self) -> TrajectoryBatch:
        updates = {
            field.name: getattr(self, field.name).pin_memory()
            for field in fields(self)
            if isinstance(getattr(self, field.name), Tensor)
        }
        updates["condition_ids"] = {
            name: value.pin_memory() for name, value in self.condition_ids.items()
        }
        return replace(self, **updates)

    def select_time_window(self, start: int, end: int) -> TrajectoryBatch:
        time_steps = self.rgb.shape[1]
        if start < 0 or end > time_steps or start >= end:
            raise ValueError("time window is outside the trajectory")
        if not start <= self.context_index < end - 1:
            raise ValueError("time window must contain context and a following action")
        result = replace(
            self,
            rgb=self.rgb[:, start:end],
            camera_mask=self.camera_mask[:, start:end],
            proprio=self.proprio[:, start:end],
            proprio_mask=self.proprio_mask[:, start:end],
            action=self.action[:, start : end - 1],
            action_dimension_mask=self.action_dimension_mask[:, start : end - 1],
            observation_valid_mask=self.observation_valid_mask[:, start:end],
            action_valid_mask=self.action_valid_mask[:, start : end - 1],
            transition_valid_mask=self.transition_valid_mask[:, start : end - 1],
            delta_time=self.delta_time[:, start : end - 1],
            context_index=self.context_index - start,
        )
        result.validate()
        return result

    @staticmethod
    def _require_bool_shape(name: str, tensor: Tensor, shape: tuple[int, ...]) -> None:
        if tensor.dtype is not torch.bool or tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} must be bool with shape {shape}")

    @staticmethod
    def _require_finite(name: str, tensor: Tensor, mask: Tensor) -> None:
        if not torch.isfinite(tensor[mask]).all().item():
            raise ValueError(f"{name} contains non-finite values where mask is true")
