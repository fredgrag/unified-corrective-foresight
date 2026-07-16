from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _all_nonempty_strings(values: Any) -> bool:
    return all(isinstance(value, str) and bool(value.strip()) for value in values)


@dataclass(frozen=True, slots=True)
class ActionSpec:
    schema_version: int
    spec_id: str
    dimension: int
    names: tuple[str, ...]
    units: tuple[str, ...]
    action_space: str
    mode: str
    rotation_representation: str
    arm_count: int
    gripper_indices: tuple[int, ...]
    control_mode: str
    frequency_hz: float
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported ActionSpec schema_version: {self.schema_version}")
        _require_nonempty("spec_id", self.spec_id)
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")

        sized_fields = {
            "names": self.names,
            "units": self.units,
            "normalization_mean": self.normalization_mean,
            "normalization_std": self.normalization_std,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }
        for name, values in sized_fields.items():
            if len(values) != self.dimension:
                raise ValueError(
                    f"{name} length {len(values)} does not match dimension {self.dimension}"
                )

        if len(set(self.names)) != len(self.names):
            raise ValueError("action dimension names must be unique")
        if not _all_nonempty_strings(self.names):
            raise ValueError("action dimension names must be nonempty strings")
        if not _all_nonempty_strings(self.units):
            raise ValueError("action units must be nonempty strings")
        if self.action_space not in {"joint", "end_effector"}:
            raise ValueError("action_space must be 'joint' or 'end_effector'")
        if self.mode not in {"absolute", "relative", "delta"}:
            raise ValueError("mode must be absolute, relative, or delta")
        _require_nonempty("rotation_representation", self.rotation_representation)
        if self.arm_count <= 0:
            raise ValueError("arm_count must be positive")
        if len(set(self.gripper_indices)) != len(self.gripper_indices) or any(
            index < 0 or index >= self.dimension for index in self.gripper_indices
        ):
            raise ValueError("gripper_indices must be unique and within action dimension")
        _require_nonempty("control_mode", self.control_mode)
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")

        numeric_fields = {
            "normalization_mean": self.normalization_mean,
            "normalization_std": self.normalization_std,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }
        for name, values in numeric_fields.items():
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain only finite values")
        if any(value <= 1e-6 for value in self.normalization_std):
            raise ValueError("normalization_std values must be positive and greater than 1e-6")
        if any(lower >= upper for lower, upper in zip(self.minimum, self.maximum)):
            raise ValueError("every minimum must be strictly less than its maximum")

    @property
    def content_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec_id": self.spec_id,
            "dimension": self.dimension,
            "names": list(self.names),
            "units": list(self.units),
            "action_space": self.action_space,
            "mode": self.mode,
            "rotation_representation": self.rotation_representation,
            "arm_count": self.arm_count,
            "gripper_indices": list(self.gripper_indices),
            "control_mode": self.control_mode,
            "frequency_hz": self.frequency_hz,
            "normalization_mean": list(self.normalization_mean),
            "normalization_std": list(self.normalization_std),
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
        }

    def normalize(self, action: Tensor, mask: Tensor | None = None) -> Tensor:
        mask = self._validate_tensor(action, mask, "action")
        mean = action.new_tensor(self.normalization_mean)
        std = action.new_tensor(self.normalization_std)
        normalized = (action - mean) / std
        return torch.where(mask, normalized, torch.zeros_like(normalized))

    def denormalize(self, normalized: Tensor, mask: Tensor | None = None) -> Tensor:
        mask = self._validate_tensor(normalized, mask, "normalized action")
        mean = normalized.new_tensor(self.normalization_mean)
        std = normalized.new_tensor(self.normalization_std)
        action = normalized * std + mean
        return torch.where(mask, action, torch.zeros_like(action))

    def clamp_to_bounds(self, action: Tensor) -> Tensor:
        self._validate_tensor(action, None, "action")
        lower = action.new_tensor(self.minimum)
        upper = action.new_tensor(self.maximum)
        return torch.maximum(torch.minimum(action, upper), lower)

    def _validate_tensor(
        self, tensor: Tensor, mask: Tensor | None, tensor_name: str
    ) -> Tensor:
        if tensor.ndim == 0 or tensor.shape[-1] != self.dimension:
            raise ValueError(
                f"{tensor_name} final action dimension must be {self.dimension}, "
                f"got shape {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point():
            raise ValueError(f"{tensor_name} must be floating point")
        if mask is None:
            mask = torch.ones_like(tensor, dtype=torch.bool)
        if mask.dtype is not torch.bool or mask.shape != tensor.shape:
            raise ValueError(f"{tensor_name} mask must be bool with identical shape")
        if not torch.isfinite(tensor[mask]).all().item():
            raise ValueError(f"{tensor_name} contains non-finite values in valid dimensions")
        return mask


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    schema_version: int
    dataset_id: str
    repo_id: str | None
    local_root: str | None
    revision: str
    fps: float
    camera_features: Mapping[str, str]
    optional_camera_roles: tuple[str, ...]
    proprio_features: tuple[str, ...]
    proprio_required: bool
    task_feature: str
    language_feature: str | None
    embodiment_id: str
    action_feature: str
    action_spec_id: str
    sample_weight: float
    split_episode_ids: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported DatasetSpec schema_version: {self.schema_version}")
        _require_nonempty("dataset_id", self.dataset_id)
        if (self.repo_id is None) == (self.local_root is None):
            raise ValueError("exactly one of repo_id and local_root must be set")
        if self.repo_id is not None:
            _require_nonempty("repo_id", self.repo_id)
        if self.local_root is not None and not Path(self.local_root).is_absolute():
            raise ValueError("local_root must be an absolute path")
        _require_nonempty("revision", self.revision)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")

        cameras = dict(self.camera_features)
        if not cameras:
            raise ValueError("camera_features must declare at least one camera")
        if not _all_nonempty_strings(cameras) or not _all_nonempty_strings(
            cameras.values()
        ):
            raise ValueError(
                "camera_features roles and feature names must be nonempty strings"
            )
        if len(set(cameras.values())) != len(cameras):
            raise ValueError("camera_features must map roles to unique features")
        if not set(self.optional_camera_roles).issubset(cameras):
            raise ValueError("optional_camera_roles must be declared in camera_features")
        if len(set(self.optional_camera_roles)) != len(self.optional_camera_roles):
            raise ValueError("optional_camera_roles must be unique")
        object.__setattr__(self, "camera_features", MappingProxyType(cameras))

        if self.proprio_required and not self.proprio_features:
            raise ValueError("proprio_features cannot be empty when proprio_required is true")
        if len(set(self.proprio_features)) != len(self.proprio_features):
            raise ValueError("proprio_features must be unique")
        if not _all_nonempty_strings(self.proprio_features):
            raise ValueError("proprio_features must be nonempty strings")

        _require_nonempty("task_feature", self.task_feature)
        if self.language_feature is not None:
            _require_nonempty("language_feature", self.language_feature)
        _require_nonempty("embodiment_id", self.embodiment_id)
        _require_nonempty("action_feature", self.action_feature)
        _require_nonempty("action_spec_id", self.action_spec_id)
        if not math.isfinite(self.sample_weight) or self.sample_weight <= 0:
            raise ValueError("sample_weight must be finite and positive")

        splits = {name: tuple(episodes) for name, episodes in self.split_episode_ids.items()}
        if "train" not in splits:
            raise ValueError("split_episode_ids must include train")
        seen: set[int] = set()
        for split_name, episode_ids in splits.items():
            _require_nonempty("split name", split_name)
            if len(set(episode_ids)) != len(episode_ids) or any(
                not isinstance(episode_id, int) or episode_id < 0
                for episode_id in episode_ids
            ):
                raise ValueError("episode IDs must be unique nonnegative integers per split")
            overlap = seen.intersection(episode_ids)
            if overlap:
                raise ValueError(
                    f"episode splits must be disjoint; repeated IDs: {sorted(overlap)}"
                )
            seen.update(episode_ids)
        object.__setattr__(self, "split_episode_ids", MappingProxyType(splits))

    @property
    def content_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "repo_id": self.repo_id,
            "local_root": self.local_root,
            "revision": self.revision,
            "fps": self.fps,
            "camera_features": dict(self.camera_features),
            "optional_camera_roles": list(self.optional_camera_roles),
            "proprio_features": list(self.proprio_features),
            "proprio_required": self.proprio_required,
            "task_feature": self.task_feature,
            "language_feature": self.language_feature,
            "embodiment_id": self.embodiment_id,
            "action_feature": self.action_feature,
            "action_spec_id": self.action_spec_id,
            "sample_weight": self.sample_weight,
            "split_episode_ids": {
                name: list(episode_ids)
                for name, episode_ids in self.split_episode_ids.items()
            },
        }
