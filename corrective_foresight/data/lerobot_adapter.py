from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
from typing import Collection, Mapping

import torch
from torch import Tensor

from corrective_foresight.config.schema import ActionSpec, DatasetSpec


@dataclass(slots=True)
class TrajectorySample:
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
    task_text: str | None
    condition_ids: Mapping[str, Tensor]
    dataset_id: str
    action_spec_id: str


def build_delta_timestamps(
    dataset_spec: DatasetSpec,
    available_features: Collection[str],
    context_steps: int = 2,
    action_horizon: int = 8,
) -> tuple[dict[str, list[float]], int]:
    if context_steps <= 0:
        raise ValueError("context_steps must be positive")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    available = set(available_features)
    context_index = context_steps - 1
    observation_indices = list(range(-context_index, action_horizon + 1))
    action_indices = list(range(-context_index, action_horizon))
    observation_times = [index / dataset_spec.fps for index in observation_indices]
    action_times = [index / dataset_spec.fps for index in action_indices]

    required_camera_features = {
        feature
        for role, feature in dataset_spec.camera_features.items()
        if role not in dataset_spec.optional_camera_roles
    }
    missing_cameras = required_camera_features - available
    if missing_cameras:
        raise ValueError(f"missing required camera features: {sorted(missing_cameras)}")
    missing_proprio = set(dataset_spec.proprio_features) - available
    if missing_proprio:
        raise ValueError(f"missing declared proprio features: {sorted(missing_proprio)}")
    if dataset_spec.action_feature not in available:
        raise ValueError(f"missing required action feature: {dataset_spec.action_feature}")

    timestamps = {
        feature: list(observation_times)
        for feature in (
            *dataset_spec.camera_features.values(),
            *dataset_spec.proprio_features,
        )
        if feature in available
    }
    timestamps[dataset_spec.action_feature] = action_times
    return timestamps, context_index


def derive_validity_masks(
    item: Mapping[str, object],
    required_observation_features: tuple[str, ...],
    action_feature: str,
) -> tuple[Tensor, Tensor, Tensor]:
    if not required_observation_features:
        raise ValueError("at least one observation feature is required for validity")
    observation_pads: list[Tensor] = []
    observation_length: int | None = None
    for feature in required_observation_features:
        key = f"{feature}_is_pad"
        if key not in item:
            raise ValueError(f"missing LeRobot padding key: {key}")
        pad = torch.as_tensor(item[key], dtype=torch.bool)
        if pad.ndim != 1:
            raise ValueError(f"{key} must be one-dimensional")
        if observation_length is None:
            observation_length = pad.numel()
        elif pad.numel() != observation_length:
            raise ValueError("observation padding length mismatch")
        observation_pads.append(pad)

    action_key = f"{action_feature}_is_pad"
    if action_key not in item:
        raise ValueError(f"missing LeRobot padding key: {action_key}")
    action_pad = torch.as_tensor(item[action_key], dtype=torch.bool)
    if action_pad.ndim != 1:
        raise ValueError(f"{action_key} must be one-dimensional")
    if action_pad.numel() != observation_length - 1:
        raise ValueError("action padding length must equal observation length minus one")

    observation_valid = ~torch.stack(observation_pads).any(dim=0)
    action_valid = ~action_pad
    transition_valid = (
        observation_valid[:-1] & observation_valid[1:] & action_valid
    )
    return observation_valid, action_valid, transition_valid


def extract_episode_ids(episode_records: Collection[Mapping[str, object]]) -> set[int]:
    episode_ids: set[int] = set()
    for position, record in enumerate(episode_records):
        if "episode_index" not in record:
            raise ValueError(f"episode metadata record {position} lacks episode_index")
        episode_index = record["episode_index"]
        if not isinstance(episode_index, Integral) or int(episode_index) < 0:
            raise ValueError(
                f"episode metadata record {position} has invalid episode_index"
            )
        episode_id = int(episode_index)
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode_index in metadata: {episode_id}")
        episode_ids.add(episode_id)
    return episode_ids


class LeRobotTrajectoryAdapter:
    def __init__(
        self,
        dataset_spec: DatasetSpec,
        action_spec: ActionSpec,
        split: str,
        context_steps: int = 2,
        action_horizon: int = 8,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
    ) -> None:
        from lerobot.datasets.feature_utils import check_delta_timestamps
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if dataset_spec.action_spec_id != action_spec.spec_id:
            raise ValueError(
                "DatasetSpec and ActionSpec IDs differ: "
                f"{dataset_spec.action_spec_id} != {action_spec.spec_id}"
            )
        if split not in dataset_spec.split_episode_ids:
            raise ValueError(f"unknown dataset split: {split}")
        if not math.isfinite(tolerance_s) or tolerance_s <= 0:
            raise ValueError("tolerance_s must be finite and positive")

        self.dataset_spec = dataset_spec
        self.action_spec = action_spec
        self.split = split
        self.camera_roles = tuple(sorted(dataset_spec.camera_features))
        self._video_backend = video_backend
        repo_id = dataset_spec.repo_id or dataset_spec.dataset_id
        root = Path(dataset_spec.local_root) if dataset_spec.local_root else None
        revision = dataset_spec.revision if dataset_spec.repo_id is not None else None
        episodes = list(dataset_spec.split_episode_ids[split])

        metadata_dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            revision=revision,
            video_backend=video_backend,
        )
        self._validate_metadata(metadata_dataset)
        available_features = set(metadata_dataset.features)
        delta_timestamps, self.context_index = build_delta_timestamps(
            dataset_spec,
            available_features,
            context_steps=context_steps,
            action_horizon=action_horizon,
        )
        check_delta_timestamps(
            delta_timestamps=delta_timestamps,
            fps=metadata_dataset.fps,
            tolerance_s=tolerance_s,
        )
        self.delta_timestamps = delta_timestamps
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            revision=revision,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            video_backend=video_backend,
        )
        self._camera_shape = self._resolve_camera_shape(self.dataset.features)
        self._present_camera_roles = {
            role
            for role, feature in dataset_spec.camera_features.items()
            if feature in available_features
        }
        required_observation_features = [
            feature
            for role, feature in dataset_spec.camera_features.items()
            if role not in dataset_spec.optional_camera_roles
        ]
        if dataset_spec.proprio_required:
            required_observation_features.extend(dataset_spec.proprio_features)
        if not required_observation_features:
            required_observation_features.extend(
                dataset_spec.camera_features[role]
                for role in self.camera_roles
                if role in self._present_camera_roles
            )
        self._required_observation_features = tuple(required_observation_features)
        self.consumed_keys = self._build_consumed_keys()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TrajectorySample:
        item = self.dataset[index]
        observation_valid, action_valid, transition_valid = derive_validity_masks(
            item,
            required_observation_features=self._required_observation_features,
            action_feature=self.dataset_spec.action_feature,
        )
        time_steps = observation_valid.numel()

        camera_tensors: list[Tensor] = []
        camera_masks: list[Tensor] = []
        for role in self.camera_roles:
            feature = self.dataset_spec.camera_features[role]
            if role in self._present_camera_roles:
                camera = self._as_chw_video(item[feature], feature)
                pad = torch.as_tensor(item[f"{feature}_is_pad"], dtype=torch.bool)
                camera_mask = ~pad
            else:
                channels, height, width = self._camera_shape
                camera = torch.zeros(time_steps, channels, height, width)
                camera_mask = torch.zeros(time_steps, dtype=torch.bool)
            camera_tensors.append(camera)
            camera_masks.append(camera_mask)
        rgb = torch.stack(camera_tensors, dim=1)
        camera_mask = torch.stack(camera_masks, dim=1)

        proprio_parts: list[Tensor] = []
        proprio_masks: list[Tensor] = []
        for feature in self.dataset_spec.proprio_features:
            values = torch.as_tensor(item[feature], dtype=torch.float32)
            if values.shape[0] != time_steps:
                raise ValueError(f"{feature} window length does not match observations")
            values = values.reshape(time_steps, -1)
            valid = ~torch.as_tensor(item[f"{feature}_is_pad"], dtype=torch.bool)
            proprio_parts.append(values)
            proprio_masks.append(valid[:, None].expand_as(values))
        if proprio_parts:
            proprio = torch.cat(proprio_parts, dim=-1)
            proprio_mask = torch.cat(proprio_masks, dim=-1)
        else:
            proprio = torch.empty(time_steps, 0, dtype=torch.float32)
            proprio_mask = torch.empty(time_steps, 0, dtype=torch.bool)

        action = torch.as_tensor(
            item[self.dataset_spec.action_feature], dtype=torch.float32
        ).reshape(time_steps - 1, -1)
        action_dimension_mask = action_valid[:, None].expand_as(action).clone()
        normalized_action = self.action_spec.normalize(action, action_dimension_mask)
        delta_time = torch.full(
            (time_steps - 1,), 1.0 / self.dataset_spec.fps, dtype=torch.float32
        )
        task_text = item[self.dataset_spec.task_feature]
        if self.dataset_spec.language_feature is not None:
            task_text = item[self.dataset_spec.language_feature]
        if task_text is not None and not isinstance(task_text, str):
            raise ValueError("task/language feature must resolve to a string")

        return TrajectorySample(
            rgb=rgb,
            camera_mask=camera_mask,
            proprio=proprio,
            proprio_mask=proprio_mask,
            action=normalized_action,
            action_dimension_mask=action_dimension_mask,
            observation_valid_mask=observation_valid,
            action_valid_mask=action_valid,
            transition_valid_mask=transition_valid,
            delta_time=delta_time,
            context_index=self.context_index,
            task_text=task_text,
            condition_ids={},
            dataset_id=self.dataset_spec.dataset_id,
            action_spec_id=self.action_spec.spec_id,
        )

    def _validate_metadata(self, dataset: object) -> None:
        from lerobot.datasets.dataset_metadata import CODEBASE_VERSION

        if dataset.meta.info.get("codebase_version") != CODEBASE_VERSION:
            raise ValueError(
                "LeRobot codebase version mismatch: "
                f"expected {CODEBASE_VERSION}, got "
                f"{dataset.meta.info.get('codebase_version')}"
            )
        if not math.isclose(dataset.fps, self.dataset_spec.fps, abs_tol=1e-9):
            raise ValueError(
                f"dataset FPS {dataset.fps} does not match DatasetSpec "
                f"{self.dataset_spec.fps}"
            )
        if not math.isclose(
            self.action_spec.frequency_hz, self.dataset_spec.fps, abs_tol=1e-9
        ):
            raise ValueError("ActionSpec frequency_hz must match DatasetSpec fps")

        features = dataset.features
        action_key = self.dataset_spec.action_feature
        if action_key not in features:
            raise ValueError(f"missing required action feature: {action_key}")
        action_feature = features[action_key]
        action_shape = tuple(action_feature.get("shape", ()))
        if action_shape != (self.action_spec.dimension,):
            raise ValueError(
                f"action feature shape {action_shape} does not match ActionSpec "
                f"dimension {self.action_spec.dimension}"
            )
        action_names = action_feature.get("names")
        if action_names is None or tuple(action_names) != self.action_spec.names:
            raise ValueError("action feature names do not match ActionSpec names")

        for role, feature in self.dataset_spec.camera_features.items():
            if feature not in features:
                if role in self.dataset_spec.optional_camera_roles:
                    continue
                raise ValueError(f"missing required camera feature: {feature}")
            feature_info = features[feature]
            if feature_info.get("dtype") not in {"video", "image"}:
                raise ValueError(f"camera feature {feature} is not image/video")
            shape = tuple(feature_info.get("shape", ()))
            if len(shape) != 3 or shape[-1] != 3:
                raise ValueError(f"camera feature {feature} must have HWC RGB shape")

        for feature in self.dataset_spec.proprio_features:
            if feature not in features:
                raise ValueError(f"missing declared proprio feature: {feature}")
            shape = tuple(features[feature].get("shape", ()))
            if not shape or math.prod(shape) <= 0:
                raise ValueError(f"proprio feature {feature} has invalid shape {shape}")
        if (
            self.dataset_spec.language_feature is not None
            and self.dataset_spec.language_feature not in features
        ):
            raise ValueError(
                f"missing language feature: {self.dataset_spec.language_feature}"
            )

        ucf_metadata = dataset.meta.info.get("ucf")
        if not isinstance(ucf_metadata, Mapping):
            raise ValueError("missing UCF training-split normalization metadata")
        action_stats = ucf_metadata.get("train_action_stats")
        if not isinstance(action_stats, Mapping):
            raise ValueError("missing UCF train_action_stats metadata")
        if action_stats.get("feature") != action_key:
            raise ValueError("UCF train_action_stats feature does not match ActionSpec")
        count = action_stats.get("count")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("UCF train_action_stats count must be positive")
        for stat_name, expected in (
            ("mean", self.action_spec.normalization_mean),
            ("std", self.action_spec.normalization_std),
        ):
            if stat_name not in action_stats:
                raise ValueError(f"missing action statistic: {action_key}.{stat_name}")
            actual = torch.as_tensor(action_stats[stat_name], dtype=torch.float64).reshape(-1)
            expected_tensor = torch.tensor(expected, dtype=torch.float64)
            if actual.shape != expected_tensor.shape or not torch.allclose(
                actual, expected_tensor, atol=1e-6, rtol=1e-6
            ):
                raise ValueError(
                    f"ActionSpec normalization_{stat_name} does not match "
                    "training-split stats"
                )

        available_episodes = extract_episode_ids(dataset.meta.episodes)
        requested_episodes = set(self.dataset_spec.split_episode_ids[self.split])
        missing_episodes = requested_episodes - available_episodes
        if missing_episodes:
            raise ValueError(f"split references missing episodes: {sorted(missing_episodes)}")

        self._resolve_camera_shape(features)

    def _resolve_camera_shape(self, features: Mapping[str, Mapping]) -> tuple[int, int, int]:
        chw_shapes = {
            (
                int(features[feature]["shape"][2]),
                int(features[feature]["shape"][0]),
                int(features[feature]["shape"][1]),
            )
            for feature in self.dataset_spec.camera_features.values()
            if feature in features
        }
        if not chw_shapes:
            raise ValueError("dataset has no declared camera features")
        if len(chw_shapes) != 1:
            raise ValueError(
                "declared cameras must share resolution for tensor collation; "
                f"got {sorted(chw_shapes)}"
            )
        return next(iter(chw_shapes))

    def _as_chw_video(self, value: object, feature: str) -> Tensor:
        tensor = torch.as_tensor(value)
        if tensor.ndim != 4:
            raise ValueError(f"camera window {feature} must be four-dimensional")
        if tensor.shape[1:] == self._camera_shape:
            result = tensor
        elif tensor.shape[-1] == 3:
            result = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"camera window {feature} shape {tuple(tensor.shape)} is not CHW/HWC RGB"
            )
        if result.dtype == torch.uint8:
            result = result.to(torch.float32).div_(255.0)
        else:
            result = result.to(torch.float32)
        if not torch.isfinite(result).all().item():
            raise ValueError(f"camera window {feature} contains non-finite values")
        if (result < 0).any().item() or (result > 1).any().item():
            raise ValueError(f"camera window {feature} must be normalized to [0,1]")
        return result.contiguous()

    def _build_consumed_keys(self) -> set[str]:
        keys = {self.dataset_spec.action_feature, self.dataset_spec.task_feature}
        keys.add(f"{self.dataset_spec.action_feature}_is_pad")
        if self.dataset_spec.language_feature is not None:
            keys.add(self.dataset_spec.language_feature)
        for role in self.camera_roles:
            if role not in self._present_camera_roles:
                continue
            feature = self.dataset_spec.camera_features[role]
            keys.update((feature, f"{feature}_is_pad"))
        for feature in self.dataset_spec.proprio_features:
            keys.update((feature, f"{feature}_is_pad"))
        return keys
