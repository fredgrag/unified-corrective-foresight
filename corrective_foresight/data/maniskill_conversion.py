from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid
import warnings

import gymnasium as gym
import h5py
import numpy as np
import torch
from corrective_foresight.config.schema import ActionSpec, DatasetSpec
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
)


MANISKILL_ACTION_NAMES = (
    "ee_delta_x",
    "ee_delta_y",
    "ee_delta_z",
    "ee_rotation_vector_x",
    "ee_rotation_vector_y",
    "ee_rotation_vector_z",
    "gripper_position",
)
MANISKILL_ACTION_UNITS = ("m", "m", "m", "rad", "rad", "rad", "m")
MANISKILL_PROPRIO_NAMES = (
    "panda_joint1_position",
    "panda_joint2_position",
    "panda_joint3_position",
    "panda_joint4_position",
    "panda_joint5_position",
    "panda_joint6_position",
    "panda_joint7_position",
    "panda_finger_joint1_position",
    "panda_finger_joint2_position",
    "panda_joint1_velocity",
    "panda_joint2_velocity",
    "panda_joint3_velocity",
    "panda_joint4_velocity",
    "panda_joint5_velocity",
    "panda_joint6_velocity",
    "panda_joint7_velocity",
    "panda_finger_joint1_velocity",
    "panda_finger_joint2_velocity",
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_qw",
    "tcp_qx",
    "tcp_qy",
    "tcp_qz",
)
UCF_PICK_CUBE_ENV_ID = "UCFPickCubeRGB-v1"
PICK_CUBE_TASK_TEXT = "Pick the red cube and place it at the green goal."


if UCF_PICK_CUBE_ENV_ID not in gym.registry:

    @register_env(UCF_PICK_CUBE_ENV_ID, max_episode_steps=50)
    class UCFPickCubeRGBEnv(PickCubeEnv):
        SUPPORTED_ROBOTS = [*PickCubeEnv.SUPPORTED_ROBOTS, "panda_wristcam"]

        def __init__(
            self,
            *args,
            robot_uids: str = "panda_wristcam",
            **kwargs,
        ) -> None:
            super().__init__(*args, robot_uids=robot_uids, **kwargs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hdf5_tree(group: h5py.Group) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            result[key] = _load_hdf5_tree(value)
        elif isinstance(value, h5py.Dataset):
            result[key] = value[:]
        else:
            raise TypeError(f"unsupported HDF5 object at {value.name}: {type(value)}")
    return result


def _iter_hdf5_datasets(group: h5py.Group):
    for value in group.values():
        if isinstance(value, h5py.Group):
            yield from _iter_hdf5_datasets(value)
        elif isinstance(value, h5py.Dataset):
            yield value


@dataclass(frozen=True, slots=True)
class ManiSkillEpisode:
    source_episode_id: int
    seed: int
    control_mode: str
    reset_kwargs: Mapping[str, Any]
    actions: np.ndarray
    env_states: tuple[Mapping[str, Any], ...]
    success: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_success: bool


class ManiSkillEpisodeReader:
    def __init__(self, h5_path: str | Path, json_path: str | Path) -> None:
        self.h5_path = Path(h5_path).expanduser().resolve()
        self.json_path = Path(json_path).expanduser().resolve()
        for name, path in (("HDF5", self.h5_path), ("JSON", self.json_path)):
            if not path.is_file():
                raise FileNotFoundError(f"ManiSkill {name} source does not exist: {path}")
        self.h5_sha256 = _sha256_file(self.h5_path)
        self.json_sha256 = _sha256_file(self.json_path)

        metadata = json.loads(self.json_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("ManiSkill trajectory JSON must contain an object")
        env_info = metadata.get("env_info")
        if not isinstance(env_info, dict) or env_info.get("env_id") != "PickCube-v1":
            raise ValueError("source env_info.env_id must be PickCube-v1")
        env_kwargs = env_info.get("env_kwargs")
        if not isinstance(env_kwargs, dict):
            raise ValueError("source env_info.env_kwargs must be an object")
        if env_kwargs.get("control_mode") != "pd_ee_delta_pose":
            raise ValueError("source control_mode must be pd_ee_delta_pose")
        episodes = metadata.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("source episodes must be a nonempty list")

        records: dict[int, Mapping[str, Any]] = {}
        with h5py.File(self.h5_path, "r") as file:
            for position, record in enumerate(episodes):
                if not isinstance(record, dict):
                    raise ValueError(f"episode record {position} must be an object")
                episode_id = record.get("episode_id")
                if not isinstance(episode_id, int) or episode_id < 0:
                    raise ValueError(f"episode record {position} has invalid episode_id")
                if episode_id in records:
                    raise ValueError(f"duplicate source episode_id: {episode_id}")
                if record.get("control_mode") != "pd_ee_delta_pose":
                    raise ValueError(
                        f"episode {episode_id} control_mode must be pd_ee_delta_pose"
                    )
                seed = record.get("episode_seed")
                if not isinstance(seed, int) or seed < 0:
                    raise ValueError(f"episode {episode_id} has invalid episode_seed")
                elapsed_steps = record.get("elapsed_steps")
                if not isinstance(elapsed_steps, int) or elapsed_steps <= 0:
                    raise ValueError(f"episode {episode_id} has invalid elapsed_steps")
                reset_kwargs = record.get("reset_kwargs")
                if not isinstance(reset_kwargs, dict):
                    raise ValueError(f"episode {episode_id} reset_kwargs must be an object")

                group_name = f"traj_{episode_id}"
                if group_name not in file:
                    raise ValueError(f"episode {episode_id} is missing HDF5 group {group_name}")
                group = file[group_name]
                if "actions" not in group or group["actions"].shape != (elapsed_steps, 7):
                    raise ValueError(
                        f"episode {episode_id} elapsed_steps does not match actions shape"
                    )
                if group["actions"].dtype.kind != "f":
                    raise ValueError(f"episode {episode_id} actions must be floating point")
                if "env_states" not in group:
                    raise ValueError(f"episode {episode_id} lacks env_states")
                state_datasets = tuple(_iter_hdf5_datasets(group["env_states"]))
                if not state_datasets or any(
                    dataset.ndim == 0 or dataset.shape[0] != elapsed_steps + 1
                    for dataset in state_datasets
                ):
                    raise ValueError(
                        f"episode {episode_id} env_states must have elapsed_steps + 1 states"
                    )
                for key in ("success", "terminated", "truncated"):
                    if key not in group or group[key].shape != (elapsed_steps,):
                        raise ValueError(
                            f"episode {episode_id} {key} must match elapsed_steps"
                        )
                    if group[key].dtype.kind != "b":
                        raise ValueError(f"episode {episode_id} {key} must be boolean")
                final_success = bool(group["success"][-1])
                if bool(record.get("success")) != final_success:
                    raise ValueError(
                        f"episode {episode_id} final success disagrees with metadata"
                    )
                records[episode_id] = MappingProxyType(dict(record))

        self.metadata = MappingProxyType(metadata)
        self.env_info = MappingProxyType(dict(env_info))
        self._records = MappingProxyType(records)
        self.episode_records = self._records
        self.episode_ids = tuple(records)

    def read_actions(self, source_episode_id: int) -> np.ndarray:
        if source_episode_id not in self._records:
            raise KeyError(f"unknown source episode_id: {source_episode_id}")
        with h5py.File(self.h5_path, "r") as file:
            result = np.asarray(
                file[f"traj_{source_episode_id}/actions"][:], dtype=np.float32
            )
        if not np.isfinite(result).all():
            raise ValueError(f"episode {source_episode_id} actions contain non-finite values")
        return result

    def read_actions_by_episode(self) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        with h5py.File(self.h5_path, "r") as file:
            for episode_id in self.episode_ids:
                actions = np.asarray(
                    file[f"traj_{episode_id}/actions"][:], dtype=np.float32
                )
                if not np.isfinite(actions).all():
                    raise ValueError(
                        f"episode {episode_id} actions contain non-finite values"
                    )
                result[episode_id] = actions
        return result

    def read_episode(self, source_episode_id: int) -> ManiSkillEpisode:
        if source_episode_id not in self._records:
            raise KeyError(f"unknown source episode_id: {source_episode_id}")
        record = self._records[source_episode_id]
        with h5py.File(self.h5_path, "r") as file:
            group = file[f"traj_{source_episode_id}"]
            actions = np.asarray(group["actions"][:], dtype=np.float32)
            states_tree = _load_hdf5_tree(group["env_states"])
            env_states = tuple(trajectory_utils.dict_to_list_of_dicts(states_tree))
            success = np.asarray(group["success"][:], dtype=np.bool_)
            terminated = np.asarray(group["terminated"][:], dtype=np.bool_)
            truncated = np.asarray(group["truncated"][:], dtype=np.bool_)
        if not np.isfinite(actions).all():
            raise ValueError(f"episode {source_episode_id} actions contain non-finite values")
        return ManiSkillEpisode(
            source_episode_id=source_episode_id,
            seed=int(record["episode_seed"]),
            control_mode=str(record["control_mode"]),
            reset_kwargs=MappingProxyType(dict(record["reset_kwargs"])),
            actions=actions,
            env_states=env_states,
            success=success,
            terminated=terminated,
            truncated=truncated,
            final_success=bool(record["success"]),
        )


def expected_panda_ee_delta_pose_contract() -> PandaEEDeltaPoseContract:
    return PandaEEDeltaPoseContract(
        control_mode="pd_ee_delta_pose",
        frequency_hz=20.0,
        dimension=7,
        frame="root_translation:root_aligned_body_rotation",
        translation_lower=(-0.1, -0.1, -0.1),
        translation_upper=(0.1, 0.1, 0.1),
        rotation_lower=(-0.1, -0.1, -0.1),
        rotation_upper=(0.1, 0.1, 0.1),
        gripper_lower=-0.01,
        gripper_upper=0.04,
        normalized_action=True,
    )


def make_pick_cube_rgb_env():
    return gym.make(
        UCF_PICK_CUBE_ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda_wristcam",
        sim_backend="physx_cpu",
        render_mode="rgb_array",
    )


def step_maniskill_env(env: object, action: object):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"__array_wrap__ must accept context and return_scalar arguments.*",
            category=DeprecationWarning,
        )
        return env.step(action)


def _broadcast_three(value: object, name: str) -> tuple[float, float, float]:
    try:
        result = np.broadcast_to(np.asarray(value, dtype=np.float64), (3,))
    except ValueError as error:
        raise ValueError(f"controller {name} must broadcast to three values") from error
    return tuple(float(item) for item in result)


def query_panda_ee_delta_pose_contract(env: object) -> PandaEEDeltaPoseContract:
    base_env = env.unwrapped
    if base_env.control_mode != "pd_ee_delta_pose":
        raise ValueError(f"unexpected control mode: {base_env.control_mode}")
    if base_env.agent.uid != "panda_wristcam":
        raise ValueError(f"unexpected robot UID: {base_env.agent.uid}")
    controller = base_env.agent.controller
    if set(controller.configs) != {"arm", "gripper"}:
        raise ValueError("Panda controller must contain exactly arm and gripper")
    arm = controller.configs["arm"]
    gripper = controller.configs["gripper"]
    action_space = env.action_space
    if tuple(action_space.shape) != (7,) or not np.allclose(
        action_space.low, -1.0
    ) or not np.allclose(action_space.high, 1.0):
        raise ValueError("Panda environment action space must be normalized 7-D [-1,1]")
    return PandaEEDeltaPoseContract(
        control_mode=base_env.control_mode,
        frequency_hz=float(base_env.control_freq),
        dimension=int(action_space.shape[0]),
        frame=str(arm.frame),
        translation_lower=_broadcast_three(arm.pos_lower, "pos_lower"),
        translation_upper=_broadcast_three(arm.pos_upper, "pos_upper"),
        rotation_lower=_broadcast_three(arm.rot_lower, "rot_lower"),
        rotation_upper=_broadcast_three(arm.rot_upper, "rot_upper"),
        gripper_lower=float(gripper.lower),
        gripper_upper=float(gripper.upper),
        normalized_action=bool(arm.normalize_action and gripper.normalize_action),
    )


def _to_numpy_unbatched(value: object, name: str) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 0 or array.shape[0] != 1:
        raise ValueError(f"{name} must have a leading singleton environment dimension")
    result = array[0]
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def extract_rgb_proprio(
    observation: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    try:
        sensor_data = observation["sensor_data"]
        agent = observation["agent"]
        extra = observation["extra"]
    except KeyError as error:
        raise ValueError(f"RGB observation is missing section: {error.args[0]}") from error
    camera_sources = {"base": "base_camera", "wrist": "hand_camera"}
    cameras: dict[str, np.ndarray] = {}
    for role, sensor_name in camera_sources.items():
        try:
            image = _to_numpy_unbatched(
                sensor_data[sensor_name]["rgb"], f"{sensor_name}.rgb"
            )
        except KeyError as error:
            raise ValueError(f"missing required {role} RGB camera") from error
        if image.shape != (128, 128, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"{sensor_name}.rgb must be uint8 HWC (128,128,3), got "
                f"{image.shape} {image.dtype}"
            )
        if int(image.max()) <= int(image.min()):
            raise ValueError(f"{sensor_name}.rgb is constant")
        cameras[role] = image.copy()

    proprio_parts = (
        _to_numpy_unbatched(agent["qpos"], "agent.qpos"),
        _to_numpy_unbatched(agent["qvel"], "agent.qvel"),
        _to_numpy_unbatched(extra["tcp_pose"], "extra.tcp_pose"),
    )
    proprio = np.concatenate(proprio_parts).astype(np.float32)
    if proprio.shape != (len(MANISKILL_PROPRIO_NAMES),):
        raise ValueError(f"unexpected Panda proprio shape: {proprio.shape}")
    return cameras, proprio


def _batch_state_tree(value: object, device: torch.device):
    if isinstance(value, Mapping):
        return {key: _batch_state_tree(child, device) for key, child in value.items()}
    tensor = torch.as_tensor(value, device=device)
    return tensor.unsqueeze(0)


def remap_source_articulation_keys(
    state: Mapping[str, Any], target_articulation_keys: Sequence[str]
) -> dict[str, Any]:
    if "articulations" not in state or not isinstance(state["articulations"], Mapping):
        raise ValueError("source state lacks articulations mapping")
    source_keys = set(state["articulations"])
    target_keys = set(target_articulation_keys)
    result = dict(state)
    if source_keys == target_keys:
        result["articulations"] = dict(state["articulations"])
        return result
    if source_keys == {"panda"} and target_keys == {"panda_wristcam"}:
        result["articulations"] = {
            "panda_wristcam": state["articulations"]["panda"]
        }
        return result
    raise ValueError(
        "source and target articulation keys differ: "
        f"source={sorted(source_keys)}, target={sorted(target_keys)}"
    )


def prepare_source_state_for_env(state: Mapping[str, Any], env: object):
    target_state = env.unwrapped.get_state_dict()
    remapped = remap_source_articulation_keys(
        state, tuple(target_state["articulations"])
    )
    if set(remapped.get("actors", ())) != set(target_state.get("actors", ())):
        raise ValueError("source and target actor keys differ")
    return _batch_state_tree(remapped, env.unwrapped.device)


def _video_feature(fps: int) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": (128, 128, 3),
        "names": ["height", "width", "channels"],
        "video.fps": fps,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }


def _save_lerobot_episode(dataset: object) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"Conversion of an array with ndim > 0 to a scalar is deprecated.*"
            ),
            category=DeprecationWarning,
        )
        dataset.save_episode(parallel_encoding=False)


@dataclass(frozen=True, slots=True)
class ManiSkillConversionResult:
    output_root: Path
    dataset_spec: DatasetSpec
    action_spec: ActionSpec
    provenance: Mapping[str, Any]


def _validate_source_splits(
    episode_ids: Sequence[int], source_splits: Mapping[str, Sequence[int]]
) -> dict[str, tuple[int, ...]]:
    required = ("train", "validation", "evaluation")
    if set(source_splits) != set(required):
        raise ValueError(f"source_splits must contain exactly {required}")
    normalized = {
        name: tuple(int(episode_id) for episode_id in source_splits[name])
        for name in required
    }
    if any(not values for values in normalized.values()):
        raise ValueError("every source split must be nonempty")
    flattened = [episode_id for values in normalized.values() for episode_id in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("source episode splits must be disjoint")
    if set(flattened) != set(episode_ids):
        raise ValueError("source episode splits must cover exactly the reader episodes")
    return normalized


def _build_specs(
    output_root: Path,
    output_splits: Mapping[str, tuple[int, ...]],
    action_stats: Mapping[str, np.ndarray | int],
    revision: str,
) -> tuple[DatasetSpec, ActionSpec]:
    mean = tuple(float(value) for value in np.asarray(action_stats["mean"]).reshape(-1))
    std = tuple(float(value) for value in np.asarray(action_stats["std"]).reshape(-1))
    action_spec = ActionSpec(
        schema_version=1,
        spec_id="maniskill.panda.pd_ee_delta_pose.physical.v1",
        dimension=7,
        names=MANISKILL_ACTION_NAMES,
        units=MANISKILL_ACTION_UNITS,
        action_space="end_effector",
        mode="delta",
        rotation_representation="axis_angle",
        arm_count=1,
        gripper_indices=(6,),
        control_mode="pd_ee_delta_pose",
        frequency_hz=20.0,
        normalization_mean=mean,
        normalization_std=std,
        minimum=(-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.01),
        maximum=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.04),
    )
    dataset_spec = DatasetSpec(
        schema_version=1,
        dataset_id="maniskill.pick_cube.panda_wristcam.v1",
        repo_id=None,
        local_root=str(output_root),
        revision=revision,
        fps=20.0,
        camera_features={
            "base": "observation.images.base",
            "wrist": "observation.images.wrist",
        },
        optional_camera_roles=(),
        proprio_features=("observation.state",),
        proprio_required=True,
        task_feature="task",
        language_feature=None,
        embodiment_id="maniskill.panda_wristcam.v1",
        action_feature="action",
        action_spec_id=action_spec.spec_id,
        sample_weight=1.0,
        split_episode_ids=output_splits,
    )
    return dataset_spec, action_spec


def _provenance(
    reader: ManiSkillEpisodeReader,
    source_splits: Mapping[str, tuple[int, ...]],
    output_splits: Mapping[str, tuple[int, ...]],
    converter_git_commit: str,
    contract: PandaEEDeltaPoseContract,
) -> dict[str, Any]:
    records = [reader.episode_records[episode_id] for episode_id in reader.episode_ids]
    success_count = sum(bool(record["success"]) for record in records)
    return {
        "schema_version": 1,
        "source": {
            "h5_path": str(reader.h5_path),
            "json_path": str(reader.json_path),
            "h5_sha256": reader.h5_sha256,
            "json_sha256": reader.json_sha256,
            "commit_info": reader.metadata.get("commit_info"),
            "env_id": "PickCube-v1",
            "control_mode": "pd_ee_delta_pose",
            "episode_count": len(reader.episode_ids),
            "success_count": success_count,
            "failure_count": len(reader.episode_ids) - success_count,
        },
        "runtime": {
            "mani_skill": importlib.metadata.version("mani_skill"),
            "lerobot": importlib.metadata.version("lerobot"),
            "h5py": importlib.metadata.version("h5py"),
            "target_env_id": UCF_PICK_CUBE_ENV_ID,
            "robot_uid": "panda_wristcam",
            "obs_mode": "rgb",
            "sim_backend": "physx_cpu",
            "camera_roles": {"base": "base_camera", "wrist": "hand_camera"},
            "camera_resolution": [128, 128],
            "fps": 20,
            "streaming_encoding": True,
            "video_files_size_in_mb": 1,
        },
        "controller_contract": {
            field.name: (
                list(getattr(contract, field.name))
                if isinstance(getattr(contract, field.name), tuple)
                else getattr(contract, field.name)
            )
            for field in fields(PandaEEDeltaPoseContract)
        },
        "source_splits": {name: list(values) for name, values in source_splits.items()},
        "output_splits": {name: list(values) for name, values in output_splits.items()},
        "episode_seeds": {
            str(episode_id): int(reader.episode_records[episode_id]["episode_seed"])
            for episode_id in reader.episode_ids
        },
        "converter_git_commit": converter_git_commit,
    }


def _write_ucf_metadata(
    dataset: object,
    provenance: Mapping[str, Any],
    action_stats: Mapping[str, np.ndarray | int],
    output_splits: Mapping[str, tuple[int, ...]],
) -> None:
    from lerobot.datasets.io_utils import write_info

    dataset.meta.info["splits"] = {
        name: f"{values[0]}:{values[-1] + 1}" for name, values in output_splits.items()
    }
    dataset.meta.info["ucf"] = {
        "schema_version": 1,
        "train_action_stats": {
            "feature": "action",
            "mean": np.asarray(action_stats["mean"]).reshape(-1).tolist(),
            "std": np.asarray(action_stats["std"]).reshape(-1).tolist(),
            "count": int(action_stats["count"]),
        },
        "provenance": provenance,
    }
    write_info(dataset.meta.info, dataset.meta.root)


def plan_maniskill_conversion(
    reader: ManiSkillEpisodeReader,
    output_root: str | Path,
    source_splits: Mapping[str, Sequence[int]],
    converter_git_commit: str,
) -> ManiSkillConversionResult:
    output_root = Path(output_root).expanduser().resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", converter_git_commit):
        raise ValueError("converter_git_commit must be a 40-character lowercase Git SHA")
    normalized_splits = _validate_source_splits(reader.episode_ids, source_splits)
    output_order = tuple(
        episode_id
        for split_name in ("train", "validation", "evaluation")
        for episode_id in normalized_splits[split_name]
    )
    source_to_output = {
        source_episode_id: output_episode_id
        for output_episode_id, source_episode_id in enumerate(output_order)
    }
    output_splits = {
        name: tuple(source_to_output[episode_id] for episode_id in values)
        for name, values in normalized_splits.items()
    }
    contract = expected_panda_ee_delta_pose_contract()
    source_actions = reader.read_actions_by_episode()
    actions_by_episode = {
        episode_id: controller_action_to_physical(
            source_actions[episode_id], contract
        )
        for episode_id in reader.episode_ids
    }
    action_stats = compute_action_statistics(
        actions_by_episode,
        normalized_splits["train"],
    )
    provenance = _provenance(
        reader,
        normalized_splits,
        output_splits,
        converter_git_commit,
        contract,
    )
    revision = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    dataset_spec, action_spec = _build_specs(
        output_root,
        output_splits,
        action_stats,
        revision,
    )
    return ManiSkillConversionResult(
        output_root=output_root,
        dataset_spec=dataset_spec,
        action_spec=action_spec,
        provenance=MappingProxyType(provenance),
    )


def validate_converted_lerobot_dataset(
    root: Path,
    repo_id: str,
    expected_episodes: int,
    expected_frames: int,
) -> dict[str, int]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.dataset as pyarrow_dataset

    reopened = LeRobotDataset(repo_id=repo_id, root=root, video_backend="torchcodec")
    if reopened.meta.total_episodes != expected_episodes:
        raise ValueError("converted LeRobot episode count mismatch")
    if reopened.meta.total_frames != expected_frames:
        raise ValueError("converted LeRobot frame count mismatch")
    video_paths = tuple(root.rglob("*.mp4"))
    for video_key in ("observation.images.base", "observation.images.wrist"):
        if not any(video_key in str(path) for path in video_paths):
            raise ValueError(f"converted LeRobot video is missing: {video_key}")
    if reopened.meta.info.get("ucf", {}).get("schema_version") != 1:
        raise ValueError("converted LeRobot provenance metadata is missing")
    provenance = reopened.meta.info["ucf"].get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("converted LeRobot provenance is invalid")
    output_splits = provenance.get("output_splits")
    if not isinstance(output_splits, Mapping):
        raise ValueError("converted LeRobot output splits are missing")
    normalized_splits = {
        name: tuple(int(episode_id) for episode_id in output_splits.get(name, ()))
        for name in ("train", "validation", "evaluation")
    }
    split_ids = [episode_id for values in normalized_splits.values() for episode_id in values]
    if len(split_ids) != len(set(split_ids)) or set(split_ids) != set(
        range(expected_episodes)
    ):
        raise ValueError("converted LeRobot splits are not disjoint and complete")

    data_paths = tuple(sorted((root / "data").rglob("*.parquet")))
    if not data_paths:
        raise ValueError("converted LeRobot dataset has no data Parquet files")
    table = pyarrow_dataset.dataset(
        [str(path) for path in data_paths], format="parquet"
    ).to_table(
        columns=[
            "index",
            "episode_index",
            "frame_index",
            "timestamp",
            "action",
            "next.success",
        ]
    )
    indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
    order = np.argsort(indices)
    indices = indices[order]
    if not np.array_equal(indices, np.arange(expected_frames)):
        raise ValueError("converted LeRobot global frame indices are not contiguous")
    episode_indices = np.asarray(
        table["episode_index"].to_numpy(), dtype=np.int64
    )[order]
    frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)[
        order
    ]
    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)[order]
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)[order]
    successes = np.asarray(
        table["next.success"].to_numpy(), dtype=np.float64
    )[order]
    if actions.shape != (expected_frames, 7) or not np.isfinite(actions).all():
        raise ValueError("converted LeRobot actions are invalid")
    if not np.isfinite(successes).all() or not np.isin(successes, (0.0, 1.0)).all():
        raise ValueError("converted LeRobot success values are not binary")

    physical_lower = np.asarray(
        (-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.01)
    )
    physical_upper = np.asarray((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.04))
    if np.any(actions < physical_lower - 1e-6) or np.any(
        actions > physical_upper + 1e-6
    ):
        raise ValueError("converted LeRobot actions exceed physical ActionSpec bounds")

    episode_records = sorted(
        reopened.meta.episodes, key=lambda record: int(record["episode_index"])
    )
    if [int(record["episode_index"]) for record in episode_records] != list(
        range(expected_episodes)
    ):
        raise ValueError("converted LeRobot episode indices are not contiguous")
    valid_transitions = 0
    final_success_count = 0
    for record in episode_records:
        episode_id = int(record["episode_index"])
        start = int(record["dataset_from_index"])
        stop = int(record["dataset_to_index"])
        length = int(record["length"])
        if stop - start != length or length <= 0:
            raise ValueError(f"converted episode {episode_id} length metadata is invalid")
        if not np.all(episode_indices[start:stop] == episode_id):
            raise ValueError(f"converted episode {episode_id} frame ownership is invalid")
        if not np.array_equal(frame_indices[start:stop], np.arange(length)):
            raise ValueError(f"converted episode {episode_id} frame indices are invalid")
        expected_timestamps = np.arange(length, dtype=np.float64) / reopened.fps
        if not np.allclose(
            timestamps[start:stop], expected_timestamps, atol=1e-7, rtol=0.0
        ):
            raise ValueError(f"converted episode {episode_id} timestamps are invalid")
        valid_transitions += length - 1
        final_success_count += int(successes[stop - 1])

    source = provenance.get("source")
    if not isinstance(source, Mapping) or final_success_count != int(
        source.get("success_count", -1)
    ):
        raise ValueError("converted LeRobot final success distribution mismatches source")

    training_mask = np.isin(episode_indices, normalized_splits["train"])
    training_actions = actions[training_mask]
    training_stats = reopened.meta.info["ucf"].get("train_action_stats")
    if not isinstance(training_stats, Mapping):
        raise ValueError("converted LeRobot training action stats are missing")
    expected_count = int(training_stats.get("count", -1))
    if training_actions.shape[0] != expected_count:
        raise ValueError("converted LeRobot training action count mismatches metadata")
    for stat_name, actual in (
        ("mean", training_actions.mean(axis=0)),
        ("std", training_actions.std(axis=0)),
    ):
        expected = np.asarray(training_stats.get(stat_name), dtype=np.float64)
        if expected.shape != (7,) or not np.allclose(
            actual, expected, atol=1e-7, rtol=1e-6
        ):
            raise ValueError(
                f"converted LeRobot training action {stat_name} mismatches metadata"
            )

    for index in {0, expected_frames - 1}:
        item = reopened[index]
        for key in ("observation.images.base", "observation.images.wrist"):
            image = torch.as_tensor(item[key])
            if image.shape != (3, 128, 128) or not torch.isfinite(image).all().item():
                raise ValueError(f"converted video frame is invalid: {key}")
        if torch.as_tensor(item["observation.state"]).shape != (25,):
            raise ValueError("converted proprio shape mismatch")
        if torch.as_tensor(item["action"]).shape != (7,):
            raise ValueError("converted action shape mismatch")
    return {
        "episodes": expected_episodes,
        "frames": expected_frames,
        "valid_transitions": valid_transitions,
        "successes": final_success_count,
        "videos": len(video_paths),
    }


def convert_to_lerobot_v3(
    reader: ManiSkillEpisodeReader,
    output_root: str | Path,
    repo_id: str,
    source_splits: Mapping[str, Sequence[int]],
    converter_git_commit: str,
) -> ManiSkillConversionResult:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace existing dataset root: {output_root}")
    normalized_splits = _validate_source_splits(reader.episode_ids, source_splits)
    output_order = tuple(
        episode_id
        for split_name in ("train", "validation", "evaluation")
        for episode_id in normalized_splits[split_name]
    )
    source_to_output = {
        source_episode_id: output_episode_id
        for output_episode_id, source_episode_id in enumerate(output_order)
    }
    output_splits = {
        name: tuple(source_to_output[episode_id] for episode_id in values)
        for name, values in normalized_splits.items()
    }

    plan = plan_maniskill_conversion(
        reader,
        output_root,
        normalized_splits,
        converter_git_commit,
    )
    contract = expected_panda_ee_delta_pose_contract()
    source_actions = reader.read_actions_by_episode()
    actions_by_episode = {
        episode_id: controller_action_to_physical(
            source_actions[episode_id], contract
        )
        for episode_id in reader.episode_ids
    }
    action_stats = compute_action_statistics(
        actions_by_episode,
        normalized_splits["train"],
    )
    provenance = dict(plan.provenance)
    dataset_spec = plan.dataset_spec
    action_spec = plan.action_spec

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    env = None
    dataset = None
    try:
        env = make_pick_cube_rgb_env()
        actual_contract = query_panda_ee_delta_pose_contract(env)
        validate_controller_contract(actual_contract, contract)
        features = {
            "observation.images.base": _video_feature(20),
            "observation.images.wrist": _video_feature(20),
            "observation.state": {
                "dtype": "float32",
                "shape": (25,),
                "names": list(MANISKILL_PROPRIO_NAMES),
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": list(MANISKILL_ACTION_NAMES),
            },
            "next.success": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["success"],
            },
        }
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=20,
            features=features,
            root=temporary_root,
            robot_type="panda_wristcam",
            use_videos=True,
            vcodec="h264",
            streaming_encoding=True,
        )
        dataset.meta.update_chunk_settings(video_files_size_in_mb=1)
        total_frames = 0
        for source_episode_id in output_order:
            episode = reader.read_episode(source_episode_id)
            reset_kwargs = dict(episode.reset_kwargs)
            reset_kwargs.pop("seed", None)
            env.reset(seed=episode.seed, **reset_kwargs)
            for step_index, source_action in enumerate(episode.actions):
                state = prepare_source_state_for_env(
                    episode.env_states[step_index], env
                )
                env.unwrapped.set_state_dict(state)
                info = env.unwrapped.get_info()
                observation = env.unwrapped.get_obs(info)
                cameras, proprio = extract_rgb_proprio(observation)

                step_maniskill_env(
                    env, torch.as_tensor(source_action, dtype=torch.float32)
                )
                next_state = prepare_source_state_for_env(
                    episode.env_states[step_index + 1], env
                )
                env.unwrapped.set_state_dict(next_state)
                replay_success = bool(env.unwrapped.get_info()["success"].item())
                source_success = bool(episode.success[step_index])
                if replay_success != source_success:
                    raise ValueError(
                        f"episode {source_episode_id} step {step_index} replay success "
                        f"{replay_success} disagrees with source {source_success}"
                    )

                dataset.add_frame(
                    {
                        "task": PICK_CUBE_TASK_TEXT,
                        "observation.images.base": cameras["base"],
                        "observation.images.wrist": cameras["wrist"],
                        "observation.state": proprio,
                        "action": actions_by_episode[source_episode_id][step_index],
                        "next.success": np.asarray([source_success], dtype=np.float32),
                    }
                )
                total_frames += 1
            _save_lerobot_episode(dataset)
        dataset.finalize()
        _write_ucf_metadata(dataset, provenance, action_stats, output_splits)
        validate_converted_lerobot_dataset(
            temporary_root,
            repo_id,
            expected_episodes=len(output_order),
            expected_frames=total_frames,
        )
        if output_root.exists():
            raise FileExistsError(
                f"dataset root appeared during conversion: {output_root}"
            )
        os.replace(temporary_root, output_root)
        directory_fd = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if dataset is not None:
            try:
                dataset.finalize()
            except Exception:
                pass
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    finally:
        if env is not None:
            env.close()

    return ManiSkillConversionResult(
        output_root=output_root,
        dataset_spec=dataset_spec,
        action_spec=action_spec,
        provenance=MappingProxyType(provenance),
    )


def _finite_tuple(name: str, values: Sequence[float], length: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True, slots=True)
class PandaEEDeltaPoseContract:
    control_mode: str
    frequency_hz: float
    dimension: int
    frame: str
    translation_lower: tuple[float, float, float]
    translation_upper: tuple[float, float, float]
    rotation_lower: tuple[float, float, float]
    rotation_upper: tuple[float, float, float]
    gripper_lower: float
    gripper_upper: float
    normalized_action: bool

    def __post_init__(self) -> None:
        if self.control_mode != "pd_ee_delta_pose":
            raise ValueError("control_mode must be pd_ee_delta_pose")
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        if self.dimension != 7:
            raise ValueError("dimension must be 7")
        if self.frame != "root_translation:root_aligned_body_rotation":
            raise ValueError("unsupported controller frame")
        for name in (
            "translation_lower",
            "translation_upper",
            "rotation_lower",
            "rotation_upper",
        ):
            object.__setattr__(self, name, _finite_tuple(name, getattr(self, name), 3))
        if any(
            lower >= upper
            for lower, upper in zip(self.translation_lower, self.translation_upper)
        ):
            raise ValueError("translation bounds must be ordered")
        if any(
            lower >= upper
            for lower, upper in zip(self.rotation_lower, self.rotation_upper)
        ):
            raise ValueError("rotation bounds must be ordered")
        if not np.allclose(
            np.negative(self.rotation_lower), self.rotation_upper, atol=1e-9, rtol=0.0
        ):
            raise ValueError("rotation bounds must be symmetric")
        if not math.isfinite(self.gripper_lower) or not math.isfinite(self.gripper_upper):
            raise ValueError("gripper bounds must be finite")
        if self.gripper_lower >= self.gripper_upper:
            raise ValueError("gripper bounds must be ordered")
        if not self.normalized_action:
            raise ValueError("pd_ee_delta_pose action must be normalized")


def validate_controller_contract(
    actual: PandaEEDeltaPoseContract,
    expected: PandaEEDeltaPoseContract,
) -> None:
    for field in fields(PandaEEDeltaPoseContract):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if isinstance(expected_value, tuple):
            matches = np.allclose(actual_value, expected_value, atol=1e-9, rtol=0.0)
        elif isinstance(expected_value, float):
            matches = math.isclose(actual_value, expected_value, abs_tol=1e-9, rel_tol=0.0)
        else:
            matches = actual_value == expected_value
        if not matches:
            raise ValueError(
                f"controller contract {field.name} mismatch: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )


def _as_action_array(action: np.ndarray | Sequence[float]) -> tuple[np.ndarray, bool]:
    result = np.asarray(action, dtype=np.float64)
    squeeze = result.ndim == 1
    if squeeze:
        result = result[None, :]
    if result.ndim != 2 or result.shape[-1] != 7:
        raise ValueError("action must have final dimension 7")
    if not np.isfinite(result).all():
        raise ValueError("action must contain only finite values")
    return result, squeeze


def _scale_normalized(
    normalized: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(normalized, -1.0, 1.0)
    return lower + (clipped + 1.0) * 0.5 * (upper - lower)


def _unscale_physical(
    physical: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    name: str,
) -> np.ndarray:
    tolerance = 1e-6
    if np.any(physical < lower - tolerance) or np.any(physical > upper + tolerance):
        raise ValueError(f"{name} is outside executable controller bounds")
    return np.clip(2.0 * (physical - lower) / (upper - lower) - 1.0, -1.0, 1.0)


def controller_action_to_physical(
    action: np.ndarray | Sequence[float],
    contract: PandaEEDeltaPoseContract,
) -> np.ndarray:
    source, squeeze = _as_action_array(action)
    translation = _scale_normalized(
        source[:, :3],
        np.asarray(contract.translation_lower),
        np.asarray(contract.translation_upper),
    )

    normalized_rotation = source[:, 3:6].copy()
    rotation_norm = np.linalg.norm(normalized_rotation, axis=-1)
    oversized = rotation_norm > 1.0
    normalized_rotation[oversized] /= rotation_norm[oversized, None]
    euler_xyz = normalized_rotation * np.asarray(contract.rotation_lower)
    rotation_vector = matrix_to_axis_angle(
        euler_angles_to_matrix(torch.as_tensor(euler_xyz, dtype=torch.float64), "XYZ")
    ).cpu().numpy()

    gripper = _scale_normalized(
        source[:, 6:7],
        np.asarray([contract.gripper_lower]),
        np.asarray([contract.gripper_upper]),
    )
    result = np.concatenate((translation, rotation_vector, gripper), axis=-1).astype(
        np.float32
    )
    return result[0] if squeeze else result


def physical_action_to_controller(
    action: np.ndarray | Sequence[float],
    contract: PandaEEDeltaPoseContract,
) -> np.ndarray:
    physical, squeeze = _as_action_array(action)
    translation = _unscale_physical(
        physical[:, :3],
        np.asarray(contract.translation_lower),
        np.asarray(contract.translation_upper),
        "translation",
    )

    euler_xyz = matrix_to_euler_angles(
        axis_angle_to_matrix(torch.as_tensor(physical[:, 3:6], dtype=torch.float64)),
        "XYZ",
    ).cpu().numpy()
    normalized_rotation = euler_xyz / np.asarray(contract.rotation_lower)
    rotation_norm = np.linalg.norm(normalized_rotation, axis=-1)
    if np.any(rotation_norm > 1.0 + 1e-5):
        raise ValueError("rotation vector is outside executable controller bounds")
    normalized_rotation[rotation_norm > 1.0] /= rotation_norm[
        rotation_norm > 1.0, None
    ]

    gripper = _unscale_physical(
        physical[:, 6:7],
        np.asarray([contract.gripper_lower]),
        np.asarray([contract.gripper_upper]),
        "gripper position",
    )
    result = np.concatenate((translation, normalized_rotation, gripper), axis=-1).astype(
        np.float32
    )
    return result[0] if squeeze else result


def split_source_episode_ids(
    source_episode_ids: Sequence[int],
    source_sha256: str,
) -> dict[str, tuple[int, ...]]:
    ids = tuple(int(episode_id) for episode_id in source_episode_ids)
    if len(ids) < 3:
        raise ValueError("at least three source episodes are required for disjoint splits")
    if len(set(ids)) != len(ids) or any(episode_id < 0 for episode_id in ids):
        raise ValueError("source episode IDs must be unique nonnegative integers")
    if not source_sha256:
        raise ValueError("source_sha256 must be nonempty")

    ranked = sorted(
        ids,
        key=lambda episode_id: hashlib.sha256(
            f"{source_sha256}:{episode_id}".encode("ascii")
        ).digest(),
    )
    count = len(ranked)
    train_count = min(count - 2, max(1, round(count * 0.90)))
    validation_count = min(
        count - train_count - 1,
        max(1, round(count * 0.05)),
    )
    return {
        "train": tuple(sorted(ranked[:train_count])),
        "validation": tuple(
            sorted(ranked[train_count : train_count + validation_count])
        ),
        "evaluation": tuple(sorted(ranked[train_count + validation_count :])),
    }


def compute_action_statistics(
    actions_by_episode: Mapping[int, np.ndarray],
    training_episode_ids: Sequence[int],
) -> dict[str, np.ndarray | int]:
    training_ids = tuple(int(episode_id) for episode_id in training_episode_ids)
    if not training_ids:
        raise ValueError("training_episode_ids must be nonempty")
    missing = set(training_ids) - set(actions_by_episode)
    if missing:
        raise ValueError(f"missing training actions for episodes: {sorted(missing)}")
    arrays = []
    for episode_id in training_ids:
        action, _ = _as_action_array(actions_by_episode[episode_id])
        arrays.append(action)
    stacked = np.concatenate(arrays, axis=0)
    return {
        "mean": stacked.mean(axis=0),
        "std": stacked.std(axis=0),
        "count": int(stacked.shape[0]),
    }
