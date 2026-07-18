from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Protocol

import numpy as np
import torch

from corrective_foresight.config.schema import ActionSpec, DatasetSpec
from corrective_foresight.conditioning.vocabulary import CONDITION_NAMESPACES
from corrective_foresight.evaluation.records import EvaluationRecord
from corrective_foresight.evaluation.video import write_rollout_video_atomic
from corrective_foresight.policy.unified_policy import (
    ActionChunkPrediction,
    PolicyObservation,
)


_SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ClosedLoopPolicy(Protocol):
    def predict_action_chunk(
        self,
        observation: PolicyObservation,
        action_spec_id: str,
        flow_seed: int,
        solver: str,
    ) -> ActionChunkPrediction: ...


class OnlineObservationAdapter(Protocol):
    def reset(self, observation: object) -> AdaptedPolicyObservation: ...
    def append(self, observation: object) -> AdaptedPolicyObservation: ...


@dataclass(frozen=True, slots=True)
class AdaptedPolicyObservation:
    policy_observation: PolicyObservation
    video_frame: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.policy_observation, PolicyObservation):
            raise ValueError("adapted observation must contain PolicyObservation")
        if (
            not isinstance(self.video_frame, np.ndarray)
            or self.video_frame.dtype != np.uint8
            or self.video_frame.ndim != 3
            or self.video_frame.shape[2] != 3
        ):
            raise ValueError("adapted video frame must be uint8 HWC RGB")


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    label: str
    directory: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.label.strip() or not Path(self.directory).is_absolute():
            raise ValueError("checkpoint label and absolute directory are required")
        if len(self.manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.manifest_sha256
        ):
            raise ValueError("checkpoint manifest_sha256 is invalid")


@dataclass(frozen=True, slots=True)
class ManiSkillEnvironmentSpec:
    env_id: str
    robot_uid: str
    obs_mode: str
    control_mode: str
    sim_backend: str

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.env_id,
                self.robot_uid,
                self.obs_mode,
                self.control_mode,
                self.sim_backend,
            )
        ):
            raise ValueError("ManiSkill environment metadata must be nonempty")


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    action_horizon: int = 8
    execution_horizon: int = 1
    temporal_ensemble: bool = False
    solver: str = "midpoint"
    solver_intervals: int = 10
    context_steps: int = 1

    def __post_init__(self) -> None:
        if self.action_horizon != 8 or self.execution_horizon != 1:
            raise ValueError(
                "primary evaluation requires action horizon 8 and execution horizon 1"
            )
        if self.temporal_ensemble is not False:
            raise ValueError("temporal ensemble must be disabled in the primary protocol")
        if self.solver != "midpoint" or self.solver_intervals != 10:
            raise ValueError("primary evaluation requires midpoint with 10 intervals")
        if type(self.context_steps) is not int or self.context_steps <= 0:
            raise ValueError("context_steps must be a positive integer")


class ManiSkillClosedLoopRunner:
    def __init__(
        self,
        *,
        env_factory: Callable[[], object],
        policy: ClosedLoopPolicy,
        observation_adapter: OnlineObservationAdapter,
        action_spec: ActionSpec,
        dataset_spec: DatasetSpec,
        checkpoint: CheckpointProvenance,
        environment_spec: ManiSkillEnvironmentSpec,
        protocol: EvaluationProtocol,
        output_root: str | Path,
        max_steps: int,
        action_to_environment: Callable[[object, np.ndarray], np.ndarray],
    ) -> None:
        if dataset_spec.action_spec_id != action_spec.spec_id:
            raise ValueError("DatasetSpec and ActionSpec identifiers do not match")
        if dataset_spec.fps != action_spec.frequency_hz:
            raise ValueError("DatasetSpec and ActionSpec frequencies do not match")
        if environment_spec.control_mode != action_spec.control_mode:
            raise ValueError("environment and ActionSpec control modes do not match")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        destination = Path(output_root)
        if not destination.is_absolute():
            raise ValueError("evaluation output_root must be absolute")
        for name, value in (
            ("env_factory", env_factory),
            ("action_to_environment", action_to_environment),
        ):
            if not callable(value):
                raise ValueError(f"{name} must be callable")
        self.env_factory = env_factory
        self.policy = policy
        self.observation_adapter = observation_adapter
        self.action_spec = action_spec
        self.dataset_spec = dataset_spec
        self.checkpoint = checkpoint
        self.environment_spec = environment_spec
        self.protocol = protocol
        self.output_root = destination
        self.max_steps = max_steps
        self.action_to_environment = action_to_environment

    def run_episode(
        self,
        seed: int,
        flow_seed_stream: Iterable[int],
        *,
        tag: str,
    ) -> EvaluationRecord:
        if type(seed) is not int or seed < 0:
            raise ValueError("environment seed must be a nonnegative integer")
        if not isinstance(tag, str) or _SAFE_TAG.fullmatch(tag) is None:
            raise ValueError("evaluation tag contains unsafe characters")
        tag_root = self.output_root / tag
        video_path = tag_root / f"seed-{seed}.mp4"
        record_path = tag_root / f"seed-{seed}.json"
        if video_path.exists() or record_path.exists():
            raise FileExistsError(
                f"evaluation artifact already exists for tag={tag} seed={seed}"
            )

        flow_seeds = iter(flow_seed_stream)
        environment = self.env_factory()
        frames: list[np.ndarray] = []
        steps: list[dict[str, object]] = []
        success = False
        total_reward = 0.0
        termination_reason = "max_steps"
        try:
            observation, _ = environment.reset(seed=seed)
            adapted = self.observation_adapter.reset(observation)
            frames.append(adapted.video_frame.copy())
            for step_index in range(self.max_steps):
                try:
                    flow_seed = next(flow_seeds)
                except StopIteration as error:
                    raise ValueError("flow_seed_stream ended before the episode") from error
                if type(flow_seed) is not int or flow_seed < 0:
                    raise ValueError("flow seeds must be nonnegative integers")
                prediction = self.policy.predict_action_chunk(
                    adapted.policy_observation,
                    self.action_spec.spec_id,
                    flow_seed,
                    self.protocol.solver,
                )
                self._validate_prediction(prediction, flow_seed)
                normalized = (
                    prediction.normalized_actions[0, 0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                physical = (
                    prediction.denormalized_actions[0, 0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                self._require_action_in_bounds(physical)
                environment_action = np.asarray(
                    self.action_to_environment(environment, physical), dtype=np.float32
                )
                if environment_action.shape != (self.action_spec.dimension,) or not np.isfinite(
                    environment_action
                ).all():
                    raise ValueError(
                        "environment action must be finite and match ActionSpec dimension"
                    )
                next_observation, reward, terminated, truncated, info = (
                    _step_maniskill_environment(environment, environment_action)
                )
                reward_value = _finite_scalar(reward, "reward")
                terminated_value = _bool_scalar(terminated, "terminated")
                truncated_value = _bool_scalar(truncated, "truncated")
                success_value = _bool_scalar(
                    info.get("success", False) if isinstance(info, Mapping) else False,
                    "success",
                )
                success = success or success_value
                total_reward += reward_value
                report = prediction.solver_report
                consistency = float(
                    prediction.consistency[0, 0].detach().float().cpu()
                )
                inverse_variance_mean = float(
                    prediction.inverse_variance[0, 0]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                )
                steps.append(
                    {
                        "index": step_index,
                        "flow_seed": flow_seed,
                        "executed_chunk_index": 0,
                        "normalized_action": normalized.tolist(),
                        "physical_action": physical.tolist(),
                        "environment_action": environment_action.tolist(),
                        "reward": reward_value,
                        "terminated": terminated_value,
                        "truncated": truncated_value,
                        "success": success_value,
                        "consistency": consistency,
                        "inverse_variance_mean": inverse_variance_mean,
                        "solver": report.solver,
                        "time_grid": list(report.time_grid),
                        "intervals": report.intervals,
                        "nfe": report.nfe,
                    }
                )
                adapted = self.observation_adapter.append(next_observation)
                frames.append(adapted.video_frame.copy())
                if terminated_value or truncated_value:
                    if success_value:
                        termination_reason = "success"
                    elif terminated_value:
                        termination_reason = "terminated"
                    else:
                        termination_reason = "truncated"
                    break
            if not steps:
                raise ValueError("evaluation episode executed no actions")
        finally:
            environment.close()

        video = write_rollout_video_atomic(
            video_path, frames, fps=self.dataset_spec.fps
        )
        consistency_mean = float(np.mean([step["consistency"] for step in steps]))
        inverse_variance_mean = float(
            np.mean([step["inverse_variance_mean"] for step in steps])
        )
        report = steps[0]
        record = EvaluationRecord(
            {
                "format_version": 1,
                "tag": tag,
                "checkpoint": {
                    "label": self.checkpoint.label,
                    "directory": self.checkpoint.directory,
                    "manifest_sha256": self.checkpoint.manifest_sha256,
                },
                "dataset": {
                    "dataset_id": self.dataset_spec.dataset_id,
                    "revision": self.dataset_spec.revision,
                    "spec_hash": self.dataset_spec.content_hash,
                },
                "action": {
                    "spec_id": self.action_spec.spec_id,
                    "spec_hash": self.action_spec.content_hash,
                },
                "environment": {
                    "env_id": self.environment_spec.env_id,
                    "robot_uid": self.environment_spec.robot_uid,
                    "obs_mode": self.environment_spec.obs_mode,
                    "control_mode": self.environment_spec.control_mode,
                    "sim_backend": self.environment_spec.sim_backend,
                    "seed": seed,
                },
                "protocol": {
                    "action_horizon": self.protocol.action_horizon,
                    "execution_horizon": self.protocol.execution_horizon,
                    "temporal_ensemble": self.protocol.temporal_ensemble,
                    "context_steps": self.protocol.context_steps,
                },
                "flow": {
                    "solver": report["solver"],
                    "time_grid": report["time_grid"],
                    "intervals": report["intervals"],
                    "nfe_per_step": report["nfe"],
                    "total_nfe": sum(int(step["nfe"]) for step in steps),
                    "seeds": [int(step["flow_seed"]) for step in steps],
                },
                "result": {
                    "success": success,
                    "total_reward": total_reward,
                    "length": len(steps),
                    "termination_reason": termination_reason,
                },
                "diagnostics": {
                    "consistency_mean": consistency_mean,
                    "inverse_variance_mean": inverse_variance_mean,
                },
                "steps": steps,
                "video": {
                    "path": str(video.path),
                    "sha256": video.sha256,
                    "frames": video.frames,
                    "fps": video.fps,
                },
            }
        )
        try:
            record.write_atomic(record_path)
        except Exception:
            video.path.unlink(missing_ok=True)
            raise
        return record

    def _validate_prediction(
        self, prediction: ActionChunkPrediction, flow_seed: int
    ) -> None:
        if not isinstance(prediction, ActionChunkPrediction):
            raise ValueError("policy must return ActionChunkPrediction")
        expected_shape = (
            1,
            self.protocol.action_horizon,
            self.action_spec.dimension,
        )
        if tuple(prediction.denormalized_actions.shape) != expected_shape:
            raise ValueError(f"policy action chunk must have shape {expected_shape}")
        report = prediction.solver_report
        if (
            report.solver != self.protocol.solver
            or report.intervals != self.protocol.solver_intervals
            or report.nfe != 20
            or report.noise_seed != flow_seed
        ):
            raise ValueError(
                "policy solver report does not match evaluation protocol"
            )

    def _require_action_in_bounds(self, action: np.ndarray) -> None:
        if action.shape != (self.action_spec.dimension,) or not np.isfinite(action).all():
            raise ValueError(
                "physical action must be finite and match ActionSpec dimension"
            )
        lower = np.asarray(self.action_spec.minimum, dtype=np.float32)
        upper = np.asarray(self.action_spec.maximum, dtype=np.float32)
        if np.any(action < lower - 1e-6) or np.any(action > upper + 1e-6):
            raise ValueError("policy physical action is outside ActionSpec bounds")


class ManiSkillOnlineObservationAdapter:
    def __init__(
        self,
        *,
        dataset_spec: DatasetSpec,
        condition_ids: Mapping[str, torch.Tensor],
        task_text: str | None,
        device: torch.device | str,
        context_steps: int,
        video_camera_role: str = "base",
    ) -> None:
        self.camera_roles = tuple(sorted(dataset_spec.camera_features))
        if set(self.camera_roles) != {"base", "wrist"}:
            raise ValueError(
                "ManiSkill online DatasetSpec must declare exactly base and wrist cameras"
            )
        if dataset_spec.optional_camera_roles:
            raise ValueError("ManiSkill online cameras cannot be optional")
        if dataset_spec.proprio_features != ("observation.state",):
            raise ValueError(
                "ManiSkill online proprio must map exactly observation.state"
            )
        if set(condition_ids) != set(CONDITION_NAMESPACES):
            raise ValueError(
                f"condition IDs must contain exactly {sorted(CONDITION_NAMESPACES)}"
            )
        if task_text is not None and (
            not isinstance(task_text, str) or not task_text.strip()
        ):
            raise ValueError("task_text must be None or a nonempty string")
        if type(context_steps) is not int or context_steps <= 0:
            raise ValueError("context_steps must be a positive integer")
        if video_camera_role not in self.camera_roles:
            raise ValueError("video camera role must be declared in DatasetSpec")
        self.dataset_spec = dataset_spec
        self.device = torch.device(device)
        self.context_steps = context_steps
        self.video_camera_role = video_camera_role
        self.task_text = task_text
        self.condition_ids = {
            name: self._validate_condition_id(name, condition_ids[name])
            for name in CONDITION_NAMESPACES
        }
        self._camera_history: list[dict[str, np.ndarray]] = []
        self._proprio_history: list[np.ndarray] = []

    def reset(self, observation: object) -> AdaptedPolicyObservation:
        self._camera_history.clear()
        self._proprio_history.clear()
        return self.append(observation)

    def append(self, observation: object) -> AdaptedPolicyObservation:
        if not isinstance(observation, Mapping):
            raise ValueError("ManiSkill RGB observation must be a mapping")
        from corrective_foresight.data.maniskill_conversion import (
            extract_rgb_proprio,
        )

        cameras, proprio = extract_rgb_proprio(observation)
        if set(cameras) != set(self.camera_roles):
            raise ValueError(
                "online camera roles do not match the conversion DatasetSpec"
            )
        self._camera_history.append(cameras)
        self._proprio_history.append(proprio)
        self._camera_history = self._camera_history[-self.context_steps :]
        self._proprio_history = self._proprio_history[-self.context_steps :]

        rgb_array = np.stack(
            [
                np.stack([frame[role] for role in self.camera_roles], axis=0)
                for frame in self._camera_history
            ],
            axis=0,
        )
        rgb = (
            torch.from_numpy(rgb_array.copy())
            .permute(0, 1, 4, 2, 3)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
            / 255.0
        )
        proprio_tensor = torch.from_numpy(
            np.stack(self._proprio_history, axis=0)
        ).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        batch_size, time_steps, views = rgb.shape[:3]
        policy_observation = PolicyObservation(
            rgb=rgb,
            camera_mask=torch.ones(
                (batch_size, time_steps, views),
                dtype=torch.bool,
                device=self.device,
            ),
            proprio=proprio_tensor,
            proprio_mask=torch.ones_like(proprio_tensor, dtype=torch.bool),
            observation_valid_mask=torch.ones(
                (batch_size, time_steps), dtype=torch.bool, device=self.device
            ),
            task_text=(self.task_text,),
            condition_ids=self.condition_ids,
        )
        return AdaptedPolicyObservation(
            policy_observation=policy_observation,
            video_frame=cameras[self.video_camera_role].copy(),
        )

    def _validate_condition_id(
        self, name: str, value: torch.Tensor
    ) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype is not torch.long
            or value.shape != (1,)
            or value.item() < 0
        ):
            raise ValueError(f"condition ID {name} must be nonnegative int64 [1]")
        return value.detach().clone().to(self.device)


def physical_action_to_maniskill_controller(
    environment: object, action: np.ndarray
) -> np.ndarray:
    from corrective_foresight.data.maniskill_conversion import (
        expected_panda_ee_delta_pose_contract,
        physical_action_to_controller,
        query_panda_ee_delta_pose_contract,
    )

    contract = query_panda_ee_delta_pose_contract(environment)
    expected = expected_panda_ee_delta_pose_contract()
    if contract != expected:
        raise ValueError("live ManiSkill controller contract differs from conversion")
    return physical_action_to_controller(action, contract)


def _step_maniskill_environment(environment: object, action: np.ndarray):
    from corrective_foresight.data.maniskill_conversion import step_maniskill_env

    return step_maniskill_env(environment, action)


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        result = float(value.detach().cpu().item())
    else:
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f"{name} must be scalar")
        result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bool_scalar(value: object, name: str) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        return bool(value.detach().cpu().item())
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    return bool(array.reshape(-1)[0])
