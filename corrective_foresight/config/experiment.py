from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from corrective_foresight.config.loader import load_action_spec, load_dataset_spec
from corrective_foresight.model.world_action_transformer import WorldActionConfig


_EXPERIMENT_FIELDS = {
    "schema_version",
    "experiment_id",
    "stage",
    "seed",
    "model_config",
    "dataset_specs",
    "action_specs",
    "task_texts",
    "artifacts",
    "output_root",
    "training",
    "optimized_objectives",
    "evaluation",
}
_TRAINING_FIELDS = {
    "batch_size_per_rank",
    "accumulation_steps",
    "learning_rate",
    "weight_decay",
    "max_grad_norm",
    "warmup_steps",
    "total_steps",
    "checkpoint_interval",
    "validation_interval",
    "ema_tau",
    "bf16",
}
_EVALUATION_FIELDS = {
    "env_id",
    "robot_uid",
    "obs_mode",
    "control_mode",
    "sim_backend",
    "context_steps",
    "action_horizon",
    "execution_horizon",
    "temporal_ensemble",
    "solver",
    "solver_intervals",
    "max_steps",
    "output_root",
}
_ARTIFACT_FIELDS = {
    "dinov3_snapshot",
    "language_tensor",
    "language_metadata",
}
_OBJECTIVES = {
    "world_pretrain": {"dynamics_loss": 1.0},
    "unified": {
        "dynamics_loss": 1.0,
        "inverse_loss": 1.0,
        "action_cycle_loss": 0.1,
        "policy_flow_loss": 1.0,
    },
}


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    dinov3_snapshot: Path
    language_tensor: Path
    language_metadata: Path


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size_per_rank: int
    accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    warmup_steps: int
    total_steps: int
    checkpoint_interval: int
    validation_interval: int
    ema_tau: float
    bf16: bool


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    env_id: str
    robot_uid: str
    obs_mode: str
    control_mode: str
    sim_backend: str
    context_steps: int
    action_horizon: int
    execution_horizon: int
    temporal_ensemble: bool
    solver: str
    solver_intervals: int
    max_steps: int
    output_root: Path

    @property
    def nfe_per_step(self) -> int:
        return 2 * self.solver_intervals


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    source_path: Path
    experiment_id: str
    stage: str
    seed: int
    model: WorldActionConfig
    dataset_specs: tuple[Path, ...]
    action_specs: tuple[Path, ...]
    task_texts: Mapping[str, tuple[str, ...]]
    artifacts: ArtifactConfig
    output_root: Path
    training: TrainingConfig
    optimized_objectives: Mapping[str, float]
    evaluation: EvaluationConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("experiment config must be a physical file")
    source = source.resolve(strict=True)
    project_root = source.parents[2]
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = _strict_mapping(raw, _EXPERIMENT_FIELDS, "experiment")
    if root["schema_version"] != 1:
        raise ValueError("experiment schema_version must be 1")
    experiment_id = _nonempty_string(root["experiment_id"], "experiment_id")
    stage = _nonempty_string(root["stage"], "stage")
    if stage not in _OBJECTIVES:
        raise ValueError(f"unsupported training stage: {stage}")
    seed = _positive_int(root["seed"], "seed", allow_zero=True)

    model_path = _project_file(project_root, root["model_config"], "model_config")
    with model_path.open("r", encoding="utf-8") as stream:
        model_raw = yaml.safe_load(stream)
    if not isinstance(model_raw, Mapping):
        raise ValueError("model config must be a mapping")
    model = WorldActionConfig.from_mapping(dict(model_raw))
    dataset_paths = _project_files(project_root, root["dataset_specs"], "dataset_specs")
    action_paths = _project_files(project_root, root["action_specs"], "action_specs")

    raw_tasks = root["task_texts"]
    if not isinstance(raw_tasks, Mapping) or not raw_tasks:
        raise ValueError("task_texts must be a nonempty mapping")
    task_texts: dict[str, tuple[str, ...]] = {}
    for raw_dataset_id, raw_values in raw_tasks.items():
        dataset_id = _nonempty_string(raw_dataset_id, "task_texts dataset ID")
        values = _string_tuple(raw_values, f"task_texts[{dataset_id}]")
        if values != tuple(sorted(values)):
            raise ValueError(f"task_texts[{dataset_id}] must be sorted")
        task_texts[dataset_id] = values

    artifacts_raw = _strict_mapping(root["artifacts"], _ARTIFACT_FIELDS, "artifacts")
    artifacts = ArtifactConfig(
        **{
            name: _absolute_path(artifacts_raw[name], f"artifacts.{name}")
            for name in sorted(_ARTIFACT_FIELDS)
        }
    )
    output_root = _absolute_path(root["output_root"], "output_root")
    training = _training_config(root["training"])
    objectives_raw = root["optimized_objectives"]
    if not isinstance(objectives_raw, Mapping):
        raise ValueError("optimized_objectives must be a mapping")
    objectives = {str(name): float(value) for name, value in objectives_raw.items()}
    if objectives != _OBJECTIVES[stage]:
        raise ValueError(
            f"{stage} optimized objectives must be exactly {_OBJECTIVES[stage]}"
        )
    evaluation = _evaluation_config(root["evaluation"])

    dataset_specs = tuple(load_dataset_spec(item) for item in dataset_paths)
    action_specs = tuple(load_action_spec(item) for item in action_paths)
    if set(task_texts) != {spec.dataset_id for spec in dataset_specs}:
        raise ValueError("task_texts keys must match DatasetSpec identifiers")
    action_by_id = {spec.spec_id: spec for spec in action_specs}
    if len(action_by_id) != len(action_specs):
        raise ValueError("ActionSpec identifiers must be unique")
    for dataset_spec in dataset_specs:
        if dataset_spec.action_spec_id not in action_by_id:
            raise ValueError("DatasetSpec references an undeclared ActionSpec")
        action_spec = action_by_id[dataset_spec.action_spec_id]
        if action_spec.control_mode != evaluation.control_mode:
            raise ValueError("evaluation and ActionSpec control modes differ")
        if dataset_spec.fps != action_spec.frequency_hz:
            raise ValueError("DatasetSpec and ActionSpec frequencies differ")

    return ExperimentConfig(
        source_path=source,
        experiment_id=experiment_id,
        stage=stage,
        seed=seed,
        model=model,
        dataset_specs=dataset_paths,
        action_specs=action_paths,
        task_texts=MappingProxyType(task_texts),
        artifacts=artifacts,
        output_root=output_root,
        training=training,
        optimized_objectives=MappingProxyType(objectives),
        evaluation=evaluation,
    )


def _training_config(value: object) -> TrainingConfig:
    raw = _strict_mapping(value, _TRAINING_FIELDS, "training")
    integer_values = {
        name: _positive_int(raw[name], f"training.{name}", allow_zero=name == "warmup_steps")
        for name in (
            "batch_size_per_rank",
            "accumulation_steps",
            "warmup_steps",
            "total_steps",
            "checkpoint_interval",
            "validation_interval",
        )
    }
    if integer_values["warmup_steps"] >= integer_values["total_steps"]:
        raise ValueError("training warmup_steps must be less than total_steps")
    learning_rate = _finite_float(raw["learning_rate"], "training.learning_rate")
    weight_decay = _finite_float(raw["weight_decay"], "training.weight_decay", allow_zero=True)
    max_grad_norm = _finite_float(raw["max_grad_norm"], "training.max_grad_norm")
    ema_tau = _finite_float(raw["ema_tau"], "training.ema_tau")
    if not 0.0 < ema_tau < 1.0:
        raise ValueError("training.ema_tau must satisfy 0 < value < 1")
    if type(raw["bf16"]) is not bool:
        raise ValueError("training.bf16 must be bool")
    return TrainingConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        ema_tau=ema_tau,
        bf16=raw["bf16"],
        **integer_values,
    )


def _evaluation_config(value: object) -> EvaluationConfig:
    raw = _strict_mapping(value, _EVALUATION_FIELDS, "evaluation")
    strings = {
        name: _nonempty_string(raw[name], f"evaluation.{name}")
        for name in (
            "env_id",
            "robot_uid",
            "obs_mode",
            "control_mode",
            "sim_backend",
            "solver",
        )
    }
    integers = {
        name: _positive_int(raw[name], f"evaluation.{name}")
        for name in (
            "context_steps",
            "action_horizon",
            "execution_horizon",
            "solver_intervals",
            "max_steps",
        )
    }
    if type(raw["temporal_ensemble"]) is not bool:
        raise ValueError("evaluation.temporal_ensemble must be bool")
    if (
        integers["action_horizon"] != 8
        or integers["execution_horizon"] != 1
        or raw["temporal_ensemble"]
        or strings["solver"] != "midpoint"
        or integers["solver_intervals"] != 10
    ):
        raise ValueError(
            "primary evaluation requires H=8, E=1, ensemble off, midpoint/10"
        )
    return EvaluationConfig(
        temporal_ensemble=raw["temporal_ensemble"],
        output_root=_absolute_path(raw["output_root"], "evaluation.output_root"),
        **strings,
        **integers,
    )


def _strict_mapping(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{name} fields must be exactly {sorted(fields)}, got {actual}")
    return dict(value)


def _project_files(project_root: Path, value: object, name: str) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    paths = tuple(_project_file(project_root, item, name) for item in value)
    if len(set(paths)) != len(paths):
        raise ValueError(f"{name} contains duplicate paths")
    return paths


def _project_file(project_root: Path, value: object, name: str) -> Path:
    text = _nonempty_string(value, name)
    relative = Path(text)
    if relative.is_absolute():
        raise ValueError(f"{name} must be relative to the project root")
    candidate = project_root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{name} must resolve to a physical project file: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the project root") from error
    return resolved


def _absolute_path(value: object, name: str) -> Path:
    path = Path(_nonempty_string(value, name))
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result = tuple(_nonempty_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates")
    return result


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _finite_float(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (not allow_zero and result == 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return result
