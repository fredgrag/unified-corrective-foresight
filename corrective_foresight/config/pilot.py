from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from corrective_foresight.config.experiment import (
    ExperimentConfig,
    load_experiment_config,
)
from corrective_foresight.config.loader import load_action_spec, load_dataset_spec
from corrective_foresight.config.schema import ActionSpec, DatasetSpec


_PILOT_FIELDS = {
    "schema_version",
    "pilot_id",
    "world_experiment",
    "unified_experiment",
    "world_steps",
    "unified_steps",
    "checkpoint_interval",
    "validation_interval",
    "validation_batches_per_rank",
    "evaluation_steps",
    "evaluation_seeds",
    "gpu_indices",
    "output_root",
}


@dataclass(frozen=True, slots=True)
class PilotConfig:
    source_path: Path
    pilot_id: str
    world_experiment: Path
    unified_experiment: Path
    world_steps: int
    unified_steps: int
    checkpoint_interval: int
    validation_interval: int
    validation_batches_per_rank: int
    evaluation_steps: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    gpu_indices: tuple[int, ...]
    output_root: Path
    world_config: ExperimentConfig
    unified_config: ExperimentConfig
    world_dataset_specs: tuple[DatasetSpec, ...]
    unified_dataset_specs: tuple[DatasetSpec, ...]
    world_action_specs: tuple[ActionSpec, ...]
    unified_action_specs: tuple[ActionSpec, ...]

    @property
    def effective_global_batch(self) -> int:
        world = (
            self.world_config.training.batch_size_per_rank
            * len(self.gpu_indices)
            * self.world_config.training.accumulation_steps
        )
        unified = (
            self.unified_config.training.batch_size_per_rank
            * len(self.gpu_indices)
            * self.unified_config.training.accumulation_steps
        )
        if world != unified:
            raise ValueError("world and unified effective global batches differ")
        return world


def load_pilot_config(path: str | Path) -> PilotConfig:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("pilot config must be a physical file")
    source = source.resolve(strict=True)
    project_root = source.parents[2]
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = _strict_mapping(raw, "pilot")
    if root["schema_version"] != 1:
        raise ValueError("pilot schema_version must be 1")
    pilot_id = _string(root["pilot_id"], "pilot_id")
    output_root = _absolute_path(root["output_root"], "output_root")
    if output_root.exists() and output_root.is_symlink():
        raise ValueError("pilot output_root cannot be a symbolic link")

    world_steps = _positive_int(root["world_steps"], "world_steps")
    unified_steps = _positive_int(root["unified_steps"], "unified_steps")
    checkpoint_interval = _positive_int(
        root["checkpoint_interval"], "checkpoint_interval"
    )
    validation_interval = _positive_int(
        root["validation_interval"], "validation_interval"
    )
    for name, interval in (
        ("checkpoint_interval", checkpoint_interval),
        ("validation_interval", validation_interval),
    ):
        if world_steps % interval or unified_steps % interval:
            raise ValueError(f"{name} must divide both stage endpoints")
    validation_batches_per_rank = _positive_int(
        root["validation_batches_per_rank"], "validation_batches_per_rank"
    )
    evaluation_steps = _int_tuple(root["evaluation_steps"], "evaluation_steps")
    expected_steps = (0, world_steps, unified_steps)
    if evaluation_steps != expected_steps:
        raise ValueError(
            f"evaluation_steps must be exactly {expected_steps}, got {evaluation_steps}"
        )
    evaluation_seeds = _int_tuple(root["evaluation_seeds"], "evaluation_seeds")
    if evaluation_seeds != tuple(range(10)):
        raise ValueError("evaluation_seeds must be exactly 0 through 9")
    gpu_indices = _int_tuple(root["gpu_indices"], "gpu_indices")
    if gpu_indices != (0, 1, 2, 3):
        raise ValueError("gpu_indices must be exactly [0, 1, 2, 3]")

    world_experiment = _project_file(
        project_root, root["world_experiment"], "world_experiment"
    )
    unified_experiment = _project_file(
        project_root, root["unified_experiment"], "unified_experiment"
    )
    if world_experiment == unified_experiment:
        raise ValueError("world_experiment and unified_experiment must differ")
    world_config = load_experiment_config(world_experiment)
    unified_config = load_experiment_config(unified_experiment)
    if world_config.stage != "world_pretrain":
        raise ValueError("world_experiment must declare world_pretrain stage")
    if unified_config.stage != "unified":
        raise ValueError("unified_experiment must declare unified stage")
    if world_config.model != unified_config.model:
        raise ValueError("world and unified model configs must match exactly")
    if world_config.artifacts != unified_config.artifacts:
        raise ValueError("world and unified artifact configs must match exactly")
    world_evaluation = asdict(world_config.evaluation)
    unified_evaluation = asdict(unified_config.evaluation)
    world_evaluation.pop("output_root")
    unified_evaluation.pop("output_root")
    if world_evaluation != unified_evaluation:
        raise ValueError("world and unified evaluation configs must match exactly")
    if world_config.output_root == unified_config.output_root:
        raise ValueError("world and unified output roots must differ")

    world_dataset_specs = tuple(
        load_dataset_spec(item) for item in world_config.dataset_specs
    )
    unified_dataset_specs = tuple(
        load_dataset_spec(item) for item in unified_config.dataset_specs
    )
    world_action_specs = tuple(load_action_spec(item) for item in world_config.action_specs)
    unified_action_specs = tuple(
        load_action_spec(item) for item in unified_config.action_specs
    )
    if tuple(spec.content_hash for spec in world_dataset_specs) != tuple(
        spec.content_hash for spec in unified_dataset_specs
    ):
        raise ValueError("world and unified DatasetSpec content must match exactly")
    if tuple(spec.content_hash for spec in world_action_specs) != tuple(
        spec.content_hash for spec in unified_action_specs
    ):
        raise ValueError("world and unified ActionSpec content must match exactly")
    for spec in (*world_dataset_specs, *unified_dataset_specs):
        if any(
            split not in spec.split_episode_ids or not spec.split_episode_ids[split]
            for split in ("train", "validation", "evaluation")
        ):
            raise ValueError("pilot DatasetSpecs must declare nonempty train/validation/evaluation splits")

    config = PilotConfig(
        source_path=source,
        pilot_id=pilot_id,
        world_experiment=world_experiment,
        unified_experiment=unified_experiment,
        world_steps=world_steps,
        unified_steps=unified_steps,
        checkpoint_interval=checkpoint_interval,
        validation_interval=validation_interval,
        validation_batches_per_rank=validation_batches_per_rank,
        evaluation_steps=evaluation_steps,
        evaluation_seeds=evaluation_seeds,
        gpu_indices=gpu_indices,
        output_root=output_root,
        world_config=world_config,
        unified_config=unified_config,
        world_dataset_specs=world_dataset_specs,
        unified_dataset_specs=unified_dataset_specs,
        world_action_specs=world_action_specs,
        unified_action_specs=unified_action_specs,
    )
    if config.effective_global_batch != 64:
        raise ValueError("pilot effective global batch must be exactly 64")
    if config.checkpoint_interval != 250 or config.validation_interval != 100:
        raise ValueError("pilot checkpoint/validation intervals must be 250/100")
    return config


def _strict_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PILOT_FIELDS:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(
            f"{name} fields must be exactly {sorted(_PILOT_FIELDS)}, got {actual}"
        )
    return dict(value)


def _project_file(project_root: Path, value: object, name: str) -> Path:
    relative = Path(_string(value, name))
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
    path = Path(_string(value, name))
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result = tuple(value)
    if any(type(item) is not int or item < 0 for item in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be sorted and contain no duplicates")
    return result
