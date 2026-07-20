from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from corrective_foresight.config.experiment import (
    ExperimentConfig,
    load_experiment_config,
)


_FIELDS = {
    "schema_version",
    "pilot_id",
    "world_experiment",
    "unified_experiment",
    "world_steps",
    "audit_steps",
    "unified_gate_steps",
    "unified_total_steps",
    "validation_interval",
    "checkpoint_interval",
    "validation_batches_per_rank",
    "protected_lr_multiplier",
    "conflict_log_interval",
    "conflict_cosine_threshold",
    "conflict_measurement_minimum",
    "dynamics_degradation_ratio",
    "world_copy_improvement_target",
    "required_success_gain",
    "evaluation_seeds",
    "gpu_indices",
    "output_root",
    "tracking",
}
_TRACKING_FIELDS = {
    "enabled",
    "project",
    "group",
    "log_interval",
    "upload_checkpoints",
}


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    enabled: bool
    project: str
    group: str
    log_interval: int
    upload_checkpoints: bool


@dataclass(frozen=True, slots=True)
class ConflictFixConfig:
    source_path: Path
    pilot_id: str
    world_experiment: Path
    unified_experiment: Path
    output_root: Path
    world_steps: int
    audit_steps: int
    unified_gate_steps: int
    unified_total_steps: int
    validation_interval: int
    checkpoint_interval: int
    validation_batches_per_rank: int
    protected_lr_multiplier: float
    conflict_log_interval: int
    conflict_cosine_threshold: float
    conflict_measurement_minimum: int
    dynamics_degradation_ratio: float
    world_copy_improvement_target: float
    required_success_gain: int
    evaluation_seeds: tuple[int, ...]
    gpu_indices: tuple[int, ...]
    tracking: TrackingConfig
    world_config: ExperimentConfig
    unified_config: ExperimentConfig

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


def load_conflict_fix_config(path: str | Path) -> ConflictFixConfig:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("conflict-fix config must be a physical file")
    source = source.resolve(strict=True)
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = _strict_mapping(raw, _FIELDS, "conflict-fix")
    if root["schema_version"] != 1:
        raise ValueError("conflict-fix schema_version must be 1")

    pilot_id = _string(root["pilot_id"], "pilot_id")
    _require_equal(pilot_id, "maniskill.pick_cube.conflict_fix.v2", "pilot_id")
    integers = {
        name: _positive_int(root[name], name)
        for name in (
            "world_steps",
            "audit_steps",
            "unified_gate_steps",
            "unified_total_steps",
            "validation_interval",
            "checkpoint_interval",
            "validation_batches_per_rank",
            "conflict_log_interval",
            "conflict_measurement_minimum",
            "required_success_gain",
        )
    }
    approved_integers = {
        "world_steps": 5000,
        "audit_steps": 500,
        "unified_gate_steps": 5000,
        "unified_total_steps": 20000,
        "validation_interval": 250,
        "checkpoint_interval": 1000,
        "validation_batches_per_rank": 8,
        "conflict_log_interval": 10,
        "conflict_measurement_minimum": 8,
        "required_success_gain": 2,
    }
    for name, expected in approved_integers.items():
        _require_equal(integers[name], expected, name)

    finite = {
        name: _finite_float(root[name], name)
        for name in (
            "protected_lr_multiplier",
            "conflict_cosine_threshold",
            "dynamics_degradation_ratio",
            "world_copy_improvement_target",
        )
    }
    approved_finite = {
        "protected_lr_multiplier": 0.1,
        "conflict_cosine_threshold": -0.05,
        "dynamics_degradation_ratio": 1.2,
        "world_copy_improvement_target": 0.05,
    }
    for name, expected in approved_finite.items():
        _require_equal(finite[name], expected, name)

    evaluation_seeds = _int_tuple(root["evaluation_seeds"], "evaluation_seeds")
    gpu_indices = _int_tuple(root["gpu_indices"], "gpu_indices")
    _require_equal(evaluation_seeds, tuple(range(10)), "evaluation_seeds")
    _require_equal(gpu_indices, (0, 1, 2, 3), "gpu_indices")

    tracking_raw = _strict_mapping(root["tracking"], _TRACKING_FIELDS, "tracking")
    if type(tracking_raw["enabled"]) is not bool or not tracking_raw["enabled"]:
        raise ValueError("tracking.enabled must be true")
    if (
        type(tracking_raw["upload_checkpoints"]) is not bool
        or tracking_raw["upload_checkpoints"]
    ):
        raise ValueError("tracking.upload_checkpoints must be false")
    tracking = TrackingConfig(
        enabled=True,
        project=_string(tracking_raw["project"], "tracking.project"),
        group=_string(tracking_raw["group"], "tracking.group"),
        log_interval=_positive_int(
            tracking_raw["log_interval"], "tracking.log_interval"
        ),
        upload_checkpoints=False,
    )
    _require_equal(
        tracking.project,
        "unified-corrective-foresight",
        "tracking.project",
    )
    _require_equal(
        tracking.group,
        "maniskill-pickcube-conflict-fix-v2",
        "tracking.group",
    )
    _require_equal(tracking.log_interval, 10, "tracking.log_interval")

    output_root = Path(_string(root["output_root"], "output_root"))
    if not output_root.is_absolute() or output_root.is_symlink():
        raise ValueError("output_root must be an absolute non-symlink path")

    project_root = source.parents[2]
    world_experiment = _project_file(
        project_root, root["world_experiment"], "world_experiment"
    )
    unified_experiment = _project_file(
        project_root, root["unified_experiment"], "unified_experiment"
    )
    world_config = load_experiment_config(world_experiment)
    unified_config = load_experiment_config(unified_experiment)
    _validate_experiments(world_config, unified_config)

    config = ConflictFixConfig(
        source_path=source,
        pilot_id=pilot_id,
        world_experiment=world_experiment,
        unified_experiment=unified_experiment,
        output_root=output_root,
        evaluation_seeds=evaluation_seeds,
        gpu_indices=gpu_indices,
        tracking=tracking,
        world_config=world_config,
        unified_config=unified_config,
        **integers,
        **finite,
    )
    if config.effective_global_batch != 64:
        raise ValueError("conflict-fix effective global batch must be exactly 64")
    return config


def _validate_experiments(
    world: ExperimentConfig,
    unified: ExperimentConfig,
) -> None:
    if world.stage != "world_pretrain" or unified.stage != "unified":
        raise ValueError("conflict-fix experiment stages are invalid")
    if world.model != unified.model or world.artifacts != unified.artifacts:
        raise ValueError("conflict-fix model and artifacts must match")
    if world.dataset_specs != unified.dataset_specs:
        raise ValueError("conflict-fix DatasetSpecs must match")
    if world.action_specs != unified.action_specs:
        raise ValueError("conflict-fix ActionSpecs must match")
    expected_world = {
        "learning_rate": 0.0001,
        "warmup_steps": 500,
        "total_steps": 5000,
        "checkpoint_interval": 1000,
        "validation_interval": 250,
    }
    expected_unified = {
        "learning_rate": 0.00005,
        "warmup_steps": 500,
        "total_steps": 20000,
        "checkpoint_interval": 1000,
        "validation_interval": 250,
    }
    for name, expected in expected_world.items():
        _require_equal(getattr(world.training, name), expected, f"world.{name}")
    for name, expected in expected_unified.items():
        _require_equal(
            getattr(unified.training, name), expected, f"unified.{name}"
        )


def _strict_mapping(
    value: object,
    fields: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(
            f"{name} fields must be exactly {sorted(fields)}, got {actual}"
        )
    return dict(value)


def _project_file(project_root: Path, value: object, name: str) -> Path:
    relative = Path(_string(value, name))
    if relative.is_absolute():
        raise ValueError(f"{name} must be project-relative")
    candidate = project_root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{name} must resolve to a physical project file")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"{name} escapes project root") from error
    return resolved


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result = tuple(value)
    if any(type(item) is not int or item < 0 for item in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _require_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be exactly {expected!r}, got {actual!r}")
