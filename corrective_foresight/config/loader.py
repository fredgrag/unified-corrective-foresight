from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from corrective_foresight.config.schema import ActionSpec, DatasetSpec


SpecType = TypeVar("SpecType", ActionSpec, DatasetSpec)


def _read_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must contain a YAML mapping")
    return dict(value)


def _check_fields(data: Mapping[str, Any], spec_type: type[SpecType]) -> None:
    expected = {field.name for field in fields(spec_type)}
    actual = set(data)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"unknown fields for {spec_type.__name__}: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing fields for {spec_type.__name__}: {', '.join(missing)}")


def load_action_spec(path: str | Path) -> ActionSpec:
    data = _read_mapping(path)
    _check_fields(data, ActionSpec)
    for name in (
        "names",
        "units",
        "gripper_indices",
        "normalization_mean",
        "normalization_std",
        "minimum",
        "maximum",
    ):
        data[name] = tuple(data[name])
    return ActionSpec(**data)


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    data = _read_mapping(path)
    _check_fields(data, DatasetSpec)
    data["camera_features"] = dict(data["camera_features"])
    data["optional_camera_roles"] = tuple(data["optional_camera_roles"])
    data["proprio_features"] = tuple(data["proprio_features"])
    data["split_episode_ids"] = {
        str(name): tuple(episode_ids)
        for name, episode_ids in dict(data["split_episode_ids"]).items()
    }
    return DatasetSpec(**data)
