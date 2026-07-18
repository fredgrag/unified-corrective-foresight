from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
import uuid


TOP_LEVEL_FIELDS = {
    "format_version",
    "tag",
    "checkpoint",
    "dataset",
    "action",
    "environment",
    "protocol",
    "flow",
    "result",
    "diagnostics",
    "steps",
    "video",
}


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    value: Mapping[str, object]

    def __post_init__(self) -> None:
        value = dict(self.value)
        if set(value) != TOP_LEVEL_FIELDS:
            raise ValueError(
                f"evaluation record fields must be exactly {sorted(TOP_LEVEL_FIELDS)}"
            )
        if value["format_version"] != 1:
            raise ValueError("unsupported evaluation record format_version")
        if not _nonempty_string(value["tag"]):
            raise ValueError("evaluation tag must be a nonempty string")
        _require_mapping_fields(
            "checkpoint", value["checkpoint"], {"label", "directory", "manifest_sha256"}
        )
        _require_mapping_fields(
            "dataset", value["dataset"], {"dataset_id", "revision", "spec_hash"}
        )
        _require_mapping_fields("action", value["action"], {"spec_id", "spec_hash"})
        _require_mapping_fields(
            "environment",
            value["environment"],
            {"env_id", "robot_uid", "obs_mode", "control_mode", "sim_backend", "seed"},
        )
        _require_mapping_fields(
            "protocol",
            value["protocol"],
            {"action_horizon", "execution_horizon", "temporal_ensemble", "context_steps"},
        )
        _require_mapping_fields(
            "flow",
            value["flow"],
            {"solver", "time_grid", "intervals", "nfe_per_step", "total_nfe", "seeds"},
        )
        _require_mapping_fields(
            "result",
            value["result"],
            {"success", "total_reward", "length", "termination_reason"},
        )
        _require_mapping_fields(
            "diagnostics",
            value["diagnostics"],
            {"consistency_mean", "inverse_variance_mean"},
        )
        _require_mapping_fields("video", value["video"], {"path", "sha256", "frames", "fps"})
        steps = value["steps"]
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise ValueError("evaluation steps must be a sequence")
        expected_step_fields = {
            "index",
            "flow_seed",
            "executed_chunk_index",
            "normalized_action",
            "physical_action",
            "environment_action",
            "reward",
            "terminated",
            "truncated",
            "success",
            "consistency",
            "inverse_variance_mean",
            "solver",
            "time_grid",
            "intervals",
            "nfe",
        }
        for index, step in enumerate(steps):
            _require_mapping_fields(f"steps[{index}]", step, expected_step_fields)
            if step["index"] != index or step["executed_chunk_index"] != 0:
                raise ValueError("evaluation steps must be ordered and execute chunk index zero")
        _validate_semantics(value)
        _reject_nonfinite(value)
        object.__setattr__(self, "value", MappingProxyType(value))

    def to_json(self) -> str:
        return json.dumps(
            dict(self.value),
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    def write_atomic(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"evaluation record already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        try:
            with temporary.open("x", encoding="utf-8") as file:
                file.write(self.to_json())
                file.flush()
                os.fsync(file.fileno())
            os.rename(temporary, destination)
            _fsync_directory(destination.parent)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def _validate_semantics(value: Mapping[str, object]) -> None:
    checkpoint = value["checkpoint"]
    dataset = value["dataset"]
    action = value["action"]
    environment = value["environment"]
    protocol = value["protocol"]
    flow = value["flow"]
    result = value["result"]
    video = value["video"]
    assert isinstance(checkpoint, Mapping)
    assert isinstance(dataset, Mapping)
    assert isinstance(action, Mapping)
    assert isinstance(environment, Mapping)
    assert isinstance(protocol, Mapping)
    assert isinstance(flow, Mapping)
    assert isinstance(result, Mapping)
    assert isinstance(video, Mapping)
    for name, item in (
        ("checkpoint label", checkpoint["label"]),
        ("checkpoint directory", checkpoint["directory"]),
        ("dataset ID", dataset["dataset_id"]),
        ("dataset revision", dataset["revision"]),
        ("action spec ID", action["spec_id"]),
        ("environment ID", environment["env_id"]),
        ("termination reason", result["termination_reason"]),
        ("video path", video["path"]),
    ):
        if not _nonempty_string(item):
            raise ValueError(f"{name} must be a nonempty string")
    for name, digest in (
        ("checkpoint manifest", checkpoint["manifest_sha256"]),
        ("DatasetSpec", dataset["spec_hash"]),
        ("ActionSpec", action["spec_hash"]),
        ("video", video["sha256"]),
    ):
        if not _sha256(digest):
            raise ValueError(f"{name} SHA256 is invalid")
    if protocol != {
        "action_horizon": 8,
        "execution_horizon": 1,
        "temporal_ensemble": False,
        "context_steps": protocol["context_steps"],
    } or type(protocol["context_steps"]) is not int or protocol["context_steps"] <= 0:
        raise ValueError("evaluation protocol must use H=8, E=1, and ensemble off")
    length = result["length"]
    steps = value["steps"]
    if type(length) is not int or length <= 0 or length != len(steps):
        raise ValueError("evaluation result length must match nonempty steps")
    seeds = flow["seeds"]
    if not isinstance(seeds, list) or len(seeds) != length:
        raise ValueError("flow seeds must contain one seed per executed step")
    if flow["solver"] != "midpoint" or flow["intervals"] != 10:
        raise ValueError("primary flow protocol must use midpoint with 10 intervals")
    if flow["nfe_per_step"] != 20 or flow["total_nfe"] != 20 * length:
        raise ValueError("flow NFE accounting is inconsistent")
    if type(video["frames"]) is not int or video["frames"] != length + 1:
        raise ValueError("rollout video must contain the initial frame plus every step")


def _require_mapping_fields(name: str, value: object, fields: set[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields must be exactly {sorted(fields)}")


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_nonfinite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("evaluation record cannot contain non-finite values")


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
