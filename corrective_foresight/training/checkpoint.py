from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import random
import shutil
from typing import Any, Protocol
import uuid

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import Tensor

from corrective_foresight.config.schema import ActionSpec, DatasetSpec
from corrective_foresight.policy.unified_policy import (
    UnifiedCorrectiveForesightPolicy,
)
from corrective_foresight.training.run_manifest import RunManifest
from corrective_foresight.training.trainer import Trainer


PAYLOAD_FILES = {
    "online_model.safetensors",
    "ema_target.safetensors",
    "trainer_state.pt",
    "rng_state.pt",
    "mixer_state.pt",
}
ALL_FILES = PAYLOAD_FILES | {"manifest.json"}


class StatefulMixer(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class CheckpointState:
    policy: UnifiedCorrectiveForesightPolicy
    trainer: Trainer
    mixer: StatefulMixer
    flow_generator: torch.Generator
    epoch: int
    global_step: int
    optimizer_step: int
    cycle_warmup_step: int
    dataset_specs: tuple[DatasetSpec, ...]
    action_specs: tuple[ActionSpec, ...]
    lerobot_version: str
    backbone_provenance: Mapping[str, str]
    dataset_revisions: Mapping[str, str]
    model_config: Mapping[str, Any]
    loss_config: Mapping[str, Any]
    git_commit: str
    git_dirty: bool

    def __post_init__(self) -> None:
        if self.trainer.policy is not self.policy:
            raise ValueError("checkpoint trainer and policy must share identity")
        for name in ("epoch", "global_step", "optimizer_step", "cycle_warmup_step"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"checkpoint {name} must be nonnegative")
        if not self.dataset_specs or not self.action_specs:
            raise ValueError("checkpoint requires dataset and action specs")
        dataset_ids = tuple(spec.dataset_id for spec in self.dataset_specs)
        action_spec_ids = tuple(spec.spec_id for spec in self.action_specs)
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("checkpoint DatasetSpec identifiers must be unique")
        if len(set(action_spec_ids)) != len(action_spec_ids):
            raise ValueError("checkpoint ActionSpec identifiers must be unique")
        if set(self.dataset_revisions) != set(dataset_ids):
            raise ValueError("dataset revisions must match DatasetSpec identifiers")
        if set(action_spec_ids) != set(self.policy.world_action_model.action_adapters.specs):
            raise ValueError("checkpoint ActionSpecs must match model adapter registry")
        if any(spec.action_spec_id not in action_spec_ids for spec in self.dataset_specs):
            raise ValueError("every DatasetSpec must reference a checkpoint ActionSpec")
        if self.optimizer_step > self.global_step:
            raise ValueError("optimizer_step cannot exceed global_step")
        if self.cycle_warmup_step > self.global_step:
            raise ValueError("cycle_warmup_step cannot exceed global_step")
        if self.policy.last_ema_step != self.optimizer_step - 1:
            raise ValueError("EMA step must equal optimizer_step - 1")


@dataclass(frozen=True, slots=True)
class ExpectedCheckpointContract:
    policy: UnifiedCorrectiveForesightPolicy
    trainer: Trainer
    mixer: StatefulMixer
    flow_generator: torch.Generator
    condition_vocabulary_hash: str
    dataset_spec_hashes: Mapping[str, str]
    action_spec_hashes: Mapping[str, str]
    lerobot_version: str
    backbone_provenance: Mapping[str, str]
    dataset_revisions: Mapping[str, str]
    model_config: Mapping[str, Any]
    loss_config: Mapping[str, Any]
    git_commit: str
    git_dirty: bool

    @classmethod
    def from_state(cls, state: CheckpointState) -> ExpectedCheckpointContract:
        return cls(
            policy=state.policy,
            trainer=state.trainer,
            mixer=state.mixer,
            flow_generator=state.flow_generator,
            condition_vocabulary_hash=state.policy.condition_encoder.vocabulary_hash,
            dataset_spec_hashes={
                spec.dataset_id: spec.content_hash for spec in state.dataset_specs
            },
            action_spec_hashes={
                spec.spec_id: spec.content_hash for spec in state.action_specs
            },
            lerobot_version=state.lerobot_version,
            backbone_provenance=dict(state.backbone_provenance),
            dataset_revisions=dict(state.dataset_revisions),
            model_config=dict(state.model_config),
            loss_config=dict(state.loss_config),
            git_commit=state.git_commit,
            git_dirty=state.git_dirty,
        )


@dataclass(frozen=True, slots=True)
class ExpectedPolicyCheckpointContract:
    policy: UnifiedCorrectiveForesightPolicy
    condition_vocabulary_hash: str
    dataset_spec_hashes: Mapping[str, str]
    action_spec_hashes: Mapping[str, str]
    lerobot_version: str
    backbone_provenance: Mapping[str, str]
    dataset_revisions: Mapping[str, str]
    model_config: Mapping[str, Any]
    loss_config: Mapping[str, Any]

    @classmethod
    def from_state(cls, state: CheckpointState) -> ExpectedPolicyCheckpointContract:
        return cls(
            policy=state.policy,
            condition_vocabulary_hash=state.policy.condition_encoder.vocabulary_hash,
            dataset_spec_hashes={
                spec.dataset_id: spec.content_hash for spec in state.dataset_specs
            },
            action_spec_hashes={
                spec.spec_id: spec.content_hash for spec in state.action_specs
            },
            lerobot_version=state.lerobot_version,
            backbone_provenance=dict(state.backbone_provenance),
            dataset_revisions=dict(state.dataset_revisions),
            model_config=dict(state.model_config),
            loss_config=dict(state.loss_config),
        )


@dataclass(frozen=True, slots=True)
class ResumeState:
    epoch: int
    global_step: int
    optimizer_step: int
    cycle_warmup_step: int
    manifest: RunManifest


@dataclass(frozen=True, slots=True)
class LegacyLoadReport:
    matched_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    match_fraction: float


def save_checkpoint_atomic(path: str | Path, state: CheckpointState) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"checkpoint destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        save_file(
            _online_state(state.policy),
            str(temporary / "online_model.safetensors"),
        )
        save_file(
            _cpu_state(state.policy.ema_state_target.adapter.state_dict()),
            str(temporary / "ema_target.safetensors"),
        )
        torch.save(state.trainer.state_dict(), temporary / "trainer_state.pt")
        torch.save(state.mixer.state_dict(), temporary / "mixer_state.pt")
        torch.save(_capture_rng_state(state.flow_generator), temporary / "rng_state.pt")
        for name in PAYLOAD_FILES:
            _fsync_file(temporary / name)
        manifest = _build_manifest(state, temporary)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        os.rename(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_checkpoint_strict(
    path: str | Path,
    expected: ExpectedCheckpointContract,
) -> ResumeState:
    source = Path(path)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("checkpoint path must be a physical directory")
    entries = {item.name for item in source.iterdir()}
    if entries != ALL_FILES:
        raise ValueError(
            f"checkpoint files must be exactly {sorted(ALL_FILES)}, got {sorted(entries)}"
        )
    if any(item.is_symlink() for item in source.iterdir()):
        raise ValueError("checkpoint payload cannot contain symbolic links")
    manifest = RunManifest.from_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    _validate_expected_contract(manifest, expected)
    _verify_payloads(source, manifest)

    online = load_file(str(source / "online_model.safetensors"), device="cpu")
    ema = load_file(str(source / "ema_target.safetensors"), device="cpu")
    expected_online = _online_state(expected.policy)
    expected_ema = _cpu_state(expected.policy.ema_state_target.adapter.state_dict())
    _validate_state_mapping(expected_online, online, "online model")
    _validate_state_mapping(expected_ema, ema, "EMA target")
    trainer_state = torch.load(
        source / "trainer_state.pt", map_location="cpu", weights_only=False
    )
    mixer_state = torch.load(
        source / "mixer_state.pt", map_location="cpu", weights_only=False
    )
    rng_state = torch.load(
        source / "rng_state.pt", map_location="cpu", weights_only=False
    )
    _validate_rng_state(rng_state)

    _copy_policy_state(expected.policy, online)
    _copy_module_state(expected.policy.ema_state_target.adapter, ema)
    expected.trainer.load_state_dict(trainer_state)
    expected.mixer.load_state_dict(mixer_state)
    _restore_rng_state(rng_state, expected.flow_generator)
    steps = manifest.value["steps"]
    return ResumeState(
        epoch=manifest.value["epoch"],
        global_step=steps["global_step"],
        optimizer_step=steps["optimizer_step"],
        cycle_warmup_step=steps["cycle_warmup_step"],
        manifest=manifest,
    )


def load_policy_checkpoint_strict(
    path: str | Path,
    expected: ExpectedPolicyCheckpointContract,
) -> ResumeState:
    source = Path(path)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("checkpoint path must be a physical directory")
    entries = {item.name for item in source.iterdir()}
    if entries != ALL_FILES:
        raise ValueError(
            f"checkpoint files must be exactly {sorted(ALL_FILES)}, got {sorted(entries)}"
        )
    if any(item.is_symlink() for item in source.iterdir()):
        raise ValueError("checkpoint payload cannot contain symbolic links")
    manifest = RunManifest.from_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    _validate_expected_policy_contract(manifest, expected)
    _verify_payloads(source, manifest)

    online = load_file(str(source / "online_model.safetensors"), device="cpu")
    ema = load_file(str(source / "ema_target.safetensors"), device="cpu")
    expected_online = _online_state(expected.policy)
    expected_ema = _cpu_state(expected.policy.ema_state_target.adapter.state_dict())
    _validate_state_mapping(expected_online, online, "online model")
    _validate_state_mapping(expected_ema, ema, "EMA target")
    _copy_policy_state(expected.policy, online)
    _copy_module_state(expected.policy.ema_state_target.adapter, ema)
    steps = manifest.value["steps"]
    expected.policy.restore_ema_step(steps["optimizer_step"] - 1)
    return ResumeState(
        epoch=manifest.value["epoch"],
        global_step=steps["global_step"],
        optimizer_step=steps["optimizer_step"],
        cycle_warmup_step=steps["cycle_warmup_step"],
        manifest=manifest,
    )


def load_policy_checkpoint_warmstart(
    path: str | Path,
    expected: ExpectedPolicyCheckpointContract,
) -> ResumeState:
    source = Path(path)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("checkpoint path must be a physical directory")
    entries = {item.name for item in source.iterdir()}
    if entries != ALL_FILES:
        raise ValueError(
            f"checkpoint files must be exactly {sorted(ALL_FILES)}, got {sorted(entries)}"
        )
    if any(item.is_symlink() for item in source.iterdir()):
        raise ValueError("checkpoint payload cannot contain symbolic links")
    manifest = RunManifest.from_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    _validate_expected_policy_contract(manifest, expected, include_loss_config=False)
    _verify_payloads(source, manifest)
    online = load_file(str(source / "online_model.safetensors"), device="cpu")
    ema = load_file(str(source / "ema_target.safetensors"), device="cpu")
    _validate_state_mapping(_online_state(expected.policy), online, "online model")
    _validate_state_mapping(
        _cpu_state(expected.policy.ema_state_target.adapter.state_dict()),
        ema,
        "EMA target",
    )
    _copy_policy_state(expected.policy, online)
    _copy_module_state(expected.policy.ema_state_target.adapter, ema)
    expected.policy.restore_ema_step(-1)
    steps = manifest.value["steps"]
    return ResumeState(
        epoch=0,
        global_step=0,
        optimizer_step=0,
        cycle_warmup_step=0,
        manifest=manifest,
    )


def load_legacy_weights_explicit(
    policy: UnifiedCorrectiveForesightPolicy,
    path: str | Path,
    *,
    allowed_keys: set[str],
    minimum_match_fraction: float,
    max_unexpected_keys: int,
) -> LegacyLoadReport:
    if not allowed_keys or any(not isinstance(key, str) or not key for key in allowed_keys):
        raise ValueError("legacy allowed_keys must be an explicit nonempty string set")
    if (
        not 0.0 < minimum_match_fraction <= 1.0
        or type(max_unexpected_keys) is not int
        or max_unexpected_keys < 0
    ):
        raise ValueError("legacy load thresholds are invalid")
    destination = policy.state_dict()
    unknown_allowed = allowed_keys - set(destination)
    if unknown_allowed:
        raise ValueError(f"legacy allowlist contains unknown keys: {sorted(unknown_allowed)}")
    source = load_file(str(path), device="cpu")
    unexpected = tuple(sorted(set(source) - allowed_keys))
    if len(unexpected) > max_unexpected_keys:
        raise ValueError(
            f"unexpected legacy keys exceed threshold: {list(unexpected)}"
        )
    matched = tuple(sorted(set(source) & allowed_keys))
    match_fraction = len(matched) / len(allowed_keys)
    if match_fraction < minimum_match_fraction:
        raise ValueError(
            f"legacy match fraction {match_fraction:.6f} is below "
            f"{minimum_match_fraction:.6f}"
        )
    for name in matched:
        value = source[name]
        expected_value = destination[name]
        if (
            value.shape != expected_value.shape
            or value.dtype != expected_value.dtype
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError(f"legacy tensor contract mismatch for {name}")
    for name in matched:
        destination[name].copy_(
            source[name].to(
                device=destination[name].device,
                dtype=destination[name].dtype,
            )
        )
    return LegacyLoadReport(
        matched_keys=matched,
        unexpected_keys=unexpected,
        match_fraction=match_fraction,
    )


def _online_state(policy: UnifiedCorrectiveForesightPolicy) -> dict[str, Tensor]:
    return _cpu_state(
        {
            name: value
            for name, value in policy.state_dict().items()
            if not name.startswith("ema_state_target.")
            and not name.startswith("online_state_encoder.backbone.")
        }
    )


def _cpu_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in sorted(state.items())
    }


def _build_manifest(state: CheckpointState, directory: Path) -> RunManifest:
    vocabulary = state.policy.condition_encoder.vocabulary
    files = {
        name: {
            "sha256": _file_hash(directory / name),
            "size": (directory / name).stat().st_size,
        }
        for name in sorted(PAYLOAD_FILES)
    }
    return RunManifest(
        {
            "format_version": 1,
            "epoch": state.epoch,
            "steps": {
                "global_step": state.global_step,
                "optimizer_step": state.optimizer_step,
                "cycle_warmup_step": state.cycle_warmup_step,
            },
            "git": {"commit": state.git_commit, "dirty": state.git_dirty},
            "runtime": {"lerobot_version": state.lerobot_version},
            "backbone_provenance": dict(state.backbone_provenance),
            "dataset_revisions": dict(sorted(state.dataset_revisions.items())),
            "model_config": dict(state.model_config),
            "loss_config": dict(state.loss_config),
            "condition_vocabulary": {
                "content": vocabulary.to_dict(),
                "content_hash": vocabulary.content_hash,
            },
            "dataset_specs": {
                spec.dataset_id: {
                    "content": spec.to_dict(),
                    "content_hash": spec.content_hash,
                }
                for spec in sorted(
                    state.dataset_specs, key=lambda item: item.dataset_id
                )
            },
            "action_specs": {
                spec.spec_id: {
                    "content": spec.to_dict(),
                    "content_hash": spec.content_hash,
                }
                for spec in sorted(
                    state.action_specs, key=lambda item: item.spec_id
                )
            },
            "files": files,
        }
    )


def _validate_expected_contract(
    manifest: RunManifest,
    expected: ExpectedCheckpointContract,
) -> None:
    value = manifest.value
    actual = {
        "condition_vocabulary_hash": value["condition_vocabulary"]["content_hash"],
        "dataset_spec_hashes": {
            name: entry["content_hash"]
            for name, entry in value["dataset_specs"].items()
        },
        "action_spec_hashes": {
            name: entry["content_hash"]
            for name, entry in value["action_specs"].items()
        },
        "lerobot_version": value["runtime"]["lerobot_version"],
        "backbone_provenance": dict(value["backbone_provenance"]),
        "dataset_revisions": dict(value["dataset_revisions"]),
        "model_config": dict(value["model_config"]),
        "loss_config": dict(value["loss_config"]),
        "git_commit": value["git"]["commit"],
        "git_dirty": value["git"]["dirty"],
    }
    wanted = {
        "condition_vocabulary_hash": expected.condition_vocabulary_hash,
        "dataset_spec_hashes": dict(expected.dataset_spec_hashes),
        "action_spec_hashes": dict(expected.action_spec_hashes),
        "lerobot_version": expected.lerobot_version,
        "backbone_provenance": dict(expected.backbone_provenance),
        "dataset_revisions": dict(expected.dataset_revisions),
        "model_config": dict(expected.model_config),
        "loss_config": dict(expected.loss_config),
        "git_commit": expected.git_commit,
        "git_dirty": expected.git_dirty,
    }
    for name in wanted:
        if actual[name] != wanted[name]:
            raise ValueError(
                f"checkpoint contract mismatch for {name}: "
                f"expected {wanted[name]!r}, got {actual[name]!r}"
            )


def _validate_expected_policy_contract(
    manifest: RunManifest,
    expected: ExpectedPolicyCheckpointContract,
    *,
    include_loss_config: bool = True,
) -> None:
    value = manifest.value
    actual = {
        "condition_vocabulary_hash": value["condition_vocabulary"]["content_hash"],
        "dataset_spec_hashes": {
            name: entry["content_hash"]
            for name, entry in value["dataset_specs"].items()
        },
        "action_spec_hashes": {
            name: entry["content_hash"]
            for name, entry in value["action_specs"].items()
        },
        "lerobot_version": value["runtime"]["lerobot_version"],
        "backbone_provenance": dict(value["backbone_provenance"]),
        "dataset_revisions": dict(value["dataset_revisions"]),
        "model_config": dict(value["model_config"]),
        "loss_config": dict(value["loss_config"]),
    }
    wanted = {
        "condition_vocabulary_hash": expected.condition_vocabulary_hash,
        "dataset_spec_hashes": dict(expected.dataset_spec_hashes),
        "action_spec_hashes": dict(expected.action_spec_hashes),
        "lerobot_version": expected.lerobot_version,
        "backbone_provenance": dict(expected.backbone_provenance),
        "dataset_revisions": dict(expected.dataset_revisions),
        "model_config": dict(expected.model_config),
        "loss_config": dict(expected.loss_config),
    }
    if not include_loss_config:
        actual.pop("loss_config")
        wanted.pop("loss_config")
    for name in wanted:
        if actual[name] != wanted[name]:
            raise ValueError(
                f"checkpoint contract mismatch for {name}: "
                f"expected {wanted[name]!r}, got {actual[name]!r}"
            )


def _verify_payloads(source: Path, manifest: RunManifest) -> None:
    files = manifest.value["files"]
    if set(files) != PAYLOAD_FILES:
        raise ValueError("manifest payload file list is incomplete")
    for name, metadata in files.items():
        path = source / name
        if path.stat().st_size != metadata["size"]:
            raise ValueError(f"checkpoint size mismatch for {name}")
        if _file_hash(path) != metadata["sha256"]:
            raise ValueError(f"checkpoint hash mismatch for {name}")


def _validate_state_mapping(
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
    name: str,
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{name} tensor keys do not match checkpoint contract")
    for key in expected:
        if (
            actual[key].shape != expected[key].shape
            or actual[key].dtype != expected[key].dtype
            or not torch.isfinite(actual[key]).all().item()
        ):
            raise ValueError(f"{name} tensor contract is invalid for {key}")


def _copy_policy_state(
    policy: UnifiedCorrectiveForesightPolicy,
    source: Mapping[str, Tensor],
) -> None:
    destination = policy.state_dict()
    for name, value in source.items():
        destination[name].copy_(
            value.to(device=destination[name].device, dtype=destination[name].dtype)
        )


def _copy_module_state(
    module: torch.nn.Module,
    source: Mapping[str, Tensor],
) -> None:
    destination = module.state_dict()
    for name, value in source.items():
        destination[name].copy_(
            value.to(device=destination[name].device, dtype=destination[name].dtype)
        )


def _capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "version": 1,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "flow_generator": generator.get_state(),
    }


def _validate_rng_state(state: object) -> None:
    required = {
        "version",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "flow_generator",
    }
    if not isinstance(state, Mapping) or set(state) != required or state["version"] != 1:
        raise ValueError("RNG state fields do not match checkpoint contract")
    if (
        not isinstance(state["torch_cpu"], Tensor)
        or state["torch_cpu"].dtype is not torch.uint8
        or not isinstance(state["flow_generator"], Tensor)
        or state["flow_generator"].dtype is not torch.uint8
        or not isinstance(state["torch_cuda"], list)
    ):
        raise ValueError("RNG tensor state is invalid")


def _restore_rng_state(
    state: Mapping[str, Any],
    flow_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    elif state["torch_cuda"]:
        raise ValueError("checkpoint contains CUDA RNG on a CPU-only runtime")
    flow_generator.set_state(state["flow_generator"])


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
