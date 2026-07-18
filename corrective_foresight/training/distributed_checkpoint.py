from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist

from corrective_foresight.policy.unified_policy import UnifiedCorrectiveForesightPolicy
from corrective_foresight.training.checkpoint import (
    CheckpointState,
    ExpectedCheckpointContract,
    ResumeState,
    StatefulMixer,
    _copy_module_state,
    _copy_policy_state,
    _cpu_state,
    _file_hash,
    _fsync_directory,
    _fsync_file,
    _online_state,
    _validate_expected_contract,
    _validate_state_mapping,
)
from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.run_manifest import RunManifest


DISTRIBUTED_FORMAT_VERSION = 2
_SHARED_PAYLOAD_FILES = {
    "online_model.safetensors",
    "ema_target.safetensors",
    "trainer_state.pt",
}


def rank_runtime_name(rank: int) -> str:
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    return f"rank-{rank:04d}-runtime.pt"


def save_distributed_checkpoint_atomic(
    path: str | Path,
    state: CheckpointState,
    context: DistributedContext,
) -> None:
    _require_context(context)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"checkpoint destination already exists: {destination}")

    setup = [{"path": "", "error": None}]
    if context.rank == 0:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
            temporary.mkdir(mode=0o700)
            setup[0]["path"] = str(temporary)
        except Exception as error:
            setup[0]["error"] = repr(error)
    dist.broadcast_object_list(setup, src=0)
    if setup[0]["error"] is not None:
        raise RuntimeError(f"distributed checkpoint setup failed: {setup[0]['error']}")
    temporary = Path(setup[0]["path"])
    local_error: str | None = None
    try:
        if context.rank == 0:
            save_file(_online_state(state.policy), str(temporary / "online_model.safetensors"))
            save_file(
                _cpu_state(state.policy.ema_state_target.adapter.state_dict()),
                str(temporary / "ema_target.safetensors"),
            )
            torch.save(state.trainer.state_dict(), temporary / "trainer_state.pt")
            for name in _SHARED_PAYLOAD_FILES:
                _fsync_file(temporary / name)
        runtime = {
            "version": 1,
            "rank": context.rank,
            "world_size": context.world_size,
            "mixer_state": state.mixer.state_dict(),
            "rng_state": _capture_rank_rng_state(state.flow_generator, context),
        }
        torch.save(runtime, temporary / rank_runtime_name(context.rank))
        _fsync_file(temporary / rank_runtime_name(context.rank))
    except Exception as error:
        local_error = repr(error)

    errors: list[str | None] = [None] * context.world_size
    dist.all_gather_object(errors, local_error)
    if any(error is not None for error in errors):
        if context.rank == 0:
            shutil.rmtree(temporary, ignore_errors=True)
        dist.barrier()
        raise RuntimeError(f"distributed checkpoint payload write failed: {errors}")

    if context.rank == 0:
        try:
            manifest = _build_distributed_manifest(state, temporary, context.world_size)
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(manifest.to_json(), encoding="utf-8")
            _fsync_file(manifest_path)
            _fsync_directory(temporary)
            os.rename(temporary, destination)
            _fsync_directory(destination.parent)
        except Exception as error:
            shutil.rmtree(temporary, ignore_errors=True)
            local_error = repr(error)
    publication_errors: list[str | None] = [None] * context.world_size
    dist.all_gather_object(publication_errors, local_error)
    dist.barrier()
    if any(error is not None for error in publication_errors):
        raise RuntimeError(
            f"distributed checkpoint publication failed: {publication_errors}"
        )


def load_distributed_checkpoint_strict(
    path: str | Path,
    expected: ExpectedCheckpointContract,
    context: DistributedContext,
) -> ResumeState:
    _require_context(context)
    source = Path(path)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("checkpoint path must be a physical directory")
    manifest = RunManifest.from_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.value["format_version"] != DISTRIBUTED_FORMAT_VERSION:
        raise ValueError("distributed loader requires checkpoint manifest format_version 2")
    distributed = manifest.value["distributed"]
    if distributed["world_size"] != context.world_size:
        raise ValueError("checkpoint world size does not match current process group")
    rank_files = tuple(distributed["rank_payloads"])
    expected_files = _SHARED_PAYLOAD_FILES | set(rank_files) | {"manifest.json"}
    entries = {item.name for item in source.iterdir()}
    if entries != expected_files:
        raise ValueError(
            f"distributed checkpoint files must be exactly {sorted(expected_files)}, "
            f"got {sorted(entries)}"
        )
    if any(item.is_symlink() for item in source.iterdir()):
        raise ValueError("distributed checkpoint payload cannot contain symbolic links")
    _validate_expected_contract(manifest, expected)
    _verify_distributed_payloads(source, manifest, expected_files - {"manifest.json"})

    online = load_file(str(source / "online_model.safetensors"), device="cpu")
    ema = load_file(str(source / "ema_target.safetensors"), device="cpu")
    _validate_state_mapping(_online_state(expected.policy), online, "online model")
    _validate_state_mapping(
        _cpu_state(expected.policy.ema_state_target.adapter.state_dict()),
        ema,
        "EMA target",
    )
    trainer_state = torch.load(
        source / "trainer_state.pt", map_location="cpu", weights_only=False
    )
    runtime = torch.load(
        source / rank_runtime_name(context.rank), map_location="cpu", weights_only=False
    )
    _validate_rank_runtime(runtime, context)
    _validate_rank_rng_state(runtime["rng_state"])
    _copy_policy_state(expected.policy, online)
    _copy_module_state(expected.policy.ema_state_target.adapter, ema)
    expected.trainer.load_state_dict(trainer_state)
    expected.mixer.load_state_dict(runtime["mixer_state"])
    _restore_rank_rng_state(runtime["rng_state"], expected.flow_generator, context)
    steps = manifest.value["steps"]
    return ResumeState(
        epoch=manifest.value["epoch"],
        global_step=steps["global_step"],
        optimizer_step=steps["optimizer_step"],
        cycle_warmup_step=steps["cycle_warmup_step"],
        manifest=manifest,
    )


def _require_context(context: DistributedContext) -> None:
    if not isinstance(context, DistributedContext):
        raise ValueError("context must be DistributedContext")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed checkpoint requires an initialized process group")
    if dist.get_rank() != context.rank or dist.get_world_size() != context.world_size:
        raise ValueError("DistributedContext does not match the process group")


def _capture_rank_rng_state(
    generator: torch.Generator,
    context: DistributedContext,
) -> dict[str, object]:
    cuda_state = []
    if context.device.type == "cuda" and torch.cuda.is_available():
        cuda_state = [torch.cuda.get_rng_state(context.local_rank)]
    return {
        "version": 2,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
        "flow_generator": generator.get_state(),
        "local_rank": context.local_rank,
    }


def _validate_rank_runtime(value: object, context: DistributedContext) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"version", "rank", "world_size", "mixer_state", "rng_state"}
        or value["version"] != 1
        or value["rank"] != context.rank
        or value["world_size"] != context.world_size
        or not isinstance(value["mixer_state"], Mapping)
    ):
        raise ValueError("rank runtime fields do not match checkpoint contract")


def _validate_rank_rng_state(value: object) -> None:
    required = {
        "version",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "flow_generator",
        "local_rank",
    }
    if not isinstance(value, Mapping) or set(value) != required or value["version"] != 2:
        raise ValueError("rank RNG state fields do not match checkpoint contract")
    if (
        not isinstance(value["torch_cpu"], torch.Tensor)
        or value["torch_cpu"].dtype is not torch.uint8
        or not isinstance(value["flow_generator"], torch.Tensor)
        or value["flow_generator"].dtype is not torch.uint8
        or not isinstance(value["torch_cuda"], list)
        or type(value["local_rank"]) is not int
        or value["local_rank"] < 0
    ):
        raise ValueError("rank RNG tensor state is invalid")


def _restore_rank_rng_state(
    value: Mapping[str, object],
    flow_generator: torch.Generator,
    context: DistributedContext,
) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    cuda_state = value["torch_cuda"]
    if context.device.type == "cuda":
        if len(cuda_state) != 1:
            raise ValueError("rank checkpoint CUDA RNG state is missing")
        torch.cuda.set_rng_state(cuda_state[0], device=context.local_rank)
    elif cuda_state:
        raise ValueError("CPU rank checkpoint contains CUDA RNG state")
    flow_generator.set_state(value["flow_generator"])


def _build_distributed_manifest(
    state: CheckpointState,
    directory: Path,
    world_size: int,
) -> RunManifest:
    vocabulary = state.policy.condition_encoder.vocabulary
    payload_files = sorted(
        _SHARED_PAYLOAD_FILES | {rank_runtime_name(rank) for rank in range(world_size)}
    )
    files = {
        name: {
            "sha256": _file_hash(directory / name),
            "size": (directory / name).stat().st_size,
        }
        for name in payload_files
    }
    return RunManifest(
        {
            "format_version": DISTRIBUTED_FORMAT_VERSION,
            "distributed": {
                "world_size": world_size,
                "rank_payloads": [rank_runtime_name(rank) for rank in range(world_size)],
            },
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
                for spec in sorted(state.dataset_specs, key=lambda item: item.dataset_id)
            },
            "action_specs": {
                spec.spec_id: {
                    "content": spec.to_dict(),
                    "content_hash": spec.content_hash,
                }
                for spec in sorted(state.action_specs, key=lambda item: item.spec_id)
            },
            "files": files,
        }
    )


def _verify_distributed_payloads(
    source: Path,
    manifest: RunManifest,
    expected_files: set[str],
) -> None:
    files = manifest.value["files"]
    if set(files) != expected_files:
        raise ValueError("distributed manifest payload file list is incomplete")
    for name, metadata in files.items():
        path = source / name
        if path.stat().st_size != metadata["size"]:
            raise ValueError(f"checkpoint size mismatch for {name}")
        if _file_hash(path) != metadata["sha256"]:
            raise ValueError(f"checkpoint hash mismatch for {name}")
