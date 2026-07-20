from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import uuid

import torch
import torch.distributed as dist

from corrective_foresight.config.conflict_fix import TrackingConfig
from corrective_foresight.training.distributed import DistributedContext


_METADATA_FIELDS = {
    "format_version",
    "entity",
    "project",
    "group",
    "run_id",
    "mode",
    "last_optimizer_step",
    "sync_complete",
}
_SECRET_KEYS = {
    "api_key",
    "wandb_api_key",
    "password",
    "secret",
    "credential",
    "credentials",
    "proxy",
    "http_proxy",
    "https_proxy",
    "access_token",
    "auth_token",
}


@dataclass(frozen=True, slots=True)
class TrackingMetadata:
    format_version: int
    entity: str
    project: str
    group: str
    run_id: str
    mode: str
    last_optimizer_step: int
    sync_complete: bool

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("tracking metadata format_version must be 1")
        if any(
            not isinstance(value, str) or not value
            for value in (self.project, self.group, self.run_id, self.mode)
        ):
            raise ValueError("tracking metadata string fields are invalid")
        if not isinstance(self.entity, str):
            raise ValueError("tracking metadata entity must be a string")
        if self.mode not in {"online", "offline"}:
            raise ValueError("tracking metadata mode must be online or offline")
        if (
            type(self.last_optimizer_step) is not int
            or self.last_optimizer_step < -1
        ):
            raise ValueError("tracking last_optimizer_step must be at least -1")
        if type(self.sync_complete) is not bool:
            raise ValueError("tracking sync_complete must be bool")

    @classmethod
    def from_mapping(cls, value: object) -> TrackingMetadata:
        if not isinstance(value, Mapping) or set(value) != _METADATA_FIELDS:
            raise ValueError("tracking metadata fields do not match contract")
        return cls(**dict(value))


class WandbTracker:
    def __init__(
        self,
        *,
        config: TrackingConfig,
        rank: int,
        output_root: Path,
        enabled: bool,
        run: object | None,
        metadata: TrackingMetadata | None,
    ) -> None:
        self.config = config
        self.rank = rank
        self.output_root = output_root
        self.enabled = enabled
        self.run = run
        self._metadata = metadata
        self._finished = False
        self._network_incomplete = False

    @classmethod
    def start(
        cls,
        *,
        config: TrackingConfig,
        rank: int,
        backend: object | None = None,
        output_root: str | Path,
        job_type: str = "training",
        sanitized_run_config: Mapping[str, object] | None = None,
        mode: str = "online",
    ) -> WandbTracker:
        root = _validate_start_inputs(config, rank, output_root, job_type, mode)
        sanitized = _sanitize_run_config(sanitized_run_config or {})
        if rank != 0:
            return cls(
                config=config,
                rank=rank,
                output_root=root,
                enabled=False,
                run=None,
                metadata=None,
            )
        metadata_path = root / "tracking-metadata.json"
        if metadata_path.exists() or metadata_path.is_symlink():
            raise FileExistsError(f"tracking metadata already exists: {metadata_path}")
        backend = _resolve_backend(backend)
        run = backend.init(
            project=config.project,
            group=config.group,
            job_type=job_type,
            config=sanitized,
        )
        run_id = getattr(run, "id", None)
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("W&B run did not provide a run ID")
        entity = getattr(run, "entity", "")
        if not isinstance(entity, str):
            raise RuntimeError("W&B run entity is invalid")
        metadata = TrackingMetadata(
            format_version=1,
            entity=entity,
            project=config.project,
            group=config.group,
            run_id=run_id,
            mode=mode,
            last_optimizer_step=-1,
            sync_complete=False,
        )
        try:
            _write_metadata_atomic(metadata_path, metadata)
        except Exception:
            try:
                run.finish()
            except Exception:
                pass
            raise
        return cls(
            config=config,
            rank=rank,
            output_root=root,
            enabled=True,
            run=run,
            metadata=metadata,
        )

    @classmethod
    def resume(
        cls,
        *,
        config: TrackingConfig,
        rank: int,
        backend: object | None = None,
        output_root: str | Path,
        job_type: str = "training",
        sanitized_run_config: Mapping[str, object] | None = None,
    ) -> WandbTracker:
        root = _validate_start_inputs(
            config,
            rank,
            output_root,
            job_type,
            "online",
        )
        sanitized = _sanitize_run_config(sanitized_run_config or {})
        if rank != 0:
            return cls(
                config=config,
                rank=rank,
                output_root=root,
                enabled=False,
                run=None,
                metadata=None,
            )
        metadata_path = root / "tracking-metadata.json"
        metadata = _read_metadata(metadata_path)
        if metadata.project != config.project or metadata.group != config.group:
            raise ValueError("tracking metadata project/group mismatch")
        backend = _resolve_backend(backend)
        run = backend.init(
            project=config.project,
            group=config.group,
            job_type=job_type,
            id=metadata.run_id,
            resume="must",
            config=sanitized,
        )
        if getattr(run, "id", None) != metadata.run_id:
            raise RuntimeError("resumed W&B run ID does not match metadata")
        active = TrackingMetadata(
            **{
                **asdict(metadata),
                "sync_complete": False,
            }
        )
        try:
            _write_metadata_atomic(metadata_path, active)
        except Exception:
            try:
                run.finish()
            except Exception:
                pass
            raise
        return cls(
            config=config,
            rank=rank,
            output_root=root,
            enabled=True,
            run=run,
            metadata=active,
        )

    @property
    def run_id(self) -> str | None:
        return self._metadata.run_id if self._metadata is not None else None

    @property
    def last_optimizer_step(self) -> int:
        return (
            self._metadata.last_optimizer_step
            if self._metadata is not None
            else -1
        )

    def log(
        self,
        metrics: Mapping[str, float],
        *,
        optimizer_step: int,
    ) -> None:
        if not self.enabled:
            return
        if self._finished:
            raise RuntimeError("W&B tracker is already finished")
        if type(optimizer_step) is not int or optimizer_step <= self.last_optimizer_step:
            raise ValueError("W&B optimizer steps must be strictly increasing")
        values = _validate_metrics(metrics)
        try:
            self.run.log(values, step=optimizer_step)
        except Exception:
            self._network_incomplete = True
        self._update_metadata(
            last_optimizer_step=optimizer_step,
            sync_complete=False,
        )

    def log_distributed(
        self,
        metrics: Mapping[str, float],
        *,
        optimizer_step: int,
        context: DistributedContext,
    ) -> None:
        if type(optimizer_step) is not int or optimizer_step <= 0:
            raise ValueError("distributed W&B optimizer_step must be positive")
        if optimizer_step % self.config.log_interval:
            return
        if context.rank != self.rank:
            raise ValueError("tracker rank and distributed context disagree")
        reduced = reduce_scalar_metrics(metrics, context)
        if self.rank == 0:
            self.log(reduced, optimizer_step=optimizer_step)

    def log_validation(
        self,
        metrics: Mapping[str, float],
        *,
        optimizer_step: int,
    ) -> None:
        if not self.enabled:
            return
        if self._finished:
            raise RuntimeError("W&B tracker is already finished")
        if optimizer_step != self.last_optimizer_step:
            raise ValueError(
                "W&B validation must use the current optimizer step"
            )
        values = _validate_metrics(metrics)
        if any(not name.startswith("validation/") for name in values):
            raise ValueError("W&B validation metrics require validation/ prefix")
        try:
            self.run.log(values, step=optimizer_step)
        except Exception:
            self._network_incomplete = True
        self._update_metadata(sync_complete=False)

    def finish(self, *, sync_complete: bool) -> None:
        if type(sync_complete) is not bool:
            raise ValueError("sync_complete must be bool")
        if not self.enabled:
            return
        if self._finished:
            raise RuntimeError("W&B tracker is already finished")
        actual_sync = sync_complete and not self._network_incomplete
        try:
            self.run.finish()
        except Exception:
            actual_sync = False
        self._update_metadata(sync_complete=actual_sync)
        self._finished = True

    def record_external_sync_success(self) -> None:
        if not self.enabled:
            return
        if not self._finished:
            raise RuntimeError("external sync can only be recorded after finish")
        self._update_metadata(sync_complete=True)
        self._network_incomplete = False

    def log_checkpoint_artifact(self, path: str | Path) -> None:
        raise ValueError("checkpoint artifact upload is forbidden")

    def _update_metadata(
        self,
        *,
        last_optimizer_step: int | None = None,
        sync_complete: bool | None = None,
    ) -> None:
        if self._metadata is None:
            raise RuntimeError("enabled tracker has no metadata")
        values = asdict(self._metadata)
        if last_optimizer_step is not None:
            values["last_optimizer_step"] = last_optimizer_step
        if sync_complete is not None:
            values["sync_complete"] = sync_complete
        metadata = TrackingMetadata(**values)
        _write_metadata_atomic(
            self.output_root / "tracking-metadata.json",
            metadata,
        )
        self._metadata = metadata


def reduce_scalar_metrics(
    metrics: Mapping[str, float],
    context: DistributedContext,
) -> dict[str, float]:
    if not isinstance(context, DistributedContext):
        raise ValueError("metric reduction requires DistributedContext")
    values = _validate_metrics(metrics)
    names = tuple(values)
    if context.world_size == 1:
        return values
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_rank() != context.rank
        or dist.get_world_size() != context.world_size
    ):
        raise ValueError("metric reduction context does not match process group")
    gathered: list[tuple[str, ...] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, names)
    if any(item != names for item in gathered):
        raise ValueError("distributed metric names differ across ranks")
    payload = torch.tensor(
        [values[name] for name in names],
        dtype=torch.float64,
        device=context.device,
    )
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    payload.div_(context.world_size)
    result = {
        name: float(payload[index].item())
        for index, name in enumerate(names)
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("distributed metric reduction produced non-finite values")
    return result


def _validate_start_inputs(
    config: TrackingConfig,
    rank: int,
    output_root: str | Path,
    job_type: str,
    mode: str,
) -> Path:
    if not isinstance(config, TrackingConfig):
        raise ValueError("W&B tracker requires TrackingConfig")
    if (
        type(config.enabled) is not bool
        or not config.enabled
        or type(config.upload_checkpoints) is not bool
        or config.upload_checkpoints
        or type(config.log_interval) is not int
        or config.log_interval <= 0
        or not isinstance(config.project, str)
        or not config.project
        or not isinstance(config.group, str)
        or not config.group
    ):
        raise ValueError("W&B tracking config is not approved")
    if type(rank) is not int or rank < 0:
        raise ValueError("W&B tracker rank must be nonnegative")
    if not isinstance(job_type, str) or not job_type:
        raise ValueError("W&B job_type must be nonempty")
    if mode not in {"online", "offline"}:
        raise ValueError("W&B mode must be online or offline")
    root = Path(output_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("W&B output_root must be a physical directory")
    return root


def _resolve_backend(backend: object | None) -> object:
    if backend is None:
        import wandb

        backend = wandb
        if getattr(backend, "__version__", None) != "0.24.2":
            raise RuntimeError("production W&B backend must be version 0.24.2")
    if not callable(getattr(backend, "init", None)):
        raise ValueError("W&B backend must provide init")
    return backend


def _validate_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("W&B metrics must be a nonempty mapping")
    if any(not isinstance(name, str) or not name for name in metrics):
        raise ValueError("W&B metric names must be nonempty strings")
    result: dict[str, float] = {}
    for name in sorted(metrics):
        value = metrics[name]
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("W&B metrics must contain finite scalar values")
        result[name] = float(value)
    return result


def _sanitize_run_config(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("W&B run config must be a mapping")

    def sanitize(item: object, path: tuple[str, ...]) -> object:
        if isinstance(item, Mapping):
            result: dict[str, object] = {}
            if any(not isinstance(key, str) or not key for key in item):
                raise ValueError("W&B config keys must be nonempty strings")
            for key in sorted(item):
                if key.lower() in _SECRET_KEYS:
                    raise ValueError(f"secret W&B config key is forbidden: {key}")
                result[key] = sanitize(item[key], (*path, key))
            return result
        if isinstance(item, (list, tuple)):
            return [sanitize(element, path) for element in item]
        if isinstance(item, Path):
            return str(item)
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float) and math.isfinite(item):
            return item
        raise ValueError(f"W&B config value at {'.'.join(path)} is invalid")

    sanitized = sanitize(value, ())
    if not isinstance(sanitized, dict):
        raise RuntimeError("sanitized W&B config is not a mapping")
    return sanitized


def _read_metadata(path: Path) -> TrackingMetadata:
    if path.is_symlink() or not path.is_file():
        raise ValueError("tracking metadata must be a physical file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("tracking metadata is unreadable") from error
    return TrackingMetadata.from_mapping(value)


def _write_metadata_atomic(path: Path, metadata: TrackingMetadata) -> None:
    if path.is_symlink():
        raise ValueError("tracking metadata cannot be a symbolic link")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("tracking metadata parent must be a physical directory")
    temporary = parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(asdict(metadata), ensure_ascii=True, sort_keys=True)
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
