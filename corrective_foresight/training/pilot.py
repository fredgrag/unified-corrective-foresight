from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import uuid

import torch.distributed as dist

from corrective_foresight.training.stages import TrainingStage
from corrective_foresight.training.validation import ValidationRecord


ValidationCallback = Callable[[int], ValidationRecord]
CheckpointCallback = Callable[[int], None]


class PilotController:
    def __init__(
        self,
        *,
        stage: TrainingStage | str,
        checkpoint_interval: int,
        validation_interval: int,
        output_root: str | Path,
        rank: int,
        world_size: int,
        validate: ValidationCallback,
        save_checkpoint: CheckpointCallback,
    ) -> None:
        self.stage = TrainingStage.parse(stage)
        if type(checkpoint_interval) is not int or checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be a positive integer")
        if type(validation_interval) is not int or validation_interval <= 0:
            raise ValueError("validation_interval must be a positive integer")
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        if type(world_size) is not int or world_size <= 0 or rank >= world_size:
            raise ValueError("world_size/rank are invalid")
        if not callable(validate) or not callable(save_checkpoint):
            raise ValueError("pilot callbacks must be callable")
        root = Path(output_root)
        if not root.is_absolute():
            raise ValueError("pilot output_root must be absolute")
        if world_size > 1:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed pilot controller requires process group")
            if dist.get_rank() != rank or dist.get_world_size() != world_size:
                raise ValueError("pilot controller rank/world_size mismatch")
        setup = {"error": None}
        if rank == 0:
            try:
                if root.exists() or root.is_symlink():
                    raise FileExistsError(f"pilot output_root already exists: {root}")
                root.parent.mkdir(parents=True, exist_ok=True)
                root.mkdir()
            except Exception as error:
                setup["error"] = repr(error)
        if world_size > 1:
            setup_payload = [setup]
            dist.broadcast_object_list(setup_payload, src=0)
            setup = setup_payload[0]
        if setup["error"] is not None:
            if "FileExistsError" in setup["error"]:
                raise FileExistsError(setup["error"])
            raise RuntimeError(f"pilot output setup failed: {setup['error']}")
        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval
        self.output_root = root
        self.rank = rank
        self.world_size = world_size
        self.validate = validate
        self.save_checkpoint = save_checkpoint
        self._last_step = 0
        self._checkpoint_steps: set[int] = set()

    def on_optimizer_step(self, step: int, result: object | None = None) -> None:
        if type(step) is not int or step <= self._last_step:
            raise ValueError("pilot optimizer steps must be strictly increasing")
        self._last_step = step
        if step % self.validation_interval == 0:
            record = self.validate(step)
            if not isinstance(record, ValidationRecord):
                raise ValueError("pilot validation callback must return ValidationRecord")
            if record.global_step != step or record.stage != self.stage.value:
                raise ValueError("pilot validation record does not match current step/stage")
            if record.world_size != self.world_size:
                raise ValueError("pilot validation record world_size mismatch")
            if self.rank == 0:
                self._write_validation(record)
        if step % self.checkpoint_interval == 0:
            self._save_checkpoint_once(step)

    def finish(self, final_step: int) -> None:
        if type(final_step) is not int or final_step < self._last_step:
            raise ValueError("pilot final_step must be at least the last optimizer step")
        if final_step == 0:
            return
        if final_step not in self._checkpoint_steps:
            self._save_checkpoint_once(final_step)

    def save_initial(self) -> None:
        if self._last_step != 0:
            raise ValueError("initial pilot checkpoint must be saved before optimizer steps")
        self._save_checkpoint_once(0)

    def _save_checkpoint_once(self, step: int) -> None:
        if step in self._checkpoint_steps:
            return
        self.save_checkpoint(step)
        self._checkpoint_steps.add(step)

    def _write_validation(self, record: ValidationRecord) -> None:
        directory = self.output_root / "validation"
        if directory.exists() or directory.is_symlink():
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("validation output directory is not a physical directory")
        else:
            directory.mkdir()
        destination = directory / f"{record.stage}-{record.global_step:06d}.json"
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"validation output already exists: {destination}")
        payload = {
            "global_step": record.global_step,
            "stage": record.stage,
            "world_size": record.world_size,
            "batches_per_rank": record.batches_per_rank,
            "samples": record.samples,
            "metrics": dict(record.metrics),
        }
        temporary = directory / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, destination)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
