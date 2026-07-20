from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import uuid

import torch.distributed as dist

from corrective_foresight.training.gates import (
    PilotStepDecision,
    ValidationStopMonitor,
    write_stop_report,
)
from corrective_foresight.training.stages import TrainingStage
from corrective_foresight.training.validation import ValidationRecord


ValidationCallback = Callable[[int], ValidationRecord]
CheckpointCallback = Callable[[int], Mapping[str, object] | None]
_STOP_REPORT_CONTEXT_FIELDS = {
    "resolved_config",
    "git_commit",
    "data_spec_hashes",
}


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
        stop_monitor: ValidationStopMonitor | None = None,
        stop_report_context: Mapping[str, object] | None = None,
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
        if stop_monitor is not None and not isinstance(
            stop_monitor,
            ValidationStopMonitor,
        ):
            raise ValueError("stop_monitor must be ValidationStopMonitor")
        if stop_monitor is not None and self.stage is not TrainingStage.UNIFIED:
            raise ValueError("validation stop monitor is unified-only")
        if stop_monitor is not None and not isinstance(
            stop_report_context,
            Mapping,
        ):
            raise ValueError("stop_report_context is required with stop_monitor")
        if stop_monitor is not None:
            self._validate_stop_report_context(stop_report_context)
        if stop_monitor is None and stop_report_context is not None:
            raise ValueError("stop_report_context requires stop_monitor")
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
        self.stop_monitor = stop_monitor
        self.stop_report_context = dict(stop_report_context or {})
        self._last_step = 0
        self._checkpoint_steps: set[int] = set()
        self._checkpoint_metadata: dict[int, dict[str, object]] = {}
        self._stop_decision = PilotStepDecision(False, None)

    def on_optimizer_step(
        self,
        step: int,
        result: object | None = None,
    ) -> PilotStepDecision:
        if self._stop_decision.should_stop:
            raise RuntimeError("pilot controller already stopped")
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
            if self.stop_monitor is not None:
                decision = (
                    self.stop_monitor.observe(record)
                    if self.rank == 0
                    else PilotStepDecision(False, None)
                )
                decision = self._broadcast_decision(decision)
                if decision.should_stop:
                    checkpoint_metadata = self._save_checkpoint_once(step)
                    if set(checkpoint_metadata) != {
                        "checkpoint_manifest_sha256"
                    }:
                        raise ValueError(
                            "stopping checkpoint must return only its manifest hash"
                        )
                    if self.rank == 0:
                        write_stop_report(
                            self.output_root / "stop-report.json",
                            decision,
                            context={
                                **self.stop_report_context,
                                **checkpoint_metadata,
                                "stage": self.stage.value,
                                "checkpoint_step": step,
                            },
                        )
                    self._stop_decision = decision
                    return decision
        if step % self.checkpoint_interval == 0:
            self._save_checkpoint_once(step)
        return PilotStepDecision(False, None)

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

    def _save_checkpoint_once(self, step: int) -> Mapping[str, object]:
        if step in self._checkpoint_steps:
            return self._checkpoint_metadata[step]
        metadata = self.save_checkpoint(step)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(
                "checkpoint callback must return a metadata mapping or None"
            )
        self._checkpoint_steps.add(step)
        self._checkpoint_metadata[step] = dict(metadata or {})
        return self._checkpoint_metadata[step]

    def _broadcast_decision(
        self,
        decision: PilotStepDecision,
    ) -> PilotStepDecision:
        if self.world_size == 1:
            return decision
        payload = [decision]
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        if not isinstance(result, PilotStepDecision):
            raise RuntimeError("distributed stop decision is invalid")
        return result

    @staticmethod
    def _validate_stop_report_context(
        context: Mapping[str, object] | None,
    ) -> None:
        if context is None or set(context) != _STOP_REPORT_CONTEXT_FIELDS:
            raise ValueError(
                "stop_report_context fields must be exactly "
                f"{sorted(_STOP_REPORT_CONTEXT_FIELDS)}"
            )
        resolved = context["resolved_config"]
        hashes = context["data_spec_hashes"]
        commit = context["git_commit"]
        if not isinstance(resolved, Mapping) or not resolved:
            raise ValueError("stop_report_context resolved_config is invalid")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValueError("stop_report_context git_commit is invalid")
        if (
            not isinstance(hashes, Mapping)
            or not hashes
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for name, value in hashes.items()
            )
        ):
            raise ValueError("stop_report_context data_spec_hashes are invalid")

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
