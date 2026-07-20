from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Protocol

import torch
import torch.distributed as dist

from corrective_foresight.data.conditioning import ConditionedTrajectoryDataset
from corrective_foresight.data.lerobot_adapter import LeRobotTrajectoryAdapter
from corrective_foresight.data.mixer import BalancedLeRobotMixer
from corrective_foresight.data.stateful_sampler import StatefulDistributedBatchSampler
from corrective_foresight.data.batch import TrajectoryBatch
from corrective_foresight.config.conflict_fix import (
    load_conflict_fix_config,
)
from corrective_foresight.config.pilot import load_pilot_config
from corrective_foresight.runtime import (
    ProductionRuntime,
    assemble_runtime_policy,
    expected_policy_checkpoint_contract,
    load_production_runtime,
)
from corrective_foresight.training.checkpoint import (
    CheckpointState,
    ExpectedCheckpointContract,
    load_checkpoint_strict,
    load_policy_checkpoint_warmstart,
    save_checkpoint_atomic,
)
from corrective_foresight.training.distributed_checkpoint import (
    load_distributed_checkpoint_strict,
    load_distributed_policy_checkpoint_strict,
    save_distributed_checkpoint_atomic,
)
from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.conflict_fix_phase import (
    resolve_conflict_fix_phase,
)
from corrective_foresight.training.gates import (
    PilotStepDecision,
    ValidationStopMonitor,
)
from corrective_foresight.training.pilot import PilotController
from corrective_foresight.training.stages import TrainingStage
from corrective_foresight.training.trainer import Trainer, TrainerConfig, TrainStepResult
from corrective_foresight.training.validation import (
    ValidationConfig,
    ValidationRecord,
    ValidationRunner,
)
from corrective_foresight.tracking.wandb_tracker import WandbTracker


class BatchSource(Protocol):
    def next_batch(self) -> TrajectoryBatch: ...


class _IndexedTrajectoryDataset:
    def __init__(self, dataset, indices: tuple[int, ...]) -> None:
        if not indices:
            raise ValueError("indexed dataset requires at least one valid index")
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


MetricLogger = Callable[[dict[str, float | int | str]], None]
OptimizerStepCallback = Callable[
    [int, TrainStepResult],
    PilotStepDecision | None,
]


def run_training(
    *,
    batch_source: BatchSource,
    trainer: Trainer,
    stage: TrainingStage | str,
    optimizer_steps: int,
    start_global_step: int,
    generator_seed: int,
    metric_logger: MetricLogger,
    flow_generator: torch.Generator | None = None,
    optimizer_step_callback: OptimizerStepCallback | None = None,
    wandb_tracker: WandbTracker | None = None,
) -> int:
    stage = TrainingStage.parse(stage)
    if type(optimizer_steps) is not int or optimizer_steps <= 0:
        raise ValueError("optimizer_steps must be a positive integer")
    if type(start_global_step) is not int or start_global_step < 0:
        raise ValueError("start_global_step must be a nonnegative integer")
    if type(generator_seed) is not int or generator_seed < 0:
        raise ValueError("generator_seed must be a nonnegative integer")
    if not callable(metric_logger):
        raise ValueError("metric_logger must be callable")
    if optimizer_step_callback is not None and not callable(optimizer_step_callback):
        raise ValueError("optimizer_step_callback must be callable")
    if wandb_tracker is not None and not isinstance(
        wandb_tracker,
        WandbTracker,
    ):
        raise ValueError("wandb_tracker must be WandbTracker")
    device = trainer.distributed_context.device
    if flow_generator is None:
        generator = torch.Generator(device=device).manual_seed(generator_seed)
    else:
        if flow_generator.device != device:
            raise ValueError("flow_generator device must match trainer device")
        generator = flow_generator
    global_step = start_global_step
    final_step = start_global_step + optimizer_steps
    optimizer_step_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while global_step < final_step:
        batch = batch_source.next_batch().to(device)
        result = trainer.train_step(
            batch,
            stage,
            global_step,
            generator,
        )
        if not result.optimizer_stepped:
            continue
        record: dict[str, float | int | str] = {
            "stage": stage.value,
            "dataset_id": batch.dataset_id,
            "global_step": global_step,
            "optimizer_step": global_step + 1,
            "learning_rate": result.learning_rate,
            "pre_clip_gradient_norm": float(
                result.pre_clip_gradient_norm.item()
            ),
        }
        record.update(
            {name: float(value.item()) for name, value in result.output.metrics.items()}
        )
        record.update(
            {
                f"gradient_norm/{name}": float(value.item())
                for name, value in result.per_loss_gradient_norms.items()
            }
        )
        record.update(
            {
                f"learning_rate/{name}": value
                for name, value in result.learning_rates.items()
            }
        )
        record.update(result.gradient_metrics)
        elapsed = time.perf_counter() - optimizer_step_started
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("optimizer step duration must be finite and positive")
        global_samples = (
            int(batch.rgb.shape[0])
            * trainer.distributed_context.world_size
            * trainer.config.accumulation_steps
        )
        pre_clip_norm = float(result.pre_clip_gradient_norm.item())
        clip_coefficient = min(
            1.0,
            trainer.config.max_grad_norm / max(pre_clip_norm, 1e-12),
        )
        record.update(
            {
                "optimizer_step_time_seconds": elapsed,
                "samples_per_second": global_samples / elapsed,
                "peak_gpu_memory_bytes": (
                    float(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0.0
                ),
                "gradient_clip/coefficient": clip_coefficient,
                "gradient_clip/frequency": float(clip_coefficient < 1.0),
            }
        )
        metric_logger(record)
        optimizer_step = global_step + 1
        if wandb_tracker is not None:
            wandb_tracker.log_distributed(
                {
                    name: float(value)
                    for name, value in record.items()
                    if isinstance(value, (int, float))
                },
                optimizer_step=optimizer_step,
                context=trainer.distributed_context,
            )
        global_step += 1
        if optimizer_step_callback is not None:
            decision = optimizer_step_callback(global_step, result)
            if decision is not None and not isinstance(
                decision,
                PilotStepDecision,
            ):
                raise ValueError(
                    "optimizer_step_callback must return PilotStepDecision or None"
                )
            if decision is not None and decision.should_stop:
                break
        optimizer_step_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    return global_step


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", choices=("world_pretrain", "unified"))
    parser.add_argument("--pilot-config", type=Path)
    parser.add_argument("--pilot-stage", choices=("world_pretrain", "unified"))
    parser.add_argument("--conflict-fix-config", type=Path)
    parser.add_argument(
        "--conflict-fix-phase",
        choices=(
            "world_pretrain",
            "gradient_audit",
            "unified_gate",
            "unified_continue",
        ),
    )
    parser.add_argument("--audit-decision", type=Path)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args(argv)
    sources = sum(
        value is not None
        for value in (
            args.config,
            args.pilot_config,
            args.conflict_fix_config,
        )
    )
    if sources != 1:
        parser.error(
            "exactly one of --config, --pilot-config, or "
            "--conflict-fix-config is required"
        )
    if args.conflict_fix_config is not None:
        if args.conflict_fix_phase is None:
            parser.error(
                "--conflict-fix-phase is required with --conflict-fix-config"
            )
        forbidden = {
            "--stage": args.stage,
            "--pilot-stage": args.pilot_stage,
            "--steps": args.steps,
            "--output-checkpoint": args.output_checkpoint,
        }
        used = [name for name, value in forbidden.items() if value is not None]
        if args.initialize_only:
            used.append("--initialize-only")
        if used:
            parser.error(
                f"conflict-fix phases forbid schedule overrides: {sorted(used)}"
            )
        phase = args.conflict_fix_phase
        if phase == "world_pretrain":
            if any(
                value is not None
                for value in (
                    args.init_checkpoint,
                    args.resume,
                    args.audit_decision,
                )
            ):
                parser.error("world_pretrain conflict-fix phase must be fresh")
        elif phase == "gradient_audit":
            if args.init_checkpoint is None:
                parser.error("gradient_audit requires --init-checkpoint")
            if args.resume is not None or args.audit_decision is not None:
                parser.error(
                    "gradient_audit forbids --resume and --audit-decision"
                )
        elif phase == "unified_gate":
            if args.init_checkpoint is None or args.audit_decision is None:
                parser.error(
                    "unified_gate requires --init-checkpoint and --audit-decision"
                )
            if args.resume is not None:
                parser.error("unified_gate forbids --resume")
        elif phase == "unified_continue":
            if args.resume is None or args.audit_decision is None:
                parser.error(
                    "unified_continue requires --resume and --audit-decision"
                )
            if args.init_checkpoint is not None:
                parser.error("unified_continue forbids --init-checkpoint")
    elif args.conflict_fix_phase is not None or args.audit_decision is not None:
        parser.error(
            "--conflict-fix-phase/--audit-decision require "
            "--conflict-fix-config"
        )
    if args.pilot_config is not None and args.pilot_stage is None:
        parser.error("--pilot-stage is required with --pilot-config")
    if args.pilot_config is not None and args.stage is not None:
        parser.error("--stage cannot be combined with --pilot-config")
    if args.pilot_config is not None and args.steps is not None:
        parser.error("--steps cannot be combined with --pilot-config")
    if args.resume is not None and args.init_checkpoint is not None:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if args.initialize_only and (args.resume is not None or args.init_checkpoint is not None):
        parser.error("--initialize-only cannot be combined with checkpoint loading")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pilot = load_pilot_config(args.pilot_config) if args.pilot_config else None
    conflict_fix = (
        load_conflict_fix_config(args.conflict_fix_config)
        if args.conflict_fix_config
        else None
    )
    phase_plan = (
        resolve_conflict_fix_phase(
            conflict_fix,
            args.conflict_fix_phase,
            init_checkpoint=args.init_checkpoint,
            resume=args.resume,
            audit_decision=args.audit_decision,
        )
        if conflict_fix is not None
        else None
    )
    if phase_plan is not None:
        runtime = load_production_runtime(phase_plan.experiment)
        stage = phase_plan.stage
        target_steps = phase_plan.target_step
        stage_output_root = phase_plan.output_root
    elif pilot is None:
        runtime = load_production_runtime(args.config)
        stage = args.stage or runtime.config.stage
        target_steps = args.steps or runtime.config.training.total_steps
        stage_output_root = runtime.config.output_root
    else:
        stage = args.pilot_stage
        experiment = (
            pilot.world_experiment
            if stage == "world_pretrain"
            else pilot.unified_experiment
        )
        runtime = load_production_runtime(experiment)
        target_steps = (
            pilot.world_steps if stage == "world_pretrain" else pilot.unified_steps
        )
        stage_output_root = pilot.output_root / stage
    if stage != runtime.config.stage:
        raise ValueError("--stage must match the experiment config stage")
    if not torch.cuda.is_available():
        raise RuntimeError("production training requires CUDA")
    context = _initialize_distributed_context()
    if conflict_fix is not None and context.world_size != len(
        conflict_fix.gpu_indices
    ):
        raise RuntimeError("conflict-fix phases require exactly four ranks")
    device = context.device
    torch.manual_seed(runtime.config.seed)
    torch.cuda.manual_seed_all(runtime.config.seed)
    policy = assemble_runtime_policy(runtime, device=device)
    mixer = _build_mixer(runtime, context, split="train")
    training = runtime.config.training
    gradient_mode = phase_plan.gradient_mode if phase_plan else "ordinary"
    protected_lr_multiplier = (
        1.0
        if phase_plan is None or phase_plan.phase == "world_pretrain"
        else conflict_fix.protected_lr_multiplier
    )
    trainer = Trainer(
        policy,
        TrainerConfig(
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            accumulation_steps=training.accumulation_steps,
            max_grad_norm=training.max_grad_norm,
            warmup_steps=training.warmup_steps,
            total_steps=training.total_steps,
            bf16=training.bf16,
            ddp=context.world_size > 1,
            stage=stage,
            protected_lr_multiplier=protected_lr_multiplier,
            gradient_mode=gradient_mode,
            gradient_diagnostic_interval=(
                conflict_fix.conflict_log_interval if conflict_fix else 10
            ),
            pcgrad_seed=runtime.config.seed,
        ),
        rank=context.rank,
        distributed_context=context,
    )
    flow_generator = torch.Generator(device=device).manual_seed(runtime.config.seed + 300)
    state = _checkpoint_state(runtime, policy, trainer, mixer, flow_generator, context)
    start_global_step = 0
    if args.resume is not None:
        resume_path = args.resume.resolve(strict=True)
        if pilot is not None or phase_plan is not None or context.world_size > 1:
            resume = load_distributed_checkpoint_strict(
                resume_path,
                ExpectedCheckpointContract.from_state(state),
                context,
            )
        else:
            resume = load_checkpoint_strict(
                resume_path,
                ExpectedCheckpointContract.from_state(state),
            )
        start_global_step = resume.global_step
    elif args.init_checkpoint is not None:
        init_path = args.init_checkpoint.resolve(strict=True)
        if (pilot is not None or phase_plan is not None) and context.world_size > 1:
            load_distributed_policy_checkpoint_strict(
                init_path,
                expected_policy_checkpoint_contract(runtime, policy),
                context,
            )
        else:
            load_policy_checkpoint_warmstart(
                init_path,
                expected_policy_checkpoint_contract(runtime, policy),
            )
        policy.restore_ema_step(-1)

    controller = None
    wandb_tracker = None
    managed_run = pilot is not None or phase_plan is not None
    if managed_run:
        validation_batches_per_rank = (
            pilot.validation_batches_per_rank
            if pilot is not None
            else conflict_fix.validation_batches_per_rank
        )
        validation_mixer = _build_mixer(
            runtime,
            context,
            split="validation",
            sampler_seed_offset=1000,
            mixer_seed_offset=2000,
        )
        validation_runner = ValidationRunner(
            policy=policy,
            validation_mixer=validation_mixer,
            context=context,
            config=ValidationConfig(
                batches_per_rank=validation_batches_per_rank,
                generator_seed=runtime.config.seed + 400,
            ),
        )

        def save_pilot_checkpoint(step: int) -> dict[str, object]:
            destination = stage_output_root / "checkpoints" / f"{stage}-{step:06d}"
            checkpoint_state = _checkpoint_state(
                runtime,
                policy,
                trainer,
                mixer,
                flow_generator,
                context,
                global_step=step,
            )
            if context.world_size > 1:
                save_distributed_checkpoint_atomic(destination, checkpoint_state, context)
            elif context.rank == 0:
                save_checkpoint_atomic(destination, checkpoint_state)
            if context.rank == 0:
                print(f"checkpoint={destination.resolve()}")
            manifest = destination / "manifest.json"
            if manifest.is_symlink() or not manifest.is_file():
                raise RuntimeError("published checkpoint manifest is unavailable")
            return {
                "checkpoint_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest()
            }

        def observe_validation(record) -> None:
            if wandb_tracker is None:
                return
            wandb_tracker.log_validation(
                {
                    f"validation/{name}": value
                    for name, value in record.metrics.items()
                },
                optimizer_step=record.global_step,
            )

        stop_monitor = None
        stop_report_context = None
        if phase_plan is not None and phase_plan.phase in {
            "unified_gate",
            "unified_continue",
        }:
            world_record = _read_validation_record(
                conflict_fix.output_root
                / "world_pretrain/validation/world_pretrain-005000.json"
            )
            stop_monitor = ValidationStopMonitor(
                world_dynamics_loss=world_record.metrics["dynamics_loss"]
            )
            if phase_plan.phase == "unified_continue":
                validation_directory = (
                    conflict_fix.output_root / "unified/validation"
                )
                if (
                    validation_directory.is_symlink()
                    or not validation_directory.is_dir()
                ):
                    raise ValueError(
                        "unified continuation validation history is unavailable"
                    )
                history_paths = tuple(
                    sorted(validation_directory.glob("unified-*.json"))
                )
                if (
                    not history_paths
                    or history_paths[-1].name != "unified-005000.json"
                ):
                    raise ValueError(
                        "unified continuation validation history is incomplete"
                    )
                for path in history_paths:
                    decision = stop_monitor.observe(
                        _read_validation_record(path)
                    )
                    if decision.should_stop:
                        raise ValueError(
                            "validation history contains an automatic stop"
                        )
            stop_report_context = {
                "resolved_config": _json_safe(conflict_fix),
                "git_commit": _git_commit(),
                "data_spec_hashes": {
                    **{
                        f"dataset:{spec.dataset_id}": spec.content_hash
                        for spec in runtime.dataset_specs
                    },
                    **{
                        f"action:{spec.spec_id}": spec.content_hash
                        for spec in runtime.action_specs
                    },
                },
            }

        controller = PilotController(
            stage=stage,
            checkpoint_interval=(
                pilot.checkpoint_interval
                if pilot is not None
                else conflict_fix.checkpoint_interval
            ),
            validation_interval=(
                pilot.validation_interval
                if pilot is not None
                else conflict_fix.validation_interval
            ),
            output_root=stage_output_root,
            rank=context.rank,
            world_size=context.world_size,
            validate=lambda step: validation_runner.evaluate(stage, step),
            save_checkpoint=save_pilot_checkpoint,
            stop_monitor=stop_monitor,
            stop_report_context=stop_report_context,
            resume_existing_output=(
                phase_plan.resume_existing_output if phase_plan else False
            ),
            save_final_checkpoint=(
                phase_plan.save_final_checkpoint if phase_plan else True
            ),
            validation_observer=observe_validation,
        )
        if phase_plan is not None:
            tracking_config = {
                "pilot": _json_safe(conflict_fix),
                "experiment": _json_safe(runtime.config),
                "phase": phase_plan.phase,
                "gradient_mode": phase_plan.gradient_mode,
                "audit_decision_sha256": phase_plan.audit_decision_sha256,
            }
            tracker_factory = (
                WandbTracker.resume
                if phase_plan.start_mode == "resume"
                else WandbTracker.start
            )
            wandb_tracker = tracker_factory(
                config=conflict_fix.tracking,
                rank=context.rank,
                output_root=stage_output_root,
                job_type=phase_plan.job_type,
                sanitized_run_config=tracking_config,
            )
        if args.initialize_only:
            controller.save_initial()
            final_step = 0
        else:
            if start_global_step > target_steps:
                raise ValueError("resume step exceeds pilot stage target")
            final_step = target_steps
            if start_global_step == 0 and (
                phase_plan is None
                or phase_plan.phase != "gradient_audit"
            ):
                controller.save_initial()
            if phase_plan is not None and start_global_step == 0:
                wandb_tracker.log(
                    {"phase/initialized": 1.0},
                    optimizer_step=0,
                )
                controller.validate_initial()
    elif args.initialize_only:
        final_step = 0
    else:
        final_step = start_global_step + target_steps
    if not args.initialize_only:
        metrics_path = args.metrics_output or (
            stage_output_root / "metrics" / f"rank-{context.rank}.jsonl"
        )
        logger = _jsonl_logger(metrics_path) if context.rank == 0 else (lambda _: None)
        final_step = run_training(
            batch_source=mixer,
            trainer=trainer,
            stage=stage,
            optimizer_steps=final_step - start_global_step,
            start_global_step=start_global_step,
            generator_seed=runtime.config.seed + 300,
            metric_logger=logger,
            flow_generator=flow_generator,
            optimizer_step_callback=(controller.on_optimizer_step if controller else None),
            wandb_tracker=wandb_tracker,
        )
    if controller is not None:
        controller.finish(final_step)
    elif context.world_size > 1:
        dist.barrier()
    if controller is None and context.rank == 0:
        destination = args.output_checkpoint
        if destination is None:
            label = "untrained" if args.initialize_only else f"{final_step:06d}"
            destination = stage_output_root / "checkpoints" / f"{stage}-{label}"
        final_state = _checkpoint_state(
            runtime, policy, trainer, mixer, flow_generator, context,
            global_step=final_step,
        )
        save_checkpoint_atomic(destination.resolve(), final_state)
        print(f"checkpoint={destination.resolve()}")
    if wandb_tracker is not None:
        wandb_tracker.finish(sync_complete=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _read_validation_record(path: Path) -> ValidationRecord:
    if path.is_symlink() or not path.is_file():
        raise ValueError("required validation record must be a physical file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("required validation record is unreadable") from error
    expected = {
        "global_step",
        "stage",
        "world_size",
        "batches_per_rank",
        "samples",
        "metrics",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("validation record fields do not match contract")
    try:
        return ValidationRecord(**value)
    except TypeError as error:
        raise ValueError("validation record values do not match contract") from error


def _json_safe(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("resolved config mapping keys must be strings")
        return {key: _json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"resolved config contains unsupported value {type(value)!r}")


def _git_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("Git commit is not a full SHA-1")
    return commit


def _initialize_distributed_context() -> DistributedContext:
    if "RANK" not in os.environ:
        return DistributedContext.single_process("cuda:0")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError("distributed launch must set LOCAL_RANK")
    torch.cuda.set_device(local_rank)
    return DistributedContext.from_initialized_process_group(
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
    )


def _build_mixer(
    runtime: ProductionRuntime,
    context: DistributedContext,
    *,
    split: str = "train",
    sampler_seed_offset: int = 0,
    mixer_seed_offset: int = 0,
) -> BalancedLeRobotMixer:
    datasets = {}
    samplers = {}
    weights = {}
    action_by_id = {spec.spec_id: spec for spec in runtime.action_specs}
    for index, dataset_spec in enumerate(runtime.dataset_specs):
        action_spec = action_by_id[dataset_spec.action_spec_id]
        adapter = LeRobotTrajectoryAdapter(
            dataset_spec,
            action_spec,
            split=split,
            context_steps=runtime.config.evaluation.context_steps,
            action_horizon=runtime.config.evaluation.action_horizon,
            video_backend="torchcodec",
        )
        adapter_for_mixer = _IndexedTrajectoryDataset(
            adapter,
            adapter.full_dynamics_indices,
        )
        conditioned = ConditionedTrajectoryDataset(
            adapter_for_mixer,
            dataset_spec=dataset_spec,
            action_spec=action_spec,
            vocabulary=runtime.vocabulary,
        )
        datasets[dataset_spec.dataset_id] = conditioned
        samplers[dataset_spec.dataset_id] = StatefulDistributedBatchSampler(
            dataset_size=len(conditioned),
            batch_size=runtime.config.training.batch_size_per_rank,
            seed=runtime.config.seed + 100 + sampler_seed_offset + index,
            rank=context.rank,
            world_size=context.world_size,
        )
        weights[dataset_spec.dataset_id] = dataset_spec.sample_weight
    return BalancedLeRobotMixer(
        datasets=datasets,
        samplers=samplers,
        weights=weights,
        generator=torch.Generator().manual_seed(
            runtime.config.seed + 200 + mixer_seed_offset
        ),
    )


def _checkpoint_state(
    runtime: ProductionRuntime,
    policy,
    trainer: Trainer,
    mixer: BalancedLeRobotMixer,
    flow_generator: torch.Generator,
    context: DistributedContext,
    *,
    global_step: int | None = None,
) -> CheckpointState:
    current_step = trainer.policy.last_ema_step + 1 if global_step is None else global_step
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout
    )
    return CheckpointState(
        policy=policy,
        trainer=trainer,
        mixer=mixer,
        flow_generator=flow_generator,
        epoch=0,
        global_step=current_step,
        optimizer_step=current_step,
        cycle_warmup_step=min(current_step, 5000),
        dataset_specs=runtime.dataset_specs,
        action_specs=runtime.action_specs,
        lerobot_version="0.5.1",
        backbone_provenance=runtime.backbone_provenance,
        dataset_revisions={spec.dataset_id: spec.revision for spec in runtime.dataset_specs},
        model_config=asdict(runtime.config.model),
        loss_config=asdict(policy.objective.config),
        git_commit=commit,
        git_dirty=dirty,
    )


def _jsonl_logger(path: Path) -> MetricLogger:
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(record: dict[str, float | int | str]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    return log


if __name__ == "__main__":
    main()
