from __future__ import annotations

import argparse
from dataclasses import asdict
from collections.abc import Callable
import json
import os
from pathlib import Path
import subprocess
from typing import Protocol

import torch
import torch.distributed as dist

from corrective_foresight.data.conditioning import ConditionedTrajectoryDataset
from corrective_foresight.data.lerobot_adapter import LeRobotTrajectoryAdapter
from corrective_foresight.data.mixer import BalancedLeRobotMixer
from corrective_foresight.data.stateful_sampler import StatefulDistributedBatchSampler
from corrective_foresight.data.batch import TrajectoryBatch
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
from corrective_foresight.training.pilot import PilotController
from corrective_foresight.training.stages import TrainingStage
from corrective_foresight.training.trainer import Trainer, TrainerConfig, TrainStepResult
from corrective_foresight.training.validation import ValidationConfig, ValidationRunner


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
OptimizerStepCallback = Callable[[int, TrainStepResult], None]


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
    device = trainer.distributed_context.device
    if flow_generator is None:
        generator = torch.Generator(device=device).manual_seed(generator_seed)
    else:
        if flow_generator.device != device:
            raise ValueError("flow_generator device must match trainer device")
        generator = flow_generator
    global_step = start_global_step
    final_step = start_global_step + optimizer_steps
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
        metric_logger(record)
        global_step += 1
        if optimizer_step_callback is not None:
            optimizer_step_callback(global_step, result)
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
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args(argv)
    if (args.config is None) == (args.pilot_config is None):
        parser.error("exactly one of --config or --pilot-config is required")
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
    if pilot is None:
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
    device = context.device
    torch.manual_seed(runtime.config.seed)
    torch.cuda.manual_seed_all(runtime.config.seed)
    policy = assemble_runtime_policy(runtime, device=device)
    mixer = _build_mixer(runtime, context, split="train")
    training = runtime.config.training
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
        ),
        rank=context.rank,
        distributed_context=context,
    )
    flow_generator = torch.Generator(device=device).manual_seed(runtime.config.seed + 300)
    state = _checkpoint_state(runtime, policy, trainer, mixer, flow_generator, context)
    start_global_step = 0
    if args.resume is not None:
        resume_path = args.resume.resolve(strict=True)
        if pilot is not None or context.world_size > 1:
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
        if pilot is not None and context.world_size > 1:
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
    if pilot is not None:
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
                batches_per_rank=pilot.validation_batches_per_rank,
                generator_seed=runtime.config.seed + 400,
            ),
        )

        def save_pilot_checkpoint(step: int) -> None:
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

        controller = PilotController(
            stage=stage,
            checkpoint_interval=pilot.checkpoint_interval,
            validation_interval=pilot.validation_interval,
            output_root=stage_output_root,
            rank=context.rank,
            world_size=context.world_size,
            validate=lambda step: validation_runner.evaluate(stage, step),
            save_checkpoint=save_pilot_checkpoint,
        )
        if args.initialize_only:
            controller.save_initial()
            final_step = 0
        else:
            if start_global_step > target_steps:
                raise ValueError("resume step exceeds pilot stage target")
            final_step = target_steps
            if start_global_step == 0:
                controller.save_initial()
    elif args.initialize_only:
        final_step = 0
    else:
        final_step = start_global_step + target_steps
    if not args.initialize_only:
        metrics_path = args.metrics_output or (
            stage_output_root / "metrics" / f"rank-{context.rank}.jsonl"
        )
        logger = _jsonl_logger(metrics_path) if context.rank == 0 else (lambda _: None)
        run_training(
            batch_source=mixer,
            trainer=trainer,
            stage=stage,
            optimizer_steps=final_step - start_global_step,
            start_global_step=start_global_step,
            generator_seed=runtime.config.seed + 300,
            metric_logger=logger,
            flow_generator=flow_generator,
            optimizer_step_callback=(controller.on_optimizer_step if controller else None),
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
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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
