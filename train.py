from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

from corrective_foresight.data.batch import TrajectoryBatch
from corrective_foresight.training.stages import TrainingStage
from corrective_foresight.training.trainer import Trainer


class BatchSource(Protocol):
    def next_batch(self) -> TrajectoryBatch: ...


MetricLogger = Callable[[dict[str, float | int | str]], None]


def run_training(
    *,
    batch_source: BatchSource,
    trainer: Trainer,
    stage: TrainingStage | str,
    optimizer_steps: int,
    start_global_step: int,
    generator_seed: int,
    metric_logger: MetricLogger,
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
    device = trainer.distributed_context.device
    generator = torch.Generator(device=device).manual_seed(generator_seed)
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
        metric_logger(record)
        global_step += 1
    return global_step
