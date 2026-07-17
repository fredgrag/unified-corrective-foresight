from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import math
from types import MappingProxyType

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from corrective_foresight.data.batch import TrajectoryBatch
from corrective_foresight.policy.unified_policy import (
    UnifiedCorrectiveForesightPolicy,
)
from corrective_foresight.training.distributed import (
    DistributedContext,
    wrap_ddp,
)
from corrective_foresight.training.numerics import (
    NumericalGuardError,
    clip_and_validate_gradients,
    per_loss_gradient_norms,
    require_finite_objective,
)
from corrective_foresight.training.stages import TrainingOutput, TrainingStage


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    learning_rate: float
    weight_decay: float
    accumulation_steps: int
    max_grad_norm: float
    warmup_steps: int
    total_steps: int
    bf16: bool
    ddp: bool

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("optimizer learning_rate/weight_decay are invalid")
        if type(self.accumulation_steps) is not int or self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be a positive integer")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if (
            type(self.warmup_steps) is not int
            or type(self.total_steps) is not int
            or self.warmup_steps < 0
            or self.total_steps <= self.warmup_steps
        ):
            raise ValueError("scheduler steps require 0 <= warmup < total")
        if type(self.bf16) is not bool or type(self.ddp) is not bool:
            raise ValueError("bf16 and ddp flags must be bool")


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    output: TrainingOutput
    optimizer_stepped: bool
    pre_clip_gradient_norm: Tensor | None
    per_loss_gradient_norms: Mapping[str, Tensor]
    learning_rate: float


class Trainer:
    def __init__(
        self,
        policy: UnifiedCorrectiveForesightPolicy,
        config: TrainerConfig,
        *,
        rank: int,
        distributed_context: DistributedContext | None = None,
    ) -> None:
        if not isinstance(policy, UnifiedCorrectiveForesightPolicy):
            raise ValueError("Trainer requires UnifiedCorrectiveForesightPolicy")
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        self.policy = policy
        self.config = config
        self.rank = rank
        parameter = next(
            (item for item in policy.parameters() if item.requires_grad),
            None,
        )
        if parameter is None:
            raise ValueError("policy has no trainable parameters")
        context = distributed_context or DistributedContext.single_process(
            parameter.device
        )
        if context.rank != rank:
            raise ValueError("Trainer rank and distributed context disagree")
        self.distributed_context = context
        self.train_module = wrap_ddp(policy, context, enabled=config.ddp)
        trainable = tuple(
            item for item in self.train_module.parameters() if item.requires_grad
        )
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            device=context.device.type,
            enabled=False,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._learning_rate_multiplier,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._micro_steps = 0

    def train_step(
        self,
        batch: TrajectoryBatch,
        stage: TrainingStage | str,
        global_step: int,
        generator: torch.Generator,
    ) -> TrainStepResult:
        stage = TrainingStage.parse(stage)
        self.train_module.train()
        next_micro_step = self._micro_steps + 1
        optimizer_boundary = next_micro_step % self.config.accumulation_steps == 0
        synchronization = (
            self.train_module.no_sync()
            if isinstance(self.train_module, DistributedDataParallel)
            and not optimizer_boundary
            else nullcontext()
        )
        device_type = self.distributed_context.device.type
        autocast_enabled = self.config.bf16 and device_type == "cuda"
        with synchronization:
            try:
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    output = self.train_module(
                        batch,
                        stage,
                        global_step,
                        generator,
                    )
            except ValueError as error:
                if "finite" not in str(error):
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                raise NumericalGuardError(
                    f"{error}; dataset={batch.dataset_id}; rank={self.rank}; "
                    f"step={global_step}"
                ) from error
            loss = output.loss.float()
            require_finite_objective(
                loss,
                objective_name="total_loss",
                dataset_id=batch.dataset_id,
                rank=self.rank,
                global_step=global_step,
            )
            shared_parameters = tuple(
                self.policy.world_action_model.transformer.parameters()
            )
            gradient_norms = per_loss_gradient_norms(
                output.losses,
                shared_parameters,
                dataset_id=batch.dataset_id,
                rank=self.rank,
                global_step=global_step,
            )
            self.scaler.scale(
                loss / self.config.accumulation_steps
            ).backward()
        self._micro_steps = next_micro_step

        if not optimizer_boundary:
            return TrainStepResult(
                output=output,
                optimizer_stepped=False,
                pre_clip_gradient_norm=None,
                per_loss_gradient_norms=MappingProxyType(gradient_norms),
                learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            )

        try:
            self.scaler.unscale_(self.optimizer)
            pre_clip_norm = clip_and_validate_gradients(
                tuple(self.train_module.named_parameters()),
                max_norm=self.config.max_grad_norm,
                dataset_id=batch.dataset_id,
                rank=self.rank,
                global_step=global_step,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.policy.update_ema(global_step)
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.optimizer.zero_grad(set_to_none=True)
        return TrainStepResult(
            output=output,
            optimizer_stepped=True,
            pre_clip_gradient_norm=pre_clip_norm,
            per_loss_gradient_norms=MappingProxyType(gradient_norms),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )

    def _learning_rate_multiplier(self, step: int) -> float:
        if self.config.warmup_steps and step < self.config.warmup_steps:
            return (step + 1) / self.config.warmup_steps
        progress = (
            step - self.config.warmup_steps
        ) / (self.config.total_steps - self.config.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
