from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from types import MappingProxyType

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from corrective_foresight.data.batch import TrajectoryBatch
from corrective_foresight.model.objectives import action_cycle_weight
from corrective_foresight.policy.unified_policy import (
    UnifiedCorrectiveForesightPolicy,
)
from corrective_foresight.training.distributed import (
    DistributedContext,
    wrap_ddp,
)
from corrective_foresight.training.gradient_conflict import (
    GlobalObjectiveGradients,
    ObjectiveGradientAccumulator,
    project_pcgrad,
)
from corrective_foresight.training.numerics import (
    NumericalGuardError,
    clip_and_validate_gradients,
    per_loss_gradient_norms,
    require_finite_objective,
)
from corrective_foresight.training.parameter_groups import (
    build_optimizer_parameter_groups,
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
    stage: str = "unified"
    protected_lr_multiplier: float = 1.0
    gradient_mode: str = "ordinary"
    gradient_diagnostic_interval: int = 10
    pcgrad_seed: int = 0

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
        TrainingStage.parse(self.stage)
        if (
            not math.isfinite(self.protected_lr_multiplier)
            or not 0.0 <= self.protected_lr_multiplier <= 1.0
        ):
            raise ValueError(
                "protected_lr_multiplier must be finite within [0, 1]"
            )
        if self.gradient_mode not in {"ordinary", "audit", "pcgrad"}:
            raise ValueError(
                "gradient_mode must be ordinary, audit, or pcgrad"
            )
        if self.stage == "world_pretrain" and (
            self.gradient_mode != "ordinary"
            or self.protected_lr_multiplier != 1.0
        ):
            raise ValueError(
                "world_pretrain requires ordinary gradient_mode and "
                "protected_lr_multiplier 1.0"
            )
        if (
            type(self.gradient_diagnostic_interval) is not int
            or self.gradient_diagnostic_interval <= 0
        ):
            raise ValueError(
                "gradient_diagnostic_interval must be a positive integer"
            )
        if type(self.pcgrad_seed) is not int or self.pcgrad_seed < 0:
            raise ValueError("pcgrad_seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    output: TrainingOutput
    optimizer_stepped: bool
    pre_clip_gradient_norm: Tensor | None
    per_loss_gradient_norms: Mapping[str, Tensor]
    learning_rate: float
    learning_rates: Mapping[str, float]
    gradient_metrics: Mapping[str, float]


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
        parameter_groups = build_optimizer_parameter_groups(policy)
        self.parameter_groups = parameter_groups
        self.shared_parameters = tuple(
            parameter
            for parameter in policy.world_action_model.transformer.parameters()
            if parameter.requires_grad
        )
        self._gradient_accumulator: ObjectiveGradientAccumulator | None = None
        self.optimizer = torch.optim.AdamW(
            (
                {
                    "params": parameter_groups.protected,
                    "lr": config.learning_rate
                    * config.protected_lr_multiplier,
                    "name": "protected",
                },
                {
                    "params": parameter_groups.action,
                    "lr": config.learning_rate,
                    "name": "action",
                },
            ),
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
        capture_objectives = stage is TrainingStage.UNIFIED and (
            self.config.gradient_mode == "pcgrad"
            or (
                self.config.gradient_mode == "audit"
                and (global_step + 1)
                % self.config.gradient_diagnostic_interval
                == 0
            )
        )
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
            if capture_objectives:
                if self._gradient_accumulator is None:
                    self._gradient_accumulator = ObjectiveGradientAccumulator(
                        self.shared_parameters,
                        tuple(output.losses),
                    )
                self._gradient_accumulator.add(
                    output.losses,
                    accumulation_steps=self.config.accumulation_steps,
                )
                gradient_norms: dict[str, Tensor] = {}
            else:
                if self._gradient_accumulator is not None:
                    raise RuntimeError(
                        "objective-gradient capture changed within optimizer step"
                    )
                gradient_norms = per_loss_gradient_norms(
                    output.losses,
                    self.shared_parameters,
                    dataset_id=batch.dataset_id,
                    rank=self.rank,
                    global_step=global_step,
                )
            self.scaler.scale(
                loss / self.config.accumulation_steps
            ).backward()
        self._micro_steps = next_micro_step

        if not optimizer_boundary:
            learning_rates = self._learning_rates()
            return TrainStepResult(
                output=output,
                optimizer_stepped=False,
                pre_clip_gradient_norm=None,
                per_loss_gradient_norms=MappingProxyType(gradient_norms),
                learning_rate=learning_rates["action"],
                learning_rates=learning_rates,
                gradient_metrics=MappingProxyType({}),
            )

        try:
            gradient_metrics: dict[str, float] = {}
            if capture_objectives:
                if self._gradient_accumulator is None:
                    raise RuntimeError("objective-gradient accumulator is missing")
                global_gradients = self._gradient_accumulator.finalize(
                    self.distributed_context,
                    self._effective_objective_weights(
                        stage,
                        global_step,
                        tuple(output.losses),
                    ),
                )
                self._gradient_accumulator = None
                gradient_metrics.update(
                    self._diagnostic_metrics(global_gradients)
                )
                diagnostics = global_gradients.diagnostics()
                gradient_norms = {
                    name: loss.new_tensor(diagnostics.raw_norms[name])
                    for name in global_gradients.names
                }
            self.scaler.unscale_(self.optimizer)
            if capture_objectives and self.config.gradient_mode == "pcgrad":
                projected = project_pcgrad(
                    global_gradients,
                    seed=self.config.pcgrad_seed,
                    global_step=global_step,
                )
                for parameter, value in zip(
                    self.shared_parameters,
                    projected.gradient,
                    strict=True,
                ):
                    parameter.grad = value.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                gradient_metrics.update(
                    {
                        "pcgrad/enabled": 1.0,
                        "pcgrad/removed_fraction": projected.removed_fraction,
                        "pcgrad/projected_norm": projected.projected_norm,
                    }
                )
            elif capture_objectives:
                gradient_metrics["pcgrad/enabled"] = 0.0
            pre_clip_norm = clip_and_validate_gradients(
                tuple(self.train_module.named_parameters()),
                max_norm=self.config.max_grad_norm,
                dataset_id=batch.dataset_id,
                rank=self.rank,
                global_step=global_step,
            )
            if capture_objectives:
                gradient_metrics["gradient_clip/coefficient"] = min(
                    1.0,
                    self.config.max_grad_norm
                    / max(float(pre_clip_norm.detach().item()), 1e-12),
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            if self.config.protected_lr_multiplier > 0.0:
                self.policy.update_ema(global_step)
        except Exception:
            if self._gradient_accumulator is not None:
                self._gradient_accumulator.reset()
                self._gradient_accumulator = None
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.optimizer.zero_grad(set_to_none=True)
        learning_rates = self._learning_rates()
        return TrainStepResult(
            output=output,
            optimizer_stepped=True,
            pre_clip_gradient_norm=pre_clip_norm,
            per_loss_gradient_norms=MappingProxyType(gradient_norms),
            learning_rate=learning_rates["action"],
            learning_rates=learning_rates,
            gradient_metrics=MappingProxyType(gradient_metrics),
        )

    def _learning_rates(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                str(group["name"]): float(group["lr"])
                for group in self.optimizer.param_groups
            }
        )

    @staticmethod
    def _effective_objective_weights(
        stage: TrainingStage,
        global_step: int,
        names: tuple[str, ...],
    ) -> Mapping[str, float]:
        if stage is TrainingStage.WORLD_PRETRAIN:
            weights = {"dynamics_loss": 1.0}
        else:
            weights = {
                "dynamics_loss": 1.0,
                "inverse_action_loss": 1.0,
                "action_cycle_loss": action_cycle_weight(global_step),
                "policy_flow_loss": 1.0,
            }
        if set(weights) != set(names):
            raise ValueError("objective names do not match the stage contract")
        return MappingProxyType(weights)

    @staticmethod
    def _diagnostic_metrics(
        gradients: GlobalObjectiveGradients,
    ) -> dict[str, float]:
        diagnostics = gradients.diagnostics()
        metrics = {
            **{
                f"gradient_norm/raw/{name}": value
                for name, value in diagnostics.raw_norms.items()
            },
            **{
                f"gradient_norm/effective/{name}": value
                for name, value in diagnostics.effective_norms.items()
            },
            **{
                f"gradient_cosine/{pair}": value
                for pair, value in diagnostics.cosines.items()
            },
            "gradient_cosine/minimum_dynamics": (
                diagnostics.minimum_dynamics_cosine
            ),
            "gradient_norm/ordinary_effective": (
                diagnostics.ordinary_effective_norm
            ),
        }
        return metrics

    def _learning_rate_multiplier(self, step: int) -> float:
        if self.config.warmup_steps and step < self.config.warmup_steps:
            return (step + 1) / self.config.warmup_steps
        progress = (
            step - self.config.warmup_steps
        ) / (self.config.total_steps - self.config.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def state_dict(self) -> dict[str, object]:
        if self._micro_steps % self.config.accumulation_steps != 0:
            raise RuntimeError("trainer checkpoints require an optimizer boundary")
        if self._gradient_accumulator is not None:
            raise RuntimeError("trainer has pending objective gradients")
        return {
            "version": 2,
            "config": asdict(self.config),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "micro_steps": self._micro_steps,
            "last_ema_step": self.policy.last_ema_step,
            "gradient_accumulator_pending": False,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {
            "version",
            "config",
            "optimizer",
            "scheduler",
            "scaler",
            "micro_steps",
            "last_ema_step",
            "gradient_accumulator_pending",
        }
        if set(state) != required:
            raise ValueError("trainer state fields do not match checkpoint contract")
        if state["version"] != 2 or state["config"] != asdict(self.config):
            raise ValueError("trainer state configuration mismatch")
        if state["gradient_accumulator_pending"] is not False:
            raise ValueError("checkpoint has pending objective gradients")
        micro_steps = state["micro_steps"]
        last_ema_step = state["last_ema_step"]
        if type(micro_steps) is not int or micro_steps < 0:
            raise ValueError("trainer micro_steps must be nonnegative")
        if micro_steps % self.config.accumulation_steps != 0:
            raise ValueError(
                "trainer micro_steps must be at an optimizer boundary"
            )
        if type(last_ema_step) is not int or last_ema_step < -1:
            raise ValueError("trainer last_ema_step is invalid")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self._micro_steps = micro_steps
        self._gradient_accumulator = None
        self.policy.restore_ema_step(last_ema_step)
