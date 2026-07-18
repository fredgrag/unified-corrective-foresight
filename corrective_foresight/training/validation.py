from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import math
import random
from types import MappingProxyType

import numpy as np
import torch
import torch.distributed as dist

from corrective_foresight.data.batch import TrajectoryBatch
from corrective_foresight.policy.unified_policy import UnifiedCorrectiveForesightPolicy
from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.stages import TrainingStage


_REQUIRED_METRICS = {
    TrainingStage.WORLD_PRETRAIN: frozenset({"dynamics_loss", "visual_loss"}),
    TrainingStage.UNIFIED: frozenset(
        {
            "dynamics_loss",
            "visual_loss",
            "policy_flow_loss",
            "action_loss",
            "inverse_action_loss",
            "self_correction_cycle_loss",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    batches_per_rank: int
    generator_seed: int

    def __post_init__(self) -> None:
        if type(self.batches_per_rank) is not int or self.batches_per_rank <= 0:
            raise ValueError("validation batches_per_rank must be a positive integer")
        if type(self.generator_seed) is not int or self.generator_seed < 0:
            raise ValueError("validation generator_seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    global_step: int
    stage: str
    world_size: int
    batches_per_rank: int
    samples: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if type(self.global_step) is not int or self.global_step < 0:
            raise ValueError("validation global_step must be nonnegative")
        TrainingStage.parse(self.stage)
        if type(self.world_size) is not int or self.world_size <= 0:
            raise ValueError("validation world_size must be positive")
        if type(self.batches_per_rank) is not int or self.batches_per_rank <= 0:
            raise ValueError("validation batches_per_rank must be positive")
        if type(self.samples) is not int or self.samples <= 0:
            raise ValueError("validation samples must be positive")
        metrics = dict(self.metrics)
        if not metrics or tuple(metrics) != tuple(sorted(metrics)):
            raise ValueError("validation metrics must be a nonempty sorted mapping")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, float)
            or not math.isfinite(value)
            for name, value in metrics.items()
        ):
            raise ValueError("validation metrics must contain finite floats")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))


class ValidationRunner:
    def __init__(
        self,
        *,
        policy: UnifiedCorrectiveForesightPolicy,
        validation_mixer,
        context: DistributedContext,
        config: ValidationConfig,
    ) -> None:
        if not isinstance(policy, UnifiedCorrectiveForesightPolicy):
            raise ValueError("ValidationRunner requires UnifiedCorrectiveForesightPolicy")
        if not isinstance(context, DistributedContext):
            raise ValueError("ValidationRunner requires DistributedContext")
        if not isinstance(config, ValidationConfig):
            raise ValueError("ValidationRunner requires ValidationConfig")
        if not callable(getattr(validation_mixer, "next_batch", None)):
            raise ValueError("validation_mixer must provide next_batch()")
        if not callable(getattr(validation_mixer, "state_dict", None)):
            raise ValueError("validation_mixer must provide state_dict()")
        if not callable(getattr(validation_mixer, "load_state_dict", None)):
            raise ValueError("validation_mixer must provide load_state_dict()")
        if context.world_size > 1 and (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() != context.rank
            or dist.get_world_size() != context.world_size
        ):
            raise ValueError("validation context does not match initialized process group")
        self.policy = policy
        self.validation_mixer = validation_mixer
        self.context = context
        self.config = config

    def evaluate(
        self,
        stage: TrainingStage | str,
        global_step: int,
    ) -> ValidationRecord:
        stage = TrainingStage.parse(stage)
        if type(global_step) is not int or global_step < 0:
            raise ValueError("validation global_step must be nonnegative")
        mixer_state = copy.deepcopy(self.validation_mixer.state_dict())
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        )
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        was_training = self.policy.training
        generator = torch.Generator(device=self.context.device).manual_seed(
            self.config.generator_seed
        )
        try:
            self.policy.eval()
            sums: dict[str, float] = {}
            local_samples = 0
            metric_names: tuple[str, ...] | None = None
            with torch.no_grad():
                for _ in range(self.config.batches_per_rank):
                    batch = self.validation_mixer.next_batch()
                    if not isinstance(batch, TrajectoryBatch):
                        raise ValueError("validation mixer must return TrajectoryBatch")
                    batch = batch.to(self.context.device)
                    output = self.policy.compute_training_objective(
                        batch,
                        stage,
                        global_step,
                        generator,
                    )
                    required = _REQUIRED_METRICS[stage]
                    if not required.issubset(output.metrics):
                        missing = sorted(required - set(output.metrics))
                        raise ValueError(
                            f"validation output is missing required metrics: {missing}"
                        )
                    names = tuple(sorted(output.metrics))
                    if metric_names is None:
                        metric_names = names
                    elif names != metric_names:
                        raise ValueError("validation metric names changed across batches")
                    samples = int(batch.rgb.shape[0])
                    if samples <= 0:
                        raise ValueError("validation batch must contain samples")
                    local_samples += samples
                    for name, value in output.metrics.items():
                        if (
                            not isinstance(value, torch.Tensor)
                            or value.ndim != 0
                            or not value.is_floating_point()
                            or not torch.isfinite(value).item()
                        ):
                            raise ValueError(f"validation metric {name} is not finite")
                        sums[name] = sums.get(name, 0.0) + float(
                            value.detach().to(dtype=torch.float64).item()
                        ) * samples
            if metric_names is None or local_samples <= 0:
                raise ValueError("validation produced no batches")
            self._validate_metric_names(metric_names)
            total_samples, total_sums = self._reduce(sums, metric_names, local_samples)
            metrics = {
                name: float(total_sums[index] / total_samples)
                for index, name in enumerate(metric_names)
            }
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError("distributed validation produced non-finite metrics")
            return ValidationRecord(
                global_step=global_step,
                stage=stage.value,
                world_size=self.context.world_size,
                batches_per_rank=self.config.batches_per_rank,
                samples=total_samples,
                metrics=metrics,
            )
        finally:
            self.validation_mixer.load_state_dict(mixer_state)
            torch.set_rng_state(cpu_rng)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_rng)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            self.policy.train(was_training)

    def _reduce(
        self,
        sums: Mapping[str, float],
        names: tuple[str, ...],
        local_samples: int,
    ) -> tuple[int, list[float]]:
        if self.context.world_size == 1:
            return local_samples, [sums[name] for name in names]
        device = self.context.device
        payload = torch.tensor(
            [float(local_samples), *(sums[name] for name in names)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        total_samples = int(payload[0].item())
        if total_samples <= 0:
            raise ValueError("distributed validation produced no samples")
        return total_samples, [float(value.item()) for value in payload[1:]]

    def _validate_metric_names(self, names: tuple[str, ...]) -> None:
        if self.context.world_size == 1:
            return
        gathered: list[tuple[str, ...] | None] = [None] * self.context.world_size
        dist.all_gather_object(gathered, names)
        if any(item != names for item in gathered):
            raise ValueError("distributed validation metric names differ across ranks")
