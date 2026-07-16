from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from corrective_foresight.data.batch import TrajectoryBatch
from corrective_foresight.data.collate import collate_trajectory_samples
from corrective_foresight.data.lerobot_adapter import TrajectorySample
from corrective_foresight.data.stateful_sampler import (
    StatefulDistributedBatchSampler,
)


class TrajectoryDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> TrajectorySample: ...


def sample_weighted_dataset_id(
    dataset_ids: Sequence[str], weights: Tensor, generator: torch.Generator
) -> str:
    if not dataset_ids:
        raise ValueError("dataset_ids cannot be empty")
    if weights.ndim != 1 or weights.numel() != len(dataset_ids):
        raise ValueError("weights must be one-dimensional and match dataset_ids")
    if not weights.is_floating_point():
        raise ValueError("weights must be floating point")
    if not torch.isfinite(weights).all().item() or (weights <= 0).any().item():
        raise ValueError("all dataset weights must be finite and positive")
    index = int(torch.multinomial(weights, 1, generator=generator).item())
    return dataset_ids[index]


class BalancedLeRobotMixer:
    def __init__(
        self,
        datasets: Mapping[str, TrajectoryDataset],
        samplers: Mapping[str, StatefulDistributedBatchSampler],
        weights: Mapping[str, float],
        generator: torch.Generator | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("datasets cannot be empty")
        dataset_keys = set(datasets)
        if set(samplers) != dataset_keys or set(weights) != dataset_keys:
            raise ValueError("datasets, samplers, and weights must have identical keys")
        if any(
            not math.isfinite(float(weight)) or float(weight) <= 0
            for weight in weights.values()
        ):
            raise ValueError("all dataset weights must be finite and positive")
        for name, dataset in datasets.items():
            if samplers[name].dataset_size != len(dataset):
                raise ValueError(f"sampler dataset_size mismatch for {name}")

        self.dataset_ids = tuple(sorted(dataset_keys))
        self.datasets = dict(datasets)
        self.samplers = dict(samplers)
        self.weights = torch.tensor(
            [float(weights[name]) for name in self.dataset_ids], dtype=torch.float64
        )
        self.generator = generator or torch.Generator().manual_seed(0)
        self.step = 0
        self.last_dataset_id: str | None = None

    def next_batch(self) -> TrajectoryBatch:
        dataset_id = self._synchronized_dataset_id()
        sampler = self.samplers[dataset_id]
        indices = sampler.next_batch()
        samples = [self.datasets[dataset_id][index] for index in indices]
        batch = collate_trajectory_samples(samples)
        if batch.rgb.shape[0] != sampler.batch_size:
            raise RuntimeError(
                f"mixer produced batch size {batch.rgb.shape[0]}, expected "
                f"{sampler.batch_size}"
            )
        self._validate_distributed_contract(batch)
        self.last_dataset_id = dataset_id
        self.step += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "step": self.step,
            "dataset_ids": list(self.dataset_ids),
            "weights": self.weights.tolist(),
            "generator_state": self.generator.get_state().clone(),
            "samplers": {
                name: self.samplers[name].state_dict() for name in self.dataset_ids
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "version",
            "step",
            "dataset_ids",
            "weights",
            "generator_state",
            "samplers",
        }
        if set(state) != required:
            raise ValueError("mixer state fields do not match the required contract")
        if state["version"] != 1:
            raise ValueError(f"unsupported mixer state version: {state['version']}")
        if tuple(state["dataset_ids"]) != self.dataset_ids:
            raise ValueError("mixer dataset ID contract mismatch")
        state_weights = torch.tensor(state["weights"], dtype=torch.float64)
        if not torch.equal(state_weights, self.weights):
            raise ValueError("mixer weight contract mismatch")
        step = state["step"]
        if not isinstance(step, int) or step < 0:
            raise ValueError("mixer step must be a nonnegative integer")
        sampler_states = state["samplers"]
        if set(sampler_states) != set(self.dataset_ids):
            raise ValueError("mixer sampler state keys do not match datasets")

        generator_state = state["generator_state"]
        if not isinstance(generator_state, Tensor) or generator_state.dtype != torch.uint8:
            raise ValueError("mixer generator_state must be a uint8 tensor")
        self.generator.set_state(generator_state.cpu())
        for name in self.dataset_ids:
            self.samplers[name].load_state_dict(sampler_states[name])
        self.step = step
        self.last_dataset_id = None

    def _synchronized_dataset_id(self) -> str:
        if not dist.is_available() or not dist.is_initialized():
            return sample_weighted_dataset_id(
                self.dataset_ids, self.weights, self.generator
            )
        backend = str(dist.get_backend()).lower()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if "nccl" in backend
            else torch.device("cpu")
        )
        selected_index = torch.full((1,), -1, dtype=torch.long, device=device)
        if dist.get_rank() == 0:
            dataset_id = sample_weighted_dataset_id(
                self.dataset_ids, self.weights, self.generator
            )
            selected_index[0] = self.dataset_ids.index(dataset_id)
        dist.broadcast(selected_index, src=0)
        index = int(selected_index.item())
        if index < 0 or index >= len(self.dataset_ids):
            raise RuntimeError(f"rank zero broadcast invalid dataset index: {index}")
        return self.dataset_ids[index]

    def _validate_distributed_contract(self, batch: TrajectoryBatch) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        local_contract = (
            batch.dataset_id,
            batch.action_spec_id,
            int(batch.rgb.shape[0]),
            self.step,
        )
        contracts: list[tuple[str, str, int, int] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(contracts, local_contract)
        if any(contract != local_contract for contract in contracts):
            raise RuntimeError(
                f"distributed dataset contract mismatch at step {self.step}, "
                f"rank {dist.get_rank()}: {contracts}"
            )
