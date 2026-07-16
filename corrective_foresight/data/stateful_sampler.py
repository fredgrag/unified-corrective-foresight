from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch


class StatefulDistributedBatchSampler:
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be within world_size")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.epoch = 0
        self.cursor = 0
        self._cached_epoch: int | None = None
        self._cached_indices: list[int] = []
        if self._usable_local_size() < batch_size:
            raise ValueError(
                "dataset cannot provide one full local batch for every distributed rank"
            )

    def __len__(self) -> int:
        return self._usable_local_size() // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self.epoch
        while self.epoch == epoch and self.cursor < self._usable_local_size():
            yield self.next_batch()

    def next_batch(self) -> list[int]:
        usable_size = self._usable_local_size()
        if self.cursor + self.batch_size > usable_size:
            self.epoch += 1
            self.cursor = 0
            self._cached_epoch = None
        indices = self._local_indices()
        start = self.cursor
        end = start + self.batch_size
        batch = indices[start:end]
        if len(batch) != self.batch_size:
            raise RuntimeError("stateful sampler failed to construct a full local batch")
        self.cursor = end
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "version",
            "dataset_size",
            "batch_size",
            "seed",
            "rank",
            "world_size",
            "drop_last",
            "epoch",
            "cursor",
        }
        if set(state) != required:
            raise ValueError("sampler state fields do not match the required contract")
        contract = (
            state["version"],
            state["dataset_size"],
            state["batch_size"],
            state["seed"],
            state["rank"],
            state["world_size"],
            state["drop_last"],
        )
        expected = (
            1,
            self.dataset_size,
            self.batch_size,
            self.seed,
            self.rank,
            self.world_size,
            self.drop_last,
        )
        if contract != expected:
            raise ValueError(
                f"sampler contract mismatch: expected {expected}, got {contract}"
            )
        epoch = state["epoch"]
        cursor = state["cursor"]
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be a nonnegative integer")
        if (
            not isinstance(cursor, int)
            or cursor < 0
            or cursor > self._usable_local_size()
            or cursor % self.batch_size != 0
        ):
            raise ValueError("sampler cursor is invalid for the batch contract")
        self.epoch = epoch
        self.cursor = cursor
        self._cached_epoch = None
        self._cached_indices = []

    def _usable_local_size(self) -> int:
        if self.drop_last:
            total_size = (self.dataset_size // self.world_size) * self.world_size
        else:
            total_size = (
                (self.dataset_size + self.world_size - 1) // self.world_size
            ) * self.world_size
        local_size = total_size // self.world_size
        return (local_size // self.batch_size) * self.batch_size

    def _local_indices(self) -> list[int]:
        if self._cached_epoch == self.epoch:
            return self._cached_indices
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
        if self.drop_last:
            total_size = (self.dataset_size // self.world_size) * self.world_size
            permutation = permutation[:total_size]
        else:
            total_size = (
                (self.dataset_size + self.world_size - 1) // self.world_size
            ) * self.world_size
            padding = total_size - len(permutation)
            if padding:
                repeats = (padding + len(permutation) - 1) // len(permutation)
                permutation.extend((permutation * repeats)[:padding])
        local = permutation[self.rank : len(permutation) : self.world_size]
        local = local[: self._usable_local_size()]
        self._cached_epoch = self.epoch
        self._cached_indices = local
        return local
