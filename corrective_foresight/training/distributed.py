from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    def __post_init__(self) -> None:
        if self.rank < 0 or self.world_size <= 0 or self.rank >= self.world_size:
            raise ValueError("invalid distributed rank/world_size")
        if self.local_rank < 0:
            raise ValueError("local_rank must be nonnegative")

    @classmethod
    def single_process(cls, device: torch.device | str) -> DistributedContext:
        return cls(rank=0, world_size=1, local_rank=0, device=torch.device(device))

    @classmethod
    def from_initialized_process_group(
        cls,
        *,
        local_rank: int,
        device: torch.device | str,
    ) -> DistributedContext:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed process group is not initialized")
        return cls(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=local_rank,
            device=torch.device(device),
        )


def wrap_ddp(
    module: nn.Module,
    context: DistributedContext,
    *,
    enabled: bool,
) -> nn.Module:
    if not enabled:
        return module
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("DDP requested without an initialized process group")
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    return DistributedDataParallel(
        module,
        device_ids=device_ids,
        output_device=context.local_rank if device_ids is not None else None,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
