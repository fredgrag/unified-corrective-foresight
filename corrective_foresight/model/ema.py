from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn

from corrective_foresight.model.state_encoder import (
    OnlineStateEncoder,
    StateAdapter,
    _encode_state,
)


class EMAStateTarget(nn.Module):
    def __init__(self, online: OnlineStateEncoder, tau: float) -> None:
        super().__init__()
        if not 0.0 <= tau < 1.0:
            raise ValueError("EMA tau must satisfy 0 <= tau < 1")
        object.__setattr__(self, "_online", online)
        object.__setattr__(self, "_backbone", online.backbone)
        self.tau = float(tau)
        self.adapter: StateAdapter = deepcopy(online.adapter)
        self.adapter.requires_grad_(False)
        self.train(False)

    @property
    def online(self) -> OnlineStateEncoder:
        return object.__getattribute__(self, "_online")

    @property
    def backbone(self) -> nn.Module:
        return object.__getattribute__(self, "_backbone")

    def train(self, mode: bool = True) -> EMAStateTarget:
        super().train(False)
        self.backbone.train(False)
        self.adapter.eval()
        return self

    @torch.no_grad()
    def update(self) -> None:
        online_parameters = dict(self.online.adapter.named_parameters())
        target_parameters = dict(self.adapter.named_parameters())
        if online_parameters.keys() != target_parameters.keys():
            raise RuntimeError("online and EMA adapter parameter contracts differ")
        for name, target in target_parameters.items():
            online = online_parameters[name].detach()
            target.mul_(self.tau).add_(online, alpha=1.0 - self.tau)

        online_buffers = dict(self.online.adapter.named_buffers())
        target_buffers = dict(self.adapter.named_buffers())
        if online_buffers.keys() != target_buffers.keys():
            raise RuntimeError("online and EMA adapter buffer contracts differ")
        for name, target in target_buffers.items():
            target.copy_(online_buffers[name])

    @torch.no_grad()
    def encode_target(
        self,
        rgb: Tensor,
        camera_mask: Tensor,
        proprio: Tensor,
        proprio_mask: Tensor,
    ) -> Tensor:
        targets = _encode_state(
            backbone=self.backbone,
            adapter=self.adapter,
            rgb=rgb,
            camera_mask=camera_mask,
            proprio=proprio,
            proprio_mask=proprio_mask,
        )
        return targets.detach()

    @staticmethod
    def delta_target(target_states: Tensor) -> Tensor:
        if target_states.ndim != 4 or target_states.shape[1] < 2:
            raise ValueError("target states must have shape [B,T>=2,9,D]")
        if target_states.shape[2] != 9:
            raise ValueError("target states must contain exactly nine state tokens")
        if not torch.isfinite(target_states).all().item():
            raise ValueError("target states must be finite")
        return (target_states[:, 1:] - target_states[:, :-1]).detach()

    def encode_delta_target(
        self,
        rgb: Tensor,
        camera_mask: Tensor,
        proprio: Tensor,
        proprio_mask: Tensor,
    ) -> Tensor:
        return self.delta_target(
            self.encode_target(rgb, camera_mask, proprio, proprio_mask)
        )
