from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from corrective_foresight.model.dinov3_backbone import PatchGrid


class DeterministicPatchBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.register_parameter(
            "channel_projection",
            nn.Parameter(
                torch.linspace(0.25, 1.25, 3 * hidden_size).reshape(3, hidden_size),
                requires_grad=False,
            ),
        )
        self.eval()

    def train(self, mode: bool = True) -> DeterministicPatchBackbone:
        super().train(False)
        return self

    def forward(self, rgb: Tensor) -> PatchGrid:
        pooled = F.adaptive_avg_pool2d(rgb, (4, 8)).permute(0, 2, 3, 1)
        tokens = pooled @ self.channel_projection
        return PatchGrid(
            tokens=tokens,
            image_size=tuple(rgb.shape[-2:]),
            patch_size=(rgb.shape[-2] // 4, rgb.shape[-1] // 8),
        )
