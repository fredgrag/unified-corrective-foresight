from __future__ import annotations

import torch
from torch import Tensor, nn

from corrective_foresight.model.spatial_resampler import AnchoredSpatialResampler


STATE_TOKEN_COUNT = 9


class StateAdapter(nn.Module):
    def __init__(
        self,
        spatial_resampler: AnchoredSpatialResampler,
        proprio_dimension: int,
        hidden_size: int = 768,
    ) -> None:
        super().__init__()
        if proprio_dimension <= 0 or hidden_size <= 0:
            raise ValueError("state adapter dimensions must be positive")
        if spatial_resampler.hidden_size != hidden_size:
            raise ValueError("spatial resampler and state adapter hidden sizes must match")
        self.spatial_resampler = spatial_resampler
        self.proprio_dimension = proprio_dimension
        self.hidden_size = hidden_size
        self.proprio_projection = nn.Linear(proprio_dimension, hidden_size)
        self.null_proprio = nn.Parameter(torch.empty(hidden_size))
        self.output_norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.null_proprio, std=0.02)

    def forward(
        self,
        patch_grids: Tensor,
        camera_mask: Tensor,
        proprio: Tensor,
        proprio_mask: Tensor,
    ) -> Tensor:
        batch_size, time_steps = patch_grids.shape[:2]
        if proprio.shape != (batch_size, time_steps, self.proprio_dimension):
            raise ValueError(
                "proprio must have shape "
                f"[B,T,{self.proprio_dimension}], got {tuple(proprio.shape)}"
            )
        if not proprio.is_floating_point():
            raise ValueError("proprio must be floating point")
        if proprio_mask.dtype is not torch.bool or proprio_mask.shape != proprio.shape:
            raise ValueError("proprio_mask must be bool and match proprio")
        if not torch.isfinite(proprio[proprio_mask]).all().item():
            raise ValueError("valid proprio values must be finite")
        safe_proprio = torch.where(proprio_mask, proprio, 0.0)
        projected_proprio = self.proprio_projection(safe_proprio)
        has_proprio = proprio_mask.any(dim=-1)
        proprio_token = torch.where(
            has_proprio[..., None],
            projected_proprio,
            self.null_proprio[None, None, :],
        )
        spatial_tokens = self.spatial_resampler(patch_grids, camera_mask)
        states = torch.cat((spatial_tokens, proprio_token[:, :, None, :]), dim=2)
        if states.shape[2] != STATE_TOKEN_COUNT:
            raise RuntimeError("state adapter did not produce exactly nine tokens")
        return self.output_norm(states)


class OnlineStateEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, adapter: StateAdapter) -> None:
        super().__init__()
        backbone_hidden_size = getattr(backbone, "hidden_size", None)
        if backbone_hidden_size != adapter.hidden_size:
            raise ValueError("backbone and state adapter hidden sizes must match")
        self.backbone = backbone
        self.adapter = adapter
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> OnlineStateEncoder:
        super().train(mode)
        self.backbone.train(False)
        return self

    def forward(
        self,
        rgb: Tensor,
        camera_mask: Tensor,
        proprio: Tensor,
        proprio_mask: Tensor,
    ) -> Tensor:
        return _encode_state(
            backbone=self.backbone,
            adapter=self.adapter,
            rgb=rgb,
            camera_mask=camera_mask,
            proprio=proprio,
            proprio_mask=proprio_mask,
        )


def _encode_state(
    *,
    backbone: nn.Module,
    adapter: StateAdapter,
    rgb: Tensor,
    camera_mask: Tensor,
    proprio: Tensor,
    proprio_mask: Tensor,
) -> Tensor:
    if rgb.ndim != 6 or rgb.shape[3] != 3:
        raise ValueError("rgb must have shape [B,T,V,3,H,W]")
    if not rgb.is_floating_point():
        raise ValueError("rgb must be floating point")
    batch_size, time_steps, views = rgb.shape[:3]
    if min((batch_size, time_steps, views)) <= 0:
        raise ValueError("rgb batch, time, and view dimensions must be positive")
    if camera_mask.dtype is not torch.bool or camera_mask.shape != (
        batch_size,
        time_steps,
        views,
    ):
        raise ValueError("camera_mask must be bool with shape [B,T,V]")
    valid_rgb = camera_mask[..., None, None, None].expand_as(rgb)
    if not torch.isfinite(rgb[valid_rgb]).all().item():
        raise ValueError("valid RGB values must be finite")
    safe_rgb = torch.where(valid_rgb, rgb, 0.0)
    flattened_rgb = safe_rgb.reshape(-1, *safe_rgb.shape[3:])
    with torch.no_grad():
        patch_grid = backbone(flattened_rgb)
    tokens = patch_grid.tokens
    if tokens.shape[0] != batch_size * time_steps * views:
        raise ValueError("backbone patch batch does not match RGB batch")
    patch_grids = tokens.reshape(
        batch_size,
        time_steps,
        views,
        *tokens.shape[1:],
    )
    return adapter(patch_grids, camera_mask, proprio, proprio_mask)
