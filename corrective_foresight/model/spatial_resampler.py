from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ANCHOR_GRID = (2, 4)
ANCHOR_COUNT = ANCHOR_GRID[0] * ANCHOR_GRID[1]


def adaptive_anchor_pool(patch_grid: Tensor) -> Tensor:
    if patch_grid.ndim != 4 or min(patch_grid.shape) <= 0:
        raise ValueError("patch_grid must have shape [N,Gh,Gw,D]")
    if not patch_grid.is_floating_point() or not torch.isfinite(patch_grid).all().item():
        raise ValueError("patch_grid must contain finite floating-point values")
    channels_first = patch_grid.permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(channels_first, ANCHOR_GRID)
    return pooled.flatten(2).transpose(1, 2)


class AnchoredSpatialResampler(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        max_cameras: int = 8,
        num_attention_heads: int = 12,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or max_cameras <= 0 or num_attention_heads <= 0:
            raise ValueError("resampler dimensions must be positive")
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size = hidden_size
        self.max_cameras = max_cameras
        self.camera_embedding = nn.Embedding(max_cameras, hidden_size)
        self.anchor_queries = nn.Parameter(torch.empty(ANCHOR_COUNT, hidden_size))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        self.residual_projection = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.anchor_queries, std=0.02)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(self, patch_grids: Tensor, camera_mask: Tensor) -> Tensor:
        if patch_grids.ndim != 6:
            raise ValueError("patch_grids must have shape [B,T,V,Gh,Gw,D]")
        batch_size, time_steps, views, grid_height, grid_width, hidden_size = (
            patch_grids.shape
        )
        if min((batch_size, time_steps, views, grid_height, grid_width)) <= 0:
            raise ValueError("patch grid dimensions must be positive")
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"patch hidden dimension must be {self.hidden_size}, got {hidden_size}"
            )
        if camera_mask.dtype is not torch.bool or camera_mask.shape != (
            batch_size,
            time_steps,
            views,
        ):
            raise ValueError("camera_mask must be bool with shape [B,T,V]")
        if views > self.max_cameras:
            raise ValueError(
                f"received {views} cameras but max_cameras is {self.max_cameras}"
            )
        if (~camera_mask.any(dim=-1)).any().item():
            raise ValueError("every state requires at least one valid camera")
        valid_values = camera_mask[..., None, None, None].expand_as(patch_grids)
        if not torch.isfinite(patch_grids[valid_values]).all().item():
            raise ValueError("valid camera patches must be finite")

        safe_patches = torch.where(valid_values, patch_grids, 0.0)
        flattened_states = batch_size * time_steps
        flattened_cameras = safe_patches.reshape(
            flattened_states * views,
            grid_height,
            grid_width,
            hidden_size,
        )
        per_camera_anchors = adaptive_anchor_pool(flattened_cameras).reshape(
            flattened_states, views, ANCHOR_COUNT, hidden_size
        )
        camera_ids = torch.arange(views, device=patch_grids.device)
        camera_tokens = self.camera_embedding(camera_ids)[None, :, None, :]
        per_camera_anchors = per_camera_anchors + camera_tokens
        flat_camera_mask = camera_mask.reshape(flattened_states, views)
        anchor_weights = flat_camera_mask[:, :, None, None].to(patch_grids.dtype)
        deterministic_base = (per_camera_anchors * anchor_weights).sum(dim=1)
        deterministic_base = deterministic_base / anchor_weights.sum(dim=1)

        patch_count = grid_height * grid_width
        per_camera_patches = safe_patches.reshape(
            flattened_states, views, patch_count, hidden_size
        )
        per_camera_patches = per_camera_patches + camera_tokens
        per_camera_patches = torch.where(
            flat_camera_mask[:, :, None, None], per_camera_patches, 0.0
        )
        keys = per_camera_patches.reshape(
            flattened_states, views * patch_count, hidden_size
        )
        key_padding_mask = ~flat_camera_mask[:, :, None].expand(
            flattened_states, views, patch_count
        ).reshape(flattened_states, views * patch_count)
        queries = self.anchor_queries[None].expand(flattened_states, -1, -1)
        attended, _ = self.cross_attention(
            query=queries,
            key=keys,
            value=keys,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        anchors = deterministic_base + self.residual_projection(attended)
        return anchors.reshape(
            batch_size, time_steps, ANCHOR_COUNT, hidden_size
        )
