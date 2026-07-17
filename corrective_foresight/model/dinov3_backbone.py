from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import DINOv3ViTImageProcessorFast, DINOv3ViTModel

from corrective_foresight.model.dinov3_artifact import (
    DinoV3ArtifactError,
    verify_snapshot,
)


DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_MODEL_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
_PRODUCTION_CONFIG = {
    "architectures": ["DINOv3ViTModel"],
    "model_type": "dinov3_vit",
    "hidden_size": 768,
    "patch_size": 16,
    "num_register_tokens": 4,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
}


@dataclass(frozen=True, slots=True)
class PatchGrid:
    tokens: Tensor
    image_size: tuple[int, int]
    patch_size: tuple[int, int]

    def __post_init__(self) -> None:
        if self.tokens.ndim != 4 or min(self.tokens.shape) <= 0:
            raise ValueError("patch tokens must have shape [N,Gh,Gw,D] with positive dimensions")
        if not self.tokens.is_floating_point():
            raise ValueError("patch tokens must be floating point")
        if not torch.isfinite(self.tokens).all().item():
            raise ValueError("patch tokens must be finite")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain two positive dimensions")
        if len(self.patch_size) != 2 or min(self.patch_size) <= 0:
            raise ValueError("patch_size must contain two positive dimensions")
        expected_grid = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        if self.image_size[0] % self.patch_size[0] or self.image_size[1] % self.patch_size[1]:
            raise ValueError("processed image dimensions must be divisible by patch_size")
        if tuple(self.tokens.shape[1:3]) != expected_grid:
            raise ValueError(
                "patch grid shape disagrees with image_size and patch_size: "
                f"expected {expected_grid}, got {tuple(self.tokens.shape[1:3])}"
            )


class DinoV3PatchBackbone(nn.Module):
    def __init__(self, model: nn.Module, processor: Any) -> None:
        super().__init__()
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("DINOv3 model config must declare a positive hidden_size")
        self.model = model
        self.processor = processor
        self.hidden_size = hidden_size
        self.model.requires_grad_(False)
        self.train(False)

    @classmethod
    def from_pretrained_exact(
        cls,
        snapshot_dir: Path,
        device: torch.device | str = "cpu",
    ) -> DinoV3PatchBackbone:
        snapshot_dir = snapshot_dir.resolve(strict=True)
        verify_snapshot(snapshot_dir)
        processor = DINOv3ViTImageProcessorFast.from_pretrained(
            str(snapshot_dir),
            local_files_only=True,
        )
        model = DINOv3ViTModel.from_pretrained(
            str(snapshot_dir),
            local_files_only=True,
        )
        _validate_production_config(model.config)
        return cls(model=model.to(device), processor=processor)

    def train(self, mode: bool = True) -> DinoV3PatchBackbone:
        super().train(False)
        self.model.eval()
        return self

    def forward(self, rgb: Tensor) -> PatchGrid:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.shape[0] <= 0:
            raise ValueError("DINOv3 RGB input must have shape [N,3,H,W]")
        if not rgb.is_floating_point() or not torch.isfinite(rgb).all().item():
            raise ValueError("DINOv3 RGB input must be finite floating point")
        processed = self.processor(
            images=rgb,
            return_tensors="pt",
            do_rescale=False,
        )
        if "pixel_values" not in processed:
            raise ValueError("DINOv3 processor did not return pixel_values")
        pixel_values = processed["pixel_values"]
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("DINOv3 processor returned invalid pixel_values")
        model_device = next(self.model.parameters(), rgb).device
        pixel_values = pixel_values.to(model_device)

        with torch.no_grad():
            output = self.model(pixel_values=pixel_values)
        hidden = output.last_hidden_state
        patch_size = _dimension_pair(getattr(self.model.config, "patch_size", None))
        image_size = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
        if image_size[0] % patch_size[0] or image_size[1] % patch_size[1]:
            raise ValueError("processed DINOv3 image dimensions are not patch aligned")
        grid_height = image_size[0] // patch_size[0]
        grid_width = image_size[1] // patch_size[1]
        patch_count = grid_height * grid_width
        register_count = getattr(self.model.config, "num_register_tokens", None)
        if not isinstance(register_count, int) or register_count < 0:
            raise ValueError("DINOv3 config must declare num_register_tokens")
        prefix_count = 1 + register_count
        if hidden.ndim != 3 or hidden.shape[0] != rgb.shape[0]:
            raise ValueError("DINOv3 returned an invalid hidden-state shape")
        if hidden.shape[1] != prefix_count + patch_count:
            raise ValueError(
                "DINOv3 patch-token count disagrees with processed image grid: "
                f"expected {patch_count}, got {hidden.shape[1] - prefix_count}"
            )
        if hidden.shape[2] != self.hidden_size:
            raise ValueError("DINOv3 hidden dimension disagrees with model config")
        patches = hidden[:, prefix_count:].reshape(
            hidden.shape[0], grid_height, grid_width, self.hidden_size
        )
        return PatchGrid(
            tokens=patches.detach(),
            image_size=image_size,
            patch_size=patch_size,
        )


def _dimension_pair(value: object) -> tuple[int, int]:
    if isinstance(value, int) and value > 0:
        return value, value
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(item, int) and item > 0 for item in value)
    ):
        return int(value[0]), int(value[1])
    raise ValueError("DINOv3 config must declare a positive patch_size")


def _validate_production_config(config: object) -> None:
    for name, expected in _PRODUCTION_CONFIG.items():
        actual = getattr(config, name, None)
        if actual != expected:
            raise DinoV3ArtifactError(
                f"DINOv3 config {name} must be {expected!r}, got {actual!r}"
            )
