from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.dinov3_artifact import (
    DinoV3ArtifactError,
    default_snapshot_path,
)
from corrective_foresight.model.dinov3_backbone import DinoV3PatchBackbone
from corrective_foresight.model.spatial_resampler import AnchoredSpatialResampler
from corrective_foresight.model.state_encoder import OnlineStateEncoder, StateAdapter


class RealDinoV3IntegrationTest(unittest.TestCase):
    def test_exact_revision_cuda_state_encoder_and_backward(self) -> None:
        if not torch.cuda.is_available():
            self.fail("real DINOv3 gate requires CUDA and cannot be skipped")
        device = torch.device("cuda:0")
        try:
            backbone = DinoV3PatchBackbone.from_pretrained_exact(
                default_snapshot_path(), device=device
            )
        except (DinoV3ArtifactError, OSError) as error:
            self.fail(
                "verified ModelScope DINOv3 snapshot is unavailable; run "
                "`python scripts/download_dinov3.py --artifact-root "
                f"artifacts/models`: {error}"
            )
        adapter = StateAdapter(
            spatial_resampler=AnchoredSpatialResampler(
                hidden_size=768,
                max_cameras=1,
                num_attention_heads=12,
            ),
            proprio_dimension=7,
            hidden_size=768,
        ).to(device)
        encoder = OnlineStateEncoder(backbone=backbone, adapter=adapter).to(device)
        first = torch.linspace(0.0, 1.0, 3 * 224 * 224, device=device).reshape(
            3, 224, 224
        )
        second = first.flip(-1)
        rgb = torch.stack((first, second))[None, :, None]
        camera_mask = torch.ones(1, 2, 1, dtype=torch.bool, device=device)
        proprio = torch.randn(1, 2, 7, device=device)
        proprio_mask = torch.ones_like(proprio, dtype=torch.bool)

        patch_grid = backbone(rgb.flatten(0, 2))
        states = encoder(rgb, camera_mask, proprio, proprio_mask)
        states.square().mean().backward()

        self.assertEqual(states.shape, (1, 2, 9, 768))
        self.assertGreater(patch_grid.tokens.var(dim=(1, 2)).mean().item(), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in backbone.parameters())
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in adapter.parameters())
        )


if __name__ == "__main__":
    unittest.main()
