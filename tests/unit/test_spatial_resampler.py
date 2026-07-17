from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.spatial_resampler import (
    AnchoredSpatialResampler,
    adaptive_anchor_pool,
)


class SpatialResamplerTest(unittest.TestCase):
    def test_adaptive_pool_has_stable_row_major_two_by_four_order(self) -> None:
        expected = torch.arange(8.0).reshape(2, 4)
        grid = expected.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

        anchors = adaptive_anchor_pool(grid[None, :, :, None])

        self.assertEqual(anchors.shape, (1, 8, 1))
        torch.testing.assert_close(anchors[0, :, 0], torch.arange(8.0))

    def test_masked_camera_cannot_affect_base_or_cross_attention(self) -> None:
        torch.manual_seed(7)
        resampler = AnchoredSpatialResampler(
            hidden_size=12,
            max_cameras=2,
            num_attention_heads=3,
        )
        with torch.no_grad():
            torch.nn.init.normal_(resampler.residual_projection.weight)
            torch.nn.init.normal_(resampler.residual_projection.bias)
        patches = torch.randn(1, 2, 2, 4, 8, 12)
        camera_mask = torch.tensor([[[True, False], [True, False]]])

        baseline = resampler(patches, camera_mask)
        changed = patches.clone()
        changed[:, :, 1] = torch.randn_like(changed[:, :, 1]) * 1_000_000
        result = resampler(changed, camera_mask)

        torch.testing.assert_close(result, baseline, rtol=0.0, atol=0.0)

    def test_rejects_any_state_with_no_valid_camera(self) -> None:
        resampler = AnchoredSpatialResampler(
            hidden_size=8,
            max_cameras=2,
            num_attention_heads=2,
        )
        patches = torch.zeros(1, 2, 2, 4, 8, 8)
        camera_mask = torch.tensor([[[True, False], [False, False]]])

        with self.assertRaisesRegex(ValueError, "at least one valid camera"):
            resampler(patches, camera_mask)


if __name__ == "__main__":
    unittest.main()
