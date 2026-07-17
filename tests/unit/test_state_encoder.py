from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.spatial_resampler import AnchoredSpatialResampler
from corrective_foresight.model.state_encoder import OnlineStateEncoder, StateAdapter
from tests.unit.fakes import DeterministicPatchBackbone


def make_encoder(hidden_size: int = 768, proprio_dimension: int = 5) -> OnlineStateEncoder:
    backbone = DeterministicPatchBackbone(hidden_size)
    adapter = StateAdapter(
        spatial_resampler=AnchoredSpatialResampler(
            hidden_size=hidden_size,
            max_cameras=2,
            num_attention_heads=12 if hidden_size == 768 else 3,
        ),
        proprio_dimension=proprio_dimension,
        hidden_size=hidden_size,
    )
    return OnlineStateEncoder(backbone=backbone, adapter=adapter)


class StateEncoderTest(unittest.TestCase):
    def test_produces_nine_tokens_with_adapter_gradients_only(self) -> None:
        torch.manual_seed(11)
        encoder = make_encoder()
        rgb = torch.randn(1, 2, 2, 3, 8, 16, requires_grad=True)
        camera_mask = torch.ones(1, 2, 2, dtype=torch.bool)
        proprio = torch.randn(1, 2, 5)
        proprio_mask = torch.ones_like(proprio, dtype=torch.bool)

        states = encoder(rgb, camera_mask, proprio, proprio_mask)
        states.square().mean().backward()

        self.assertEqual(states.shape, (1, 2, 9, 768))
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                for parameter in encoder.adapter.parameters()
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in encoder.backbone.parameters())
        )
        self.assertIsNone(rgb.grad)

    def test_invalid_rgb_and_proprio_values_are_mask_invariant(self) -> None:
        torch.manual_seed(13)
        encoder = make_encoder(hidden_size=12, proprio_dimension=3)
        encoder.eval()
        rgb = torch.randn(1, 2, 2, 3, 8, 16)
        camera_mask = torch.tensor([[[True, False], [True, False]]])
        proprio = torch.randn(1, 2, 3)
        proprio_mask = torch.zeros_like(proprio, dtype=torch.bool)

        baseline = encoder(rgb, camera_mask, proprio, proprio_mask)
        changed_rgb = rgb.clone()
        changed_rgb[:, :, 1] = torch.nan
        changed_proprio = torch.full_like(proprio, torch.nan)
        result = encoder(
            changed_rgb,
            camera_mask,
            changed_proprio,
            proprio_mask,
        )

        torch.testing.assert_close(result, baseline, rtol=0.0, atol=0.0)

    def test_train_mode_never_changes_backbone_to_training(self) -> None:
        encoder = make_encoder(hidden_size=12, proprio_dimension=3)

        encoder.train()

        self.assertTrue(encoder.training)
        self.assertTrue(encoder.adapter.training)
        self.assertFalse(encoder.backbone.training)


if __name__ == "__main__":
    unittest.main()
