from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.ema import EMAStateTarget
from tests.unit.test_state_encoder import make_encoder


class EMAStateTargetTest(unittest.TestCase):
    def test_shares_only_backbone_and_initializes_all_adapter_parameters(self) -> None:
        online = make_encoder(hidden_size=12, proprio_dimension=3)
        ema = EMAStateTarget(online=online, tau=0.8)

        self.assertIs(ema.backbone, online.backbone)
        self.assertIsNot(ema.adapter, online.adapter)
        online_parameters = dict(online.adapter.named_parameters())
        target_parameters = dict(ema.adapter.named_parameters())
        self.assertEqual(online_parameters.keys(), target_parameters.keys())
        self.assertIn("spatial_resampler.camera_embedding.weight", target_parameters)
        self.assertIn("null_proprio", target_parameters)
        for name, target in target_parameters.items():
            torch.testing.assert_close(target, online_parameters[name])
            self.assertFalse(target.requires_grad)

    def test_update_uses_declared_tau_for_every_adapter_parameter(self) -> None:
        online = make_encoder(hidden_size=12, proprio_dimension=3)
        ema = EMAStateTarget(online=online, tau=0.75)
        old_target = {
            name: parameter.detach().clone()
            for name, parameter in ema.adapter.named_parameters()
        }
        with torch.no_grad():
            for index, parameter in enumerate(online.adapter.parameters(), start=1):
                parameter.add_(index / 10.0)
        new_online = dict(online.adapter.named_parameters())

        ema.update()

        for name, target in ema.adapter.named_parameters():
            expected = old_target[name] * 0.75 + new_online[name].detach() * 0.25
            torch.testing.assert_close(target, expected)

    def test_public_target_and_temporal_delta_are_stop_gradient(self) -> None:
        online = make_encoder(hidden_size=12, proprio_dimension=3)
        ema = EMAStateTarget(online=online, tau=0.9)
        rgb = torch.randn(1, 3, 2, 3, 8, 16, requires_grad=True)
        camera_mask = torch.ones(1, 3, 2, dtype=torch.bool)
        proprio = torch.randn(1, 3, 3, requires_grad=True)
        proprio_mask = torch.ones_like(proprio, dtype=torch.bool)

        states = ema.encode_target(rgb, camera_mask, proprio, proprio_mask)
        delta = EMAStateTarget.delta_target(states)

        self.assertEqual(states.shape, (1, 3, 9, 12))
        self.assertFalse(states.requires_grad)
        self.assertFalse(delta.requires_grad)
        torch.testing.assert_close(delta, states[:, 1:] - states[:, :-1])
        self.assertIsNone(rgb.grad)
        self.assertIsNone(proprio.grad)


if __name__ == "__main__":
    unittest.main()
