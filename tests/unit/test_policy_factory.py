from __future__ import annotations

import unittest

import torch

from corrective_foresight.policy.factory import assemble_unified_policy
from corrective_foresight.model.world_action_transformer import WorldActionConfig
from tests.unit.fakes import DeterministicPatchBackbone
from tests.unit.policy_fakes import make_vocabulary
from tests.unit.test_action_spec import make_action_spec


class PolicyFactoryTest(unittest.TestCase):
    def test_assembles_one_shared_policy_with_injected_frozen_backbone(self) -> None:
        backbone = DeterministicPatchBackbone(hidden_size=12)
        config = WorldActionConfig(
            hidden_size=12,
            num_layers=2,
            num_attention_heads=3,
            mlp_ratio=2,
            dropout=0.0,
            state_tokens=9,
            action_horizon=8,
            max_time_steps=32,
            time_fourier_bands=4,
            gradient_checkpointing=False,
            precision="float32",
            cycle_noise_std=0.01,
            cycle_dropout=0.05,
        )

        policy = assemble_unified_policy(
            backbone=backbone,
            action_specs=(make_action_spec(),),
            vocabulary=make_vocabulary(),
            world_action_config=config,
            language_cache=None,
            language_embedding_dim=4,
            proprio_dimension=3,
            max_cameras=2,
            ema_tau=0.99,
            device="cpu",
        )

        self.assertIs(policy.online_state_encoder.backbone, backbone)
        self.assertIs(policy.ema_state_target.online, policy.online_state_encoder)
        self.assertIs(policy.objective.model, policy.world_action_model)
        self.assertEqual(policy.world_action_model.config, config)
        self.assertEqual(
            set(policy.world_action_model.action_adapters.specs),
            {"test.ee_delta.v1"},
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))
        self.assertEqual(next(policy.parameters()).device, torch.device("cpu"))

    def test_rejects_backbone_config_hidden_dimension_mismatch(self) -> None:
        config = WorldActionConfig(
            hidden_size=12,
            num_layers=2,
            num_attention_heads=3,
            mlp_ratio=2,
            dropout=0.0,
            state_tokens=9,
            action_horizon=8,
            max_time_steps=32,
            time_fourier_bands=4,
            gradient_checkpointing=False,
            precision="float32",
            cycle_noise_std=0.01,
            cycle_dropout=0.05,
        )
        with self.assertRaisesRegex(ValueError, "backbone hidden"):
            assemble_unified_policy(
                backbone=DeterministicPatchBackbone(hidden_size=8),
                action_specs=(make_action_spec(),),
                vocabulary=make_vocabulary(),
                world_action_config=config,
                language_cache=None,
                language_embedding_dim=4,
                proprio_dimension=3,
                max_cameras=1,
                ema_tau=0.99,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
