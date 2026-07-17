from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.ema import EMAStateTarget
from corrective_foresight.model.objectives import CorrectiveForesightObjective
from corrective_foresight.model.spatial_resampler import AnchoredSpatialResampler
from corrective_foresight.model.state_encoder import OnlineStateEncoder, StateAdapter
from tests.unit.fakes import DeterministicPatchBackbone
from tests.unit.objective_fakes import (
    ACTION_SPEC_ID,
    make_objective_inputs,
    make_objective_model,
)


def has_gradient(loss: torch.Tensor, parameter: torch.Tensor, *, retain: bool) -> bool:
    gradient = torch.autograd.grad(
        loss,
        parameter,
        retain_graph=retain,
        allow_unused=True,
    )[0]
    return gradient is not None and gradient.abs().sum().item() > 0.0


class GradientRoutingTest(unittest.TestCase):
    def test_each_objective_reaches_its_declared_model_paths(self) -> None:
        model = make_objective_model()
        result = CorrectiveForesightObjective(model)(
            make_objective_inputs(),
            flow_generator=torch.Generator().manual_seed(109),
        )
        adapter = model.action_adapters.resolve(ACTION_SPEC_ID)
        shared = model.transformer.layers[0].attention.in_proj_weight

        self.assertTrue(
            has_gradient(
                result.losses["dynamics_loss"],
                model.delta_projection.weight,
                retain=True,
            )
        )
        self.assertTrue(
            has_gradient(
                result.losses["dynamics_loss"],
                adapter.action_input_projection.weight,
                retain=True,
            )
        )
        self.assertTrue(
            has_gradient(
                result.losses["inverse_action_loss"],
                adapter.inverse_mean_head.weight,
                retain=True,
            )
        )
        self.assertTrue(
            has_gradient(
                result.losses["inverse_action_loss"],
                shared,
                retain=True,
            )
        )
        self.assertTrue(
            has_gradient(
                result.losses["action_cycle_loss"],
                model.delta_projection.weight,
                retain=True,
            )
        )
        self.assertTrue(
            has_gradient(
                result.losses["action_cycle_loss"],
                adapter.inverse_mean_head.weight,
                retain=True,
            )
        )
        self.assertTrue(
            has_gradient(
                result.losses["policy_flow_loss"],
                adapter.flow_state_projection.weight,
                retain=True,
            )
        )
        self.assertTrue(
            has_gradient(
                result.losses["policy_flow_loss"],
                adapter.flow_velocity_head.weight,
                retain=False,
            )
        )

    def test_total_reaches_online_adapter_but_not_backbone_or_ema_target(self) -> None:
        backbone = DeterministicPatchBackbone(12)
        online = OnlineStateEncoder(
            backbone,
            StateAdapter(
                AnchoredSpatialResampler(
                    hidden_size=12,
                    max_cameras=1,
                    num_attention_heads=3,
                ),
                proprio_dimension=3,
                hidden_size=12,
            ),
        )
        ema = EMAStateTarget(online, tau=0.99)
        rgb = torch.randn(1, 9, 1, 3, 8, 16)
        camera_mask = torch.ones(1, 9, 1, dtype=torch.bool)
        proprio = torch.randn(1, 9, 3)
        proprio_mask = torch.ones_like(proprio, dtype=torch.bool)
        online_states = online(rgb, camera_mask, proprio, proprio_mask)
        target_states = ema.encode_target(rgb, camera_mask, proprio, proprio_mask)
        inputs = make_objective_inputs(
            online_states=online_states,
            target_states=target_states,
        )

        result = CorrectiveForesightObjective(make_objective_model())(
            inputs,
            flow_generator=torch.Generator().manual_seed(113),
        )
        result.total_loss.backward()

        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
                for parameter in online.adapter.parameters()
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))
        self.assertFalse(target_states.requires_grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in ema.adapter.parameters())
        )


if __name__ == "__main__":
    unittest.main()
