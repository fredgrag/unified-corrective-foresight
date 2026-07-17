from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import unittest

import torch
import yaml

from corrective_foresight.model.world_action_transformer import (
    WorldActionConfig,
    WorldActionTransformer,
)
from corrective_foresight.model.token_types import TokenRole
from tests.unit.test_action_spec import make_action_spec


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_config() -> WorldActionConfig:
    return WorldActionConfig(
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
        cycle_noise_std=0.0,
        cycle_dropout=0.0,
    )


def make_model() -> WorldActionTransformer:
    torch.manual_seed(31)
    return WorldActionTransformer(test_config(), (make_action_spec(),))


class WorldActionTransformerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()
        self.batch_size = 2
        self.condition = torch.randn(2, 6, 12)
        self.spec_ids = ("test.ee_delta.v1", "test.ee_delta.v1")

    def test_predict_delta_keeps_nine_tokens_and_recurses_state(self) -> None:
        initial_state = torch.randn(2, 9, 12)
        actions = torch.randn(2, 4, 2)
        delta_time = torch.full((2, 4), 0.1)

        prediction = self.model.predict_delta(
            self.condition,
            initial_state,
            actions,
            delta_time,
            self.spec_ids,
        )

        self.assertEqual(prediction.delta.shape, (2, 4, 9, 12))
        self.assertEqual(prediction.states.shape, (2, 5, 9, 12))
        torch.testing.assert_close(
            prediction.states[:, 1:],
            prediction.states[:, :-1] + prediction.delta,
        )

    def test_inverse_outputs_bounded_diagonal_gaussian(self) -> None:
        states = torch.randn(2, 3, 9, 12)
        target_delta = torch.randn(2, 3, 9, 12)

        prediction = self.model.predict_inverse(
            self.condition, states, target_delta, self.spec_ids
        )

        self.assertEqual(prediction.mean.shape, (2, 3, 2))
        self.assertEqual(prediction.log_variance.shape, (2, 3, 2))
        self.assertGreaterEqual(prediction.log_variance.min().item(), -10.0)
        self.assertLessEqual(prediction.log_variance.max().item(), 2.0)

    def test_cycle_keeps_predicted_delta_differentiable(self) -> None:
        states = torch.randn(2, 3, 9, 12)
        predicted_delta = torch.randn(2, 3, 9, 12, requires_grad=True)

        prediction = self.model.predict_cycle(
            self.condition, states, predicted_delta, self.spec_ids
        )
        prediction.mean.square().mean().backward()

        self.assertEqual(prediction.mean.shape, (2, 3, 2))
        self.assertIsNotNone(predicted_delta.grad)
        self.assertGreater(predicted_delta.grad.abs().sum().item(), 0.0)

    def test_policy_velocity_has_eight_grounded_actions(self) -> None:
        observed_states = torch.randn(2, 3, 9, 12)
        noisy_actions = torch.randn(2, 8, 2)
        flow_time = torch.rand(2, 1, 1)

        prediction = self.model.predict_policy_velocity(
            self.condition,
            observed_states,
            noisy_actions,
            flow_time,
            self.spec_ids,
        )

        self.assertEqual(prediction.velocity.shape, (2, 8, 2))

    def test_policy_observation_mask_becomes_state_key_padding(self) -> None:
        observed_states = torch.randn(2, 2, 9, 12)
        observed_valid = torch.tensor([[False, True], [True, True]])
        captured = []
        handle = self.model.transformer.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0])
        )
        try:
            self.model.predict_policy_velocity(
                self.condition,
                observed_states,
                torch.randn(2, 8, 2),
                torch.rand(2, 1, 1),
                self.spec_ids,
                observed_state_valid_mask=observed_valid,
            )
        finally:
            handle.remove()

        view = captured[0]
        state_indices = view.indices(TokenRole.STATE)
        self.assertTrue(view.key_padding_mask[0, state_indices[:9]].all().item())
        self.assertFalse(view.key_padding_mask[0, state_indices[9:]].any().item())

    def test_delta_time_fourier_embedding_changes_dynamics(self) -> None:
        self.model.eval()
        initial_state = torch.randn(2, 9, 12)
        actions = torch.randn(2, 1, 2)
        first = self.model.predict_delta(
            self.condition,
            initial_state,
            actions,
            torch.full((2, 1), 0.05),
            self.spec_ids,
        )
        second = self.model.predict_delta(
            self.condition,
            initial_state,
            actions,
            torch.full((2, 1), 0.2),
            self.spec_ids,
        )

        self.assertGreater((first.delta - second.delta).abs().max().item(), 1e-6)

    def test_dynamics_start_time_changes_discrete_timestep_embedding(self) -> None:
        self.model.eval()
        initial_state = torch.randn(2, 9, 12)
        actions = torch.randn(2, 1, 2)
        delta_time = torch.full((2, 1), 0.1)

        first = self.model.predict_delta(
            self.condition,
            initial_state,
            actions,
            delta_time,
            self.spec_ids,
            start_time=0,
        )
        second = self.model.predict_delta(
            self.condition,
            initial_state,
            actions,
            delta_time,
            self.spec_ids,
            start_time=3,
        )

        self.assertGreater((first.delta - second.delta).abs().max().item(), 1e-6)

    def test_inverse_and_cycle_share_one_delta_layernorm(self) -> None:
        calls: list[torch.Tensor] = []
        handle = self.model.delta_norm.register_forward_hook(
            lambda module, inputs, output: calls.append(output)
        )
        states = torch.randn(2, 1, 9, 12)
        delta = torch.randn(2, 1, 9, 12)
        try:
            self.model.predict_inverse(self.condition, states, delta, self.spec_ids)
            self.model.predict_cycle(self.condition, states, delta, self.spec_ids)
        finally:
            handle.remove()

        self.assertEqual(len(calls), 2)

    def test_inverse_preserves_transition_timestep_embeddings_when_flattened(self) -> None:
        condition = self.condition[:1]
        state = torch.randn(1, 1, 9, 12).expand(-1, 2, -1, -1).clone()
        delta = torch.randn(1, 1, 9, 12).expand(-1, 2, -1, -1).clone()
        captured = []
        handle = self.model.transformer.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0])
        )
        try:
            self.model.predict_inverse(
                condition,
                state,
                delta,
                (self.spec_ids[0],),
            )
        finally:
            handle.remove()

        self.assertEqual(len(captured), 1)
        view = captured[0]
        condition_indices = view.indices(TokenRole.CONDITION)
        state_indices = view.indices(TokenRole.STATE)
        torch.testing.assert_close(
            view.tokens[0, condition_indices], view.tokens[1, condition_indices]
        )
        self.assertGreater(
            (
                view.tokens[0, state_indices]
                - view.tokens[1, state_indices]
            ).abs().max().item(),
            1e-6,
        )

    def test_invalid_temporal_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one transition"):
            self.model.predict_delta(
                self.condition,
                torch.randn(2, 9, 12),
                torch.empty(2, 0, 2),
                torch.empty(2, 0),
                self.spec_ids,
            )
        with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
            self.model.predict_policy_velocity(
                self.condition,
                torch.randn(2, 2, 9, 12),
                torch.randn(2, 8, 2),
                torch.full((2,), 1.1),
                self.spec_ids,
            )

    def test_production_yaml_is_frozen_and_rejects_one_layer(self) -> None:
        path = PROJECT_ROOT / "configs/model/unified_base.yaml"
        config = WorldActionConfig.from_mapping(yaml.safe_load(path.read_text()))

        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.mlp_ratio, 4)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.state_tokens, 9)
        self.assertEqual(config.action_horizon, 8)
        self.assertTrue(config.gradient_checkpointing)
        self.assertEqual(config.precision, "bf16")
        with self.assertRaisesRegex(ValueError, "at least two layers"):
            WorldActionConfig.from_mapping(
                {**yaml.safe_load(path.read_text()), "num_layers": 1}
            )
        with self.assertRaisesRegex(ValueError, "model dimensions"):
            WorldActionConfig.from_mapping(
                {**yaml.safe_load(path.read_text()), "num_attention_heads": 0}
            )

    def test_gradient_checkpointing_flag_reaches_shared_transformer(self) -> None:
        config = replace(test_config(), gradient_checkpointing=True)
        model = WorldActionTransformer(config, (make_action_spec(),))

        self.assertTrue(model.transformer.gradient_checkpointing)

    def test_gradient_checkpointing_supports_dynamics_backward(self) -> None:
        config = replace(test_config(), gradient_checkpointing=True)
        model = WorldActionTransformer(config, (make_action_spec(),))
        prediction = model.predict_delta(
            torch.randn(1, 3, 12),
            torch.randn(1, 9, 12),
            torch.randn(1, 2, 2),
            torch.full((1, 2), 0.1),
            ("test.ee_delta.v1",),
        )

        prediction.delta.square().mean().backward()

        self.assertIsNotNone(model.delta_projection.weight.grad)
        self.assertGreater(model.delta_projection.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
