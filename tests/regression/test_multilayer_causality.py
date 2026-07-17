from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.token_types import (
    TokenMetadata,
    TokenRole,
    TokenView,
)
from corrective_foresight.model.token_views import (
    build_forward_view,
    build_inverse_view,
    build_policy_view,
)
from corrective_foresight.model.transformer import CausalTokenTransformer


def metadata() -> tuple[TokenMetadata, ...]:
    return (
        TokenMetadata(TokenRole.CONDITION, -1, 0, True, True),
        TokenMetadata(TokenRole.CONDITION, -1, 0, True, True),
        TokenMetadata(TokenRole.STATE, 0, 1, True, False),
        TokenMetadata(TokenRole.STATE, 0, 1, True, False),
        TokenMetadata(TokenRole.DELTA_QUERY, 1, 2, True, False),
        TokenMetadata(TokenRole.STATE, 1, 3, True, False),
    )


class MultilayerCausalityTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.model = CausalTokenTransformer(
            hidden_size=8,
            num_layers=3,
            num_attention_heads=2,
            mlp_ratio=2,
            dropout=0.0,
        ).eval()
        self.base = torch.randn(1, 6, 8)
        self.valid = torch.zeros(1, 6, dtype=torch.bool)

    def encode(
        self, values: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        view = TokenView(
            tokens=values,
            metadata=metadata(),
            key_padding_mask=self.valid if key_padding_mask is None else key_padding_mask,
        )
        return self.model(view)

    def test_future_state_cannot_reach_query_directly_or_through_conditions(self) -> None:
        baseline = self.encode(self.base)
        changed = self.base.clone()
        changed[:, 5] += 1000.0
        result = self.encode(changed)

        torch.testing.assert_close(result[:, 4], baseline[:, 4], rtol=0.0, atol=1e-6)
        torch.testing.assert_close(result[:, :2], baseline[:, :2], rtol=0.0, atol=1e-6)

    def test_allowed_past_state_perturbation_changes_query(self) -> None:
        baseline = self.encode(self.base)
        changed = self.base.clone()
        changed[:, 2, 0] += 5.0
        result = self.encode(changed)

        self.assertGreater((result[:, 4] - baseline[:, 4]).abs().max().item(), 1e-5)

    def test_same_frame_state_tokens_fuse_bidirectionally(self) -> None:
        baseline = self.encode(self.base)
        changed = self.base.clone()
        changed[:, 3, 0] += 5.0
        result = self.encode(changed)

        self.assertGreater((result[:, 2] - baseline[:, 2]).abs().max().item(), 1e-5)

    def test_invalid_key_cannot_relay_even_with_nonfinite_content(self) -> None:
        key_padding = self.valid.clone()
        key_padding[:, 2] = True
        baseline_values = self.base.clone()
        baseline_values[:, 2] = 0.0
        baseline = self.encode(baseline_values, key_padding)
        changed = self.base.clone()
        changed[:, 2] = torch.nan
        result = self.encode(changed, key_padding)

        torch.testing.assert_close(result[:, 4], baseline[:, 4], rtol=0.0, atol=1e-6)
        self.assertTrue(torch.isfinite(result).all().item())

    def test_forward_target_delta_perturbation_has_no_view_entry(self) -> None:
        condition = torch.randn(1, 2, 8)
        state = torch.randn(1, 3, 8)
        action = torch.randn(1, 1, 8)
        queries = torch.randn(1, 3, 8)
        target_delta = torch.randn(1, 3, 8)
        view = build_forward_view(condition, state, action, queries)
        baseline = self.model(view)
        target_delta.add_(1000.0)
        result = self.model(build_forward_view(condition, state, action, queries))

        self.assertNotIn(TokenRole.TARGET_DELTA, tuple(item.role for item in view.metadata))
        torch.testing.assert_close(result, baseline, rtol=0.0, atol=1e-6)

    def test_inverse_demonstrated_action_perturbation_has_no_view_entry(self) -> None:
        condition = torch.randn(1, 2, 8)
        state = torch.randn(1, 3, 8)
        target_delta = torch.randn(1, 3, 8)
        action_query = torch.randn(1, 1, 8)
        demonstrated_action = torch.randn(1, 1, 8)
        view = build_inverse_view(condition, state, target_delta, action_query)
        baseline = self.model(view)
        demonstrated_action.add_(1000.0)
        result = self.model(
            build_inverse_view(condition, state, target_delta, action_query)
        )

        self.assertNotIn(TokenRole.ACTION, tuple(item.role for item in view.metadata))
        torch.testing.assert_close(result, baseline, rtol=0.0, atol=1e-6)

    def test_policy_future_observation_and_action_have_no_view_entry(self) -> None:
        condition = torch.randn(1, 2, 8)
        observed_states = torch.randn(1, 2, 3, 8)
        flow_state = torch.randn(1, 8, 8)
        flow_time = torch.randn(1, 8, 8)
        horizon = torch.randn(1, 8, 8)
        future_observation = torch.randn(1, 3, 8)
        demonstrated_actions = torch.randn(1, 8, 8)
        view = build_policy_view(
            condition, observed_states, flow_state, flow_time, horizon
        )
        baseline = self.model(view)
        future_observation.add_(1000.0)
        demonstrated_actions.add_(1000.0)
        result = self.model(
            build_policy_view(
                condition, observed_states, flow_state, flow_time, horizon
            )
        )

        view_roles = tuple(item.role for item in view.metadata)
        self.assertNotIn(TokenRole.ACTION, view_roles)
        self.assertNotIn(TokenRole.TARGET_DELTA, view_roles)
        torch.testing.assert_close(result, baseline, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
