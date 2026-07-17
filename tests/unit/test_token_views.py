from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.attention_contract import compile_attention_mask
from corrective_foresight.model.token_types import TokenRole
from corrective_foresight.model.token_views import (
    build_cycle_view,
    build_forward_view,
    build_inverse_view,
    build_policy_view,
)


def tokens(value: float, count: int, hidden_size: int = 6) -> torch.Tensor:
    return torch.full((2, count, hidden_size), value)


def roles(view) -> tuple[TokenRole, ...]:
    return tuple(metadata.role for metadata in view.metadata)


class TokenViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.conditions = tokens(1.0, 2)
        self.state = tokens(2.0, 3)
        self.action = tokens(3.0, 1)
        self.delta = tokens(4.0, 3)
        self.delta_queries = tokens(5.0, 3)
        self.action_query = tokens(6.0, 1)

    def test_forward_contains_no_target_delta_or_future_state(self) -> None:
        view = build_forward_view(
            self.conditions,
            self.state,
            self.action,
            self.delta_queries,
            semantic_time=7,
        )

        self.assertEqual(
            roles(view),
            (
                TokenRole.CONDITION,
                TokenRole.CONDITION,
                TokenRole.STATE,
                TokenRole.STATE,
                TokenRole.STATE,
                TokenRole.ACTION,
                TokenRole.DELTA_QUERY,
                TokenRole.DELTA_QUERY,
                TokenRole.DELTA_QUERY,
            ),
        )
        self.assertNotIn(TokenRole.TARGET_DELTA, roles(view))
        self.assertEqual({item.semantic_time for item in view.metadata[2:]}, {7, 8})

    def test_inverse_contains_target_delta_but_no_demonstrated_action(self) -> None:
        view = build_inverse_view(
            self.conditions,
            self.state,
            self.delta,
            self.action_query,
            semantic_time=3,
        )

        self.assertEqual(
            roles(view),
            (
                TokenRole.CONDITION,
                TokenRole.CONDITION,
                TokenRole.STATE,
                TokenRole.STATE,
                TokenRole.STATE,
                TokenRole.TARGET_DELTA,
                TokenRole.TARGET_DELTA,
                TokenRole.TARGET_DELTA,
                TokenRole.ACTION_QUERY,
            ),
        )
        self.assertNotIn(TokenRole.ACTION, roles(view))

    def test_cycle_marks_predicted_delta_and_excludes_target_delta(self) -> None:
        view = build_cycle_view(
            self.conditions,
            self.state,
            self.delta,
            self.action_query,
            semantic_time=4,
        )

        self.assertEqual(roles(view).count(TokenRole.PREDICTED_DELTA), 3)
        self.assertNotIn(TokenRole.TARGET_DELTA, roles(view))
        self.assertNotIn(TokenRole.ACTION, roles(view))

    def test_policy_queries_sum_all_generative_components(self) -> None:
        observed_states = torch.full((2, 3, 3, 6), 2.0)
        flow_state = tokens(10.0, 8)
        flow_time = tokens(20.0, 8)
        horizon = tokens(30.0, 8)

        view = build_policy_view(
            self.conditions,
            observed_states,
            flow_state,
            flow_time,
            horizon,
            start_time=5,
        )

        self.assertEqual(roles(view).count(TokenRole.STATE), 9)
        self.assertEqual(roles(view).count(TokenRole.POLICY_QUERY), 8)
        self.assertNotIn(TokenRole.ACTION, roles(view))
        self.assertNotIn(TokenRole.TARGET_DELTA, roles(view))
        query_indices = view.indices(TokenRole.POLICY_QUERY)
        torch.testing.assert_close(
            view.tokens[:, query_indices], torch.full((2, 8, 6), 60.0)
        )
        self.assertEqual(
            [view.metadata[index].semantic_time for index in query_indices],
            list(range(8, 16)),
        )

    def test_per_sample_validity_becomes_key_padding(self) -> None:
        state_valid = torch.tensor([[True, False, True], [True, True, True]])
        view = build_forward_view(
            self.conditions,
            self.state,
            self.action,
            self.delta_queries,
            state_valid=state_valid,
        )

        self.assertTrue(view.key_padding_mask[0, 3].item())
        self.assertFalse(view.key_padding_mask[1, 3].item())
        self.assertFalse(view.key_padding_mask[:, :2].any().item())

    def test_attention_contract_is_block_causal_with_condition_isolation(self) -> None:
        view = build_forward_view(
            self.conditions,
            self.state,
            self.action,
            self.delta_queries,
        )
        allowed = compile_attention_mask(view.metadata)
        condition_index = 0
        state_indices = view.indices(TokenRole.STATE)
        action_index = view.indices(TokenRole.ACTION)[0]
        query_indices = view.indices(TokenRole.DELTA_QUERY)

        self.assertTrue(allowed[condition_index, :2].all().item())
        self.assertFalse(allowed[condition_index, 2:].any().item())
        self.assertTrue(allowed[state_indices[0], state_indices].all().item())
        self.assertFalse(allowed[state_indices[0], action_index].item())
        self.assertFalse(allowed[state_indices[0], query_indices].any().item())
        self.assertTrue(allowed[action_index, state_indices].all().item())
        self.assertTrue(allowed[query_indices[0], :].all().item())

    def test_policy_requires_exactly_eight_query_components(self) -> None:
        observed_states = torch.zeros(2, 1, 3, 6)
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            build_policy_view(
                self.conditions,
                observed_states,
                tokens(1.0, 7),
                tokens(1.0, 7),
                tokens(1.0, 7),
            )


if __name__ == "__main__":
    unittest.main()
