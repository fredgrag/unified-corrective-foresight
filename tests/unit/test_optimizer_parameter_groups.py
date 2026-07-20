from __future__ import annotations

import unittest

from corrective_foresight.training.parameter_groups import (
    build_optimizer_parameter_groups,
)
from tests.unit.policy_fakes import make_policy


class OptimizerParameterGroupsTest(unittest.TestCase):
    def test_groups_are_exhaustive_disjoint_and_semantic(self) -> None:
        policy = make_policy()
        groups = build_optimizer_parameter_groups(policy)
        protected = {id(item) for item in groups.protected}
        action = {id(item) for item in groups.action}
        trainable = {id(item) for item in policy.parameters() if item.requires_grad}

        self.assertFalse(protected & action)
        self.assertEqual(protected | action, trainable)
        self.assertIn(
            id(policy.world_action_model.delta_projection.weight),
            protected,
        )
        self.assertIn(id(policy.world_action_model.delta_queries), protected)
        self.assertIn(id(policy.world_action_model.policy_queries), action)
        self.assertIn(id(policy.world_action_model.action_query), action)
        self.assertIn(
            id(policy.world_action_model.policy_horizon_embedding.weight),
            action,
        )
        adapter = next(
            iter(policy.world_action_model.action_adapters.adapters.values())
        )
        self.assertIn(id(adapter.action_input_projection.weight), protected)
        self.assertIn(id(adapter.flow_velocity_head.weight), action)
        self.assertIn(id(adapter.inverse_mean_head.weight), action)
        self.assertTrue(
            all(
                not item.requires_grad
                for item in policy.online_state_encoder.backbone.parameters()
            )
        )

    def test_group_order_and_parameter_order_are_stable(self) -> None:
        first = build_optimizer_parameter_groups(make_policy())
        second = build_optimizer_parameter_groups(make_policy())

        self.assertEqual(first.names, ("protected", "action"))
        self.assertEqual(
            tuple(item.shape for item in first.protected),
            tuple(item.shape for item in second.protected),
        )
        self.assertEqual(
            tuple(item.shape for item in first.action),
            tuple(item.shape for item in second.action),
        )


if __name__ == "__main__":
    unittest.main()
