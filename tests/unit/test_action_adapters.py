from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from corrective_foresight.model.action_adapters import ActionAdapterRegistry
from tests.unit.test_action_spec import make_action_spec


def second_action_spec():
    return replace(
        make_action_spec(),
        spec_id="test.joint.v1",
        dimension=3,
        names=("joint_0", "joint_1", "gripper"),
        units=("rad", "rad", "unitless"),
        action_space="joint",
        mode="absolute",
        gripper_indices=(2,),
        control_mode="pd_joint_pos",
        normalization_mean=(0.0, 0.5, 1.0),
        normalization_std=(1.0, 2.0, 4.0),
        minimum=(-2.0, -2.0, 0.0),
        maximum=(2.0, 2.0, 1.0),
    )


class ActionAdapterRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first = make_action_spec()
        self.second = second_action_spec()
        self.registry = ActionAdapterRegistry((self.first, self.second), hidden_size=12)

    def test_incompatible_specs_own_separate_complete_adapters(self) -> None:
        first = self.registry.resolve(self.first.spec_id)
        second = self.registry.resolve(self.second.spec_id)

        self.assertIsNot(first, second)
        for name in (
            "action_input_projection",
            "flow_state_projection",
            "flow_velocity_head",
            "inverse_mean_head",
            "inverse_logvar_head",
        ):
            self.assertIsNot(getattr(first, name), getattr(second, name))
        self.assertEqual(first.action_input_projection.in_features, 2)
        self.assertEqual(second.action_input_projection.in_features, 3)
        self.assertEqual(first.flow_velocity_head.out_features, 2)
        self.assertEqual(second.flow_velocity_head.out_features, 3)

    def test_projection_and_heads_preserve_schema_dimensions(self) -> None:
        adapter = self.registry.resolve(self.second.spec_id)
        action = torch.randn(2, 8, 3)
        hidden = torch.randn(2, 8, 12)

        self.assertEqual(adapter.project_action(action).shape, (2, 8, 12))
        self.assertEqual(adapter.project_flow_state(action).shape, (2, 8, 12))
        self.assertEqual(adapter.predict_flow_velocity(hidden).shape, (2, 8, 3))
        mean, log_variance = adapter.predict_inverse(hidden[:, :1])
        self.assertEqual(mean.shape, (2, 1, 3))
        self.assertEqual(log_variance.shape, (2, 1, 3))

    def test_each_adapter_uses_its_own_actionspec_normalization(self) -> None:
        first = self.registry.resolve(self.first.spec_id)
        second = self.registry.resolve(self.second.spec_id)

        torch.testing.assert_close(
            first.normalize(torch.tensor([[[3.0, 6.0]]])),
            torch.tensor([[[1.0, 1.0]]]),
        )
        torch.testing.assert_close(
            second.normalize(torch.tensor([[[1.0, 2.5, 5.0]]])),
            torch.tensor([[[1.0, 1.0, 1.0]]]),
        )

    def test_unknown_and_mixed_batch_fail_before_adapter_use(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown ActionSpec"):
            self.registry.resolve("missing")
        with self.assertRaisesRegex(ValueError, "mixed ActionSpec"):
            self.registry.resolve_batch((self.first.spec_id, self.second.spec_id))
        self.assertIs(
            self.registry.resolve_batch((self.first.spec_id, self.first.spec_id)),
            self.registry.resolve(self.first.spec_id),
        )

    def test_duplicate_spec_id_is_rejected_even_if_contents_differ(self) -> None:
        duplicate = replace(self.first, frequency_hz=20.0)
        with self.assertRaisesRegex(ValueError, "duplicate ActionSpec"):
            ActionAdapterRegistry((self.first, duplicate), hidden_size=12)


if __name__ == "__main__":
    unittest.main()
