from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from corrective_foresight.config.schema import ActionSpec


def make_action_spec() -> ActionSpec:
    return ActionSpec(
        schema_version=1,
        spec_id="test.ee_delta.v1",
        dimension=2,
        names=("x", "gripper"),
        units=("m", "unitless"),
        action_space="end_effector",
        mode="delta",
        rotation_representation="none",
        arm_count=1,
        gripper_indices=(1,),
        control_mode="pd_ee_delta_pose",
        frequency_hz=10.0,
        normalization_mean=(1.0, 2.0),
        normalization_std=(2.0, 4.0),
        minimum=(-1.0, -2.0),
        maximum=(3.0, 4.0),
    )


class ActionSpecTest(unittest.TestCase):
    def test_rejects_nonpositive_normalization_std(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalization_std.*positive"):
            replace(make_action_spec(), normalization_std=(1.0, 0.0))

    def test_rejects_duplicate_dimension_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "names.*unique"):
            replace(make_action_spec(), names=("x", "x"))

    def test_rejects_nonstring_dimension_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "names.*nonempty strings"):
            replace(make_action_spec(), names=("x", None))

    def test_rejects_out_of_range_gripper_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "gripper_indices"):
            replace(make_action_spec(), gripper_indices=(2,))

    def test_rejects_inverted_action_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum.*maximum"):
            replace(make_action_spec(), minimum=(-1.0, 5.0))

    def test_normalizes_and_denormalizes_only_valid_dimensions(self) -> None:
        spec = make_action_spec()
        action = torch.tensor([[3.0, float("nan")]])
        mask = torch.tensor([[True, False]])

        normalized = spec.normalize(action, mask)
        restored = spec.denormalize(normalized, mask)

        torch.testing.assert_close(normalized, torch.tensor([[1.0, 0.0]]))
        torch.testing.assert_close(restored, torch.tensor([[3.0, 0.0]]))

    def test_rejects_wrong_action_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "final action dimension"):
            make_action_spec().normalize(torch.zeros(1, 3))

    def test_content_hash_is_stable_and_content_sensitive(self) -> None:
        spec = make_action_spec()

        self.assertEqual(spec.content_hash, make_action_spec().content_hash)
        self.assertNotEqual(
            spec.content_hash,
            replace(spec, frequency_hz=20.0).content_hash,
        )


if __name__ == "__main__":
    unittest.main()
