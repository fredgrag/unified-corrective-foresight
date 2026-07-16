from __future__ import annotations

import unittest

import torch

from corrective_foresight.config.schema import DatasetSpec
from corrective_foresight.data.lerobot_adapter import (
    build_delta_timestamps,
    derive_validity_masks,
    extract_episode_ids,
)


def make_dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        schema_version=1,
        dataset_id="test/window.v1",
        repo_id="test/window",
        local_root=None,
        revision="revision",
        fps=10.0,
        camera_features={
            "base": "observation.images.base",
            "wrist": "observation.images.wrist",
        },
        optional_camera_roles=("wrist",),
        proprio_features=("observation.state",),
        proprio_required=True,
        task_feature="task",
        language_feature=None,
        embodiment_id="robot.v1",
        action_feature="action",
        action_spec_id="action.v1",
        sample_weight=1.0,
        split_episode_ids={"train": (0,)},
    )


class LeRobotMaskTest(unittest.TestCase):
    def test_extracts_episode_ids_from_lerobot_metadata_records(self) -> None:
        records = [
            {"episode_index": 0, "length": 12},
            {"episode_index": 2, "length": 9},
        ]

        self.assertEqual(extract_episode_ids(records), {0, 2})

    def test_builds_exact_observation_and_action_windows(self) -> None:
        timestamps, context_index = build_delta_timestamps(
            make_dataset_spec(),
            available_features={
                "observation.images.base",
                "observation.state",
                "action",
            },
            context_steps=2,
            action_horizon=8,
        )

        self.assertEqual(context_index, 1)
        self.assertEqual(
            timestamps["observation.images.base"],
            [index / 10.0 for index in range(-1, 9)],
        )
        self.assertEqual(
            timestamps["observation.state"],
            [index / 10.0 for index in range(-1, 9)],
        )
        self.assertEqual(
            timestamps["action"],
            [index / 10.0 for index in range(-1, 8)],
        )
        self.assertNotIn("observation.images.wrist", timestamps)

    def test_derives_transition_masks_from_lerobot_padding(self) -> None:
        item = {
            "observation.images.base_is_pad": torch.tensor(
                [True, False, False, False]
            ),
            "observation.state_is_pad": torch.tensor([True, False, False, False]),
            "action_is_pad": torch.tensor([True, False, False]),
        }

        observation, action, transition = derive_validity_masks(
            item,
            required_observation_features=(
                "observation.images.base",
                "observation.state",
            ),
            action_feature="action",
        )

        torch.testing.assert_close(
            observation, torch.tensor([False, True, True, True])
        )
        torch.testing.assert_close(action, torch.tensor([False, True, True]))
        torch.testing.assert_close(transition, torch.tensor([False, True, True]))

    def test_missing_required_padding_key_fails_closed(self) -> None:
        item = {
            "observation.images.base_is_pad": torch.zeros(4, dtype=torch.bool),
            "action_is_pad": torch.zeros(3, dtype=torch.bool),
        }

        with self.assertRaisesRegex(ValueError, "observation.state_is_pad"):
            derive_validity_masks(
                item,
                required_observation_features=(
                    "observation.images.base",
                    "observation.state",
                ),
                action_feature="action",
            )

    def test_padding_length_mismatch_fails_closed(self) -> None:
        item = {
            "observation.images.base_is_pad": torch.zeros(4, dtype=torch.bool),
            "observation.state_is_pad": torch.zeros(3, dtype=torch.bool),
            "action_is_pad": torch.zeros(3, dtype=torch.bool),
        }

        with self.assertRaisesRegex(ValueError, "padding length"):
            derive_validity_masks(
                item,
                required_observation_features=(
                    "observation.images.base",
                    "observation.state",
                ),
                action_feature="action",
            )


if __name__ == "__main__":
    unittest.main()
