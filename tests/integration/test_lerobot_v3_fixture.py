from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from corrective_foresight.data.collate import collate_trajectory_samples
from corrective_foresight.data.lerobot_adapter import LeRobotTrajectoryAdapter
from tests.fixtures.create_lerobot_v3_fixture import (
    EPISODE_LENGTH,
    HEIGHT,
    WIDTH,
    create_lerobot_v3_fixture,
)


class LeRobotV3FixtureIntegrationTest(unittest.TestCase):
    def test_real_video_parquet_window_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tiny_lerobot_v3"
            dataset_spec, action_spec = create_lerobot_v3_fixture(root)
            adapter = LeRobotTrajectoryAdapter(
                dataset_spec=dataset_spec,
                action_spec=action_spec,
                split="train",
                context_steps=2,
                action_horizon=8,
                video_backend="torchcodec",
            )

            self.assertEqual(len(adapter), EPISODE_LENGTH)
            self.assertEqual(len(adapter.full_dynamics_indices), 5)
            for index in adapter.full_dynamics_indices:
                mask = adapter[index].transition_valid_mask
                self.assertTrue(mask.unfold(0, 8, 1).all(dim=-1).any().item())
            sample = adapter[0]
            batch = collate_trajectory_samples([sample])
            batch.validate(expected_action_spec_id=action_spec.spec_id)

            self.assertEqual(batch.rgb.shape, (1, 10, 2, 3, HEIGHT, WIDTH))
            self.assertEqual(batch.action.shape, (1, 9, 2))
            self.assertEqual(batch.proprio.shape, (1, 10, 3))
            self.assertEqual(batch.context_index, 1)
            self.assertEqual(batch.task_text, ("fixture task 0",))
            self.assertFalse(batch.observation_valid_mask[0, 0].item())
            self.assertFalse(batch.action_valid_mask[0, 0].item())
            self.assertFalse(batch.transition_valid_mask[0, 0].item())
            self.assertTrue(batch.transition_valid_mask[0, 1:].all().item())
            self.assertTrue((batch.rgb >= 0).all().item())
            self.assertTrue((batch.rgb <= 1).all().item())
            self.assertGreater(batch.rgb[0, 1, 0].std().item(), 0.0)

            self.assertEqual(len(list(root.rglob("*.mp4"))), 2)
            self.assertGreaterEqual(len(list(root.rglob("*.parquet"))), 3)
            self.assertEqual(len(adapter.dataset.meta.episodes), 2)
            for episode in adapter.dataset.meta.episodes:
                for video_key in (
                    "observation.images.base",
                    "observation.images.wrist",
                ):
                    self.assertIn(f"videos/{video_key}/from_timestamp", episode)
                    self.assertIn(f"videos/{video_key}/to_timestamp", episode)
            self.assertEqual(
                adapter.consumed_keys,
                {
                    "action",
                    "action_is_pad",
                    "observation.images.base",
                    "observation.images.base_is_pad",
                    "observation.images.wrist",
                    "observation.images.wrist_is_pad",
                    "observation.state",
                    "observation.state_is_pad",
                    "task",
                },
            )

    def test_collate_rejects_mixed_action_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tiny_lerobot_v3"
            dataset_spec, action_spec = create_lerobot_v3_fixture(root)
            adapter = LeRobotTrajectoryAdapter(dataset_spec, action_spec, split="train")
            first = adapter[1]
            second = adapter[2]
            second.action_spec_id = "other.action.v1"

            with self.assertRaisesRegex(ValueError, "same ActionSpec"):
                collate_trajectory_samples([first, second])


if __name__ == "__main__":
    unittest.main()
