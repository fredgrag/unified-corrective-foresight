from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from corrective_foresight.data.batch import TrajectoryBatch


def make_batch() -> TrajectoryBatch:
    batch_size, time, views, proprio_dim, action_dim = 2, 12, 2, 3, 2
    return TrajectoryBatch(
        rgb=torch.zeros(batch_size, time, views, 3, 8, 8),
        camera_mask=torch.ones(batch_size, time, views, dtype=torch.bool),
        proprio=torch.zeros(batch_size, time, proprio_dim),
        proprio_mask=torch.ones(batch_size, time, proprio_dim, dtype=torch.bool),
        action=torch.zeros(batch_size, time - 1, action_dim),
        action_dimension_mask=torch.ones(
            batch_size, time - 1, action_dim, dtype=torch.bool
        ),
        observation_valid_mask=torch.ones(batch_size, time, dtype=torch.bool),
        action_valid_mask=torch.ones(batch_size, time - 1, dtype=torch.bool),
        transition_valid_mask=torch.ones(batch_size, time - 1, dtype=torch.bool),
        delta_time=torch.full((batch_size, time - 1), 0.1),
        context_index=2,
        task_text=("task a", "task b"),
        condition_ids={"task": torch.tensor([0, 1], dtype=torch.long)},
        dataset_id="test.dataset.v1",
        action_spec_id="test.ee_delta.v1",
    )


class TrajectoryBatchTest(unittest.TestCase):
    def test_valid_batch_passes_strict_validation(self) -> None:
        make_batch().validate(expected_action_spec_id="test.ee_delta.v1")

    def test_rejects_action_schema_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same ActionSpec"):
            make_batch().validate(expected_action_spec_id="other.action.v1")

    def test_rejects_transition_with_invalid_endpoint(self) -> None:
        batch = make_batch()
        observations = batch.observation_valid_mask.clone()
        observations[0, 3] = False
        cameras = batch.camera_mask.clone()
        cameras[0, 3] = False
        proprio = batch.proprio_mask.clone()
        proprio[0, 3] = False

        with self.assertRaisesRegex(ValueError, "transition_valid_mask"):
            replace(
                batch,
                observation_valid_mask=observations,
                camera_mask=cameras,
                proprio_mask=proprio,
            ).validate()

    def test_rejects_wrong_camera_mask_shape(self) -> None:
        batch = make_batch()

        with self.assertRaisesRegex(ValueError, "camera_mask"):
            replace(batch, camera_mask=batch.camera_mask[:, :-1]).validate()

    def test_rejects_nonpositive_valid_delta_time(self) -> None:
        batch = make_batch()
        delta_time = batch.delta_time.clone()
        delta_time[0, 0] = 0.0

        with self.assertRaisesRegex(ValueError, "delta_time.*positive"):
            replace(batch, delta_time=delta_time).validate()

    def test_moves_tensors_without_changing_metadata(self) -> None:
        moved = make_batch().to("cpu")

        self.assertEqual(moved.dataset_id, "test.dataset.v1")
        self.assertEqual(moved.task_text, ("task a", "task b"))
        self.assertEqual(moved.rgb.device.type, "cpu")
        self.assertEqual(moved.condition_ids["task"].device.type, "cpu")

    def test_selects_time_window_and_rebases_context(self) -> None:
        selected = make_batch().select_time_window(start=1, end=12)

        self.assertEqual(selected.rgb.shape[1], 11)
        self.assertEqual(selected.action.shape[1], 10)
        self.assertEqual(selected.context_index, 1)
        selected.validate()


if __name__ == "__main__":
    unittest.main()
