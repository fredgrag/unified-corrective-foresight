from __future__ import annotations

import unittest

import torch

from corrective_foresight.data.mixer import (
    BalancedLeRobotMixer,
    sample_weighted_dataset_id,
)
from corrective_foresight.data.stateful_sampler import (
    StatefulDistributedBatchSampler,
)
from tests.fixtures.fake_trajectory_dataset import FakeTrajectoryDataset


def make_mixer(seed: int = 7) -> BalancedLeRobotMixer:
    datasets = {
        "alpha": FakeTrajectoryDataset("alpha", "alpha.action.v1"),
        "beta": FakeTrajectoryDataset("beta", "beta.action.v1"),
    }
    samplers = {
        name: StatefulDistributedBatchSampler(len(dataset), batch_size=2, seed=11)
        for name, dataset in datasets.items()
    }
    generator = torch.Generator().manual_seed(seed)
    return BalancedLeRobotMixer(
        datasets=datasets,
        samplers=samplers,
        weights={"alpha": 1.0, "beta": 3.0},
        generator=generator,
    )


class BalancedLeRobotMixerTest(unittest.TestCase):
    def test_weighted_selector_is_seeded_and_balanced(self) -> None:
        ids = ("alpha", "beta")
        weights = torch.tensor([1.0, 3.0], dtype=torch.float64)
        first_generator = torch.Generator().manual_seed(23)
        second_generator = torch.Generator().manual_seed(23)

        first = [
            sample_weighted_dataset_id(ids, weights, first_generator)
            for _ in range(20_000)
        ]
        second = [
            sample_weighted_dataset_id(ids, weights, second_generator)
            for _ in range(20_000)
        ]

        self.assertEqual(first, second)
        beta_fraction = first.count("beta") / len(first)
        self.assertAlmostEqual(beta_fraction, 0.75, delta=0.02)

    def test_each_batch_has_one_dataset_and_action_schema(self) -> None:
        mixer = make_mixer()

        for _ in range(20):
            batch = mixer.next_batch()
            self.assertEqual(batch.rgb.shape[0], 2)
            self.assertTrue(
                all(text.startswith(f"{batch.dataset_id}:") for text in batch.task_text)
            )
            self.assertEqual(
                batch.action_spec_id,
                f"{batch.dataset_id}.action.v1",
            )

    def test_mixer_state_resumes_dataset_and_sample_sequence(self) -> None:
        uninterrupted = make_mixer(seed=31)
        for _ in range(4):
            uninterrupted.next_batch()
        state = uninterrupted.state_dict()
        expected = [
            (batch.dataset_id, batch.task_text)
            for batch in (uninterrupted.next_batch() for _ in range(10))
        ]

        resumed = make_mixer(seed=999)
        resumed.load_state_dict(state)
        actual = [
            (batch.dataset_id, batch.task_text)
            for batch in (resumed.next_batch() for _ in range(10))
        ]

        self.assertEqual(actual, expected)
        self.assertEqual(resumed.step, uninterrupted.step)

    def test_rejects_nonpositive_weight(self) -> None:
        dataset = FakeTrajectoryDataset("alpha", "alpha.action.v1")
        sampler = StatefulDistributedBatchSampler(len(dataset), batch_size=2, seed=1)

        with self.assertRaisesRegex(ValueError, "weights.*positive"):
            BalancedLeRobotMixer(
                datasets={"alpha": dataset},
                samplers={"alpha": sampler},
                weights={"alpha": 0.0},
            )


if __name__ == "__main__":
    unittest.main()
