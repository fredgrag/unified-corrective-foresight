from __future__ import annotations

import unittest

from corrective_foresight.data.stateful_sampler import (
    StatefulDistributedBatchSampler,
)


class StatefulDistributedBatchSamplerTest(unittest.TestCase):
    def test_same_seed_produces_same_batches(self) -> None:
        first = StatefulDistributedBatchSampler(32, batch_size=2, seed=17)
        second = StatefulDistributedBatchSampler(32, batch_size=2, seed=17)

        self.assertEqual(
            [first.next_batch() for _ in range(8)],
            [second.next_batch() for _ in range(8)],
        )

    def test_rank_shards_are_disjoint_within_epoch(self) -> None:
        rank_zero = StatefulDistributedBatchSampler(
            32, batch_size=2, seed=9, rank=0, world_size=2
        )
        rank_one = StatefulDistributedBatchSampler(
            32, batch_size=2, seed=9, rank=1, world_size=2
        )
        zero_indices = {
            index for _ in range(8) for index in rank_zero.next_batch()
        }
        one_indices = {index for _ in range(8) for index in rank_one.next_batch()}

        self.assertTrue(zero_indices.isdisjoint(one_indices))
        self.assertEqual(zero_indices | one_indices, set(range(32)))

    def test_state_dict_resumes_exact_next_batch(self) -> None:
        uninterrupted = StatefulDistributedBatchSampler(40, batch_size=2, seed=3)
        for _ in range(5):
            uninterrupted.next_batch()
        state = uninterrupted.state_dict()
        expected = [uninterrupted.next_batch() for _ in range(7)]

        resumed = StatefulDistributedBatchSampler(40, batch_size=2, seed=3)
        resumed.load_state_dict(state)
        actual = [resumed.next_batch() for _ in range(7)]

        self.assertEqual(actual, expected)

    def test_next_batch_rolls_to_new_deterministic_epoch(self) -> None:
        sampler = StatefulDistributedBatchSampler(8, batch_size=2, seed=4)
        first_epoch = [sampler.next_batch() for _ in range(4)]
        next_batch = sampler.next_batch()

        self.assertEqual(sampler.epoch, 1)
        self.assertNotEqual(next_batch, first_epoch[0])

    def test_rejects_dataset_too_small_for_rank_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "full local batch"):
            StatefulDistributedBatchSampler(
                3, batch_size=2, seed=0, rank=0, world_size=2
            )

    def test_rejects_incompatible_resume_contract(self) -> None:
        source = StatefulDistributedBatchSampler(16, batch_size=2, seed=0)
        target = StatefulDistributedBatchSampler(16, batch_size=4, seed=0)

        with self.assertRaisesRegex(ValueError, "sampler contract mismatch"):
            target.load_state_dict(source.state_dict())


if __name__ == "__main__":
    unittest.main()
