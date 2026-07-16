from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist

from corrective_foresight.data.mixer import BalancedLeRobotMixer
from corrective_foresight.data.stateful_sampler import (
    StatefulDistributedBatchSampler,
)
from tests.fixtures.fake_trajectory_dataset import FakeTrajectoryDataset


class DistributedMixerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.owns_process_group = not dist.is_initialized()
        cls.init_file: Path | None = None
        if not cls.owns_process_group:
            return
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            dist.init_process_group("gloo")
        else:
            handle, path = tempfile.mkstemp(prefix="ucf-gloo-")
            os.close(handle)
            Path(path).unlink()
            cls.init_file = Path(path)
            dist.init_process_group(
                "gloo",
                init_method=f"file://{path}",
                rank=0,
                world_size=1,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        if cls.init_file is not None:
            cls.init_file.unlink(missing_ok=True)

    def test_ranks_share_dataset_choice_and_keep_disjoint_samples(self) -> None:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        datasets = {
            "alpha": FakeTrajectoryDataset("alpha", "alpha.action.v1", size=64),
            "beta": FakeTrajectoryDataset("beta", "beta.action.v1", size=64),
        }
        samplers = {
            name: StatefulDistributedBatchSampler(
                len(dataset),
                batch_size=2,
                seed=41,
                rank=rank,
                world_size=world_size,
            )
            for name, dataset in datasets.items()
        }
        mixer = BalancedLeRobotMixer(
            datasets=datasets,
            samplers=samplers,
            weights={"alpha": 1.0, "beta": 1.0},
            generator=torch.Generator().manual_seed(123),
        )

        for _ in range(8):
            batch = mixer.next_batch()
            dataset_ids: list[str | None] = [None] * world_size
            dist.all_gather_object(dataset_ids, batch.dataset_id)
            self.assertEqual(len(set(dataset_ids)), 1)

            local_indices = [int(text.rsplit(":", 1)[1]) for text in batch.task_text]
            rank_indices: list[list[int] | None] = [None] * world_size
            dist.all_gather_object(rank_indices, local_indices)
            flattened = [index for indices in rank_indices for index in indices]
            self.assertEqual(len(flattened), len(set(flattened)))

    def test_action_schema_mismatch_across_ranks_fails_closed(self) -> None:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        action_spec_id = (
            "alpha.action.v1"
            if rank == 0 or world_size == 1
            else "wrong.action.v1"
        )
        dataset = FakeTrajectoryDataset("alpha", action_spec_id, size=16)
        sampler = StatefulDistributedBatchSampler(
            len(dataset),
            batch_size=2,
            seed=5,
            rank=rank,
            world_size=world_size,
        )
        mixer = BalancedLeRobotMixer(
            datasets={"alpha": dataset},
            samplers={"alpha": sampler},
            weights={"alpha": 1.0},
            generator=torch.Generator().manual_seed(3),
        )

        if world_size == 1:
            self.assertEqual(mixer.next_batch().action_spec_id, "alpha.action.v1")
        else:
            with self.assertRaisesRegex(
                RuntimeError, "distributed dataset contract mismatch"
            ):
                mixer.next_batch()


if __name__ == "__main__":
    unittest.main()
