from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.validation import ValidationConfig, ValidationRunner
from tests.unit.policy_fakes import make_batch, make_policy


class _ValidationMixer:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.counter = 0

    def next_batch(self):
        self.counter += 1
        return make_batch(self.policy)

    def state_dict(self):
        return {"version": 1, "counter": self.counter}

    def load_state_dict(self, state):
        if set(state) != {"version", "counter"}:
            raise ValueError("invalid validation mixer state")
        self.counter = state["counter"]


def _distributed_validation_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    result_directory: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(7001)
        policy = make_policy()
        mixer = _ValidationMixer(policy)
        mixer.counter = 11
        runner = ValidationRunner(
            policy=policy,
            validation_mixer=mixer,
            context=DistributedContext.from_initialized_process_group(
                local_rank=rank,
                device="cpu",
            ),
            config=ValidationConfig(batches_per_rank=2, generator_seed=20261017),
        )
        first = runner.evaluate("unified", global_step=100)
        second = runner.evaluate("unified", global_step=100)
        if first != second or mixer.counter != 11:
            raise AssertionError("distributed validation was not deterministic")
        result = {
            "global_step": first.global_step,
            "stage": first.stage,
            "world_size": first.world_size,
            "batches_per_rank": first.batches_per_rank,
            "samples": first.samples,
            "metrics": dict(first.metrics),
        }
        Path(result_directory, f"rank-{rank}.json").write_text(
            json.dumps(result, sort_keys=True), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


class DistributedValidationTest(unittest.TestCase):
    def test_all_ranks_return_identical_aggregated_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_directory = root / "results"
            result_directory.mkdir()
            mp.spawn(
                _distributed_validation_worker,
                args=(2, str(root / "rendezvous"), str(result_directory)),
                nprocs=2,
                join=True,
            )
            first = (result_directory / "rank-0.json").read_bytes()
            second = (result_directory / "rank-1.json").read_bytes()
            self.assertEqual(first, second)
            record = json.loads(first)
            self.assertEqual(record["world_size"], 2)
            self.assertEqual(record["batches_per_rank"], 2)
            self.assertEqual(record["samples"], 4)


if __name__ == "__main__":
    unittest.main()
