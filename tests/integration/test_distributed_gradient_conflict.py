from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.gradient_conflict import (
    ObjectiveGradientAccumulator,
)


NAMES = (
    "dynamics_loss",
    "inverse_action_loss",
    "action_cycle_loss",
    "policy_flow_loss",
)
WEIGHTS = {
    "dynamics_loss": 1.0,
    "inverse_action_loss": 1.0,
    "action_cycle_loss": 0.1,
    "policy_flow_loss": 1.0,
}


def _worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_root: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        context = DistributedContext.from_initialized_process_group(
            local_rank=rank,
            device="cpu",
        )
        parameter = nn.Parameter(torch.tensor([1.0]))
        accumulator = ObjectiveGradientAccumulator((parameter,), NAMES)
        coefficients = (1.0, 3.0) if rank == 0 else (5.0, 7.0)
        for coefficient in coefficients:
            losses = {
                name: coefficient * (index + 1) * parameter.sum()
                for index, name in enumerate(NAMES)
            }
            accumulator.add(losses, accumulation_steps=2)
        result = accumulator.finalize(context, WEIGHTS)
        payload = {
            name: float(result.raw[name][0].item()) for name in NAMES
        }
        (Path(output_root) / f"rank-{rank}.json").write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


class DistributedGradientConflictTest(unittest.TestCase):
    def test_full_microbatch_gradients_are_averaged_before_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp.spawn(
                _worker,
                args=(2, str(root / "rendezvous"), str(root)),
                nprocs=2,
                join=True,
            )
            records = [
                json.loads(
                    (root / f"rank-{rank}.json").read_text(encoding="utf-8")
                )
                for rank in range(2)
            ]

        self.assertEqual(records[0], records[1])
        self.assertEqual(records[0]["dynamics_loss"], 4.0)
        self.assertEqual(records[0]["inverse_action_loss"], 8.0)
        self.assertEqual(records[0]["action_cycle_loss"], 12.0)
        self.assertEqual(records[0]["policy_flow_loss"], 16.0)


if __name__ == "__main__":
    unittest.main()
