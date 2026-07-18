from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.checkpoint import ExpectedCheckpointContract
from corrective_foresight.training.distributed_checkpoint import (
    DISTRIBUTED_FORMAT_VERSION,
    load_distributed_checkpoint_strict,
    rank_runtime_name,
    save_distributed_checkpoint_atomic,
)
from tests.unit.test_checkpoint_validation import make_state


def _two_rank_round_trip_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    destination: str,
    result_directory: str,
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
        source = make_state()
        source.mixer.next_batch()
        save_distributed_checkpoint_atomic(destination, source, context)
        destination_state = make_state()
        resume = load_distributed_checkpoint_strict(
            destination,
            ExpectedCheckpointContract.from_state(destination_state),
            context,
        )
        result = {
            "rank": rank,
            "global_step": resume.global_step,
            "mixer_step": destination_state.mixer.state_dict()["step"],
            "flow_generator_state": destination_state.flow_generator.get_state().tolist(),
        }
        Path(result_directory, f"rank-{rank}.json").write_text(
            json.dumps(result, sort_keys=True), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


class DistributedCheckpointTest(unittest.TestCase):
    def test_rank_runtime_name_is_zero_padded_and_rejects_bad_rank(self) -> None:
        self.assertEqual(rank_runtime_name(0), "rank-0000-runtime.pt")
        self.assertEqual(rank_runtime_name(12), "rank-0012-runtime.pt")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            rank_runtime_name(-1)

    def test_single_rank_v2_save_has_shared_and_rank_payloads(self) -> None:
        if dist.is_initialized():
            self.skipTest("test owns its process group")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory) / "rendezvous"
            dist.init_process_group(
                "gloo",
                init_method=f"file://{rendezvous}",
                rank=0,
                world_size=1,
            )
            try:
                destination = Path(directory) / "checkpoint"
                state = make_state()
                save_distributed_checkpoint_atomic(
                    destination,
                    state,
                    DistributedContext.from_initialized_process_group(
                        local_rank=0,
                        device="cpu",
                    ),
                )
                self.assertEqual(
                    {item.name for item in destination.iterdir()},
                    {
                        "online_model.safetensors",
                        "ema_target.safetensors",
                        "trainer_state.pt",
                        "rank-0000-runtime.pt",
                        "manifest.json",
                    },
                )
                self.assertEqual(
                    __import__(
                        "corrective_foresight.training.run_manifest",
                        fromlist=["RunManifest"],
                    ).RunManifest.from_json(
                        (destination / "manifest.json").read_text()
                    ).value["format_version"],
                    DISTRIBUTED_FORMAT_VERSION,
                )
            finally:
                dist.destroy_process_group()

    def test_missing_or_corrupt_rank_runtime_is_rejected(self) -> None:
        if dist.is_initialized():
            self.skipTest("test owns its process group")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory) / "rendezvous"
            dist.init_process_group(
                "gloo",
                init_method=f"file://{rendezvous}",
                rank=0,
                world_size=1,
            )
            try:
                context = DistributedContext.from_initialized_process_group(
                    local_rank=0,
                    device="cpu",
                )
                destination = Path(directory) / "checkpoint"
                save_distributed_checkpoint_atomic(destination, make_state(), context)
                expected = ExpectedCheckpointContract.from_state(make_state())
                (destination / rank_runtime_name(0)).unlink()
                with self.assertRaisesRegex(ValueError, "exactly"):
                    load_distributed_checkpoint_strict(destination, expected, context)

                destination = Path(directory) / "checkpoint-corrupt"
                save_distributed_checkpoint_atomic(destination, make_state(), context)
                with (destination / rank_runtime_name(0)).open("ab") as stream:
                    stream.write(b"corrupt")
                with self.assertRaisesRegex(ValueError, "(size|hash).*rank-0000"):
                    load_distributed_checkpoint_strict(
                        destination,
                        ExpectedCheckpointContract.from_state(make_state()),
                        context,
                    )
            finally:
                dist.destroy_process_group()

    def test_two_rank_round_trip_restores_each_rank_runtime(self) -> None:
        if dist.is_initialized():
            self.skipTest("test owns its process group")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory) / "rendezvous"
            destination = Path(directory) / "checkpoint"
            result_directory = Path(directory) / "results"
            result_directory.mkdir()
            mp.spawn(
                _two_rank_round_trip_worker,
                args=(2, str(rendezvous), str(destination), str(result_directory)),
                nprocs=2,
                join=True,
            )
            self.assertEqual(
                {item.name for item in destination.iterdir()},
                {
                    "online_model.safetensors",
                    "ema_target.safetensors",
                    "trainer_state.pt",
                    "rank-0000-runtime.pt",
                    "rank-0001-runtime.pt",
                    "manifest.json",
                },
            )
            results = [
                json.loads(
                    (result_directory / f"rank-{rank}.json").read_text(
                        encoding="utf-8"
                    )
                )
                for rank in range(2)
            ]
            self.assertEqual([item["rank"] for item in results], [0, 1])
            self.assertEqual([item["global_step"] for item in results], [7, 7])
            self.assertEqual([item["mixer_step"] for item in results], [1, 1])
            self.assertEqual(
                results[0]["flow_generator_state"],
                results[1]["flow_generator_state"],
            )


if __name__ == "__main__":
    unittest.main()
