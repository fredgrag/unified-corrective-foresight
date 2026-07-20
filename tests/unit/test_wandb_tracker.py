from __future__ import annotations

import json
from pathlib import Path
import tempfile
import tomllib
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from corrective_foresight.config.conflict_fix import TrackingConfig
from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.trainer import Trainer, TrainerConfig
from corrective_foresight.tracking.wandb_tracker import (
    WandbTracker,
    reduce_scalar_metrics,
)
from tests.unit.policy_fakes import make_batch, make_policy
from train import run_training


class FakeRun:
    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.entity = "test-entity"
        self.logged: list[tuple[dict[str, float], int]] = []
        self.finished = False
        self.fail_logging = False

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        if self.fail_logging:
            raise ConnectionError("offline")
        self.logged.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    def __init__(self) -> None:
        self.init_calls = 0
        self.last_resume: str | None = None
        self.init_kwargs: list[dict[str, object]] = []
        self.runs: list[FakeRun] = []

    def init(self, **kwargs: object) -> FakeRun:
        self.init_calls += 1
        self.init_kwargs.append(dict(kwargs))
        self.last_resume = (
            kwargs.get("resume")
            if isinstance(kwargs.get("resume"), str)
            else None
        )
        run_id = (
            kwargs.get("id")
            if isinstance(kwargs.get("id"), str)
            else "run-fixed"
        )
        run = FakeRun(run_id)
        self.runs.append(run)
        return run


def tracking_config() -> TrackingConfig:
    return TrackingConfig(
        enabled=True,
        project="unified-corrective-foresight",
        group="maniskill-pickcube-conflict-fix-v2",
        log_interval=10,
        upload_checkpoints=False,
    )


def _reduce_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    result_root: str,
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
        reduced = reduce_scalar_metrics(
            {"loss": float(rank + 1), "shared": 3.0},
            context,
        )
        (Path(result_root) / f"rank-{rank}.json").write_text(
            json.dumps(reduced, sort_keys=True),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


class WandbTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "run"
        self.root.mkdir()
        self.backend = FakeWandb()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_rank_zero_initializes_and_nonzero_rank_is_noop(self) -> None:
        rank_zero = WandbTracker.start(
            config=tracking_config(),
            rank=0,
            backend=self.backend,
            output_root=self.root,
        )
        rank_one = WandbTracker.start(
            config=tracking_config(),
            rank=1,
            backend=self.backend,
            output_root=self.root,
        )

        self.assertEqual(self.backend.init_calls, 1)
        self.assertTrue(rank_zero.enabled)
        self.assertFalse(rank_one.enabled)

    def test_project_pins_approved_wandb_version(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        self.assertIn("wandb==0.24.2", project["project"]["dependencies"])

    def test_resume_must_uses_atomic_run_id(self) -> None:
        first = WandbTracker.start(
            config=tracking_config(),
            rank=0,
            backend=self.backend,
            output_root=self.root,
            job_type="unified_pilot",
            sanitized_run_config={"seed": 1901},
        )
        run_id = first.run_id
        first.finish(sync_complete=True)

        resumed = WandbTracker.resume(
            config=tracking_config(),
            rank=0,
            backend=self.backend,
            output_root=self.root,
            job_type="unified_pilot",
            sanitized_run_config={"seed": 1901},
        )

        self.assertEqual(resumed.run_id, run_id)
        self.assertEqual(self.backend.last_resume, "must")
        self.assertEqual(self.backend.init_kwargs[-1]["id"], run_id)

    def test_validation_logs_on_current_training_step_without_advancing(self) -> None:
        tracker = WandbTracker.start(
            config=tracking_config(),
            rank=0,
            backend=self.backend,
            output_root=self.root,
        )
        tracker.log({"train/loss": 1.0}, optimizer_step=250)

        tracker.log_validation(
            {"validation/dynamics_loss": 0.1},
            optimizer_step=250,
        )

        self.assertEqual(self.backend.runs[-1].logged[-1][1], 250)
        self.assertEqual(tracker.last_optimizer_step, 250)
        with self.assertRaisesRegex(ValueError, "current optimizer step"):
            tracker.log_validation(
                {"validation/dynamics_loss": 0.09},
                optimizer_step=251,
            )

    def test_log_updates_metadata_and_network_failure_does_not_raise(self) -> None:
        tracker = WandbTracker.start(
            config=tracking_config(),
            rank=0,
            backend=self.backend,
            output_root=self.root,
        )
        tracker.log({"loss": 1.0}, optimizer_step=10)
        metadata_path = self.root / "tracking-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["last_optimizer_step"], 10)
        self.assertFalse(metadata["sync_complete"])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            tracker.log({"loss": 0.9}, optimizer_step=10)

        self.backend.runs[-1].fail_logging = True
        tracker.log({"loss": 0.8}, optimizer_step=20)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["last_optimizer_step"], 20)
        self.assertFalse(metadata["sync_complete"])
        tracker.finish(sync_complete=True)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertFalse(metadata["sync_complete"])
        tracker.record_external_sync_success()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertTrue(metadata["sync_complete"])

    def test_secrets_and_checkpoint_artifacts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "secret"):
            WandbTracker.start(
                config=tracking_config(),
                rank=0,
                backend=self.backend,
                output_root=self.root,
                sanitized_run_config={"wandb_api_key": "secret"},
            )
        tracker = WandbTracker.start(
            config=tracking_config(),
            rank=0,
            backend=self.backend,
            output_root=self.root,
        )
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            tracker.log_checkpoint_artifact(self.root / "checkpoint")

    def test_two_rank_scalar_reduction_returns_global_mean(self) -> None:
        rendezvous = Path(self.temporary.name) / "reduce.rendezvous"
        result_root = Path(self.temporary.name) / "results"
        result_root.mkdir()
        mp.spawn(
            _reduce_worker,
            args=(2, str(rendezvous), str(result_root)),
            nprocs=2,
            join=True,
        )

        for rank in range(2):
            reduced = json.loads(
                (result_root / f"rank-{rank}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reduced, {"loss": 1.5, "shared": 3.0})

    def test_training_loop_logs_reduced_scalars_on_tenth_step(self) -> None:
        class RepeatingBatchSource:
            def __init__(self, batch) -> None:
                self.batch = batch

            def next_batch(self):
                return self.batch

        tracker = WandbTracker.start(
            config=tracking_config(),
            rank=0,
            backend=self.backend,
            output_root=self.root,
        )
        policy = make_policy()
        trainer = Trainer(
            policy,
            TrainerConfig(
                learning_rate=1e-3,
                weight_decay=0.0,
                accumulation_steps=1,
                max_grad_norm=1.0,
                warmup_steps=0,
                total_steps=12,
                bf16=False,
                ddp=False,
            ),
            rank=0,
        )

        final_step = run_training(
            batch_source=RepeatingBatchSource(make_batch(policy)),
            trainer=trainer,
            stage="unified",
            optimizer_steps=10,
            start_global_step=0,
            generator_seed=1901,
            metric_logger=lambda record: None,
            wandb_tracker=tracker,
        )

        self.assertEqual(final_step, 10)
        self.assertEqual(len(self.backend.runs[-1].logged), 1)
        metrics, step = self.backend.runs[-1].logged[0]
        self.assertEqual(step, 10)
        self.assertIn("learning_rate/action", metrics)
        self.assertIn("pre_clip_gradient_norm", metrics)


if __name__ == "__main__":
    unittest.main()
