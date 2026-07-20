from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import torch

from corrective_foresight.training.gates import (
    PilotStepDecision,
    ValidationStopMonitor,
)
from corrective_foresight.training.pilot import PilotController
from corrective_foresight.training.trainer import Trainer, TrainerConfig
from corrective_foresight.training.validation import ValidationRecord
from tests.unit.policy_fakes import make_batch, make_policy
from train import run_training


def _record(step: int) -> ValidationRecord:
    return ValidationRecord(
        global_step=step,
        stage="unified",
        world_size=1,
        batches_per_rank=8,
        samples=16,
        metrics={"action_loss": 1.0, "dynamics_loss": 2.0},
    )


def _stop_record(step: int) -> ValidationRecord:
    return ValidationRecord(
        global_step=step,
        stage="unified",
        world_size=1,
        batches_per_rank=8,
        samples=16,
        metrics={
            "dynamics_loss": 1.0,
            "improvement_vs_copy_last": 0.0,
            "inverse_action_loss": -2.0,
            "policy_flow_loss": 1.0,
            "self_correction_cycle_loss": 1.0,
        },
    )


class PilotControllerTest(unittest.TestCase):
    def test_callbacks_fire_at_optimizer_steps_and_validation_precedes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            events: list[tuple[str, int]] = []
            controller = PilotController(
                stage="unified",
                checkpoint_interval=250,
                validation_interval=100,
                output_root=root,
                rank=0,
                world_size=1,
                validate=lambda step: events.append(("validate", step)) or _record(step),
                save_checkpoint=lambda step: events.append(("checkpoint", step)),
            )
            for step in range(1, 501):
                controller.on_optimizer_step(step)
            controller.finish(500)

            self.assertEqual(
                [step for kind, step in events if kind == "validate"],
                [100, 200, 300, 400, 500],
            )
            self.assertEqual(
                [step for kind, step in events if kind == "checkpoint"],
                [250, 500],
            )
            at_500 = [kind for kind, step in events if step == 500]
            self.assertEqual(at_500, ["validate", "checkpoint"])
            self.assertTrue((root / "validation" / "unified-000500.json").is_file())

    def test_existing_output_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "existing"
            root.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                PilotController(
                    stage="unified",
                    checkpoint_interval=250,
                    validation_interval=100,
                    output_root=root,
                    rank=0,
                    world_size=1,
                    validate=lambda step: _record(step),
                    save_checkpoint=lambda step: None,
                )

    def test_stop_monitor_requires_complete_static_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "stop_report_context"):
                PilotController(
                    stage="unified",
                    checkpoint_interval=1000,
                    validation_interval=250,
                    output_root=Path(directory) / "pilot",
                    rank=0,
                    world_size=1,
                    validate=_stop_record,
                    save_checkpoint=lambda step: {
                        "checkpoint_manifest_sha256": "c" * 64
                    },
                    stop_monitor=ValidationStopMonitor(
                        world_dynamics_loss=1.0
                    ),
                    stop_report_context={},
                )

    def test_stop_monitor_saves_boundary_once_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            checkpoints: list[int] = []

            def save_checkpoint(step: int):
                checkpoints.append(step)
                return {"checkpoint_manifest_sha256": "c" * 64}

            controller = PilotController(
                stage="unified",
                checkpoint_interval=1000,
                validation_interval=250,
                output_root=root,
                rank=0,
                world_size=1,
                validate=_stop_record,
                save_checkpoint=save_checkpoint,
                stop_monitor=ValidationStopMonitor(world_dynamics_loss=1.0),
                stop_report_context={
                    "git_commit": "a" * 40,
                    "resolved_config": {"pilot_id": "conflict-fix-test"},
                    "data_spec_hashes": {
                        "dataset": "d" * 64,
                        "action": "e" * 64,
                    },
                },
            )

            decision = None
            for step in range(1, 1001):
                decision = controller.on_optimizer_step(step)
                if decision.should_stop:
                    break
            controller.finish(step)

            self.assertTrue(decision.should_stop)
            self.assertEqual(decision.reason, "copy_last_non_improvement")
            self.assertEqual(checkpoints, [1000])
            report = json.loads(
                (root / "stop-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["triggering_steps"], [500, 750, 1000])

    def test_training_loop_returns_actual_step_after_explicit_stop(self) -> None:
        class RepeatingBatchSource:
            def __init__(self, batch) -> None:
                self.batch = batch

            def next_batch(self):
                return self.batch

        policy = make_policy()
        trainer = Trainer(
            policy,
            TrainerConfig(
                learning_rate=1e-3,
                weight_decay=0.0,
                accumulation_steps=1,
                max_grad_norm=1.0,
                warmup_steps=0,
                total_steps=5,
                bf16=False,
                ddp=False,
            ),
            rank=0,
        )
        callback_steps: list[int] = []

        def stop_at_two(step, result):
            callback_steps.append(step)
            if step == 2:
                return PilotStepDecision(
                    True,
                    "test_stop",
                    (0, 1, 2),
                )
            return PilotStepDecision(False, None)

        final_step = run_training(
            batch_source=RepeatingBatchSource(make_batch(policy)),
            trainer=trainer,
            stage="unified",
            optimizer_steps=5,
            start_global_step=0,
            generator_seed=1901,
            metric_logger=lambda record: None,
            optimizer_step_callback=stop_at_two,
        )

        self.assertEqual(final_step, 2)
        self.assertEqual(callback_steps, [1, 2])
        self.assertEqual(policy.last_ema_step, 1)

    def test_resume_existing_output_can_skip_final_audit_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "audit"
            root.mkdir()
            checkpoints: list[int] = []
            controller = PilotController(
                stage="unified",
                checkpoint_interval=1000,
                validation_interval=250,
                output_root=root,
                rank=0,
                world_size=1,
                validate=_stop_record,
                save_checkpoint=lambda step: checkpoints.append(step),
                resume_existing_output=True,
                save_final_checkpoint=False,
            )

            controller.on_optimizer_step(1)
            controller.finish(500)

            self.assertEqual(checkpoints, [])


if __name__ == "__main__":
    unittest.main()
