from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from corrective_foresight.training.gates import (
    AuditMeasurement,
    ValidationStopMonitor,
    assess_unified_gate,
    assess_world_gate,
    decide_gradient_audit,
    negative_nll_deteriorated,
    write_audit_decision,
    write_stop_report,
)
from corrective_foresight.training.validation import ValidationRecord


def validation_record(
    step: int,
    *,
    improvement: float = 0.1,
    dynamics: float = 1.0,
    inverse: float = -2.0,
    cycle: float = 1.0,
    policy: float = 1.0,
    stage: str = "unified",
) -> ValidationRecord:
    return ValidationRecord(
        global_step=step,
        stage=stage,
        world_size=1,
        batches_per_rank=8,
        samples=16,
        metrics={
            "dynamics_loss": float(dynamics),
            "improvement_vs_copy_last": float(improvement),
            "inverse_action_loss": float(inverse),
            "policy_flow_loss": float(policy),
            "self_correction_cycle_loss": float(cycle),
        },
    )


class ConflictFixGatesTest(unittest.TestCase):
    def test_audit_requires_eight_conflicts_and_dynamics_degradation(self) -> None:
        seven = [
            AuditMeasurement(
                optimizer_step=260 + 10 * index,
                minimum_dynamics_cosine=-0.06 if index < 7 else 0.0,
            )
            for index in range(25)
        ]
        eight = [
            AuditMeasurement(
                optimizer_step=260 + 10 * index,
                minimum_dynamics_cosine=-0.06 if index < 8 else 0.0,
            )
            for index in range(25)
        ]

        self.assertFalse(
            decide_gradient_audit(seven, 0.004, 0.00481).enable_pcgrad
        )
        self.assertFalse(
            decide_gradient_audit(eight, 0.004, 0.00480).enable_pcgrad
        )
        self.assertTrue(
            decide_gradient_audit(eight, 0.004, 0.00481).enable_pcgrad
        )

    def test_audit_rejects_incomplete_or_misaligned_window(self) -> None:
        valid = [
            AuditMeasurement(260 + 10 * index, -0.06)
            for index in range(25)
        ]

        with self.assertRaisesRegex(ValueError, "exactly"):
            decide_gradient_audit(valid[:-1], 0.004, 0.005)
        with self.assertRaisesRegex(ValueError, "steps"):
            decide_gradient_audit(
                [replace(valid[0], optimizer_step=259), *valid[1:]],
                0.004,
                0.005,
            )

    def test_copy_last_and_world_dynamics_require_three_windows(self) -> None:
        copy_monitor = ValidationStopMonitor(world_dynamics_loss=1.0)
        copy_monitor.observe(validation_record(250))
        self.assertFalse(
            copy_monitor.observe(validation_record(500, improvement=0.0)).should_stop
        )
        self.assertFalse(
            copy_monitor.observe(validation_record(750, improvement=-0.1)).should_stop
        )
        copy_decision = copy_monitor.observe(
            validation_record(1000, improvement=0.0)
        )
        self.assertTrue(copy_decision.should_stop)
        self.assertEqual(copy_decision.reason, "copy_last_non_improvement")

        dynamics_monitor = ValidationStopMonitor(world_dynamics_loss=1.0)
        dynamics_monitor.observe(validation_record(250))
        for step in (500, 750):
            self.assertFalse(
                dynamics_monitor.observe(
                    validation_record(step, dynamics=1.200001)
                ).should_stop
            )
        decision = dynamics_monitor.observe(
            validation_record(1000, dynamics=1.200001)
        )
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.reason, "world_dynamics_degradation")

    def test_two_loss_deterioration_handles_negative_nll_and_resets(self) -> None:
        self.assertTrue(negative_nll_deteriorated(-1.5, -2.0))
        self.assertFalse(negative_nll_deteriorated(-1.6, -2.0))
        monitor = ValidationStopMonitor(world_dynamics_loss=10.0)
        monitor.observe(validation_record(250, inverse=-2.0, policy=1.0))
        bad = {"inverse": -1.5, "policy": 1.21}
        self.assertFalse(monitor.observe(validation_record(500, **bad)).should_stop)
        self.assertFalse(monitor.observe(validation_record(750, **bad)).should_stop)
        self.assertFalse(monitor.observe(validation_record(1000)).should_stop)
        self.assertFalse(monitor.observe(validation_record(1250, **bad)).should_stop)
        self.assertFalse(monitor.observe(validation_record(1500, **bad)).should_stop)
        decision = monitor.observe(validation_record(1750, **bad))
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.reason, "optimized_loss_deterioration")

    def test_no_stop_before_unified_step_500(self) -> None:
        monitor = ValidationStopMonitor(world_dynamics_loss=1.0)
        for step in (0, 100, 200, 300, 400):
            self.assertFalse(
                monitor.observe(
                    validation_record(
                        step,
                        improvement=-1.0,
                        dynamics=2.0,
                        inverse=1.0,
                        policy=2.0,
                    )
                ).should_stop
            )

    def test_world_gate_distinguishes_accept_marginal_and_reject(self) -> None:
        accepted = assess_world_gate(
            validation_record(5000, improvement=0.05, stage="world_pretrain")
        )
        marginal = assess_world_gate(
            validation_record(5000, improvement=0.01, stage="world_pretrain")
        )
        rejected = assess_world_gate(
            validation_record(5000, improvement=0.0, stage="world_pretrain")
        )

        self.assertEqual(accepted.outcome, "accepted")
        self.assertEqual(marginal.outcome, "marginal")
        self.assertTrue(marginal.requires_approval)
        self.assertEqual(rejected.outcome, "rejected")

    def test_unified_gate_requires_effect_and_operational_evidence(self) -> None:
        record = validation_record(5000, improvement=0.1, dynamics=1.1)
        passed = assess_unified_gate(
            final_validation=record,
            world_dynamics_loss=1.0,
            stop_rule_fired=False,
            optimized_metrics_diverged=False,
            checkpoint_verified=True,
            resume_verified=True,
            baseline_successes=3,
            unified_successes=5,
            paired_reward_changes=(
                -1.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ),
            actions_within_bounds=True,
            tracking_consistent=True,
        )
        failed = assess_unified_gate(
            final_validation=record,
            world_dynamics_loss=1.0,
            stop_rule_fired=False,
            optimized_metrics_diverged=False,
            checkpoint_verified=True,
            resume_verified=True,
            baseline_successes=3,
            unified_successes=4,
            paired_reward_changes=(
                -1.0,
                -0.8,
                -0.6,
                -0.4,
                -0.2,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ),
            actions_within_bounds=True,
            tracking_consistent=True,
        )

        self.assertTrue(passed.passed)
        self.assertFalse(failed.passed)
        self.assertIn("success_gain_below_two", failed.reasons)
        self.assertIn("median_reward_not_positive", failed.reasons)

    def test_atomic_audit_writer_rejects_existing_destination_and_symlink(self) -> None:
        decision = decide_gradient_audit(
            [AuditMeasurement(260 + 10 * index, -0.06) for index in range(25)],
            0.004,
            0.005,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "gradient-audit-decision.json"
            write_audit_decision(destination, decision)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertTrue(payload["enable_pcgrad"])
            with self.assertRaises(FileExistsError):
                write_audit_decision(destination, decision)
            link = root / "link.json"
            link.symlink_to(destination)
            with self.assertRaises(FileExistsError):
                write_audit_decision(link, decision)

    def test_stop_report_rejects_incomplete_provenance(self) -> None:
        from corrective_foresight.training.gates import PilotStepDecision

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "context"):
                write_stop_report(
                    Path(directory) / "stop-report.json",
                    PilotStepDecision(
                        True,
                        "copy_last_non_improvement",
                        (500, 750, 1000),
                    ),
                    context={},
                )


if __name__ == "__main__":
    unittest.main()
