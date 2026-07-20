from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import unittest
import json

from corrective_foresight.config.conflict_fix import load_conflict_fix_config
from corrective_foresight.training.conflict_fix_phase import (
    resolve_conflict_fix_phase,
)
from corrective_foresight.training.gates import (
    AuditMeasurement,
    assess_unified_gate,
    assess_world_gate,
    decide_gradient_audit,
    write_audit_decision,
    write_unified_gate_report,
    write_world_gate_report,
)
from corrective_foresight.training.validation import ValidationRecord
from train import _json_safe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/pilots/maniskill_conflict_fix_v2.yaml"


class ConflictFixOrchestrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "run"
        loaded = load_conflict_fix_config(CONFIG_PATH)
        self.config = replace(
            loaded,
            output_root=self.root,
            world_config=replace(
                loaded.world_config,
                output_root=self.root / "world_pretrain",
            ),
            unified_config=replace(
                loaded.unified_config,
                output_root=self.root / "unified",
            ),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _accepted_world_gate(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = ValidationRecord(
            global_step=5000,
            stage="world_pretrain",
            world_size=4,
            batches_per_rank=8,
            samples=64,
            metrics={"improvement_vs_copy_last": 0.05},
        )
        write_world_gate_report(
            self.root / "world-gate-report.json",
            assess_world_gate(record),
        )

    def _world_checkpoint(self) -> Path:
        checkpoint = (
            self.root
            / "world_pretrain/checkpoints/world_pretrain-005000"
        )
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")
        return checkpoint

    def _audit_decision(self, *, enable_pcgrad: bool = True) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        conflicts = 25 if enable_pcgrad else 0
        decision = decide_gradient_audit(
            [
                AuditMeasurement(
                    260 + 10 * index,
                    -0.06 if index < conflicts else 0.0,
                )
                for index in range(25)
            ],
            0.004,
            0.005,
        )
        destination = self.root / "gradient-audit-decision.json"
        write_audit_decision(destination, decision)
        return destination

    def test_world_phase_is_fresh_and_fixed_to_5000(self) -> None:
        plan = resolve_conflict_fix_phase(
            self.config,
            "world_pretrain",
        )

        self.assertEqual(plan.stage, "world_pretrain")
        self.assertEqual(plan.target_step, 5000)
        self.assertEqual(plan.gradient_mode, "ordinary")
        self.assertEqual(plan.start_mode, "fresh")

    def test_full_resolved_config_is_json_serializable_without_deepcopy(self) -> None:
        payload = _json_safe(self.config)

        encoded = json.dumps(payload, sort_keys=True)
        self.assertIn("maniskill.pick_cube.conflict_fix.v2", encoded)

    def test_audit_requires_accepted_world_gate_and_canonical_checkpoint(self) -> None:
        self._accepted_world_gate()
        checkpoint = self._world_checkpoint()

        plan = resolve_conflict_fix_phase(
            self.config,
            "gradient_audit",
            init_checkpoint=checkpoint,
        )

        self.assertEqual(plan.stage, "unified")
        self.assertEqual(plan.target_step, 500)
        self.assertEqual(plan.gradient_mode, "audit")
        self.assertFalse(plan.save_final_checkpoint)

    def test_unified_gate_uses_immutable_audit_decision(self) -> None:
        self._accepted_world_gate()
        checkpoint = self._world_checkpoint()
        decision = self._audit_decision(enable_pcgrad=True)

        plan = resolve_conflict_fix_phase(
            self.config,
            "unified_gate",
            init_checkpoint=checkpoint,
            audit_decision=decision,
        )

        self.assertEqual(plan.gradient_mode, "pcgrad")
        self.assertEqual(plan.target_step, 5000)
        self.assertEqual(plan.start_mode, "warmstart")
        self.assertEqual(len(plan.audit_decision_sha256), 64)

    def test_continue_requires_exact_resume_and_existing_tracking_metadata(self) -> None:
        self._accepted_world_gate()
        decision = self._audit_decision(enable_pcgrad=False)
        unified_root = self.root / "unified"
        checkpoint = unified_root / "checkpoints/unified-005000"
        checkpoint.mkdir(parents=True)
        (checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")
        (unified_root / "tracking-metadata.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "entity": "test",
                    "project": "unified-corrective-foresight",
                    "group": "maniskill-pickcube-conflict-fix-v2",
                    "run_id": "run-test",
                    "mode": "online",
                    "last_optimizer_step": 5000,
                    "sync_complete": True,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "unified gate"):
            resolve_conflict_fix_phase(
                self.config,
                "unified_continue",
                resume=checkpoint,
                audit_decision=decision,
            )
        final_validation = ValidationRecord(
            global_step=5000,
            stage="unified",
            world_size=4,
            batches_per_rank=8,
            samples=64,
            metrics={
                "dynamics_loss": 1.0,
                "improvement_vs_copy_last": 0.1,
            },
        )
        gate = assess_unified_gate(
            final_validation=final_validation,
            world_dynamics_loss=1.0,
            stop_rule_fired=False,
            optimized_metrics_diverged=False,
            checkpoint_verified=True,
            resume_verified=True,
            baseline_successes=1,
            unified_successes=3,
            paired_reward_changes=(0.1,) * 10,
            actions_within_bounds=True,
            tracking_consistent=True,
        )
        write_unified_gate_report(
            self.root / "unified-gate-report.json",
            gate,
        )

        plan = resolve_conflict_fix_phase(
            self.config,
            "unified_continue",
            resume=checkpoint,
            audit_decision=decision,
        )

        self.assertEqual(plan.gradient_mode, "ordinary")
        self.assertEqual(plan.target_step, 20000)
        self.assertEqual(plan.start_mode, "resume")
        self.assertTrue(plan.resume_existing_output)

    def test_wrong_checkpoint_and_existing_fresh_output_fail_closed(self) -> None:
        self._accepted_world_gate()
        wrong = self.root / "wrong-checkpoint"
        wrong.mkdir()
        with self.assertRaisesRegex(ValueError, "canonical"):
            resolve_conflict_fix_phase(
                self.config,
                "gradient_audit",
                init_checkpoint=wrong,
            )

        (self.root / "world_pretrain").mkdir()
        with self.assertRaisesRegex(FileExistsError, "output"):
            resolve_conflict_fix_phase(
                self.config,
                "world_pretrain",
            )

    def test_launcher_and_verifier_are_syntax_valid_and_structured(self) -> None:
        launcher = PROJECT_ROOT / "scripts/launch_maniskill_conflict_fix.sh"
        verifier = PROJECT_ROOT / "scripts/verify_maniskill_conflict_fix.sh"
        for script in (launcher, verifier):
            self.assertTrue(script.is_file())
            subprocess.run(["bash", "-n", str(script)], check=True)
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("set -o noclobber", text)
        self.assertIn("--nproc_per_node=4", text)
        self.assertIn("gradient-audit-decision.json", text)
        self.assertIn("UCF_CONTINUE_TO_20000", text)
        self.assertNotIn("grep", text)


if __name__ == "__main__":
    unittest.main()
