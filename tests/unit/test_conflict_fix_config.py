from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from corrective_foresight.config.conflict_fix import load_conflict_fix_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/pilots/maniskill_conflict_fix_v2.yaml"


class ConflictFixConfigTest(unittest.TestCase):
    def test_loads_approved_schedule_thresholds_and_tracking(self) -> None:
        config = load_conflict_fix_config(CONFIG_PATH)

        self.assertEqual(config.pilot_id, "maniskill.pick_cube.conflict_fix.v2")
        self.assertEqual(config.world_steps, 5000)
        self.assertEqual(config.audit_steps, 500)
        self.assertEqual(config.unified_gate_steps, 5000)
        self.assertEqual(config.unified_total_steps, 20000)
        self.assertEqual(config.validation_interval, 250)
        self.assertEqual(config.checkpoint_interval, 1000)
        self.assertEqual(config.validation_batches_per_rank, 8)
        self.assertEqual(config.protected_lr_multiplier, 0.0)
        self.assertEqual(config.conflict_log_interval, 10)
        self.assertEqual(config.conflict_cosine_threshold, -0.05)
        self.assertEqual(config.conflict_measurement_minimum, 8)
        self.assertEqual(config.dynamics_degradation_ratio, 1.2)
        self.assertEqual(config.world_copy_improvement_target, 0.05)
        self.assertEqual(config.required_success_gain, 2)
        self.assertEqual(config.evaluation_seeds, tuple(range(10)))
        self.assertEqual(config.gpu_indices, (0, 1, 2, 3))
        self.assertEqual(config.effective_global_batch, 64)
        self.assertEqual(config.world_config.training.warmup_steps, 500)
        self.assertEqual(config.world_config.training.total_steps, 5000)
        self.assertEqual(config.unified_config.training.warmup_steps, 500)
        self.assertEqual(config.unified_config.training.total_steps, 20000)
        self.assertEqual(config.tracking.project, "unified-corrective-foresight")
        self.assertEqual(
            config.tracking.group,
            "maniskill-pickcube-conflict-fix-v2",
        )
        self.assertEqual(config.tracking.log_interval, 10)
        self.assertFalse(config.tracking.upload_checkpoints)

    def test_rejects_unknown_field_and_noncanonical_seeds(self) -> None:
        original = CONFIG_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(dir=CONFIG_PATH.parent) as directory:
            unknown = Path(directory) / "unknown.yaml"
            unknown.write_text(original + "unknown: true\n", encoding="utf-8")
            bad_seeds = Path(directory) / "bad-seeds.yaml"
            bad_seeds.write_text(
                original.replace(
                    "evaluation_seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]",
                    "evaluation_seeds: [0, 2]",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "conflict-fix fields"):
                load_conflict_fix_config(unknown)
            with self.assertRaisesRegex(ValueError, "evaluation_seeds"):
                load_conflict_fix_config(bad_seeds)

    def test_rejects_mutated_approved_threshold(self) -> None:
        original = CONFIG_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(dir=CONFIG_PATH.parent) as directory:
            changed = Path(directory) / "changed.yaml"
            changed.write_text(
                original.replace(
                    "dynamics_degradation_ratio: 1.2",
                    "dynamics_degradation_ratio: 1.3",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "dynamics_degradation_ratio"):
                load_conflict_fix_config(changed)


if __name__ == "__main__":
    unittest.main()
