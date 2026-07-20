from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from corrective_foresight.evaluation.conflict_fix_report import (
    ValidationCandidate,
    plan_checkpoint_retention,
    select_best_validation,
)


class ConflictFixEvaluationSelectionTest(unittest.TestCase):
    def test_positive_copy_improvement_precedes_lowest_dynamics(self) -> None:
        records = (
            ValidationCandidate(250, -0.1, 0.001),
            ValidationCandidate(500, 0.01, 0.010),
            ValidationCandidate(750, 0.02, 0.008),
            ValidationCandidate(1000, 0.03, 0.009),
        )

        best = select_best_validation(records)

        self.assertEqual(best.optimizer_step, 750)

    def test_selection_fails_when_no_positive_copy_improvement_exists(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            select_best_validation(
                (
                    ValidationCandidate(250, 0.0, 0.001),
                    ValidationCandidate(500, -0.1, 0.0005),
                )
            )

    def test_retention_plan_keeps_initial_best_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in ("world_pretrain", "unified"):
                checkpoint_root = root / stage / "checkpoints"
                validation_root = root / stage / "validation"
                checkpoint_root.mkdir(parents=True)
                validation_root.mkdir()
                for step, improvement, dynamics in (
                    (0, -0.1, 1.0),
                    (1000, 0.01, 0.2),
                    (2000, 0.02, 0.1),
                    (5000, 0.03, 0.15),
                ):
                    (checkpoint_root / f"{stage}-{step:06d}").mkdir()
                    (validation_root / f"{stage}-{step:06d}.json").write_text(
                        json.dumps(
                            {
                                "metrics": {
                                    "improvement_vs_copy_last": improvement,
                                    "dynamics_loss": dynamics,
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
            audit = root / "gradient_audit/checkpoints/unified-000500"
            audit.mkdir(parents=True)

            plan = plan_checkpoint_retention(root)

            retained_names = {path.name for path in plan.retained}
            self.assertEqual(
                retained_names,
                {
                    "world_pretrain-000000",
                    "world_pretrain-002000",
                    "world_pretrain-005000",
                    "unified-000000",
                    "unified-002000",
                    "unified-005000",
                },
            )
            self.assertIn("world_pretrain-001000", {p.name for p in plan.delete})
            self.assertIn("unified-000500", {p.name for p in plan.delete})


if __name__ == "__main__":
    unittest.main()
