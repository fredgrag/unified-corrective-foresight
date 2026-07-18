from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from corrective_foresight.training.pilot import PilotController
from corrective_foresight.training.validation import ValidationRecord


def _record(step: int) -> ValidationRecord:
    return ValidationRecord(
        global_step=step,
        stage="unified",
        world_size=1,
        batches_per_rank=8,
        samples=16,
        metrics={"action_loss": 1.0, "dynamics_loss": 2.0},
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


if __name__ == "__main__":
    unittest.main()
