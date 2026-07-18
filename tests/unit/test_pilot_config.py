from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from corrective_foresight.config.pilot import load_pilot_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = PROJECT_ROOT / "configs/pilots/maniskill_stable_v1.yaml"


class PilotConfigTest(unittest.TestCase):
    def test_loads_approved_stable_pilot_contract(self) -> None:
        config = load_pilot_config(PILOT_PATH)
        self.assertEqual(config.pilot_id, "maniskill.pick_cube.stable.v1")
        self.assertEqual(config.world_steps, 1000)
        self.assertEqual(config.unified_steps, 2000)
        self.assertEqual(config.evaluation_steps, (0, 1000, 2000))
        self.assertEqual(config.evaluation_seeds, tuple(range(10)))
        self.assertEqual(config.gpu_indices, (0, 1, 2, 3))
        self.assertEqual(config.effective_global_batch, 64)
        self.assertEqual(config.world_config.model, config.unified_config.model)
        self.assertEqual(
            tuple(spec.content_hash for spec in config.world_dataset_specs),
            tuple(spec.content_hash for spec in config.unified_dataset_specs),
        )

    def test_rejects_unknown_fields_and_non_absolute_output(self) -> None:
        original = PILOT_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            unknown = Path(directory) / "unknown.yaml"
            unknown.write_text(original + "unexpected: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pilot fields"):
                load_pilot_config(unknown)

            relative = Path(directory) / "relative.yaml"
            relative.write_text(
                original.replace(
                    "output_root: /mnt/workspace/wwl/corrective-foresight/"
                    "unified_corrective_foresight/artifacts/runs/maniskill_pilot_20260718",
                    "output_root: relative/pilot",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "output_root.*absolute"):
                load_pilot_config(relative)

    def test_rejects_intervals_that_do_not_divide_both_stages(self) -> None:
        original = PILOT_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-interval.yaml"
            path.write_text(
                original.replace("checkpoint_interval: 250", "checkpoint_interval: 300"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "divide"):
                load_pilot_config(path)


if __name__ == "__main__":
    unittest.main()
