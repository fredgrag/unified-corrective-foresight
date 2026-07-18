from __future__ import annotations

from pathlib import Path
import unittest

from train import parse_args


class TrainEntrypointTest(unittest.TestCase):
    def test_parser_freezes_config_stage_and_checkpoint_controls(self) -> None:
        args = parse_args(
            [
                "--config",
                "configs/experiments/maniskill_unified.yaml",
                "--stage",
                "unified",
                "--steps",
                "12",
                "--init-checkpoint",
                "/tmp/world-pretrain",
                "--output-checkpoint",
                "/tmp/unified-checkpoint",
            ]
        )

        self.assertEqual(args.config, Path("configs/experiments/maniskill_unified.yaml"))
        self.assertEqual(args.stage, "unified")
        self.assertEqual(args.steps, 12)
        self.assertEqual(args.init_checkpoint, Path("/tmp/world-pretrain"))
        self.assertEqual(args.output_checkpoint, Path("/tmp/unified-checkpoint"))

    def test_parser_rejects_nonpositive_steps(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--config",
                    "configs/experiments/maniskill_unified.yaml",
                    "--steps",
                    "0",
                ]
            )


if __name__ == "__main__":
    unittest.main()
