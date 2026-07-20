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

    def test_parser_accepts_fresh_conflict_fix_world_phase(self) -> None:
        args = parse_args(
            [
                "--conflict-fix-config",
                "configs/pilots/maniskill_conflict_fix_v2.yaml",
                "--conflict-fix-phase",
                "world_pretrain",
            ]
        )

        self.assertEqual(args.conflict_fix_phase, "world_pretrain")
        self.assertIsNone(args.init_checkpoint)
        self.assertIsNone(args.resume)

    def test_parser_rejects_legacy_mix_and_schedule_override(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--conflict-fix-config",
                    "configs/pilots/maniskill_conflict_fix_v2.yaml",
                    "--conflict-fix-phase",
                    "world_pretrain",
                    "--config",
                    "configs/experiments/maniskill_world_pretrain_conflict_fix.yaml",
                ]
            )
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--conflict-fix-config",
                    "configs/pilots/maniskill_conflict_fix_v2.yaml",
                    "--conflict-fix-phase",
                    "world_pretrain",
                    "--steps",
                    "10",
                ]
            )

    def test_parser_enforces_phase_checkpoint_and_decision_inputs(self) -> None:
        base = [
            "--conflict-fix-config",
            "configs/pilots/maniskill_conflict_fix_v2.yaml",
        ]
        with self.assertRaises(SystemExit):
            parse_args([*base, "--conflict-fix-phase", "gradient_audit"])
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    *base,
                    "--conflict-fix-phase",
                    "unified_gate",
                    "--init-checkpoint",
                    "/tmp/world-005000",
                ]
            )
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    *base,
                    "--conflict-fix-phase",
                    "unified_continue",
                    "--audit-decision",
                    "/tmp/gradient-audit-decision.json",
                ]
            )

        continuation = parse_args(
            [
                *base,
                "--conflict-fix-phase",
                "unified_continue",
                "--resume",
                "/tmp/unified-005000",
                "--audit-decision",
                "/tmp/gradient-audit-decision.json",
            ]
        )
        self.assertEqual(continuation.resume, Path("/tmp/unified-005000"))


if __name__ == "__main__":
    unittest.main()
