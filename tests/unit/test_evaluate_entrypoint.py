from __future__ import annotations

import itertools
from pathlib import Path
import unittest

from evaluate import flow_seed_stream, parse_args


class EvaluateEntrypointTest(unittest.TestCase):
    def test_flow_seed_stream_is_reproducible_and_separate_from_environment_seed(self) -> None:
        first = tuple(itertools.islice(flow_seed_stream(101, 7), 8))
        repeated = tuple(itertools.islice(flow_seed_stream(101, 7), 8))
        other_environment = tuple(itertools.islice(flow_seed_stream(101, 8), 8))

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_environment)
        self.assertTrue(all(type(value) is int and value >= 0 for value in first))

    def test_parser_requires_optimizer_step_with_conflict_fix_tracking(self) -> None:
        args = parse_args(
            [
                "--config",
                "configs/experiments/maniskill_unified_conflict_fix.yaml",
                "--checkpoint",
                "/tmp/unified-005000",
                "--seeds",
                *[str(seed) for seed in range(10)],
                "--tag",
                "unified_5000",
                "--conflict-fix-config",
                "configs/pilots/maniskill_conflict_fix_v2.yaml",
                "--optimizer-step",
                "5000",
            ]
        )

        self.assertEqual(args.optimizer_step, 5000)
        self.assertEqual(
            args.conflict_fix_config,
            Path("configs/pilots/maniskill_conflict_fix_v2.yaml"),
        )
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--config",
                    "configs/experiments/maniskill_unified_conflict_fix.yaml",
                    "--checkpoint",
                    "/tmp/unified-005000",
                    "--seeds",
                    "0",
                    "--tag",
                    "unified_5000",
                    "--conflict-fix-config",
                    "configs/pilots/maniskill_conflict_fix_v2.yaml",
                ]
            )


if __name__ == "__main__":
    unittest.main()
