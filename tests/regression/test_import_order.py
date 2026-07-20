from __future__ import annotations

import subprocess
import sys
import unittest


class ImportOrderRegressionTest(unittest.TestCase):
    def test_model_can_be_imported_before_conflict_fix_config(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from corrective_foresight.model.objectives import "
                    "DEFAULT_OPTIMIZED_TERMS; "
                    "from corrective_foresight.config.conflict_fix import "
                    "load_conflict_fix_config"
                ),
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_conflict_fix_config_can_be_imported_before_model(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from corrective_foresight.config.conflict_fix import "
                    "load_conflict_fix_config; "
                    "from corrective_foresight.model.objectives import "
                    "DEFAULT_OPTIMIZED_TERMS"
                ),
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
