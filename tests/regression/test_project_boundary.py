from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_TRACKED_ROOTS = {
    ".venv",
    "artifacts",
    "checkpoints",
    "data",
    "datasets",
    "logs",
    "outputs",
    "wandb",
}


class ProjectBoundaryTest(unittest.TestCase):
    def test_package_is_owned_by_isolated_project(self) -> None:
        package = importlib.import_module("corrective_foresight")
        package_path = Path(package.__file__).resolve()

        self.assertTrue(package_path.is_relative_to(PROJECT_ROOT), package_path)

    def test_project_contains_no_symbolic_links(self) -> None:
        links = [
            path.relative_to(PROJECT_ROOT)
            for path in PROJECT_ROOT.rglob("*")
            if ".git" not in path.parts and path.is_symlink()
        ]

        self.assertEqual(links, [])

    def test_git_tracks_no_runtime_artifacts(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
        tracked = [Path(value.decode()) for value in result.stdout.split(b"\0") if value]
        violations = [
            path
            for path in tracked
            if path.name.startswith("._")
            or path.parts[0] in FORBIDDEN_TRACKED_ROOTS
            or "__pycache__" in path.parts
        ]

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
