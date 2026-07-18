from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    PROJECT_ROOT / "corrective_foresight",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "configs",
    PROJECT_ROOT / "train.py",
    PROJECT_ROOT / "evaluate.py",
)
_CREDENTIAL = re.compile(r"AKIA[0-9A-Z]{16}|BEGIN (?:RSA |EC )?PRIVATE KEY|hf_[A-Za-z0-9]{20,}")


class ProjectBoundaryRegressionTest(unittest.TestCase):
    def test_source_has_no_links_appledouble_credentials_or_generated_payloads(self) -> None:
        for root in SOURCE_ROOTS:
            paths = (root,) if root.is_file() else root.rglob("*")
            for path in paths:
                if "__pycache__" in path.parts:
                    continue
                self.assertFalse(path.is_symlink(), path)
                self.assertFalse(path.name.startswith("._"), path)
                self.assertNotIn(path.suffix, {".pt", ".mp4", ".safetensors"}, path)
                if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".json"}:
                    self.assertIsNone(_CREDENTIAL.search(path.read_text(encoding="utf-8")), path)

    def test_python_imports_are_project_or_external_package_only(self) -> None:
        forbidden_prefixes = (
            "unified_video_action",
            "parent_legacy",
            "corrective_foresight_legacy",
        )
        for path in (PROJECT_ROOT / "corrective_foresight").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            for module in imported:
                self.assertFalse(module.startswith(forbidden_prefixes), (path, module))


if __name__ == "__main__":
    unittest.main()
