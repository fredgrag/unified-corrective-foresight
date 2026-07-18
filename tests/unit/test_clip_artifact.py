from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from corrective_foresight.conditioning.clip_artifact import verify_clip_snapshot


class ClipArtifactTest(unittest.TestCase):
    def test_verifies_exact_physical_files_and_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            files = {
                "config.json": b"config",
                "pytorch_model.bin": b"weights",
                "tokenizer.json": b"tokenizer",
            }
            for name, value in files.items():
                (snapshot / name).write_bytes(value)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "model_id": "openai/clip-vit-base-patch32",
                        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                        "files": {
                            name: {
                                "size": len(value),
                                "sha256": hashlib.sha256(value).hexdigest(),
                            }
                            for name, value in files.items()
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            verify_clip_snapshot(snapshot, manifest_path=manifest)
            (snapshot / "config.json").write_bytes(b"damage")
            with self.assertRaisesRegex(ValueError, "(size|SHA256).*config.json"):
                verify_clip_snapshot(snapshot, manifest_path=manifest)

    def test_rejects_extra_files_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            value = b"config"
            (snapshot / "config.json").write_bytes(value)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "model_id": "openai/clip-vit-base-patch32",
                        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                        "files": {
                            "config.json": {
                                "size": len(value),
                                "sha256": hashlib.sha256(value).hexdigest(),
                            }
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (snapshot / "extra").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "files must be exactly"):
                verify_clip_snapshot(snapshot, manifest_path=manifest)
            (snapshot / "extra").unlink()
            (snapshot / "config.json").unlink()
            (snapshot / "config.json").symlink_to(root / "target")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                verify_clip_snapshot(snapshot, manifest_path=manifest)


if __name__ == "__main__":
    unittest.main()
