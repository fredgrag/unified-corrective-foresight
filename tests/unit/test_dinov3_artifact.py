from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import unquote

from corrective_foresight.model.dinov3_artifact import (
    DinoV3ArtifactError,
    DinoV3SourceManifest,
    install_snapshot,
    load_expected_manifest,
    verify_snapshot,
)


def manifest_mapping(payloads: dict[str, bytes]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "canonical": {
            "provider": "huggingface",
            "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
        },
        "source": {
            "provider": "modelscope",
            "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "revision": "23d0280ae6ee4ced592a3459674ad027d3c18906",
        },
        "files": {
            name: {"size": len(payload), "sha256": sha256(payload).hexdigest()}
            for name, payload in payloads.items()
        },
    }


def tiny_manifest(payloads: dict[str, bytes]) -> DinoV3SourceManifest:
    return DinoV3SourceManifest.from_mapping(manifest_mapping(payloads))


class FakeOpen:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.urls: list[str] = []

    def __call__(self, url: str) -> BytesIO:
        self.urls.append(url)
        return BytesIO(self.payloads[unquote(url.rsplit("/", 1)[-1])])


class DinoV3ArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.payloads = {
            "config.json": b'{"model_type":"dinov3_vit"}',
            "model.safetensors": b"deterministic-test-weights",
        }
        self.manifest = tiny_manifest(self.payloads)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def install_valid(self) -> Path:
        return install_snapshot(
            self.root,
            manifest=self.manifest,
            open_url=FakeOpen(self.payloads),
        )

    def test_expected_manifest_pins_equivalent_modelscope_snapshot(self) -> None:
        manifest = load_expected_manifest()

        self.assertEqual(
            manifest.canonical_revision,
            "5931719e67bbdb9737e363e781fb0c67687896bc",
        )
        self.assertEqual(
            manifest.source_revision,
            "23d0280ae6ee4ced592a3459674ad027d3c18906",
        )
        self.assertEqual(
            manifest.files["model.safetensors"].sha256,
            "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
        )
        self.assertEqual(manifest.files["model.safetensors"].size, 342_662_192)

    def test_installs_from_full_revision_and_promotes_atomically(self) -> None:
        fake_open = FakeOpen(self.payloads)

        snapshot = install_snapshot(
            self.root,
            manifest=self.manifest,
            open_url=fake_open,
        )

        self.assertEqual(
            {path.name for path in snapshot.iterdir()},
            {"config.json", "model.safetensors", "source-manifest.json"},
        )
        self.assertTrue(
            all(self.manifest.source_revision in url for url in fake_open.urls)
        )
        self.assertTrue(all("/resolve/" in url for url in fake_open.urls))
        self.assertFalse(any("master" in url for url in fake_open.urls))
        self.assertFalse(any("partial" in path.name for path in snapshot.iterdir()))
        verify_snapshot(snapshot, manifest=self.manifest)

    def test_valid_existing_files_are_verified_and_reused(self) -> None:
        snapshot = self.install_valid()
        fake_open = FakeOpen(self.payloads)

        second = install_snapshot(
            self.root,
            manifest=self.manifest,
            open_url=fake_open,
        )

        self.assertEqual(second, snapshot)
        self.assertEqual(fake_open.urls, [])

    def test_corrupted_file_fails_closed(self) -> None:
        snapshot = self.install_valid()
        (snapshot / "config.json").write_bytes(b"x" * len(self.payloads["config.json"]))

        with self.assertRaisesRegex(DinoV3ArtifactError, "config.json.*digest"):
            verify_snapshot(snapshot, manifest=self.manifest)

    def test_wrong_size_download_is_not_promoted(self) -> None:
        wrong_payloads = dict(self.payloads)
        wrong_payloads["config.json"] += b"truncated-or-appended"

        with self.assertRaisesRegex(DinoV3ArtifactError, "config.json.*size"):
            install_snapshot(
                self.root,
                manifest=self.manifest,
                open_url=FakeOpen(wrong_payloads),
            )

        snapshot = (
            self.root / self.manifest.source_model_id / self.manifest.source_revision
        )
        self.assertFalse((snapshot / "config.json").exists())
        self.assertFalse(any(snapshot.glob("*.partial")))

    def test_missing_extra_and_changed_local_manifest_fail_closed(self) -> None:
        snapshot = self.install_valid()
        (snapshot / "config.json").unlink()
        with self.assertRaisesRegex(DinoV3ArtifactError, "file set"):
            verify_snapshot(snapshot, manifest=self.manifest)

        snapshot = self.install_valid()
        (snapshot / "extra.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(DinoV3ArtifactError, "file set"):
            verify_snapshot(snapshot, manifest=self.manifest)
        (snapshot / "extra.bin").unlink()

        local_manifest = json.loads((snapshot / "source-manifest.json").read_text())
        local_manifest["source"]["revision"] = "0" * 40
        (snapshot / "source-manifest.json").write_text(json.dumps(local_manifest))
        with self.assertRaisesRegex(DinoV3ArtifactError, "local manifest"):
            verify_snapshot(snapshot, manifest=self.manifest)

    def test_symlink_fails_closed_even_when_target_bytes_match(self) -> None:
        snapshot = self.install_valid()
        outside = self.root / "outside-config.json"
        outside.write_bytes(self.payloads["config.json"])
        (snapshot / "config.json").unlink()
        (snapshot / "config.json").symlink_to(outside)

        with self.assertRaisesRegex(DinoV3ArtifactError, "symbolic link"):
            verify_snapshot(snapshot, manifest=self.manifest)

    def test_manifest_parser_rejects_unsafe_or_mutable_fields(self) -> None:
        valid = manifest_mapping(self.payloads)
        mutations = {
            "wrong schema": lambda data: data.update(schema_version=2),
            "wrong canonical provider": lambda data: data["canonical"].update(
                provider="modelscope"
            ),
            "wrong source provider": lambda data: data["source"].update(
                provider="huggingface"
            ),
            "mutable revision": lambda data: data["source"].update(
                revision="master"
            ),
            "path traversal": lambda data: data["files"].update(
                {"../model": {"size": 1, "sha256": "0" * 64}}
            ),
            "nonpositive size": lambda data: data["files"]["config.json"].update(
                size=0
            ),
            "nonhex digest": lambda data: data["files"]["config.json"].update(
                sha256="z" * 64
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                candidate = deepcopy(valid)
                mutate(candidate)
                with self.assertRaises(DinoV3ArtifactError):
                    DinoV3SourceManifest.from_mapping(candidate)


if __name__ == "__main__":
    unittest.main()
