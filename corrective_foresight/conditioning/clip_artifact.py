from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = (
    Path(__file__).with_name("manifests") / "clip-vit-base-patch32.json"
)
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ClipArtifactError(ValueError):
    pass


def default_clip_snapshot_path(
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    return Path(artifact_root) / Path(CLIP_MODEL_ID) / CLIP_REVISION


def verify_clip_snapshot(
    snapshot_dir: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> None:
    snapshot = Path(snapshot_dir)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ClipArtifactError("CLIP snapshot must be a physical directory")
    manifest = _load_manifest(Path(manifest_path))
    expected_files = set(manifest["files"])
    actual_files = {entry.name for entry in snapshot.iterdir()}
    if actual_files != expected_files:
        raise ClipArtifactError(
            "CLIP snapshot files must be exactly "
            f"{sorted(expected_files)}, got {sorted(actual_files)}"
        )
    for name, expected in manifest["files"].items():
        path = snapshot / name
        if path.is_symlink():
            raise ClipArtifactError(f"CLIP snapshot cannot contain symbolic link: {name}")
        if not path.is_file():
            raise ClipArtifactError(f"CLIP snapshot entry is not a physical file: {name}")
        size = path.stat().st_size
        if size != expected["size"]:
            raise ClipArtifactError(
                f"CLIP size mismatch for {name}: expected {expected['size']}, got {size}"
            )
        digest = _file_sha256(path)
        if digest != expected["sha256"]:
            raise ClipArtifactError(
                f"CLIP SHA256 mismatch for {name}: expected {expected['sha256']}, got {digest}"
            )


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClipArtifactError(f"cannot read CLIP manifest: {path}") from error
    if not isinstance(value, Mapping) or set(value) != {
        "format_version",
        "model_id",
        "revision",
        "files",
    }:
        raise ClipArtifactError("CLIP manifest fields are invalid")
    if (
        value["format_version"] != 1
        or value["model_id"] != CLIP_MODEL_ID
        or value["revision"] != CLIP_REVISION
    ):
        raise ClipArtifactError("CLIP manifest identity or revision is invalid")
    files = value["files"]
    if not isinstance(files, Mapping) or not files:
        raise ClipArtifactError("CLIP manifest files must be a nonempty mapping")
    normalized: dict[str, dict[str, object]] = {}
    for name, metadata in files.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
            or not isinstance(metadata, Mapping)
            or set(metadata) != {"size", "sha256"}
            or type(metadata["size"]) is not int
            or metadata["size"] <= 0
            or not isinstance(metadata["sha256"], str)
            or _SHA256_PATTERN.fullmatch(metadata["sha256"]) is None
        ):
            raise ClipArtifactError(f"invalid CLIP manifest entry: {name!r}")
        normalized[name] = {
            "size": metadata["size"],
            "sha256": metadata["sha256"],
        }
    return {
        "format_version": 1,
        "model_id": CLIP_MODEL_ID,
        "revision": CLIP_REVISION,
        "files": normalized,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
