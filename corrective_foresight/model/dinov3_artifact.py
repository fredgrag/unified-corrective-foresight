from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import re
from types import MappingProxyType
from typing import BinaryIO
from urllib.parse import quote
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MANIFEST_PATH = (
    Path(__file__).with_name("manifests") / "dinov3-vitb16-lvd1689m.json"
)
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models"
LOCAL_MANIFEST_NAME = "source-manifest.json"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_MODEL_ID_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)


class DinoV3ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DinoV3SourceManifest:
    canonical_model_id: str
    canonical_revision: str
    source_model_id: str
    source_revision: str
    files: Mapping[str, ArtifactFile]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> DinoV3SourceManifest:
        root = _strict_mapping(
            value,
            {"schema_version", "canonical", "source", "files"},
            "manifest",
        )
        if root["schema_version"] != 1:
            raise DinoV3ArtifactError("manifest schema_version must be 1")
        canonical = _strict_mapping(
            root["canonical"], {"provider", "model_id", "revision"}, "canonical"
        )
        source = _strict_mapping(
            root["source"], {"provider", "model_id", "revision"}, "source"
        )
        if canonical["provider"] != "huggingface":
            raise DinoV3ArtifactError("canonical provider must be huggingface")
        if source["provider"] != "modelscope":
            raise DinoV3ArtifactError("source provider must be modelscope")
        canonical_model_id = _model_id(canonical["model_id"], "canonical model_id")
        source_model_id = _model_id(source["model_id"], "source model_id")
        if canonical_model_id != source_model_id:
            raise DinoV3ArtifactError("canonical and source model ids must match")
        canonical_revision = _commit(canonical["revision"], "canonical revision")
        source_revision = _commit(source["revision"], "source revision")

        raw_files = root["files"]
        if not isinstance(raw_files, Mapping) or not raw_files:
            raise DinoV3ArtifactError("manifest files must be a nonempty mapping")
        parsed_files: dict[str, ArtifactFile] = {}
        for raw_name, raw_file in raw_files.items():
            name = _file_name(raw_name)
            file_mapping = _strict_mapping(raw_file, {"size", "sha256"}, name)
            size = file_mapping["size"]
            digest = file_mapping["sha256"]
            if type(size) is not int or size <= 0:
                raise DinoV3ArtifactError(f"{name} size must be a positive integer")
            if not isinstance(digest, str) or _DIGEST_PATTERN.fullmatch(digest) is None:
                raise DinoV3ArtifactError(f"{name} sha256 must be 64 lowercase hex digits")
            parsed_files[name] = ArtifactFile(size=size, sha256=digest)
        return cls(
            canonical_model_id=canonical_model_id,
            canonical_revision=canonical_revision,
            source_model_id=source_model_id,
            source_revision=source_revision,
            files=MappingProxyType(dict(sorted(parsed_files.items()))),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "canonical": {
                "provider": "huggingface",
                "model_id": self.canonical_model_id,
                "revision": self.canonical_revision,
            },
            "source": {
                "provider": "modelscope",
                "model_id": self.source_model_id,
                "revision": self.source_revision,
            },
            "files": {
                name: {"size": file.size, "sha256": file.sha256}
                for name, file in self.files.items()
            },
        }


OpenUrl = Callable[[str], AbstractContextManager[BinaryIO]]


def load_expected_manifest(
    path: Path = EXPECTED_MANIFEST_PATH,
) -> DinoV3SourceManifest:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DinoV3ArtifactError(f"cannot read expected manifest {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise DinoV3ArtifactError("expected manifest root must be a mapping")
    return DinoV3SourceManifest.from_mapping(value)


def default_snapshot_path(
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    manifest = load_expected_manifest()
    return artifact_root / Path(manifest.source_model_id) / manifest.source_revision


def install_snapshot(
    artifact_root: Path,
    manifest: DinoV3SourceManifest | None = None,
    open_url: OpenUrl = urlopen,
) -> Path:
    selected = manifest or load_expected_manifest()
    snapshot = artifact_root / Path(selected.source_model_id) / selected.source_revision
    snapshot.mkdir(parents=True, exist_ok=True)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise DinoV3ArtifactError(f"snapshot path is not a physical directory: {snapshot}")

    for name, expected in selected.files.items():
        destination = snapshot / name
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise DinoV3ArtifactError(f"{name} must not be a symbolic link")
            if not destination.is_file():
                raise DinoV3ArtifactError(f"{name} must be a regular file")
            try:
                _verify_file(destination, expected)
            except DinoV3ArtifactError:
                destination.unlink()
            else:
                continue
        _download_file(snapshot, name, expected, selected, open_url)

    manifest_bytes = _manifest_bytes(selected)
    manifest_path = snapshot / LOCAL_MANIFEST_NAME
    _write_atomic(manifest_path, manifest_bytes)
    verify_snapshot(snapshot, manifest=selected)
    return snapshot


def verify_snapshot(
    snapshot_dir: Path,
    manifest: DinoV3SourceManifest | None = None,
) -> None:
    selected = manifest or load_expected_manifest()
    if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
        raise DinoV3ArtifactError("snapshot must be a physical directory")
    expected_names = set(selected.files) | {LOCAL_MANIFEST_NAME}
    actual_names = {path.name for path in snapshot_dir.iterdir()}
    if actual_names != expected_names:
        raise DinoV3ArtifactError(
            "artifact file set mismatch: "
            f"expected {sorted(expected_names)}, got {sorted(actual_names)}"
        )

    local_manifest_path = snapshot_dir / LOCAL_MANIFEST_NAME
    if local_manifest_path.is_symlink() or not local_manifest_path.is_file():
        raise DinoV3ArtifactError("local manifest must be a physical regular file")
    try:
        local_value = json.loads(local_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(local_value, Mapping):
            raise DinoV3ArtifactError("local manifest root must be a mapping")
        local_manifest = DinoV3SourceManifest.from_mapping(local_value)
    except (OSError, json.JSONDecodeError, DinoV3ArtifactError) as error:
        raise DinoV3ArtifactError(f"local manifest is invalid: {error}") from error
    if local_manifest.to_mapping() != selected.to_mapping():
        raise DinoV3ArtifactError("local manifest does not match expected manifest")

    for name, expected in selected.files.items():
        _verify_file(snapshot_dir / name, expected)


def _download_file(
    snapshot: Path,
    name: str,
    expected: ArtifactFile,
    manifest: DinoV3SourceManifest,
    open_url: OpenUrl,
) -> None:
    partial = snapshot / f"{name}.partial"
    if partial.exists() or partial.is_symlink():
        if partial.is_symlink() or not partial.is_file():
            raise DinoV3ArtifactError(f"{partial.name} must be a regular file")
        partial.unlink()
    model_id = quote(manifest.source_model_id, safe="/")
    url = (
        f"https://modelscope.cn/models/{model_id}/resolve/"
        f"{manifest.source_revision}/{quote(name, safe='')}"
    )
    digest = sha256()
    size = 0
    try:
        with open_url(url) as response, partial.open("wb") as output:
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size != expected.size:
            raise DinoV3ArtifactError(
                f"{name} size mismatch: expected {expected.size}, got {size}"
            )
        actual_digest = digest.hexdigest()
        if actual_digest != expected.sha256:
            raise DinoV3ArtifactError(
                f"{name} digest mismatch: expected {expected.sha256}, got {actual_digest}"
            )
        os.replace(partial, snapshot / name)
    except DinoV3ArtifactError:
        partial.unlink(missing_ok=True)
        raise
    except Exception as error:
        partial.unlink(missing_ok=True)
        raise DinoV3ArtifactError(f"failed to download {name}: {error}") from error


def _write_atomic(destination: Path, content: bytes) -> None:
    partial = destination.with_name(f"{destination.name}.partial")
    if partial.exists() or partial.is_symlink():
        if partial.is_symlink() or not partial.is_file():
            raise DinoV3ArtifactError(f"{partial.name} must be a regular file")
        partial.unlink()
    try:
        with partial.open("wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial, destination)
    except Exception as error:
        partial.unlink(missing_ok=True)
        raise DinoV3ArtifactError(
            f"failed to write {destination.name}: {error}"
        ) from error


def _verify_file(path: Path, expected: ArtifactFile) -> None:
    if path.is_symlink():
        raise DinoV3ArtifactError(f"{path.name} must not be a symbolic link")
    if not path.is_file():
        raise DinoV3ArtifactError(f"{path.name} must be a regular file")
    size = path.stat().st_size
    if size != expected.size:
        raise DinoV3ArtifactError(
            f"{path.name} size mismatch: expected {expected.size}, got {size}"
        )
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    actual_digest = digest.hexdigest()
    if actual_digest != expected.sha256:
        raise DinoV3ArtifactError(
            f"{path.name} digest mismatch: expected {expected.sha256}, got {actual_digest}"
        )


def _manifest_bytes(manifest: DinoV3SourceManifest) -> bytes:
    text = json.dumps(manifest.to_mapping(), indent=2, sort_keys=True) + "\n"
    return text.encode("utf-8")


def _strict_mapping(
    value: object, expected_keys: set[str], context: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DinoV3ArtifactError(f"{context} must be a mapping")
    if set(value) != expected_keys:
        raise DinoV3ArtifactError(
            f"{context} fields must be exactly {sorted(expected_keys)}"
        )
    if not all(isinstance(key, str) for key in value):
        raise DinoV3ArtifactError(f"{context} field names must be strings")
    return value


def _model_id(value: object, context: str) -> str:
    if not isinstance(value, str) or _MODEL_ID_PATTERN.fullmatch(value) is None:
        raise DinoV3ArtifactError(f"{context} must be owner/name")
    return value


def _commit(value: object, context: str) -> str:
    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        raise DinoV3ArtifactError(f"{context} must be a full lowercase commit hash")
    return value


def _file_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or value == LOCAL_MANIFEST_NAME
        or value.endswith(".partial")
    ):
        raise DinoV3ArtifactError(f"unsafe artifact file name: {value!r}")
    return value
