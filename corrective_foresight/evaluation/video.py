from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import uuid

import av
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoArtifact:
    path: Path
    sha256: str
    frames: int
    fps: float


def write_rollout_video_atomic(
    path: str | Path,
    frames: Sequence[np.ndarray],
    *,
    fps: float,
) -> VideoArtifact:
    destination = Path(path)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("rollout video path must end in .mp4")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"rollout video already exists: {destination}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("rollout video FPS must be finite and positive")
    if not frames:
        raise ValueError("rollout video requires at least one frame")
    first_shape = tuple(frames[0].shape)
    if len(first_shape) != 3 or first_shape[2] != 3:
        raise ValueError("rollout frames must be HWC RGB")
    if first_shape[0] % 2 or first_shape[1] % 2:
        raise ValueError("rollout frame height and width must be even")
    for frame in frames:
        if frame.dtype != np.uint8 or tuple(frame.shape) != first_shape:
            raise ValueError("rollout frames must share one uint8 HWC RGB shape")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}.mp4"
    try:
        with av.open(str(temporary), mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=Fraction(str(float(fps))))
            stream.width = first_shape[1]
            stream.height = first_shape[0]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}
            for frame in frames:
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.rename(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return VideoArtifact(
        path=destination,
        sha256=_sha256_file(destination),
        frames=len(frames),
        fps=float(fps),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
