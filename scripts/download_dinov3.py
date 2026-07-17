#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from corrective_foresight.model.dinov3_artifact import (
    DEFAULT_ARTIFACT_ROOT,
    install_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned ModelScope DINOv3 snapshot."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Root under which the immutable provider/model/revision tree is stored.",
    )
    arguments = parser.parse_args()
    snapshot = install_snapshot(arguments.artifact_root)
    print(snapshot)


if __name__ == "__main__":
    main()
