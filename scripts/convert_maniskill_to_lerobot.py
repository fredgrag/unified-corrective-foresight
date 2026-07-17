from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch
import yaml

from corrective_foresight.config.loader import load_action_spec, load_dataset_spec
from corrective_foresight.data.lerobot_adapter import LeRobotTrajectoryAdapter
from corrective_foresight.data.maniskill_conversion import (
    ManiSkillEpisodeReader,
    convert_to_lerobot_v3,
    expected_panda_ee_delta_pose_contract,
    extract_rgb_proprio,
    make_pick_cube_rgb_env,
    plan_maniskill_conversion,
    query_panda_ee_delta_pose_contract,
    split_source_episode_ids,
    validate_controller_contract,
    validate_converted_lerobot_dataset,
)


REQUIRED_SOURCE_FIELDS = {
    "schema_version",
    "source_h5",
    "source_json",
    "source_h5_sha256",
    "source_json_sha256",
    "output_root",
    "repo_id",
    "action_spec",
    "converter_git_commit",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pinned ManiSkill PickCube trajectories to LeRobot v3."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="DatasetSpec YAML; its .source.yaml sibling pins conversion inputs.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--write-specs", action="store_true")
    return parser.parse_args()


def _source_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}.source.yaml")


def _read_source_config(dataset_path: Path) -> dict[str, Any]:
    path = _source_path(dataset_path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    actual = set(value)
    if actual != REQUIRED_SOURCE_FIELDS:
        raise ValueError(
            f"{path} fields differ: missing={sorted(REQUIRED_SOURCE_FIELDS - actual)}, "
            f"unknown={sorted(actual - REQUIRED_SOURCE_FIELDS)}"
        )
    result = dict(value)
    if result["schema_version"] != 1:
        raise ValueError("unsupported source config schema_version")
    return result


def _git_root() -> Path:
    output = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(output).resolve()


def _resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _dump_spec(path: Path, spec: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = spec.to_dict()
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _assert_specs_equal(actual: object, expected: object, name: str) -> None:
    if actual.content_hash != expected.content_hash:
        differences = []
        for field in fields(type(actual)):
            left = getattr(actual, field.name)
            right = getattr(expected, field.name)
            if left != right:
                differences.append(f"{field.name}: configured={left!r}, expected={right!r}")
        raise ValueError(f"{name} content mismatch: {'; '.join(differences)}")


def _runtime_rgb_gate() -> None:
    env = make_pick_cube_rgb_env()
    try:
        contract = query_panda_ee_delta_pose_contract(env)
        validate_controller_contract(contract, expected_panda_ee_delta_pose_contract())
        observation, _ = env.reset(seed=1729)
        cameras, proprio = extract_rgb_proprio(observation)
        if proprio.shape != (25,) or set(cameras) != {"base", "wrist"}:
            raise ValueError("ManiSkill RGB runtime gate returned the wrong features")
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    project_root = _git_root()
    dataset_path = _resolve_project_path(str(args.config), project_root)
    source = _read_source_config(dataset_path)
    if source["converter_git_commit"] == "pending":
        raise ValueError("source config converter_git_commit has not been pinned")

    reader = ManiSkillEpisodeReader(source["source_h5"], source["source_json"])
    if reader.h5_sha256 != source["source_h5_sha256"]:
        raise ValueError("pinned source HDF5 SHA256 mismatch")
    if reader.json_sha256 != source["source_json_sha256"]:
        raise ValueError("pinned source JSON SHA256 mismatch")
    source_splits = split_source_episode_ids(reader.episode_ids, reader.h5_sha256)
    output_root = _resolve_project_path(source["output_root"], project_root)
    expected = plan_maniskill_conversion(
        reader,
        output_root,
        source_splits,
        source["converter_git_commit"],
    )
    action_path = _resolve_project_path(source["action_spec"], project_root)

    if args.write_specs:
        _dump_spec(dataset_path, expected.dataset_spec)
        _dump_spec(action_path, expected.action_spec)
        print(
            json.dumps(
                {
                    "dataset_spec": str(dataset_path),
                    "dataset_spec_hash": expected.dataset_spec.content_hash,
                    "action_spec": str(action_path),
                    "action_spec_hash": expected.action_spec.content_hash,
                },
                sort_keys=True,
            )
        )
        return

    configured_dataset = load_dataset_spec(dataset_path)
    configured_action = load_action_spec(action_path)
    _assert_specs_equal(configured_dataset, expected.dataset_spec, "DatasetSpec")
    _assert_specs_equal(configured_action, expected.action_spec, "ActionSpec")
    _runtime_rgb_gate()

    if args.validate_only:
        if not output_root.is_dir():
            raise FileNotFoundError(f"converted dataset is absent: {output_root}")
        expected_frames = sum(
            int(reader.episode_records[episode_id]["elapsed_steps"])
            for episode_id in reader.episode_ids
        )
        validation = validate_converted_lerobot_dataset(
            output_root,
            source["repo_id"],
            expected_episodes=len(reader.episode_ids),
            expected_frames=expected_frames,
        )
        for split_name in ("train", "validation", "evaluation"):
            adapter = LeRobotTrajectoryAdapter(
                configured_dataset,
                configured_action,
                split=split_name,
                video_backend="torchcodec",
            )
            sample = adapter[0]
            if not torch.isfinite(sample.rgb).all().item():
                raise ValueError(f"{split_name} adapter produced non-finite RGB")
        print(
            json.dumps(
                {
                    "status": "valid",
                    "episodes": len(reader.episode_ids),
                    "frames": expected_frames,
                    "valid_transitions": validation["valid_transitions"],
                    "successes": validation["successes"],
                    "videos": validation["videos"],
                    "dataset_spec_hash": configured_dataset.content_hash,
                    "action_spec_hash": configured_action.content_hash,
                },
                sort_keys=True,
            )
        )
        return

    result = convert_to_lerobot_v3(
        reader=reader,
        output_root=output_root,
        repo_id=source["repo_id"],
        source_splits=source_splits,
        converter_git_commit=source["converter_git_commit"],
    )
    _assert_specs_equal(result.dataset_spec, configured_dataset, "published DatasetSpec")
    _assert_specs_equal(result.action_spec, configured_action, "published ActionSpec")
    print(
        json.dumps(
            {
                "status": "published",
                "root": str(result.output_root),
                "episodes": len(reader.episode_ids),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
