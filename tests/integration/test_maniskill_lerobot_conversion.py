from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from corrective_foresight.data.lerobot_adapter import LeRobotTrajectoryAdapter
from corrective_foresight.data.maniskill_conversion import (
    MANISKILL_ACTION_NAMES,
    ManiSkillEpisodeReader,
    controller_action_to_physical,
    convert_to_lerobot_v3,
    expected_panda_ee_delta_pose_contract,
    extract_rgb_proprio,
    make_pick_cube_rgb_env,
    query_panda_ee_delta_pose_contract,
    step_maniskill_env,
)


EPISODE_LENGTH = 4
SOURCE_EPISODE_IDS = (11, 22, 33)


def _unbatch_tree(value):
    if isinstance(value, dict):
        return {key: _unbatch_tree(child) for key, child in value.items()}
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return array[0].copy()


def _stack_trees(values):
    if isinstance(values[0], dict):
        return {key: _stack_trees([value[key] for value in values]) for key in values[0]}
    return np.stack(values)


def _write_hdf5_tree(group: h5py.Group, value: dict) -> None:
    for key, child in value.items():
        if isinstance(child, dict):
            _write_hdf5_tree(group.create_group(key), child)
        else:
            group.create_dataset(key, data=child)


def _raw_actions(episode_position: int) -> np.ndarray:
    base = np.asarray(
        [
            [0.20, -0.30, 0.10, 0.20, -0.10, 0.30, -0.80],
            [-0.40, 0.10, 0.30, -0.20, 0.40, 0.10, 0.70],
            [0.10, 0.40, -0.20, 0.30, 0.20, -0.40, -0.20],
            [-0.30, -0.20, 0.40, -0.10, -0.30, 0.20, 0.40],
        ],
        dtype=np.float32,
    )
    return base * (1.0 - episode_position * 0.15)


def create_recorded_source(root: Path):
    h5_path = root / "trajectory.h5"
    json_path = root / "trajectory.json"
    expected = {}
    records = []
    env = make_pick_cube_rgb_env()
    contract = query_panda_ee_delta_pose_contract(env)
    try:
        self_contract = expected_panda_ee_delta_pose_contract()
        if contract != self_contract:
            raise AssertionError((contract, self_contract))
        with h5py.File(h5_path, "w") as file:
            for position, episode_id in enumerate(SOURCE_EPISODE_IDS):
                actions = _raw_actions(position)
                observation, _ = env.reset(seed=episode_id)
                states = [_unbatch_tree(env.unwrapped.get_state_dict())]
                cameras = []
                proprios = []
                successes = []
                terminated_values = []
                truncated_values = []
                for action in actions:
                    rgb, proprio = extract_rgb_proprio(observation)
                    cameras.append(rgb)
                    proprios.append(proprio)
                    observation, _, terminated, truncated, info = step_maniskill_env(
                        env,
                        torch.as_tensor(action)
                    )
                    states.append(_unbatch_tree(env.unwrapped.get_state_dict()))
                    successes.append(bool(info["success"].item()))
                    terminated_values.append(bool(terminated.item()))
                    truncated_values.append(bool(truncated.item()))

                group = file.create_group(f"traj_{episode_id}")
                group.create_dataset("actions", data=actions)
                _write_hdf5_tree(group.create_group("env_states"), _stack_trees(states))
                group.create_dataset("success", data=np.asarray(successes, dtype=np.bool_))
                group.create_dataset(
                    "terminated", data=np.asarray(terminated_values, dtype=np.bool_)
                )
                group.create_dataset(
                    "truncated", data=np.asarray(truncated_values, dtype=np.bool_)
                )
                records.append(
                    {
                        "episode_id": episode_id,
                        "episode_seed": episode_id,
                        "control_mode": "pd_ee_delta_pose",
                        "elapsed_steps": EPISODE_LENGTH,
                        "reset_kwargs": {},
                        "success": successes[-1],
                    }
                )
                expected[episode_id] = {
                    "rgb": cameras,
                    "proprio": np.stack(proprios),
                    "action": controller_action_to_physical(actions, contract),
                    "success": np.asarray(successes, dtype=np.bool_),
                }
    finally:
        env.close()

    metadata = {
        "env_info": {
            "env_id": "PickCube-v1",
            "env_kwargs": {
                "control_mode": "pd_ee_delta_pose",
                "obs_mode": "state",
                "sim_backend": "physx_cpu",
            },
            "max_episode_steps": 50,
        },
        "commit_info": {"commit_id": "integration-fixture"},
        "episodes": records,
        "source_type": "integration_fixture",
        "source_desc": "deterministic RGB conversion fixture",
    }
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    return h5_path, json_path, expected


class ManiSkillLeRobotConversionIntegrationTest(unittest.TestCase):
    def test_real_rgb_replay_lerobot_v3_round_trip(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path, json_path, expected = create_recorded_source(root)
            reader = ManiSkillEpisodeReader(h5_path, json_path)
            output_root = root / "published"
            result = convert_to_lerobot_v3(
                reader=reader,
                output_root=output_root,
                repo_id="fixtures/maniskill_pick_cube_v1",
                source_splits={
                    "train": (11,),
                    "validation": (22,),
                    "evaluation": (33,),
                },
                converter_git_commit="a" * 40,
            )

            self.assertEqual(result.dataset_spec.split_episode_ids["train"], (0,))
            self.assertEqual(result.dataset_spec.split_episode_ids["validation"], (1,))
            self.assertEqual(result.dataset_spec.split_episode_ids["evaluation"], (2,))
            self.assertEqual(result.action_spec.names, MANISKILL_ACTION_NAMES)
            self.assertEqual(result.action_spec.rotation_representation, "axis_angle")
            self.assertEqual(result.action_spec.frequency_hz, 20.0)
            self.assertEqual(result.provenance["source"]["h5_sha256"], reader.h5_sha256)

            reopened = LeRobotDataset(
                repo_id="fixtures/maniskill_pick_cube_v1",
                root=output_root,
                video_backend="torchcodec",
            )
            self.assertEqual(reopened.meta.total_episodes, 3)
            self.assertEqual(reopened.meta.total_frames, 3 * EPISODE_LENGTH)
            video_paths = tuple(output_root.rglob("*.mp4"))
            self.assertEqual(len(video_paths), 2)
            self.assertTrue(
                any("observation.images.base" in str(path) for path in video_paths)
            )
            self.assertTrue(
                any("observation.images.wrist" in str(path) for path in video_paths)
            )
            self.assertFalse(any(path.name.startswith(".published.tmp") for path in root.iterdir()))
            self.assertEqual(reopened.meta.info["ucf"]["provenance"], result.provenance)

            item = reopened[0]
            base = item["observation.images.base"].permute(1, 2, 0).numpy()
            wrist = item["observation.images.wrist"].permute(1, 2, 0).numpy()
            expected_base = expected[11]["rgb"][0]["base"].astype(np.float32) / 255.0
            expected_wrist = expected[11]["rgb"][0]["wrist"].astype(np.float32) / 255.0
            self.assertLess(float(np.abs(base - expected_base).mean()), 0.04)
            self.assertLess(float(np.abs(wrist - expected_wrist).mean()), 0.04)
            np.testing.assert_allclose(
                item["observation.state"].numpy(), expected[11]["proprio"][0], atol=1e-6
            )
            np.testing.assert_allclose(
                item["action"].numpy(), expected[11]["action"][0], atol=1e-6
            )
            self.assertEqual(bool(item["next.success"].item()), expected[11]["success"][0])
            self.assertAlmostEqual(float(item["timestamp"]), 0.0, places=7)

            adapter = LeRobotTrajectoryAdapter(
                result.dataset_spec,
                result.action_spec,
                split="train",
                context_steps=1,
                action_horizon=2,
                video_backend="torchcodec",
            )
            sample = adapter[0]
            self.assertEqual(sample.rgb.shape, (3, 2, 3, 128, 128))
            self.assertEqual(sample.proprio.shape, (3, 25))
            self.assertEqual(sample.action.shape, (2, 7))
            self.assertTrue(sample.observation_valid_mask.all().item())
            self.assertTrue(sample.action_valid_mask.all().item())
            self.assertTrue(sample.transition_valid_mask.all().item())

            info_before = (output_root / "meta" / "info.json").read_bytes()
            with self.assertRaises(FileExistsError):
                convert_to_lerobot_v3(
                    reader=reader,
                    output_root=output_root,
                    repo_id="fixtures/maniskill_pick_cube_v1",
                    source_splits={
                        "train": (11,),
                        "validation": (22,),
                        "evaluation": (33,),
                    },
                    converter_git_commit="a" * 40,
                )
            self.assertEqual(
                (output_root / "meta" / "info.json").read_bytes(), info_before
            )


if __name__ == "__main__":
    unittest.main()
