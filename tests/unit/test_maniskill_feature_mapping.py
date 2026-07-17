from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from corrective_foresight.data.maniskill_conversion import (
    MANISKILL_ACTION_NAMES,
    MANISKILL_ACTION_UNITS,
    ManiSkillEpisodeReader,
    PandaEEDeltaPoseContract,
    compute_action_statistics,
    controller_action_to_physical,
    physical_action_to_controller,
    remap_source_articulation_keys,
    split_source_episode_ids,
    validate_controller_contract,
)


def make_contract() -> PandaEEDeltaPoseContract:
    return PandaEEDeltaPoseContract(
        control_mode="pd_ee_delta_pose",
        frequency_hz=20.0,
        dimension=7,
        frame="root_translation:root_aligned_body_rotation",
        translation_lower=(-0.1, -0.1, -0.1),
        translation_upper=(0.1, 0.1, 0.1),
        rotation_lower=(-0.1, -0.1, -0.1),
        rotation_upper=(0.1, 0.1, 0.1),
        gripper_lower=-0.01,
        gripper_upper=0.04,
        normalized_action=True,
    )


def write_source_fixture(root: Path, elapsed_steps: int = 2) -> tuple[Path, Path]:
    h5_path = root / "trajectory.h5"
    json_path = root / "trajectory.json"
    with h5py.File(h5_path, "w") as file:
        trajectory = file.create_group("traj_7")
        trajectory.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
        states = trajectory.create_group("env_states")
        actors = states.create_group("actors")
        actors.create_dataset("cube", data=np.zeros((3, 13), dtype=np.float32))
        articulations = states.create_group("articulations")
        articulations.create_dataset("panda", data=np.zeros((3, 31), dtype=np.float32))
        trajectory.create_dataset("success", data=np.asarray([False, True]))
        trajectory.create_dataset("terminated", data=np.asarray([False, False]))
        trajectory.create_dataset("truncated", data=np.asarray([False, True]))
    metadata = {
        "env_info": {
            "env_id": "PickCube-v1",
            "env_kwargs": {
                "control_mode": "pd_ee_delta_pose",
                "obs_mode": "state",
                "sim_backend": "physx_cuda",
            },
            "max_episode_steps": 50,
        },
        "commit_info": {"commit_id": "fixture"},
        "episodes": [
            {
                "episode_id": 7,
                "episode_seed": 1729,
                "control_mode": "pd_ee_delta_pose",
                "elapsed_steps": elapsed_steps,
                "reset_kwargs": {},
                "success": True,
            }
        ],
        "source_type": "fixture",
        "source_desc": "unit test",
    }
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    return h5_path, json_path


class ManiSkillFeatureMappingTest(unittest.TestCase):
    def test_freezes_physical_action_names_and_units(self) -> None:
        self.assertEqual(
            MANISKILL_ACTION_NAMES,
            (
                "ee_delta_x",
                "ee_delta_y",
                "ee_delta_z",
                "ee_rotation_vector_x",
                "ee_rotation_vector_y",
                "ee_rotation_vector_z",
                "gripper_position",
            ),
        )
        self.assertEqual(
            MANISKILL_ACTION_UNITS,
            ("m", "m", "m", "rad", "rad", "rad", "m"),
        )

    def test_maps_unbounded_policy_output_to_executed_physical_action(self) -> None:
        raw = np.asarray([2.0, -0.5, 0.0, 2.0, 0.0, 0.0, -2.0], dtype=np.float32)

        physical = controller_action_to_physical(raw, make_contract())

        np.testing.assert_allclose(
            physical,
            np.asarray([0.1, -0.05, 0.0, -0.1, 0.0, 0.0, -0.01]),
            atol=1e-6,
        )
        canonical = physical_action_to_controller(physical, make_contract())
        np.testing.assert_allclose(
            canonical,
            np.asarray([1.0, -0.5, 0.0, 1.0, 0.0, 0.0, -1.0]),
            atol=1e-5,
        )

    def test_axis_angle_round_trip_preserves_executed_rotation(self) -> None:
        raw = np.asarray(
            [
                [0.25, -0.5, 0.75, 0.2, -0.3, 0.4, 0.5],
                [-0.25, 0.5, -0.75, 3.0, 4.0, 0.0, -0.5],
            ],
            dtype=np.float32,
        )

        physical = controller_action_to_physical(raw, make_contract())
        canonical = physical_action_to_controller(physical, make_contract())
        physical_again = controller_action_to_physical(canonical, make_contract())

        np.testing.assert_allclose(physical_again, physical, atol=1e-6)
        self.assertTrue(np.all(np.linalg.norm(canonical[:, 3:6], axis=-1) <= 1.0 + 1e-6))

    def test_rejects_nonfinite_source_action(self) -> None:
        raw = np.zeros(7, dtype=np.float32)
        raw[3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            controller_action_to_physical(raw, make_contract())

    def test_rejects_real_controller_contract_mismatch(self) -> None:
        expected = make_contract()
        with self.assertRaisesRegex(ValueError, "frequency_hz"):
            validate_controller_contract(
                replace(expected, frequency_hz=10.0),
                expected,
            )
        with self.assertRaisesRegex(ValueError, "rotation_lower"):
            validate_controller_contract(
                replace(
                    expected,
                    rotation_lower=(-0.2, -0.2, -0.2),
                    rotation_upper=(0.2, 0.2, 0.2),
                ),
                expected,
            )

    def test_episode_splits_are_deterministic_disjoint_and_complete(self) -> None:
        source_ids = tuple(range(100))

        first = split_source_episode_ids(source_ids, "source-sha256")
        second = split_source_episode_ids(source_ids, "source-sha256")

        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "validation", "evaluation"})
        split_sets = [set(values) for values in first.values()]
        self.assertEqual(set.union(*split_sets), set(source_ids))
        self.assertFalse(split_sets[0] & split_sets[1])
        self.assertFalse(split_sets[0] & split_sets[2])
        self.assertFalse(split_sets[1] & split_sets[2])
        self.assertTrue(all(split_sets))

    def test_action_statistics_use_only_training_episodes(self) -> None:
        actions = {
            0: np.zeros((2, 7), dtype=np.float32),
            1: np.full((2, 7), 2.0, dtype=np.float32),
            2: np.full((2, 7), 100.0, dtype=np.float32),
        }

        stats = compute_action_statistics(actions, training_episode_ids=(0, 1))

        np.testing.assert_allclose(stats["mean"], np.ones(7), atol=0.0)
        np.testing.assert_allclose(stats["std"], np.ones(7), atol=0.0)
        self.assertEqual(stats["count"], 4)

    def test_episode_reader_validates_and_loads_hdf5_json_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            h5_path, json_path = write_source_fixture(Path(directory))
            reader = ManiSkillEpisodeReader(h5_path, json_path)

            self.assertEqual(reader.episode_ids, (7,))
            self.assertEqual(len(reader.h5_sha256), 64)
            self.assertEqual(len(reader.json_sha256), 64)
            episode = reader.read_episode(7)

        self.assertEqual(episode.source_episode_id, 7)
        self.assertEqual(episode.seed, 1729)
        self.assertEqual(episode.actions.shape, (2, 7))
        self.assertEqual(len(episode.env_states), 3)
        self.assertEqual(episode.success.tolist(), [False, True])
        self.assertEqual(episode.final_success, True)

    def test_episode_reader_rejects_metadata_length_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            h5_path, json_path = write_source_fixture(Path(directory), elapsed_steps=3)
            with self.assertRaisesRegex(ValueError, "elapsed_steps"):
                ManiSkillEpisodeReader(h5_path, json_path)

    def test_only_allows_panda_to_wristcam_articulation_key_remap(self) -> None:
        state = {
            "actors": {"cube": np.zeros(13)},
            "articulations": {"panda": np.zeros(31)},
        }

        remapped = remap_source_articulation_keys(state, ("panda_wristcam",))

        self.assertEqual(set(remapped["articulations"]), {"panda_wristcam"})
        self.assertIn("panda", state["articulations"])
        with self.assertRaisesRegex(ValueError, "articulation keys"):
            remap_source_articulation_keys(state, ("other_robot",))


if __name__ == "__main__":
    unittest.main()
