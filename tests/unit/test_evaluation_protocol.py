from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from corrective_foresight.config.schema import DatasetSpec
from corrective_foresight.evaluation.maniskill_runner import (
    AdaptedPolicyObservation,
    CheckpointProvenance,
    EvaluationProtocol,
    ManiSkillClosedLoopRunner,
    ManiSkillEnvironmentSpec,
    ManiSkillOnlineObservationAdapter,
)
from corrective_foresight.model.flow import IntegrationReport
from corrective_foresight.policy.unified_policy import (
    ActionChunkPrediction,
    PolicyObservation,
)
from tests.unit.test_action_spec import make_action_spec


def make_dataset_spec(root: Path) -> DatasetSpec:
    return DatasetSpec(
        schema_version=1,
        dataset_id="test.maniskill.v1",
        repo_id=None,
        local_root=str(root.resolve()),
        revision="dataset-revision",
        fps=10.0,
        camera_features={"base": "observation.images.base"},
        optional_camera_roles=(),
        proprio_features=("observation.state",),
        proprio_required=True,
        task_feature="task",
        language_feature=None,
        embodiment_id="test.panda",
        action_feature="action",
        action_spec_id="test.ee_delta.v1",
        sample_weight=1.0,
        split_episode_ids={"train": (0,), "validation": (1,), "evaluation": (2,)},
    )


def make_policy_observation(value: float) -> PolicyObservation:
    return PolicyObservation(
        rgb=torch.full((1, 1, 1, 3, 8, 8), value, dtype=torch.float32),
        camera_mask=torch.ones((1, 1, 1), dtype=torch.bool),
        proprio=torch.full((1, 1, 3), value, dtype=torch.float32),
        proprio_mask=torch.ones((1, 1, 3), dtype=torch.bool),
        observation_valid_mask=torch.ones((1, 1), dtype=torch.bool),
        task_text=(None,),
        condition_ids={"dataset": torch.zeros(1, dtype=torch.long)},
    )


class FakeObservationAdapter:
    def __init__(self) -> None:
        self.values: list[int] = []

    def reset(self, observation: dict) -> AdaptedPolicyObservation:
        self.values.clear()
        return self.append(observation)

    def append(self, observation: dict) -> AdaptedPolicyObservation:
        value = int(observation["value"])
        self.values.append(value)
        return AdaptedPolicyObservation(
            policy_observation=make_policy_observation(float(value)),
            video_frame=np.full((16, 16, 3), value, dtype=np.uint8),
        )


class FakeEnvironment:
    def __init__(self, *, truncate_after: int = 3) -> None:
        self.truncate_after = truncate_after
        self.reset_seeds: list[int] = []
        self.actions: list[np.ndarray] = []
        self.closed = False

    def reset(self, *, seed: int):
        self.reset_seeds.append(seed)
        self.actions.clear()
        return {"value": 10}, {"success": False}

    def step(self, action: np.ndarray):
        self.actions.append(np.asarray(action).copy())
        step = len(self.actions)
        return (
            {"value": 10 + step},
            float(step),
            False,
            step >= self.truncate_after,
            {"success": False},
        )

    def close(self) -> None:
        self.closed = True


class FakePolicy:
    def __init__(self, *, out_of_bounds: bool = False) -> None:
        self.out_of_bounds = out_of_bounds
        self.flow_seeds: list[int] = []
        self.observation_values: list[float] = []

    def predict_action_chunk(
        self,
        observation: PolicyObservation,
        action_spec_id: str,
        flow_seed: int,
        solver: str,
    ) -> ActionChunkPrediction:
        self.flow_seeds.append(flow_seed)
        self.observation_values.append(float(observation.proprio[0, -1, 0]))
        call = len(self.flow_seeds)
        actions = torch.zeros((1, 8, 2), dtype=torch.float32)
        actions[0, 0] = torch.tensor([0.25 * call, 1.0])
        actions[0, 1] = torch.tensor([2.5, 3.5])
        if self.out_of_bounds:
            actions[0, 0, 0] = 3.5
        consistency = torch.full((1, 8), 9.0)
        consistency[0, 1] = 0.0
        inverse_variance = torch.full((1, 8, 2), 2.0)
        inverse_variance[0, 1] = 0.01
        return ActionChunkPrediction(
            normalized_actions=actions.clone(),
            denormalized_actions=actions,
            consistency=consistency,
            inverse_variance=inverse_variance,
            solver_report=IntegrationReport(
                solver=solver,
                time_grid=tuple(index / 10 for index in range(11)),
                intervals=10,
                nfe=20,
                noise_seed=flow_seed,
            ),
        )


class EvaluationProtocolTest(unittest.TestCase):
    def make_runner(
        self,
        root: Path,
        environment: FakeEnvironment,
        policy: FakePolicy,
    ) -> ManiSkillClosedLoopRunner:
        return ManiSkillClosedLoopRunner(
            env_factory=lambda: environment,
            policy=policy,
            observation_adapter=FakeObservationAdapter(),
            action_spec=make_action_spec(),
            dataset_spec=make_dataset_spec(root),
            checkpoint=CheckpointProvenance(
                label="untrained_smoke",
                directory="/checkpoints/fresh",
                manifest_sha256="a" * 64,
            ),
            environment_spec=ManiSkillEnvironmentSpec(
                env_id="PickCube-v1",
                robot_uid="panda_wristcam",
                obs_mode="rgb",
                control_mode="pd_ee_delta_pose",
                sim_backend="physx_cpu",
            ),
            protocol=EvaluationProtocol(),
            output_root=root / "evaluation",
            max_steps=3,
            action_to_environment=lambda _env, action: np.asarray(action),
        )

    def test_replans_every_step_executes_only_first_action_and_records_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = FakeEnvironment()
            policy = FakePolicy()
            runner = self.make_runner(root, environment, policy)

            record = runner.run_episode(
                seed=7,
                flow_seed_stream=iter((101, 102, 103)),
                tag="untrained_smoke",
            )

            self.assertEqual(environment.reset_seeds, [7])
            self.assertEqual(policy.flow_seeds, [101, 102, 103])
            self.assertEqual(policy.observation_values, [10.0, 11.0, 12.0])
            np.testing.assert_allclose(
                np.stack(environment.actions),
                np.asarray([[0.25, 1.0], [0.5, 1.0], [0.75, 1.0]]),
            )
            self.assertTrue(environment.closed)
            self.assertEqual(record.value["protocol"]["action_horizon"], 8)
            self.assertEqual(record.value["protocol"]["execution_horizon"], 1)
            self.assertFalse(record.value["protocol"]["temporal_ensemble"])
            self.assertEqual(record.value["flow"]["seeds"], [101, 102, 103])
            self.assertEqual(record.value["flow"]["intervals"], 10)
            self.assertEqual(record.value["flow"]["nfe_per_step"], 20)
            self.assertEqual(record.value["flow"]["total_nfe"], 60)
            self.assertEqual(record.value["result"]["length"], 3)
            self.assertEqual(record.value["result"]["termination_reason"], "truncated")
            self.assertEqual(record.value["result"]["total_reward"], 6.0)
            self.assertEqual(record.value["steps"][0]["executed_chunk_index"], 0)
            self.assertEqual(record.value["steps"][0]["consistency"], 9.0)
            self.assertEqual(record.value["steps"][0]["inverse_variance_mean"], 2.0)
            self.assertTrue(Path(record.value["video"]["path"]).is_file())
            self.assertEqual(len(record.value["video"]["sha256"]), 64)
            record_path = root / "evaluation" / "untrained_smoke" / "seed-7.json"
            self.assertTrue(record_path.is_file())
            self.assertEqual(record_path.read_text(encoding="utf-8"), record.to_json())

    def test_action_outside_explicit_bounds_fails_before_environment_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = FakeEnvironment()
            runner = self.make_runner(root, environment, FakePolicy(out_of_bounds=True))

            with self.assertRaisesRegex(ValueError, "outside ActionSpec bounds"):
                runner.run_episode(
                    seed=11,
                    flow_seed_stream=iter((201, 202, 203)),
                    tag="untrained_smoke",
                )

            self.assertEqual(environment.actions, [])
            self.assertTrue(environment.closed)
            self.assertFalse((root / "evaluation" / "untrained_smoke" / "seed-11.json").exists())

    def test_protocol_rejects_nonprimary_action_or_execution_horizon(self) -> None:
        for kwargs in (
            {"action_horizon": 7},
            {"execution_horizon": 2},
            {"temporal_ensemble": True},
            {"solver": "euler"},
            {"solver_intervals": 9},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    EvaluationProtocol(**kwargs)

    def test_online_adapter_reuses_dataset_camera_and_proprio_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_spec = DatasetSpec(
                schema_version=1,
                dataset_id="test.maniskill.rgb.v1",
                repo_id=None,
                local_root=str(root),
                revision="revision",
                fps=20.0,
                camera_features={
                    "wrist": "observation.images.wrist",
                    "base": "observation.images.base",
                },
                optional_camera_roles=(),
                proprio_features=("observation.state",),
                proprio_required=True,
                task_feature="task",
                language_feature=None,
                embodiment_id="maniskill.panda_wristcam.v1",
                action_feature="action",
                action_spec_id="test.ee_delta.v1",
                sample_weight=1.0,
                split_episode_ids={
                    "train": (0,),
                    "validation": (1,),
                    "evaluation": (2,),
                },
            )
            condition_ids = {
                name: torch.tensor([index], dtype=torch.long)
                for index, name in enumerate(
                    ("dataset", "task", "embodiment", "action_spec", "control_mode")
                )
            }
            adapter = ManiSkillOnlineObservationAdapter(
                dataset_spec=dataset_spec,
                condition_ids=condition_ids,
                task_text=None,
                device="cpu",
                context_steps=2,
                video_camera_role="base",
            )

            first = adapter.reset(make_real_rgb_observation(offset=0))
            second = adapter.append(make_real_rgb_observation(offset=1))
            third = adapter.append(make_real_rgb_observation(offset=2))

            self.assertEqual(first.policy_observation.rgb.shape, (1, 1, 2, 3, 128, 128))
            self.assertEqual(second.policy_observation.rgb.shape, (1, 2, 2, 3, 128, 128))
            self.assertEqual(third.policy_observation.rgb.shape, (1, 2, 2, 3, 128, 128))
            self.assertEqual(third.policy_observation.proprio.shape, (1, 2, 25))
            self.assertTrue(third.policy_observation.camera_mask.all().item())
            self.assertTrue(third.policy_observation.proprio_mask.all().item())
            self.assertTrue(third.policy_observation.observation_valid_mask.all().item())
            expected = make_real_rgb_observation(offset=2)["sensor_data"]["base_camera"]["rgb"][0]
            np.testing.assert_array_equal(third.video_frame, expected)
            torch.testing.assert_close(
                third.policy_observation.rgb[0, -1, 0],
                torch.from_numpy(expected).permute(2, 0, 1).float() / 255.0,
            )


def make_real_rgb_observation(offset: int) -> dict:
    pixels = (
        np.arange(128 * 128 * 3, dtype=np.int64).reshape(128, 128, 3) + offset
    ) % 251
    base = pixels.astype(np.uint8)
    wrist = np.flip(base, axis=1).copy()
    return {
        "sensor_data": {
            "base_camera": {"rgb": base[None]},
            "hand_camera": {"rgb": wrist[None]},
        },
        "agent": {
            "qpos": np.full((1, 9), offset, dtype=np.float32),
            "qvel": np.full((1, 9), offset + 1, dtype=np.float32),
        },
        "extra": {
            "tcp_pose": np.full((1, 7), offset + 2, dtype=np.float32),
        },
    }


if __name__ == "__main__":
    unittest.main()
