from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from corrective_foresight.config.loader import load_action_spec, load_dataset_spec
from corrective_foresight.conditioning.language_cache import LanguageEmbeddingCache
from corrective_foresight.data.collate import collate_trajectory_samples
from corrective_foresight.data.lerobot_adapter import LeRobotTrajectoryAdapter
from corrective_foresight.data.mixer import BalancedLeRobotMixer
from corrective_foresight.data.stateful_sampler import StatefulDistributedBatchSampler
from corrective_foresight.evaluation.maniskill_runner import (
    CheckpointProvenance,
    EvaluationProtocol,
    ManiSkillClosedLoopRunner,
    ManiSkillEnvironmentSpec,
    ManiSkillOnlineObservationAdapter,
    physical_action_to_maniskill_controller,
)
from corrective_foresight.model.flow import IntegrationReport
from corrective_foresight.policy.unified_policy import ActionChunkPrediction
from corrective_foresight.training.checkpoint import (
    CheckpointState,
    ExpectedCheckpointContract,
    load_checkpoint_strict,
    save_checkpoint_atomic,
)
from corrective_foresight.training.trainer import Trainer, TrainerConfig
from tests.fixtures.fake_trajectory_dataset import FakeTrajectoryDataset
from tests.unit.policy_fakes import make_condition_ids, make_policy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTION_SPEC_PATH = (
    PROJECT_ROOT
    / "configs/actions/maniskill_panda_pd_ee_delta_pose_physical_v1.yaml"
)
DATASET_SPEC_PATH = PROJECT_ROOT / "configs/datasets/maniskill_pick_cube_v1.yaml"
WORKER_ENV = "UCF_MANISKILL_CLOSED_LOOP_WORKER"


class ScriptedPhysicalPolicy:
    def __init__(self, action_spec) -> None:
        self.action_spec = action_spec
        self.calls: list[int] = []

    def predict_action_chunk(
        self,
        observation,
        action_spec_id: str,
        flow_seed: int,
        solver: str,
    ) -> ActionChunkPrediction:
        self.calls.append(flow_seed)
        physical = torch.zeros((1, 8, self.action_spec.dimension), dtype=torch.float32)
        physical[..., 6] = 0.015
        normalized = self.action_spec.normalize(physical)
        return ActionChunkPrediction(
            normalized_actions=normalized,
            denormalized_actions=physical,
            consistency=torch.full((1, 8), 0.25, dtype=torch.float32),
            inverse_variance=torch.full_like(physical, 0.5),
            solver_report=IntegrationReport(
                solver=solver,
                time_grid=tuple(index / 10 for index in range(11)),
                intervals=10,
                nfe=20,
                noise_seed=flow_seed,
            ),
        )


def condition_ids() -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(1, dtype=torch.long)
        for name in ("dataset", "task", "embodiment", "action_spec", "control_mode")
    }


class ManiSkillClosedLoopIntegrationTest(unittest.TestCase):
    def test_real_rgb_runner_replans_and_is_reproducible(self) -> None:
        if os.environ.get(WORKER_ENV) != "1":
            environment = os.environ.copy()
            environment[WORKER_ENV] = "1"
            environment["CUDA_VISIBLE_DEVICES"] = "0"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    (
                        "tests.integration.test_maniskill_closed_loop."
                        "ManiSkillClosedLoopIntegrationTest."
                        "test_real_rgb_runner_replans_and_is_reproducible"
                    ),
                    "-v",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertNotIn("used fork_rng without explicitly specifying", output)
            return

        from corrective_foresight.data.maniskill_conversion import (
            make_pick_cube_rgb_env,
        )

        action_spec = load_action_spec(ACTION_SPEC_PATH)
        dataset_spec = load_dataset_spec(DATASET_SPEC_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            policies = []
            for run_name in ("first", "second"):
                policy = ScriptedPhysicalPolicy(action_spec)
                policies.append(policy)
                runner = ManiSkillClosedLoopRunner(
                    env_factory=make_pick_cube_rgb_env,
                    policy=policy,
                    observation_adapter=ManiSkillOnlineObservationAdapter(
                        dataset_spec=dataset_spec,
                        condition_ids=condition_ids(),
                        task_text=None,
                        device="cpu",
                        context_steps=2,
                    ),
                    action_spec=action_spec,
                    dataset_spec=dataset_spec,
                    checkpoint=CheckpointProvenance(
                        label="scripted_path_gate",
                        directory="/checkpoints/scripted-path-gate",
                        manifest_sha256="b" * 64,
                    ),
                    environment_spec=ManiSkillEnvironmentSpec(
                        env_id="PickCube-v1",
                        robot_uid="panda_wristcam",
                        obs_mode="rgb",
                        control_mode="pd_ee_delta_pose",
                        sim_backend="physx_cpu",
                    ),
                    protocol=EvaluationProtocol(context_steps=2),
                    output_root=(root / run_name).resolve(),
                    max_steps=3,
                    action_to_environment=physical_action_to_maniskill_controller,
                )
                records.append(
                    runner.run_episode(
                        seed=17,
                        flow_seed_stream=iter((701, 702, 703)),
                        tag="path_gate",
                    )
                )

            self.assertEqual(policies[0].calls, [701, 702, 703])
            self.assertEqual(policies[1].calls, [701, 702, 703])
            self.assertEqual(records[0].value["steps"], records[1].value["steps"])
            self.assertEqual(records[0].value["result"], records[1].value["result"])
            self.assertEqual(records[0].value["flow"], records[1].value["flow"])
            self.assertEqual(
                records[0].value["video"]["sha256"],
                records[1].value["video"]["sha256"],
            )
            self.assertEqual(records[0].value["result"]["length"], 3)
            self.assertEqual(records[0].value["result"]["termination_reason"], "max_steps")
            self.assertEqual(records[0].value["dataset"]["spec_hash"], dataset_spec.content_hash)
            self.assertEqual(records[0].value["action"]["spec_hash"], action_spec.content_hash)

    def test_tiny_overfit_checkpoint_produces_real_closed_loop_actions(self) -> None:
        if os.environ.get(WORKER_ENV) != "1":
            environment = os.environ.copy()
            environment[WORKER_ENV] = "1"
            environment["CUDA_VISIBLE_DEVICES"] = "0"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    (
                        "tests.integration.test_maniskill_closed_loop."
                        "ManiSkillClosedLoopIntegrationTest."
                        "test_tiny_overfit_checkpoint_produces_real_closed_loop_actions"
                    ),
                    "-v",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertNotIn("used fork_rng without explicitly specifying", output)
            return

        from corrective_foresight.data.maniskill_conversion import make_pick_cube_rgb_env

        action_spec = load_action_spec(ACTION_SPEC_PATH)
        dataset_spec = load_dataset_spec(DATASET_SPEC_PATH)
        task_text = "Pick the red cube and place it at the green goal."
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            language_cache = LanguageEmbeddingCache.write(
                root / "language.safetensors",
                root / "language.json",
                {task_text: torch.tensor([1.0, 2.0, 3.0, 4.0])},
                model_id="tests/deterministic-language",
                revision="d" * 40,
            )
            torch.manual_seed(811)
            policy = make_policy(
                max_cameras=2,
                proprio_dimension=25,
                language_cache=language_cache,
                action_spec=action_spec,
            )
            adapter = LeRobotTrajectoryAdapter(
                dataset_spec,
                action_spec,
                split="train",
                context_steps=1,
                action_horizon=8,
                video_backend="torchcodec",
            )
            identifiers = make_condition_ids(policy.condition_encoder.vocabulary, 1)
            sample = replace(
                adapter[0],
                condition_ids={name: value[0] for name, value in identifiers.items()},
            )
            batch = collate_trajectory_samples((sample,))
            policy.eval()
            with torch.no_grad():
                initial_loss = float(
                    policy.compute_training_objective(
                        batch,
                        "unified",
                        global_step=0,
                        generator=torch.Generator().manual_seed(821),
                    ).loss
                )
            trainer = make_trainer(policy, total_steps=21)
            for step in range(20):
                result = trainer.train_step(
                    batch,
                    "unified",
                    global_step=step,
                    generator=torch.Generator().manual_seed(821),
                )
                self.assertTrue(result.optimizer_stepped)
            policy.eval()
            with torch.no_grad():
                final_loss = float(
                    policy.compute_training_objective(
                        batch,
                        "unified",
                        global_step=20,
                        generator=torch.Generator().manual_seed(821),
                    ).loss
                )
            self.assertLess(final_loss, initial_loss)

            mixer = make_mixer(dataset_spec, action_spec)
            checkpoint_path = root / "tiny-overfit-checkpoint"
            state = make_checkpoint_state(
                policy=policy,
                trainer=trainer,
                mixer=mixer,
                dataset_spec=dataset_spec,
                action_spec=action_spec,
                global_step=20,
            )
            save_checkpoint_atomic(checkpoint_path, state)

            torch.manual_seed(877)
            restored_policy = make_policy(
                max_cameras=2,
                proprio_dimension=25,
                language_cache=language_cache,
                action_spec=action_spec,
            )
            restored_policy.restore_ema_step(19)
            restored_trainer = make_trainer(restored_policy, total_steps=21)
            restored_state = make_checkpoint_state(
                policy=restored_policy,
                trainer=restored_trainer,
                mixer=make_mixer(dataset_spec, action_spec),
                dataset_spec=dataset_spec,
                action_spec=action_spec,
                global_step=20,
            )
            resume = load_checkpoint_strict(
                checkpoint_path,
                ExpectedCheckpointContract.from_state(restored_state),
            )
            self.assertEqual(resume.global_step, 20)
            counting_policy = CountingPolicy(restored_policy)
            online_ids = make_condition_ids(
                restored_policy.condition_encoder.vocabulary, 1
            )
            manifest_hash = hashlib.sha256(
                (checkpoint_path / "manifest.json").read_bytes()
            ).hexdigest()
            runner = ManiSkillClosedLoopRunner(
                env_factory=make_pick_cube_rgb_env,
                policy=counting_policy,
                observation_adapter=ManiSkillOnlineObservationAdapter(
                    dataset_spec=dataset_spec,
                    condition_ids=online_ids,
                    task_text=task_text,
                    device="cpu",
                    context_steps=1,
                ),
                action_spec=action_spec,
                dataset_spec=dataset_spec,
                checkpoint=CheckpointProvenance(
                    label="tiny_overfit_smoke",
                    directory=str(checkpoint_path),
                    manifest_sha256=manifest_hash,
                ),
                environment_spec=ManiSkillEnvironmentSpec(
                    env_id="PickCube-v1",
                    robot_uid="panda_wristcam",
                    obs_mode="rgb",
                    control_mode="pd_ee_delta_pose",
                    sim_backend="physx_cpu",
                ),
                protocol=EvaluationProtocol(context_steps=1),
                output_root=(root / "model-rollout").resolve(),
                max_steps=2,
                action_to_environment=physical_action_to_maniskill_controller,
            )
            record = runner.run_episode(
                seed=23,
                flow_seed_stream=iter((901, 902)),
                tag="tiny_overfit_smoke",
            )

            self.assertEqual(counting_policy.calls, 2)
            self.assertEqual(record.value["checkpoint"]["label"], "tiny_overfit_smoke")
            self.assertEqual(record.value["result"]["length"], 2)
            self.assertTrue(all(step["physical_action"] for step in record.value["steps"]))


class CountingPolicy:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.calls = 0

    def predict_action_chunk(self, *args, **kwargs):
        self.calls += 1
        return self.policy.predict_action_chunk(*args, **kwargs)


def make_trainer(policy, *, total_steps: int) -> Trainer:
    return Trainer(
        policy,
        TrainerConfig(
            learning_rate=3e-3,
            weight_decay=0.0,
            accumulation_steps=1,
            max_grad_norm=1.0,
            warmup_steps=0,
            total_steps=total_steps,
            bf16=False,
            ddp=False,
        ),
        rank=0,
    )


def make_mixer(dataset_spec, action_spec) -> BalancedLeRobotMixer:
    dataset = FakeTrajectoryDataset(dataset_spec.dataset_id, action_spec.spec_id, size=16)
    sampler = StatefulDistributedBatchSampler(
        dataset_size=16,
        batch_size=2,
        seed=827,
    )
    return BalancedLeRobotMixer(
        datasets={dataset_spec.dataset_id: dataset},
        samplers={dataset_spec.dataset_id: sampler},
        weights={dataset_spec.dataset_id: 1.0},
        generator=torch.Generator().manual_seed(829),
    )


def make_checkpoint_state(
    *,
    policy,
    trainer,
    mixer,
    dataset_spec,
    action_spec,
    global_step: int,
) -> CheckpointState:
    return CheckpointState(
        policy=policy,
        trainer=trainer,
        mixer=mixer,
        flow_generator=torch.Generator().manual_seed(839),
        epoch=0,
        global_step=global_step,
        optimizer_step=global_step,
        cycle_warmup_step=global_step,
        dataset_specs=(dataset_spec,),
        action_specs=(action_spec,),
        lerobot_version="0.5.1",
        backbone_provenance={
            "canonical_model_id": "tests/DeterministicPatchBackbone",
            "canonical_revision": "1" * 40,
            "delivery_model_id": "tests/DeterministicPatchBackbone",
            "delivery_revision": "2" * 40,
            "weights_sha256": "3" * 64,
        },
        dataset_revisions={dataset_spec.dataset_id: dataset_spec.revision},
        model_config=asdict(policy.world_action_model.config),
        loss_config=asdict(policy.objective.config),
        git_commit="4" * 40,
        git_dirty=False,
    )


if __name__ == "__main__":
    unittest.main()
