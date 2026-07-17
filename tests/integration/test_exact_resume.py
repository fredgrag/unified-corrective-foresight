from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from corrective_foresight.conditioning.vocabulary import CONDITION_NAMESPACES
from corrective_foresight.data.lerobot_adapter import TrajectorySample
from corrective_foresight.data.mixer import BalancedLeRobotMixer
from corrective_foresight.data.stateful_sampler import (
    StatefulDistributedBatchSampler,
)
from corrective_foresight.training.checkpoint import (
    CheckpointState,
    ExpectedCheckpointContract,
    load_checkpoint_strict,
    save_checkpoint_atomic,
)
from corrective_foresight.training.stages import TrainingStage
from corrective_foresight.training.trainer import Trainer, TrainerConfig
from tests.unit.policy_fakes import make_policy
from tests.unit.test_action_spec import make_action_spec
from tests.unit.test_dataset_spec import make_dataset_spec


class ResumeDataset:
    dataset_id = "test.dataset.v1"
    action_spec_id = "test.ee_delta.v1"

    def __init__(self, policy, size: int = 16) -> None:
        self.policy = policy
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> TrajectorySample:
        time_steps = 9
        time = torch.arange(time_steps, dtype=torch.float32)
        rgb = torch.zeros(time_steps, 1, 3, 8, 16)
        rgb[:, 0, 0] = (index + 1) / self.size
        rgb[:, 0, 1] = time[:, None, None] / time_steps
        proprio = torch.stack(
            (
                torch.full_like(time, index / self.size),
                time / time_steps,
                torch.sin(time),
            ),
            dim=-1,
        )
        action = torch.stack(
            (
                torch.full((time_steps - 1,), index / 100.0),
                torch.arange(time_steps - 1, dtype=torch.float32) / 100.0,
            ),
            dim=-1,
        )
        return TrajectorySample(
            rgb=rgb,
            camera_mask=torch.ones(time_steps, 1, dtype=torch.bool),
            proprio=proprio,
            proprio_mask=torch.ones_like(proprio, dtype=torch.bool),
            action=action,
            action_dimension_mask=torch.ones_like(action, dtype=torch.bool),
            observation_valid_mask=torch.ones(time_steps, dtype=torch.bool),
            action_valid_mask=torch.ones(time_steps - 1, dtype=torch.bool),
            transition_valid_mask=torch.ones(time_steps - 1, dtype=torch.bool),
            delta_time=torch.full((time_steps - 1,), 0.1),
            context_index=0,
            task_text=None,
            condition_ids={
                name: torch.tensor(
                    self.policy.condition_encoder.vocabulary.id_for(
                        name, f"{name}.value"
                    )
                )
                for name in CONDITION_NAMESPACES
            },
            dataset_id=self.dataset_id,
            action_spec_id=self.action_spec_id,
        )


def make_runtime(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy = make_policy()
    trainer = Trainer(
        policy,
        TrainerConfig(
            learning_rate=1e-3,
            weight_decay=0.01,
            accumulation_steps=1,
            max_grad_norm=1.0,
            warmup_steps=0,
            total_steps=10,
            bf16=False,
            ddp=False,
        ),
        rank=0,
    )
    dataset = ResumeDataset(policy)
    sampler = StatefulDistributedBatchSampler(
        dataset_size=len(dataset),
        batch_size=2,
        seed=337,
    )
    mixer = BalancedLeRobotMixer(
        datasets={dataset.dataset_id: dataset},
        samplers={dataset.dataset_id: sampler},
        weights={dataset.dataset_id: 1.0},
        generator=torch.Generator().manual_seed(347),
    )
    flow_generator = torch.Generator().manual_seed(349)
    return policy, trainer, mixer, flow_generator


def checkpoint_state(policy, trainer, mixer, flow_generator) -> CheckpointState:
    return CheckpointState(
        policy=policy,
        trainer=trainer,
        mixer=mixer,
        flow_generator=flow_generator,
        epoch=0,
        global_step=1,
        optimizer_step=1,
        cycle_warmup_step=1,
        dataset_specs=(make_dataset_spec(),),
        action_specs=(make_action_spec(),),
        lerobot_version="0.5.1",
        backbone_provenance={
            "canonical_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "canonical_revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
            "delivery_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "delivery_revision": "23d0280ae6ee4ced592a3459674ad027d3c18906",
            "weights_sha256": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
        },
        dataset_revisions={"test.dataset.v1": "0123456789abcdef"},
        model_config=asdict(policy.world_action_model.config),
        loss_config=asdict(policy.objective.config),
        git_commit="b" * 40,
        git_dirty=False,
    )


def step(trainer, mixer, generator, global_step):
    batch = mixer.next_batch()
    result = trainer.train_step(
        batch,
        TrainingStage.UNIFIED,
        global_step=global_step,
        generator=generator,
    )
    return batch, result


class ExactResumeIntegrationTest(unittest.TestCase):
    def test_fresh_process_resume_matches_continuous_next_step(self) -> None:
        continuous = make_runtime(353)
        step(continuous[1], continuous[2], continuous[3], 0)
        expected_batch, expected_result = step(
            continuous[1], continuous[2], continuous[3], 1
        )
        expected_python = random.random()
        expected_numpy = float(np.random.rand())
        expected_torch = torch.rand(4)

        with tempfile.TemporaryDirectory() as directory:
            interrupted = make_runtime(353)
            step(interrupted[1], interrupted[2], interrupted[3], 0)
            path = Path(directory) / "checkpoint-000001"
            save_checkpoint_atomic(path, checkpoint_state(*interrupted))

            resumed = make_runtime(997)
            resumed[0].restore_ema_step(0)
            resume = load_checkpoint_strict(
                path,
                ExpectedCheckpointContract.from_state(
                    checkpoint_state(*resumed)
                ),
            )
            actual_batch, actual_result = step(
                resumed[1], resumed[2], resumed[3], resume.global_step
            )
            actual_python = random.random()
            actual_numpy = float(np.random.rand())
            actual_torch = torch.rand(4)

        torch.testing.assert_close(
            actual_batch.action, expected_batch.action, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            actual_result.output.loss,
            expected_result.output.loss,
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertEqual(
            set(actual_result.output.metrics),
            set(expected_result.output.metrics),
        )
        for name in expected_result.output.metrics:
            torch.testing.assert_close(
                actual_result.output.metrics[name],
                expected_result.output.metrics[name],
                rtol=1e-6,
                atol=1e-7,
            )
        for expected_parameter, actual_parameter in zip(
            continuous[0].parameters(),
            resumed[0].parameters(),
            strict=True,
        ):
            torch.testing.assert_close(
                actual_parameter,
                expected_parameter,
                rtol=1e-6,
                atol=1e-7,
            )
        self.assertEqual(
            resumed[1].scheduler.state_dict(),
            continuous[1].scheduler.state_dict(),
        )
        self.assertEqual(actual_python, expected_python)
        self.assertEqual(actual_numpy, expected_numpy)
        torch.testing.assert_close(actual_torch, expected_torch, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
