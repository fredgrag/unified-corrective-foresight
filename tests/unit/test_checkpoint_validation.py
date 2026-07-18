from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from corrective_foresight.data.mixer import BalancedLeRobotMixer
from corrective_foresight.data.stateful_sampler import (
    StatefulDistributedBatchSampler,
)
from corrective_foresight.training.checkpoint import (
    CheckpointState,
    ExpectedCheckpointContract,
    ExpectedPolicyCheckpointContract,
    load_legacy_weights_explicit,
    load_checkpoint_strict,
    load_policy_checkpoint_strict,
    save_checkpoint_atomic,
)
from corrective_foresight.training.trainer import Trainer, TrainerConfig
from tests.fixtures.fake_trajectory_dataset import FakeTrajectoryDataset
from tests.unit.policy_fakes import make_policy
from tests.unit.test_action_spec import make_action_spec
from tests.unit.test_dataset_spec import make_dataset_spec


def make_state() -> CheckpointState:
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
    policy.restore_ema_step(6)
    dataset = FakeTrajectoryDataset(
        "test.dataset.v1", "test.ee_delta.v1", size=16
    )
    sampler = StatefulDistributedBatchSampler(
        dataset_size=16,
        batch_size=2,
        seed=307,
    )
    mixer = BalancedLeRobotMixer(
        datasets={"test.dataset.v1": dataset},
        samplers={"test.dataset.v1": sampler},
        weights={"test.dataset.v1": 1.0},
        generator=torch.Generator().manual_seed(311),
    )
    return CheckpointState(
        policy=policy,
        trainer=trainer,
        mixer=mixer,
        flow_generator=torch.Generator().manual_seed(313),
        epoch=2,
        global_step=7,
        optimizer_step=7,
        cycle_warmup_step=7,
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
        git_commit="a" * 40,
        git_dirty=False,
    )


class CheckpointValidationTest(unittest.TestCase):
    def test_atomic_round_trip_restores_all_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint-000007"
            source = make_state()
            source.mixer.next_batch()
            expected_generator_state = source.flow_generator.get_state().clone()

            save_checkpoint_atomic(path, source)

            self.assertTrue(path.is_dir())
            self.assertFalse(any(item.is_symlink() for item in path.rglob("*")))
            self.assertEqual(
                {item.name for item in path.iterdir()},
                {
                    "online_model.safetensors",
                    "ema_target.safetensors",
                    "trainer_state.pt",
                    "rng_state.pt",
                    "mixer_state.pt",
                    "manifest.json",
                },
            )
            destination = make_state()
            resume = load_checkpoint_strict(
                path,
                ExpectedCheckpointContract.from_state(destination),
            )

            self.assertEqual(resume.epoch, 2)
            self.assertEqual(resume.global_step, 7)
            self.assertEqual(resume.optimizer_step, 7)
            self.assertEqual(resume.cycle_warmup_step, 7)
            self.assertEqual(destination.policy.last_ema_step, 6)
            torch.testing.assert_close(
                destination.flow_generator.get_state(),
                expected_generator_state,
            )
            destination_mixer = destination.mixer.state_dict()
            source_mixer = source.mixer.state_dict()
            self.assertEqual(destination_mixer["step"], source_mixer["step"])
            self.assertEqual(
                destination_mixer["samplers"], source_mixer["samplers"]
            )
            torch.testing.assert_close(
                destination_mixer["generator_state"],
                source_mixer["generator_state"],
            )
            for source_parameter, destination_parameter in zip(
                source.policy.parameters(),
                destination.policy.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(
                    source_parameter,
                    destination_parameter,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_policy_only_strict_load_verifies_checkpoint_and_restores_ema_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint-000007"
            source = make_state()
            save_checkpoint_atomic(path, source)
            destination = make_state()

            resume = load_policy_checkpoint_strict(
                path,
                ExpectedPolicyCheckpointContract.from_state(destination),
            )

            self.assertEqual(resume.global_step, 7)
            self.assertEqual(destination.policy.last_ema_step, 6)
            for source_parameter, destination_parameter in zip(
                source.policy.parameters(),
                destination.policy.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(
                    source_parameter,
                    destination_parameter,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_missing_manifest_field_and_corrupt_payload_fail_before_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            state = make_state()
            save_checkpoint_atomic(path, state)
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field in tuple(manifest):
                missing = dict(manifest)
                missing.pop(field)
                manifest_path.write_text(
                    json.dumps(missing, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "manifest fields"):
                    load_checkpoint_strict(
                        path,
                        ExpectedCheckpointContract.from_state(make_state()),
                    )

            second_path = Path(directory) / "checkpoint-corrupt"
            save_checkpoint_atomic(second_path, make_state())
            with (second_path / "online_model.safetensors").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "(size|hash).*online_model"):
                load_checkpoint_strict(
                    second_path,
                    ExpectedCheckpointContract.from_state(make_state()),
                )

    def test_existing_destination_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            path.mkdir()
            marker = path / "keep"
            marker.write_text("original", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                save_checkpoint_atomic(path, make_state())

            self.assertEqual(marker.read_text(encoding="utf-8"), "original")

    def test_legacy_load_requires_explicit_allowlist_and_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = make_state().policy
            destination = {
                name: value
                for name, value in policy.state_dict().items()
                if name.startswith("world_action_model.delta_projection.")
            }
            key = sorted(destination)[0]
            bad_path = Path(directory) / "bad.safetensors"
            save_file(
                {
                    key: destination[key].detach().cpu(),
                    "unexpected.weight": torch.ones(1),
                },
                str(bad_path),
            )
            with self.assertRaisesRegex(ValueError, "unexpected legacy"):
                load_legacy_weights_explicit(
                    policy,
                    bad_path,
                    allowed_keys={key},
                    minimum_match_fraction=1.0,
                    max_unexpected_keys=0,
                )

            good_path = Path(directory) / "good.safetensors"
            save_file({key: destination[key].detach().cpu()}, str(good_path))
            report = load_legacy_weights_explicit(
                policy,
                good_path,
                allowed_keys={key},
                minimum_match_fraction=1.0,
                max_unexpected_keys=0,
            )

            self.assertEqual(report.matched_keys, (key,))
            self.assertEqual(report.match_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
