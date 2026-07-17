from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist

from corrective_foresight.conditioning.language_cache import LanguageEmbeddingCache
from corrective_foresight.conditioning.vocabulary import CONDITION_NAMESPACES
from corrective_foresight.data.collate import collate_trajectory_samples
from corrective_foresight.data.lerobot_adapter import LeRobotTrajectoryAdapter
from corrective_foresight.training.stages import TrainingStage
from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.trainer import Trainer, TrainerConfig
from tests.fixtures.create_lerobot_v3_fixture import create_lerobot_v3_fixture
from tests.unit.policy_fakes import make_batch, make_policy


class OptimizerStepsIntegrationTest(unittest.TestCase):
    def test_real_lerobot_fixture_steps_both_strict_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_spec, action_spec = create_lerobot_v3_fixture(
                root / "tiny_lerobot_v3"
            )
            cache = LanguageEmbeddingCache.write(
                root / "language.safetensors",
                root / "language.json",
                {"fixture task 0": torch.tensor([1.0, 2.0, 3.0, 4.0])},
                model_id="fixture/clip",
                revision="fixture-revision",
            )
            policy = make_policy(
                max_cameras=2,
                language_cache=cache,
                action_spec=action_spec,
            )
            adapter = LeRobotTrajectoryAdapter(
                dataset_spec,
                action_spec,
                split="train",
                context_steps=2,
                action_horizon=8,
                video_backend="torchcodec",
            )
            sample = adapter[0]
            sample.condition_ids = {
                name: torch.tensor(
                    policy.condition_encoder.vocabulary.id_for(
                        name, f"{name}.value"
                    )
                )
                for name in CONDITION_NAMESPACES
            }
            batch = collate_trajectory_samples([sample])
            trainer = Trainer(
                policy,
                TrainerConfig(
                    learning_rate=1e-3,
                    weight_decay=0.01,
                    accumulation_steps=1,
                    max_grad_norm=1.0,
                    warmup_steps=0,
                    total_steps=4,
                    bf16=True,
                    ddp=False,
                ),
                rank=0,
            )
            parameter = policy.world_action_model.delta_projection.weight
            before = parameter.detach().clone()

            world = trainer.train_step(
                batch,
                TrainingStage.WORLD_PRETRAIN,
                global_step=0,
                generator=torch.Generator().manual_seed(241),
            )
            unified = trainer.train_step(
                batch,
                TrainingStage.UNIFIED,
                global_step=1,
                generator=torch.Generator().manual_seed(251),
            )

            self.assertTrue(world.optimizer_stepped)
            self.assertEqual(
                set(world.per_loss_gradient_norms), {"dynamics_loss"}
            )
            self.assertTrue(unified.optimizer_stepped)
            self.assertEqual(
                set(unified.per_loss_gradient_norms),
                {
                    "dynamics_loss",
                    "inverse_action_loss",
                    "action_cycle_loss",
                    "policy_flow_loss",
                },
            )
            self.assertFalse(torch.equal(before, parameter))
            self.assertEqual(policy.last_ema_step, 1)

    def test_production_768_twelve_layer_cuda_bf16_optimizer_smoke(self) -> None:
        if not torch.cuda.is_available():
            self.fail("production CUDA optimizer smoke requires CUDA")
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        policy = make_policy(
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            gradient_checkpointing=True,
        ).to(device)
        batch = make_batch(policy, device=device)
        trainer = Trainer(
            policy,
            TrainerConfig(
                learning_rate=1e-4,
                weight_decay=0.01,
                accumulation_steps=1,
                max_grad_norm=1.0,
                warmup_steps=0,
                total_steps=2,
                bf16=True,
                ddp=False,
            ),
            rank=0,
        )

        result = trainer.train_step(
            batch,
            TrainingStage.WORLD_PRETRAIN,
            global_step=0,
            generator=torch.Generator(device=device).manual_seed(257),
        )

        self.assertTrue(result.optimizer_stepped)
        self.assertTrue(torch.isfinite(result.output.loss).item())
        self.assertTrue(policy.world_action_model.transformer.gradient_checkpointing)
        self.assertEqual(policy.world_action_model.config.precision, "bf16")

    def test_single_rank_ddp_executes_world_pretrain_with_unused_heads(self) -> None:
        if dist.is_initialized():
            self.fail("test requires ownership of the process group")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory) / "ddp_init"
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{rendezvous}",
                rank=0,
                world_size=1,
            )
            try:
                policy = make_policy()
                batch = make_batch(policy)
                context = DistributedContext.from_initialized_process_group(
                    local_rank=0,
                    device="cpu",
                )
                trainer = Trainer(
                    policy,
                    TrainerConfig(
                        learning_rate=1e-3,
                        weight_decay=0.0,
                        accumulation_steps=1,
                        max_grad_norm=1.0,
                        warmup_steps=0,
                        total_steps=2,
                        bf16=False,
                        ddp=True,
                    ),
                    rank=0,
                    distributed_context=context,
                )

                result = trainer.train_step(
                    batch,
                    TrainingStage.WORLD_PRETRAIN,
                    global_step=0,
                    generator=torch.Generator().manual_seed(281),
                )

                self.assertTrue(result.optimizer_stepped)
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
