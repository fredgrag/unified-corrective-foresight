from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.objectives import DEFAULT_OPTIMIZED_TERMS
from corrective_foresight.training.stages import TrainingStage
from corrective_foresight.training.trainer import Trainer, TrainerConfig
from train import run_training
from tests.unit.policy_fakes import (
    make_batch,
    make_policy,
    observation_from_batch,
)


class TrainingStagesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = make_policy()
        self.batch = make_batch(self.policy)

    def test_world_pretrain_uses_only_final_dynamics_path(self) -> None:
        output = self.policy.compute_training_objective(
            self.batch,
            TrainingStage.WORLD_PRETRAIN,
            global_step=0,
            generator=torch.Generator().manual_seed(223),
        )

        self.assertEqual(output.optimized_terms, frozenset({"dynamics_loss"}))
        self.assertEqual(set(output.losses), {"dynamics_loss"})
        output.loss.backward()
        adapter = self.policy.world_action_model.action_adapters.resolve(
            self.batch.action_spec_id
        )
        self.assertIsNotNone(
            self.policy.world_action_model.delta_projection.weight.grad
        )
        self.assertIsNotNone(adapter.action_input_projection.weight.grad)
        self.assertIsNotNone(
            self.policy.world_action_model.transformer.layers[
                0
            ].attention.in_proj_weight.grad
        )
        self.assertIsNone(adapter.inverse_mean_head.weight.grad)
        self.assertIsNone(adapter.flow_velocity_head.weight.grad)

    def test_unified_uses_exact_four_terms_and_rejects_other_stages(self) -> None:
        output = self.policy.compute_training_objective(
            self.batch,
            "unified",
            global_step=2500,
            generator=torch.Generator().manual_seed(227),
        )

        self.assertEqual(output.optimized_terms, DEFAULT_OPTIMIZED_TERMS)
        torch.testing.assert_close(
            output.loss,
            output.losses["dynamics_loss"]
            + output.losses["inverse_action_loss"]
            + 0.05 * output.losses["action_cycle_loss"]
            + output.losses["policy_flow_loss"],
        )
        for invalid in ("closed_loop", "evaluate", "stage2"):
            with self.assertRaisesRegex(ValueError, "training stage"):
                TrainingStage.parse(invalid)

    def test_inference_returns_actions_and_diagnostics_without_reranking(self) -> None:
        observation = observation_from_batch(self.batch)
        first = self.policy.predict_action_chunk(
            observation,
            self.batch.action_spec_id,
            flow_seed=229,
            solver="midpoint",
        )
        second = self.policy.predict_action_chunk(
            observation,
            self.batch.action_spec_id,
            flow_seed=229,
            solver="midpoint",
        )

        self.assertEqual(first.normalized_actions.shape, (1, 8, 2))
        self.assertEqual(first.denormalized_actions.shape, (1, 8, 2))
        self.assertEqual(first.consistency.shape, (1, 8))
        self.assertEqual(first.inverse_variance.shape, (1, 8, 2))
        self.assertEqual(first.solver_report.nfe, 20)
        torch.testing.assert_close(first.normalized_actions, second.normalized_actions)
        torch.testing.assert_close(first.consistency, second.consistency)
        self.assertTrue((first.inverse_variance > 0).all().item())

    def test_ema_updates_once_per_strictly_increasing_optimizer_step(self) -> None:
        target = next(self.policy.ema_state_target.adapter.parameters())
        before = target.detach().clone()
        online = next(self.policy.online_state_encoder.adapter.parameters())
        with torch.no_grad():
            online.add_(1.0)

        self.policy.update_ema(global_step=0)

        self.assertFalse(torch.equal(before, target))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.policy.update_ema(global_step=0)

    def test_objective_does_not_duplicate_world_model_in_state_dict(self) -> None:
        keys = tuple(self.policy.state_dict())

        self.assertTrue(
            any(key.startswith("world_action_model.transformer.") for key in keys)
        )
        self.assertFalse(any(key.startswith("objective.model.") for key in keys))
        self.assertTrue(
            any(key.startswith("online_state_encoder.backbone.") for key in keys)
        )
        self.assertFalse(
            any(key.startswith("ema_state_target.backbone.") for key in keys)
        )

    def test_trainer_accumulates_then_clips_steps_schedules_and_updates_ema(self) -> None:
        trainer = Trainer(
            self.policy,
            TrainerConfig(
                learning_rate=1e-3,
                weight_decay=0.01,
                accumulation_steps=2,
                max_grad_norm=0.5,
                warmup_steps=0,
                total_steps=4,
                bf16=True,
                ddp=False,
            ),
            rank=0,
        )

        first = trainer.train_step(
            self.batch,
            TrainingStage.WORLD_PRETRAIN,
            global_step=0,
            generator=torch.Generator().manual_seed(233),
        )
        self.assertFalse(first.optimizer_stepped)
        self.assertIsNone(first.pre_clip_gradient_norm)
        self.assertEqual(self.policy.last_ema_step, -1)

        second = trainer.train_step(
            self.batch,
            TrainingStage.WORLD_PRETRAIN,
            global_step=0,
            generator=torch.Generator().manual_seed(239),
        )

        self.assertTrue(second.optimizer_stepped)
        self.assertIsNotNone(second.pre_clip_gradient_norm)
        self.assertGreater(second.pre_clip_gradient_norm.item(), 0.0)
        self.assertEqual(set(second.per_loss_gradient_norms), {"dynamics_loss"})
        self.assertEqual(self.policy.last_ema_step, 0)
        self.assertEqual(trainer.scheduler.last_epoch, 1)
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)

    def test_training_loop_logs_each_alias_once_per_optimizer_step(self) -> None:
        class RepeatingBatchSource:
            def __init__(self, batch):
                self.batch = batch

            def next_batch(self):
                return self.batch

        trainer = Trainer(
            self.policy,
            TrainerConfig(
                learning_rate=1e-3,
                weight_decay=0.0,
                accumulation_steps=1,
                max_grad_norm=1.0,
                warmup_steps=0,
                total_steps=2,
                bf16=False,
                ddp=False,
            ),
            rank=0,
        )
        records = []

        final_step = run_training(
            batch_source=RepeatingBatchSource(self.batch),
            trainer=trainer,
            stage="world_pretrain",
            optimizer_steps=1,
            start_global_step=0,
            generator_seed=283,
            metric_logger=records.append,
        )

        self.assertEqual(final_step, 1)
        self.assertEqual(len(records), 1)
        self.assertIn("dynamics_loss", records[0])
        self.assertIn("visual_loss", records[0])
        self.assertEqual(
            len([name for name in records[0] if name == "visual_loss"]),
            1,
        )


if __name__ == "__main__":
    unittest.main()
