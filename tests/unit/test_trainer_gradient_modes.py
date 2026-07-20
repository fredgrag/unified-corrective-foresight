from __future__ import annotations

import unittest

import torch

from corrective_foresight.training.trainer import (
    Trainer,
    TrainerConfig,
)
from tests.unit.policy_fakes import make_batch, make_policy


def make_mode_trainer(
    *,
    gradient_mode: str,
    diagnostic_interval: int = 10,
    learning_rate: float = 5e-5,
    protected_lr_multiplier: float = 0.1,
    warmup_steps: int = 2,
    total_steps: int = 20,
    accumulation_steps: int = 1,
) -> Trainer:
    policy = make_policy()
    return Trainer(
        policy,
        TrainerConfig(
            learning_rate=learning_rate,
            weight_decay=0.05,
            accumulation_steps=accumulation_steps,
            max_grad_norm=1.0,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            bf16=False,
            ddp=False,
            stage="unified",
            protected_lr_multiplier=protected_lr_multiplier,
            gradient_mode=gradient_mode,
            gradient_diagnostic_interval=diagnostic_interval,
            pcgrad_seed=20260720,
        ),
        rank=0,
    )


def run_optimizer_steps(trainer: Trainer, count: int):
    batch = make_batch(trainer.policy)
    generator = torch.Generator().manual_seed(1901)
    return [
        trainer.train_step(batch, "unified", step, generator)
        for step in range(count)
    ]


class TrainerGradientModesTest(unittest.TestCase):
    def test_unified_optimizer_uses_approved_group_lrs(self) -> None:
        trainer = make_mode_trainer(gradient_mode="audit")

        self.assertEqual(
            tuple(group["name"] for group in trainer.optimizer.param_groups),
            ("protected", "action"),
        )
        self.assertEqual(
            trainer.optimizer.param_groups[0]["initial_lr"],
            5e-6,
        )
        self.assertEqual(
            trainer.optimizer.param_groups[1]["initial_lr"],
            5e-5,
        )

    def test_audit_logs_only_on_tenth_optimizer_step(self) -> None:
        trainer = make_mode_trainer(
            gradient_mode="audit",
            diagnostic_interval=10,
            warmup_steps=2,
            total_steps=20,
        )

        results = run_optimizer_steps(trainer, 10)

        self.assertEqual(dict(results[8].gradient_metrics), {})
        self.assertIn(
            "gradient_cosine/dynamics_loss/inverse_action_loss",
            results[9].gradient_metrics,
        )
        self.assertIn(
            "gradient_norm/raw/dynamics_loss",
            results[9].gradient_metrics,
        )
        self.assertEqual(results[9].gradient_metrics["pcgrad/enabled"], 0.0)
        self.assertEqual(
            set(results[9].learning_rates),
            {"protected", "action"},
        )

    def test_pcgrad_reports_projection_and_updates(self) -> None:
        trainer = make_mode_trainer(
            gradient_mode="pcgrad",
            diagnostic_interval=10,
        )

        result = run_optimizer_steps(trainer, 1)[0]

        self.assertTrue(result.optimizer_stepped)
        self.assertEqual(result.gradient_metrics["pcgrad/enabled"], 1.0)
        self.assertIn("pcgrad/removed_fraction", result.gradient_metrics)
        self.assertIn("pcgrad/projected_norm", result.gradient_metrics)
        self.assertGreaterEqual(result.gradient_metrics["pcgrad/removed_fraction"], 0.0)

    def test_checkpoint_rejects_gradient_mode_or_multiplier_change(self) -> None:
        audit = make_mode_trainer(gradient_mode="audit")
        state = audit.state_dict()
        pcgrad = make_mode_trainer(gradient_mode="pcgrad")
        changed_multiplier = make_mode_trainer(
            gradient_mode="audit",
            protected_lr_multiplier=0.2,
        )

        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            pcgrad.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            changed_multiplier.load_state_dict(state)

    def test_checkpoint_rejects_non_boundary_microstep_state(self) -> None:
        trainer = make_mode_trainer(
            gradient_mode="pcgrad",
            accumulation_steps=2,
        )
        state = trainer.state_dict()
        state["micro_steps"] = 1

        with self.assertRaisesRegex(ValueError, "optimizer boundary"):
            trainer.load_state_dict(state)

    def test_invalid_gradient_configuration_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "gradient_mode"):
            make_mode_trainer(gradient_mode="dynamic")
        with self.assertRaisesRegex(ValueError, "protected_lr_multiplier"):
            make_mode_trainer(
                gradient_mode="audit",
                protected_lr_multiplier=0.0,
            )
        with self.assertRaisesRegex(ValueError, "world_pretrain"):
            TrainerConfig(
                learning_rate=5e-5,
                weight_decay=0.05,
                accumulation_steps=1,
                max_grad_norm=1.0,
                warmup_steps=2,
                total_steps=20,
                bf16=False,
                ddp=False,
                stage="world_pretrain",
                protected_lr_multiplier=0.1,
                gradient_mode="ordinary",
            )


if __name__ == "__main__":
    unittest.main()
