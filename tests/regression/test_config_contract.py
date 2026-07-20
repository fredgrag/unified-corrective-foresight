from __future__ import annotations

from pathlib import Path
import unittest

from corrective_foresight.config.experiment import load_experiment_config
from corrective_foresight.config.loader import load_action_spec, load_dataset_spec
from corrective_foresight.policy.factory import PRODUCTION_WORLD_ACTION_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "configs/experiments"


class ConfigContractTest(unittest.TestCase):
    def test_every_experiment_freezes_production_contract(self) -> None:
        paths = tuple(sorted(EXPERIMENT_ROOT.glob("*.yaml")))
        self.assertEqual(
            {path.name for path in paths},
            {
                "maniskill_unified.yaml",
                "maniskill_unified_conflict_fix.yaml",
                "maniskill_world_pretrain.yaml",
                "maniskill_world_pretrain_conflict_fix.yaml",
            },
        )
        for path in paths:
            with self.subTest(path=path.name):
                config = load_experiment_config(path)
                self.assertEqual(config.model, PRODUCTION_WORLD_ACTION_CONFIG)
                self.assertEqual(config.evaluation.action_horizon, 8)
                self.assertEqual(config.evaluation.execution_horizon, 1)
                self.assertFalse(config.evaluation.temporal_ensemble)
                self.assertEqual(config.evaluation.solver, "midpoint")
                self.assertEqual(config.evaluation.solver_intervals, 10)
                self.assertEqual(config.evaluation.nfe_per_step, 20)
                self.assertTrue(config.output_root.is_absolute())
                self.assertTrue(config.evaluation.output_root.is_absolute())
                self.assertTrue(config.artifacts.dinov3_snapshot.is_absolute())
                self.assertTrue(config.artifacts.language_tensor.is_absolute())
                self.assertTrue(config.artifacts.language_metadata.is_absolute())
                dataset_specs = tuple(
                    load_dataset_spec(spec_path) for spec_path in config.dataset_specs
                )
                action_specs = tuple(
                    load_action_spec(spec_path) for spec_path in config.action_specs
                )
                self.assertEqual(
                    {spec.content_hash for spec in dataset_specs},
                    {"3f6383665d9dc7a3476cea649d02c8be2ae735d616f7ca7aa4d4065e9ed91a9d"},
                )
                self.assertEqual(
                    {spec.content_hash for spec in action_specs},
                    {"13c05454f557d69af8f4ceab9a40318bda1203b513d6b25ee7e1c0c10a1c1000"},
                )

    def test_conflict_fix_training_schedules_are_exact(self) -> None:
        world = load_experiment_config(
            EXPERIMENT_ROOT / "maniskill_world_pretrain_conflict_fix.yaml"
        )
        unified = load_experiment_config(
            EXPERIMENT_ROOT / "maniskill_unified_conflict_fix.yaml"
        )

        self.assertEqual(world.training.learning_rate, 1e-4)
        self.assertEqual(world.training.warmup_steps, 500)
        self.assertEqual(world.training.total_steps, 5000)
        self.assertEqual(world.training.checkpoint_interval, 1000)
        self.assertEqual(world.training.validation_interval, 250)
        self.assertEqual(unified.training.learning_rate, 5e-5)
        self.assertEqual(unified.training.warmup_steps, 500)
        self.assertEqual(unified.training.total_steps, 20000)
        self.assertEqual(unified.training.checkpoint_interval, 1000)
        self.assertEqual(unified.training.validation_interval, 250)

    def test_stage_objective_whitelists_are_exact(self) -> None:
        world = load_experiment_config(
            EXPERIMENT_ROOT / "maniskill_world_pretrain.yaml"
        )
        unified = load_experiment_config(EXPERIMENT_ROOT / "maniskill_unified.yaml")

        self.assertEqual(world.stage, "world_pretrain")
        self.assertEqual(dict(world.optimized_objectives), {"dynamics_loss": 1.0})
        self.assertEqual(unified.stage, "unified")
        self.assertEqual(
            dict(unified.optimized_objectives),
            {
                "dynamics_loss": 1.0,
                "inverse_loss": 1.0,
                "action_cycle_loss": 0.1,
                "policy_flow_loss": 1.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
