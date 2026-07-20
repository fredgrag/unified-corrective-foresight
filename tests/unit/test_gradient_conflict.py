from __future__ import annotations

import unittest

import torch
from torch import nn

from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.gradient_conflict import (
    GlobalObjectiveGradients,
    ObjectiveGradientAccumulator,
    projection_order,
    project_pcgrad,
)
from corrective_foresight.training.numerics import NumericalGuardError


WEIGHTS = {
    "dynamics_loss": 1.0,
    "inverse_action_loss": 1.0,
    "action_cycle_loss": 0.1,
    "policy_flow_loss": 1.0,
}


class GradientConflictTest(unittest.TestCase):
    def test_cosines_cover_aligned_orthogonal_and_opposed(self) -> None:
        gradients = GlobalObjectiveGradients.from_flattened(
            raw={
                "dynamics_loss": torch.tensor([1.0, 0.0]),
                "inverse_action_loss": torch.tensor([-1.0, 0.0]),
                "action_cycle_loss": torch.tensor([0.0, 1.0]),
                "policy_flow_loss": torch.tensor([1.0, 0.0]),
            },
            effective_weights=WEIGHTS,
        )

        diagnostics = gradients.diagnostics()

        self.assertEqual(
            diagnostics.cosines[
                "dynamics_loss/inverse_action_loss"
            ],
            -1.0,
        )
        self.assertEqual(
            diagnostics.cosines["dynamics_loss/action_cycle_loss"],
            0.0,
        )
        self.assertEqual(
            diagnostics.cosines["dynamics_loss/policy_flow_loss"],
            1.0,
        )
        self.assertEqual(diagnostics.raw_norms["action_cycle_loss"], 1.0)
        self.assertAlmostEqual(
            diagnostics.effective_norms["action_cycle_loss"],
            0.1,
        )

    def test_zero_effective_cycle_is_excluded_but_raw_gradient_remains(self) -> None:
        weights = dict(WEIGHTS)
        weights["action_cycle_loss"] = 0.0
        gradients = GlobalObjectiveGradients.from_flattened(
            raw={
                "dynamics_loss": torch.tensor([1.0, 0.0]),
                "inverse_action_loss": torch.tensor([0.0, 1.0]),
                "action_cycle_loss": torch.tensor([1.0, 1.0]),
                "policy_flow_loss": torch.tensor([-1.0, 0.0]),
            },
            effective_weights=weights,
        )

        projected = project_pcgrad(gradients, seed=17, global_step=0)

        self.assertNotIn("action_cycle_loss", projected.active_objectives)
        self.assertIn("action_cycle_loss", projected.raw_objectives)
        self.assertEqual(projected.gradient[0].shape, (2,))
        self.assertTrue(torch.isfinite(projected.gradient[0]).all().item())

    def test_projection_is_deterministic_and_does_not_mutate_raw(self) -> None:
        gradients = GlobalObjectiveGradients.from_flattened(
            raw={
                "dynamics_loss": torch.tensor([1.0, 0.0]),
                "inverse_action_loss": torch.tensor([-2.0, 1.0]),
                "action_cycle_loss": torch.tensor([0.0, 1.0]),
                "policy_flow_loss": torch.tensor([1.0, 1.0]),
            },
            effective_weights=WEIGHTS,
        )
        before = {
            name: tuple(value.clone() for value in values)
            for name, values in gradients.raw.items()
        }

        first = project_pcgrad(gradients, seed=23, global_step=91)
        repeated = project_pcgrad(gradients, seed=23, global_step=91)

        torch.testing.assert_close(first.gradient[0], repeated.gradient[0])
        self.assertEqual(
            projection_order(gradients.active_names, seed=23, global_step=91),
            projection_order(gradients.active_names, seed=23, global_step=91),
        )
        for name, values in gradients.raw.items():
            for actual, expected in zip(values, before[name], strict=True):
                torch.testing.assert_close(actual, expected)
        self.assertGreaterEqual(first.removed_fraction, 0.0)

    def test_accumulator_uses_full_microbatch_mean(self) -> None:
        parameter = nn.Parameter(torch.tensor([2.0]))
        accumulator = ObjectiveGradientAccumulator(
            (parameter,), tuple(WEIGHTS)
        )
        for coefficient in (1.0, 3.0):
            losses = {
                name: coefficient * (index + 1) * parameter.sum()
                for index, name in enumerate(WEIGHTS)
            }
            accumulator.add(losses, accumulation_steps=2)

        result = accumulator.finalize(
            DistributedContext.single_process("cpu"),
            WEIGHTS,
        )

        self.assertEqual(result.raw["dynamics_loss"][0].item(), 2.0)
        self.assertEqual(result.raw["inverse_action_loss"][0].item(), 4.0)
        self.assertEqual(result.raw["action_cycle_loss"][0].item(), 6.0)
        self.assertEqual(result.raw["policy_flow_loss"][0].item(), 8.0)

    def test_nonfinite_or_missing_raw_gradient_fails_closed(self) -> None:
        parameter = nn.Parameter(torch.tensor([1.0]))
        accumulator = ObjectiveGradientAccumulator((parameter,), ("loss",))
        with self.assertRaisesRegex(NumericalGuardError, "no shared gradient"):
            accumulator.add(
                {"loss": torch.tensor(1.0, requires_grad=True)},
                accumulation_steps=1,
            )
        with self.assertRaisesRegex(NumericalGuardError, "non-finite"):
            GlobalObjectiveGradients.from_flattened(
                raw={"loss": torch.tensor([float("nan")])},
                effective_weights={"loss": 1.0},
            )

    def test_global_gradients_require_materialized_parameter_slots(self) -> None:
        with self.assertRaisesRegex(ValueError, "fully materialized"):
            GlobalObjectiveGradients(
                names=("loss",),
                raw={"loss": (torch.tensor([1.0]), None)},
                effective_weights={"loss": 1.0},
            )


if __name__ == "__main__":
    unittest.main()
