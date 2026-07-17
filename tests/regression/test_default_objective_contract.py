from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from corrective_foresight.model.objectives import (
    DEFAULT_OPTIMIZED_TERMS,
    REQUIRED_METRICS,
    CorrectiveForesightObjective,
    ObjectiveConfig,
)
from tests.unit.objective_fakes import make_objective_inputs, make_objective_model


class DefaultObjectiveContractTest(unittest.TestCase):
    def test_exact_four_term_whitelist_aliases_and_total(self) -> None:
        objective = CorrectiveForesightObjective(make_objective_model())
        result = objective(
            make_objective_inputs(global_step=2500),
            flow_generator=torch.Generator().manual_seed(107),
        )

        self.assertEqual(
            result.optimized_terms,
            frozenset(
                {
                    "dynamics_loss",
                    "inverse_action_loss",
                    "action_cycle_loss",
                    "policy_flow_loss",
                }
            ),
        )
        self.assertEqual(result.optimized_terms, DEFAULT_OPTIMIZED_TERMS)
        self.assertEqual(set(result.losses), set(DEFAULT_OPTIMIZED_TERMS))
        self.assertEqual(set(result.metrics), set(REQUIRED_METRICS))
        torch.testing.assert_close(
            result.total_loss,
            result.losses["dynamics_loss"]
            + result.losses["inverse_action_loss"]
            + 0.05 * result.losses["action_cycle_loss"]
            + result.losses["policy_flow_loss"],
        )
        torch.testing.assert_close(
            result.metrics["visual_loss"], result.metrics["dynamics_loss"]
        )
        torch.testing.assert_close(
            result.metrics["action_loss"], result.metrics["policy_flow_loss"]
        )
        torch.testing.assert_close(
            result.metrics["self_correction_cycle_loss"],
            result.losses["action_cycle_loss"].detach(),
        )

    def test_nondefault_or_named_extra_objective_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden objective"):
            ObjectiveConfig.from_mapping(
                {
                    "dynamics_weight": 1.0,
                    "inverse_weight": 1.0,
                    "action_cycle_weight": 0.1,
                    "policy_flow_weight": 1.0,
                    "cycle_warmup_steps": 5000,
                    "cosine_weight": 0.1,
                }
            )
        with self.assertRaisesRegex(ValueError, "fixed"):
            ObjectiveConfig(action_cycle_weight=0.2)

    def test_padded_nonfinite_sample_is_ignored_end_to_end(self) -> None:
        base = make_objective_inputs()

        def with_invalid_copy(value: torch.Tensor) -> torch.Tensor:
            invalid = torch.full_like(value, torch.nan)
            return torch.cat((value, invalid), dim=0)

        inputs = replace(
            base,
            condition_tokens=torch.cat(
                (base.condition_tokens, base.condition_tokens), dim=0
            ),
            online_states=with_invalid_copy(base.online_states),
            target_states=with_invalid_copy(base.target_states).detach(),
            normalized_actions=with_invalid_copy(base.normalized_actions),
            observation_valid_mask=torch.cat(
                (
                    base.observation_valid_mask,
                    torch.zeros_like(base.observation_valid_mask),
                ),
                dim=0,
            ),
            transition_valid_mask=torch.cat(
                (
                    base.transition_valid_mask,
                    torch.zeros_like(base.transition_valid_mask),
                ),
                dim=0,
            ),
            action_dimension_mask=torch.cat(
                (
                    base.action_dimension_mask,
                    torch.zeros_like(base.action_dimension_mask),
                ),
                dim=0,
            ),
            delta_time=torch.cat(
                (base.delta_time, torch.full_like(base.delta_time, torch.nan)),
                dim=0,
            ),
            action_spec_ids=(
                "test.ee_delta.v1",
                "test.ee_delta.v1",
            ),
        )

        result = CorrectiveForesightObjective(make_objective_model())(
            inputs,
            flow_generator=torch.Generator().manual_seed(127),
        )

        self.assertTrue(torch.isfinite(result.total_loss).item())

    def test_integer_actions_and_attached_targets_fail_at_objective_boundary(self) -> None:
        objective = CorrectiveForesightObjective(make_objective_model())
        base = make_objective_inputs()
        with self.assertRaisesRegex(ValueError, "normalized_actions.*floating"):
            objective(
                replace(base, normalized_actions=base.normalized_actions.long()),
                flow_generator=torch.Generator().manual_seed(131),
            )
        with self.assertRaisesRegex(ValueError, "stop-gradient"):
            objective(
                replace(
                    base,
                    target_states=base.target_states.clone().requires_grad_(True),
                ),
                flow_generator=torch.Generator().manual_seed(137),
            )


if __name__ == "__main__":
    unittest.main()
