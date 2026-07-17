from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from corrective_foresight.model.objectives import (
    CorrectiveForesightObjective,
    ObjectiveInputs,
)
from tests.unit.objective_fakes import make_objective_model


def deterministic_inputs() -> ObjectiveInputs:
    torch.manual_seed(263)
    batch_size = 1
    transitions = 8
    actions = torch.tensor([0.4, -0.3]).reshape(1, 1, 2).expand(
        batch_size, transitions, 2
    ).clone()
    target_delta = torch.zeros(batch_size, transitions, 9, 12)
    target_delta[..., 0::2] = actions[..., 0, None, None]
    target_delta[..., 1::2] = actions[..., 1, None, None]
    target_states = torch.cat(
        (
            torch.zeros(batch_size, 1, 9, 12),
            target_delta.cumsum(dim=1),
        ),
        dim=1,
    )
    return ObjectiveInputs(
        condition_tokens=torch.zeros(batch_size, 6, 12),
        online_states=target_states.clone(),
        target_states=target_states,
        normalized_actions=actions,
        observation_valid_mask=torch.ones(
            batch_size, transitions + 1, dtype=torch.bool
        ),
        transition_valid_mask=torch.ones(
            batch_size, transitions, dtype=torch.bool
        ),
        action_dimension_mask=torch.ones_like(actions, dtype=torch.bool),
        delta_time=torch.full((batch_size, transitions), 0.1),
        context_index=0,
        action_spec_ids=("test.ee_delta.v1",) * batch_size,
        global_step=5000,
    )


def evaluate(objective, inputs):
    objective.model.eval()
    with torch.no_grad():
        return objective(
            inputs,
            flow_generator=torch.Generator().manual_seed(269),
        )


class TinyOverfitIntegrationTest(unittest.TestCase):
    def test_deterministic_transition_and_flow_losses_overfit(self) -> None:
        torch.manual_seed(271)
        model = make_objective_model()
        objective = CorrectiveForesightObjective(model)
        inputs = deterministic_inputs()
        initial = evaluate(objective, inputs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

        model.train()
        for step in range(80):
            torch.manual_seed(277)
            result = objective(
                replace(inputs, global_step=step),
                flow_generator=torch.Generator().manual_seed(269),
            )
            optimizer.zero_grad(set_to_none=True)
            result.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        final = evaluate(objective, inputs)

        self.assertTrue(torch.isfinite(final.total_loss).item())
        self.assertLess(
            final.losses["dynamics_loss"].item(),
            0.6 * initial.losses["dynamics_loss"].item(),
        )
        self.assertLess(
            final.losses["policy_flow_loss"].item(),
            0.8 * initial.losses["policy_flow_loss"].item(),
        )
        self.assertLess(
            final.metrics["visual_mse_loss"].item(),
            final.metrics["copy_last_mse"].item(),
        )


if __name__ == "__main__":
    unittest.main()
