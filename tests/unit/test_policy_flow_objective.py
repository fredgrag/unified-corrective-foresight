from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.flow import FlowTrainingSample
from corrective_foresight.model.objectives import compute_policy_flow_losses


class PolicyFlowObjectiveTest(unittest.TestCase):
    def test_velocity_is_optimized_and_clean_action_is_diagnostic_only(self) -> None:
        epsilon = torch.zeros(1, 2, 2)
        target_action = torch.ones_like(epsilon)
        flow_time = torch.full((1, 1, 1), 0.5)
        sample = FlowTrainingSample(
            epsilon=epsilon,
            t=flow_time,
            x_t=flow_time * target_action,
            target_velocity=target_action,
        )
        predicted_velocity = torch.zeros_like(target_action, requires_grad=True)
        mask = torch.tensor([[[True, True], [True, False]]])

        result = compute_policy_flow_losses(
            predicted_velocity,
            sample,
            target_action,
            mask,
        )

        torch.testing.assert_close(result.velocity_loss, torch.tensor(1.0))
        torch.testing.assert_close(result.clean_action_mse, torch.tensor(0.25))
        result.velocity_loss.backward()
        self.assertGreater(predicted_velocity.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
