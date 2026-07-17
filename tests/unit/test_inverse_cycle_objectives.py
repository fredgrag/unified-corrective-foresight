from __future__ import annotations

import math
import unittest

import torch

from corrective_foresight.model.objectives import (
    action_cycle_weight,
    diagonal_gaussian_nll,
    masked_action_cycle_loss,
)


class InverseCycleObjectivesTest(unittest.TestCase):
    def test_diagonal_gaussian_nll_matches_closed_form(self) -> None:
        mean = torch.zeros(1, 2, 2)
        log_variance = torch.zeros_like(mean)
        target = torch.zeros_like(mean)
        mask = torch.tensor([[[True, False], [True, True]]])

        loss = diagonal_gaussian_nll(mean, log_variance, target, mask)

        self.assertAlmostEqual(loss.item(), 0.5 * math.log(2.0 * math.pi), places=6)
        with self.assertRaisesRegex(ValueError, r"\[-10, 2\]"):
            diagonal_gaussian_nll(
                mean,
                torch.full_like(mean, 2.1),
                target,
                mask,
            )
        with self.assertRaisesRegex(ValueError, "mask"):
            diagonal_gaussian_nll(
                mean,
                log_variance,
                target,
                torch.ones(1, 2, dtype=torch.bool),
            )

    def test_cycle_is_masked_smooth_l1_and_warmup_is_exact(self) -> None:
        recovered = torch.tensor([[[0.0, 3.0]]], requires_grad=True)
        action = torch.tensor([[[0.0, 1.0]]])
        mask = torch.tensor([[[False, True]]])

        loss = masked_action_cycle_loss(recovered, action, mask)
        loss.backward()

        torch.testing.assert_close(loss, torch.tensor(1.5))
        self.assertGreater(recovered.grad.abs().sum().item(), 0.0)
        self.assertEqual(action_cycle_weight(-10), 0.0)
        self.assertEqual(action_cycle_weight(0), 0.0)
        self.assertEqual(action_cycle_weight(2500), 0.05)
        self.assertEqual(action_cycle_weight(5000), 0.1)
        self.assertEqual(action_cycle_weight(9000), 0.1)


if __name__ == "__main__":
    unittest.main()
