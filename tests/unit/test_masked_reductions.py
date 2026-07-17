from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.masked_reductions import (
    masked_mean,
    masked_mse,
    masked_smooth_l1,
)


class MaskedReductionsTest(unittest.TestCase):
    def test_masked_mean_uses_true_element_count_and_float32(self) -> None:
        values = torch.tensor([1.0, 3.0, torch.nan], dtype=torch.bfloat16)
        mask = torch.tensor([True, True, False])

        result = masked_mean(values, mask, objective_name="known")

        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.tensor(2.0))

    def test_mse_and_smooth_l1_ignore_invalid_nonfinite_values(self) -> None:
        predicted = torch.tensor([1.0, 3.0, torch.nan])
        target = torch.tensor([0.0, 1.0, torch.nan])
        mask = torch.tensor([True, True, False])

        torch.testing.assert_close(
            masked_mse(predicted, target, mask, objective_name="mse"),
            torch.tensor(2.5),
        )
        torch.testing.assert_close(
            masked_smooth_l1(predicted, target, mask, objective_name="smooth"),
            torch.tensor(1.0),
        )

    def test_empty_or_nonfinite_valid_objective_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty_objective.*no valid"):
            masked_mean(
                torch.ones(2),
                torch.zeros(2, dtype=torch.bool),
                objective_name="empty_objective",
            )
        with self.assertRaisesRegex(ValueError, "broken_objective.*non-finite"):
            masked_mean(
                torch.tensor([torch.inf]),
                torch.tensor([True]),
                objective_name="broken_objective",
            )


if __name__ == "__main__":
    unittest.main()
