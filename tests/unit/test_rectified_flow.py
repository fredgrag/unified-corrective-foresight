from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.flow import (
    masked_velocity_mse,
    sample_flow_training,
)


class RectifiedFlowTest(unittest.TestCase):
    def test_seeded_training_sample_matches_exact_equations(self) -> None:
        action = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        generator = torch.Generator().manual_seed(101)
        sample = sample_flow_training(action, generator=generator)
        expected_generator = torch.Generator().manual_seed(101)
        expected_epsilon = torch.randn(action.shape, generator=expected_generator)
        expected_t = torch.rand((2, 1, 1), generator=expected_generator)

        torch.testing.assert_close(sample.epsilon, expected_epsilon)
        torch.testing.assert_close(sample.t, expected_t)
        torch.testing.assert_close(
            sample.x_t,
            (1.0 - expected_t) * expected_epsilon + expected_t * action,
        )
        torch.testing.assert_close(sample.target_velocity, action - expected_epsilon)
        predicted_velocity = torch.full_like(action, 0.25)
        torch.testing.assert_close(
            sample.clean_estimate(predicted_velocity),
            sample.x_t + (1.0 - sample.t) * predicted_velocity,
        )

    def test_mask_never_changes_random_draw_order(self) -> None:
        action = torch.randn(2, 3, 4)
        first_mask = torch.ones_like(action, dtype=torch.bool)
        second_mask = first_mask.clone()
        second_mask[:, 1:, 2:] = False

        first = sample_flow_training(
            action,
            valid_mask=first_mask,
            generator=torch.Generator().manual_seed(77),
        )
        second = sample_flow_training(
            action,
            valid_mask=second_mask,
            generator=torch.Generator().manual_seed(77),
        )

        torch.testing.assert_close(first.epsilon, second.epsilon)
        torch.testing.assert_close(first.t, second.t)

    def test_masked_reduction_is_float32_and_ignores_invalid_nan(self) -> None:
        predicted = torch.tensor([[[1.0, 9.0], [3.0, 4.0]]], dtype=torch.float16)
        target = torch.tensor(
            [[[0.0, float("nan")], [1.0, 0.0]]], dtype=torch.float16
        )
        mask = torch.tensor([[[True, False], [True, False]]])

        loss = masked_velocity_mse(predicted, target, mask)

        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, torch.tensor((1.0 + 4.0) / 2.0))

    def test_empty_velocity_mask_fails_closed(self) -> None:
        values = torch.zeros(1, 2, 3)
        with self.assertRaisesRegex(ValueError, "no valid velocity"):
            masked_velocity_mse(values, values, torch.zeros_like(values, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
