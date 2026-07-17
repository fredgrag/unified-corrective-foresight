from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.metrics import dynamics_diagnostics


class ObjectiveMetricsTest(unittest.TestCase):
    def test_perfect_prediction_beats_copy_last_by_one(self) -> None:
        shape = (1, 1, 9, 2)
        base = torch.zeros(shape)
        target = torch.ones(shape)
        predicted = target.clone()
        delta = torch.ones(shape)

        metrics = dynamics_diagnostics(
            predicted_states=predicted,
            target_states=target,
            base_states=base,
            predicted_deltas=delta,
            target_deltas=delta,
            transition_mask=torch.ones(1, 1, dtype=torch.bool),
        )

        torch.testing.assert_close(metrics["visual_mse_loss"], torch.tensor(0.0))
        torch.testing.assert_close(metrics["visual_delta_loss"], torch.tensor(0.0))
        torch.testing.assert_close(metrics["visual_cosine_loss"], torch.tensor(0.0))
        torch.testing.assert_close(metrics["copy_last_mse"], torch.tensor(1.0))
        torch.testing.assert_close(
            metrics["improvement_vs_copy_last"], torch.tensor(1.0)
        )
        for name in (
            "pred_token_std",
            "target_token_std",
            "pred_delta_std",
            "target_delta_std",
        ):
            torch.testing.assert_close(metrics[name], torch.tensor(0.0))

    def test_invalid_nonfinite_transition_is_ignored(self) -> None:
        value = torch.ones(2, 1, 9, 2)
        invalid = value.clone()
        invalid[1] = torch.nan

        metrics = dynamics_diagnostics(
            predicted_states=invalid,
            target_states=invalid,
            base_states=torch.zeros_like(value),
            predicted_deltas=invalid,
            target_deltas=invalid,
            transition_mask=torch.tensor([[True], [False]]),
        )

        torch.testing.assert_close(metrics["visual_mse_loss"], torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
