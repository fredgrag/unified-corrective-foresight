from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from corrective_foresight.model.objectives import (
    DYNAMICS_HORIZONS,
    combine_horizon_losses,
    compute_recursive_dynamics,
    horizon_chain_mask,
    normalized_horizon_weights,
)


class RecordingZeroDynamicsModel:
    def __init__(self) -> None:
        self.rollout_lengths: list[int] = []
        self.start_times: list[int] = []

    def predict_delta(
        self,
        condition_tokens,
        initial_state,
        normalized_actions,
        delta_time,
        action_spec_ids,
        *,
        start_time=0,
    ):
        del condition_tokens, delta_time, action_spec_ids
        rollout_length = normalized_actions.shape[1]
        self.rollout_lengths.append(rollout_length)
        self.start_times.append(start_time)
        delta = initial_state.new_zeros(
            initial_state.shape[0],
            rollout_length,
            *initial_state.shape[1:],
        )
        states = torch.cat(
            (
                initial_state[:, None],
                initial_state[:, None].expand(-1, rollout_length, -1, -1),
            ),
            dim=1,
        )
        return SimpleNamespace(delta=delta, states=states)


class DynamicsObjectiveTest(unittest.TestCase):
    def test_horizon_weights_are_fixed_normalized_and_applied_once(self) -> None:
        weights = normalized_horizon_weights()

        self.assertEqual(tuple(weights), DYNAMICS_HORIZONS)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        expected = {
            1: 1.0 / 2.952,
            2: 0.8 / 2.952,
            4: 0.64 / 2.952,
            8: 0.512 / 2.952,
        }
        for horizon in DYNAMICS_HORIZONS:
            self.assertAlmostEqual(weights[horizon], expected[horizon])
        losses = {horizon: torch.tensor(float(horizon)) for horizon in DYNAMICS_HORIZONS}
        expected_total = sum(expected[h] * h for h in DYNAMICS_HORIZONS)
        torch.testing.assert_close(
            combine_horizon_losses(losses),
            torch.tensor(expected_total),
        )
        with self.assertRaisesRegex(ValueError, "exactly"):
            combine_horizon_losses({1: torch.tensor(1.0)})

    def test_chain_mask_requires_every_intermediate_transition(self) -> None:
        transitions = torch.tensor([[True, True, False, True, True]])

        result = horizon_chain_mask(transitions, horizon=2)

        torch.testing.assert_close(
            result,
            torch.tensor([[True, False, False, True]]),
        )

    def test_recursive_rollout_runs_each_origin_once_and_exposes_horizons(self) -> None:
        model = RecordingZeroDynamicsModel()
        computation = compute_recursive_dynamics(
            model=model,
            condition_tokens=torch.zeros(1, 2, 4),
            online_states=torch.zeros(1, 9, 9, 4),
            target_states=torch.zeros(1, 9, 9, 4),
            normalized_actions=torch.zeros(1, 8, 2),
            transition_valid_mask=torch.ones(1, 8, dtype=torch.bool),
            action_dimension_mask=torch.ones(1, 8, 2, dtype=torch.bool),
            delta_time=torch.ones(1, 8),
            action_spec_ids=("test",),
        )

        self.assertEqual(model.rollout_lengths, [8, 7, 6, 5, 4, 3, 2, 1])
        self.assertEqual(model.start_times, list(range(8)))
        self.assertEqual(set(computation.horizon_losses), set(DYNAMICS_HORIZONS))
        self.assertEqual(computation.one_step_predicted_states.shape, (1, 8, 9, 4))
        self.assertEqual(computation.one_step_predicted_deltas.shape, (1, 8, 9, 4))
        torch.testing.assert_close(computation.loss, torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
