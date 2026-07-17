from __future__ import annotations

import math
import unittest

import torch

from corrective_foresight.model.flow import IntegrationReport, sample_rectified_flow


def constant_field(value: float):
    def field(actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.full_like(actions, value)

    return field


class FlowSolverTest(unittest.TestCase):
    def test_default_midpoint_has_ten_intervals_and_twenty_nfe(self) -> None:
        actions, report = sample_rectified_flow(
            constant_field(2.0),
            action_shape=(2, 8, 3),
            noise_seed=19,
            device="cpu",
        )
        generator = torch.Generator().manual_seed(19)
        initial = torch.randn((2, 8, 3), generator=generator)

        torch.testing.assert_close(actions, initial + 2.0)
        self.assertEqual(report.solver, "midpoint")
        self.assertEqual(report.intervals, 10)
        self.assertEqual(report.nfe, 20)
        self.assertEqual(report.noise_seed, 19)
        self.assertEqual(len(report.time_grid), 11)
        self.assertEqual(report.time_grid[0], 0.0)
        self.assertEqual(report.time_grid[-1], 1.0)

    def test_euler_and_midpoint_match_discrete_linear_field(self) -> None:
        def linear(actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            del time
            return actions

        initial = torch.ones(1, 1, 1)
        euler, euler_report = sample_rectified_flow(
            linear,
            initial_noise=initial,
            solver="euler",
            intervals=4,
            noise_seed=3,
        )
        midpoint, midpoint_report = sample_rectified_flow(
            linear,
            initial_noise=initial,
            solver="midpoint",
            intervals=4,
            noise_seed=3,
        )

        torch.testing.assert_close(euler, torch.tensor([[[1.25**4]]]))
        torch.testing.assert_close(midpoint, torch.tensor([[[(1.0 + 0.25 + 0.5 * 0.25**2) ** 4]]]))
        self.assertEqual(euler_report.nfe, 4)
        self.assertEqual(midpoint_report.nfe, 8)
        self.assertLess(abs(midpoint.item() - math.e), abs(euler.item() - math.e))

    def test_noise_seed_is_reproducible(self) -> None:
        first, first_report = sample_rectified_flow(
            constant_field(0.0), action_shape=(1, 8, 2), noise_seed=55
        )
        second, second_report = sample_rectified_flow(
            constant_field(0.0), action_shape=(1, 8, 2), noise_seed=55
        )

        torch.testing.assert_close(first, second)
        self.assertEqual(first_report, second_report)

    def test_report_rejects_inconsistent_nfe_or_time_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "NFE"):
            IntegrationReport(
                solver="midpoint",
                time_grid=(0.0, 1.0),
                intervals=1,
                nfe=1,
                noise_seed=0,
            )
        with self.assertRaisesRegex(ValueError, "time grid"):
            IntegrationReport(
                solver="euler",
                time_grid=(0.0, 0.4, 1.0),
                intervals=1,
                nfe=1,
                noise_seed=0,
            )

    def test_solver_rejects_bad_velocity_output(self) -> None:
        def wrong_shape(actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            del time
            return actions[..., :1]

        with self.assertRaisesRegex(ValueError, "velocity field shape"):
            sample_rectified_flow(
                wrong_shape, initial_noise=torch.zeros(1, 2, 3), noise_seed=0
            )


if __name__ == "__main__":
    unittest.main()
