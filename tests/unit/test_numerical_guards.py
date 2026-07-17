from __future__ import annotations

import unittest

import torch
from torch import nn

from corrective_foresight.training.numerics import (
    NumericalGuardError,
    clip_and_validate_gradients,
    per_loss_gradient_norms,
    require_finite_objective,
)


class NumericalGuardsTest(unittest.TestCase):
    def test_nonfinite_objective_identifies_origin(self) -> None:
        with self.assertRaisesRegex(
            NumericalGuardError,
            "policy_flow_loss.*fixture.dataset.*rank=2.*step=17",
        ):
            require_finite_objective(
                torch.tensor(float("nan")),
                objective_name="policy_flow_loss",
                dataset_id="fixture.dataset",
                rank=2,
                global_step=17,
            )

    def test_nonfinite_gradient_identifies_parameter_and_context(self) -> None:
        parameter = nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([float("inf")])

        with self.assertRaisesRegex(
            NumericalGuardError,
            "weight.*fixture.dataset.*rank=1.*step=9",
        ):
            clip_and_validate_gradients(
                (("weight", parameter),),
                max_norm=1.0,
                dataset_id="fixture.dataset",
                rank=1,
                global_step=9,
            )

    def test_per_loss_norms_and_preclip_norm_are_measured(self) -> None:
        parameter = nn.Parameter(torch.tensor([3.0, 4.0]))
        losses = {
            "first": parameter.square().sum(),
            "second": (2.0 * parameter).sum(),
        }

        norms = per_loss_gradient_norms(
            losses,
            (parameter,),
            dataset_id="dataset",
            rank=0,
            global_step=0,
        )
        losses["first"].backward()
        preclip = clip_and_validate_gradients(
            (("parameter", parameter),),
            max_norm=1.0,
            dataset_id="dataset",
            rank=0,
            global_step=0,
        )

        torch.testing.assert_close(norms["first"], torch.tensor(10.0))
        torch.testing.assert_close(
            norms["second"], torch.tensor(2.0).sqrt() * 2.0
        )
        torch.testing.assert_close(preclip, torch.tensor(10.0))
        self.assertLessEqual(parameter.grad.norm().item(), 1.00001)


if __name__ == "__main__":
    unittest.main()
