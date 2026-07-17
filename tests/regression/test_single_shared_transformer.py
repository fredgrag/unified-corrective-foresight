from __future__ import annotations

import unittest

import torch

from corrective_foresight.model.transformer import CausalTokenTransformer
from tests.unit.test_world_action_transformer import make_model


class SingleSharedTransformerTest(unittest.TestCase):
    def test_all_prediction_paths_call_the_only_transformer_instance(self) -> None:
        model = make_model().eval()
        shared = [
            module
            for module in model.modules()
            if isinstance(module, CausalTokenTransformer)
        ]
        self.assertEqual(len(shared), 1)
        calls: list[int] = []
        handle = shared[0].register_forward_hook(
            lambda module, inputs, output: calls.append(id(module))
        )
        condition = torch.randn(1, 6, 12)
        state = torch.randn(1, 9, 12)
        state_sequence = torch.randn(1, 2, 9, 12)
        delta = torch.randn(1, 2, 9, 12)
        spec_ids = ("test.ee_delta.v1",)
        try:
            model.predict_delta(
                condition,
                state,
                torch.randn(1, 2, 2),
                torch.full((1, 2), 0.1),
                spec_ids,
            )
            model.predict_inverse(condition, state_sequence, delta, spec_ids)
            model.predict_cycle(condition, state_sequence, delta, spec_ids)
            model.predict_policy_velocity(
                condition,
                state_sequence,
                torch.randn(1, 8, 2),
                torch.rand(1, 1, 1),
                spec_ids,
            )
        finally:
            handle.remove()

        self.assertEqual(calls, [id(shared[0])] * 5)


if __name__ == "__main__":
    unittest.main()
