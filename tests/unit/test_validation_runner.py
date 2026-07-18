from __future__ import annotations

import copy
import io
import math
from pathlib import Path
import unittest

import torch

from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.validation import ValidationConfig, ValidationRunner
from tests.unit.policy_fakes import make_batch
from tests.unit.test_checkpoint_validation import make_state


class _ValidationMixer:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.counter = 0

    def next_batch(self):
        self.counter += 1
        return make_batch(self.policy)

    def state_dict(self):
        return {"version": 1, "counter": self.counter}

    def load_state_dict(self, state):
        if set(state) != {"version", "counter"} or state["version"] != 1:
            raise ValueError("invalid fake validation mixer state")
        self.counter = state["counter"]


def _tensor_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _serialized(value: object) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


class ValidationRunnerTest(unittest.TestCase):
    def test_evaluate_is_deterministic_and_does_not_mutate_training_state(self) -> None:
        state = make_state()
        state.policy.train()
        validation_mixer = _ValidationMixer(state.policy)
        validation_mixer.counter = 9
        runner = ValidationRunner(
            policy=state.policy,
            validation_mixer=validation_mixer,
            context=DistributedContext.single_process("cpu"),
            config=ValidationConfig(batches_per_rank=2, generator_seed=20261017),
        )
        policy_before = _tensor_state(state.policy)
        ema_before = _tensor_state(state.policy.ema_state_target.adapter)
        trainer_before = copy.deepcopy(state.trainer.state_dict())
        training_mixer_before = copy.deepcopy(state.mixer.state_dict())
        rng_before = torch.get_rng_state().clone()
        mode_before = state.policy.training
        ema_step_before = state.policy.last_ema_step

        first = runner.evaluate("unified", global_step=100)
        second = runner.evaluate("unified", global_step=100)

        self.assertEqual(first, second)
        self.assertEqual(first.batches_per_rank, 2)
        self.assertEqual(first.samples, 2)
        self.assertTrue(all(math.isfinite(value) for value in first.metrics.values()))
        self.assertEqual(validation_mixer.counter, 9)
        self.assertEqual(state.policy.training, mode_before)
        self.assertEqual(state.policy.last_ema_step, ema_step_before)
        for name, value in policy_before.items():
            torch.testing.assert_close(state.policy.state_dict()[name], value, rtol=0, atol=0)
        for name, value in ema_before.items():
            torch.testing.assert_close(
                state.policy.ema_state_target.adapter.state_dict()[name],
                value,
                rtol=0,
                atol=0,
            )
        self.assertEqual(_serialized(state.trainer.state_dict()), _serialized(trainer_before))
        self.assertEqual(_serialized(state.mixer.state_dict()), _serialized(training_mixer_before))
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
