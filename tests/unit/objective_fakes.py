from __future__ import annotations

import torch

from corrective_foresight.model.objectives import ObjectiveInputs
from corrective_foresight.model.world_action_transformer import (
    WorldActionConfig,
    WorldActionTransformer,
)
from tests.unit.test_action_spec import make_action_spec


ACTION_SPEC_ID = "test.ee_delta.v1"


def make_objective_model() -> WorldActionTransformer:
    torch.manual_seed(101)
    return WorldActionTransformer(
        WorldActionConfig(
            hidden_size=12,
            num_layers=2,
            num_attention_heads=3,
            mlp_ratio=2,
            dropout=0.0,
            state_tokens=9,
            action_horizon=8,
            max_time_steps=32,
            time_fourier_bands=4,
            gradient_checkpointing=False,
            precision="float32",
            cycle_noise_std=0.01,
            cycle_dropout=0.05,
        ),
        (make_action_spec(),),
    )


def make_objective_inputs(
    *,
    online_states: torch.Tensor | None = None,
    target_states: torch.Tensor | None = None,
    global_step: int = 5000,
) -> ObjectiveInputs:
    torch.manual_seed(103)
    if online_states is None:
        online_states = torch.randn(1, 9, 9, 12, requires_grad=True)
    if target_states is None:
        target_states = torch.randn(1, 9, 9, 12)
    return ObjectiveInputs(
        condition_tokens=torch.randn(1, 6, 12),
        online_states=online_states,
        target_states=target_states.detach(),
        normalized_actions=torch.randn(1, 8, 2),
        observation_valid_mask=torch.ones(1, 9, dtype=torch.bool),
        transition_valid_mask=torch.ones(1, 8, dtype=torch.bool),
        action_dimension_mask=torch.ones(1, 8, 2, dtype=torch.bool),
        delta_time=torch.full((1, 8), 0.1),
        context_index=0,
        action_spec_ids=(ACTION_SPEC_ID,),
        global_step=global_step,
    )
