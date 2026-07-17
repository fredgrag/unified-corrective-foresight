from __future__ import annotations

import torch

from corrective_foresight.conditioning.encoder import ConditionEncoder
from corrective_foresight.conditioning.vocabulary import (
    CONDITION_NAMESPACES,
    ConditionVocabulary,
)
from corrective_foresight.data.batch import TrajectoryBatch
from corrective_foresight.model.ema import EMAStateTarget
from corrective_foresight.model.objectives import CorrectiveForesightObjective
from corrective_foresight.model.spatial_resampler import AnchoredSpatialResampler
from corrective_foresight.model.state_encoder import OnlineStateEncoder, StateAdapter
from corrective_foresight.model.world_action_transformer import (
    WorldActionConfig,
    WorldActionTransformer,
)
from corrective_foresight.policy.unified_policy import (
    PolicyObservation,
    UnifiedCorrectiveForesightPolicy,
)
from tests.unit.fakes import DeterministicPatchBackbone
from tests.unit.test_action_spec import make_action_spec


def make_vocabulary() -> ConditionVocabulary:
    return ConditionVocabulary.build(
        {name: (f"{name}.value",) for name in CONDITION_NAMESPACES}
    )


def make_condition_ids(
    vocabulary: ConditionVocabulary,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    return {
        name: torch.full(
            (batch_size,),
            vocabulary.id_for(name, f"{name}.value"),
            dtype=torch.long,
            device=device,
        )
        for name in CONDITION_NAMESPACES
    }


def make_policy(
    *,
    hidden_size: int = 12,
    num_layers: int = 2,
    num_attention_heads: int = 3,
    gradient_checkpointing: bool = False,
    max_cameras: int = 1,
    proprio_dimension: int = 3,
    language_cache=None,
    action_spec=None,
) -> UnifiedCorrectiveForesightPolicy:
    vocabulary = make_vocabulary()
    condition_encoder = ConditionEncoder(
        vocabulary,
        hidden_dim=hidden_size,
        language_embedding_dim=4,
    )
    online = OnlineStateEncoder(
        DeterministicPatchBackbone(hidden_size),
        StateAdapter(
            AnchoredSpatialResampler(
                hidden_size=hidden_size,
                max_cameras=max_cameras,
                num_attention_heads=num_attention_heads,
            ),
            proprio_dimension=proprio_dimension,
            hidden_size=hidden_size,
        ),
    )
    ema = EMAStateTarget(online, tau=0.99)
    model = WorldActionTransformer(
        WorldActionConfig(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=2 if hidden_size == 12 else 4,
            dropout=0.0 if hidden_size == 12 else 0.1,
            state_tokens=9,
            action_horizon=8,
            max_time_steps=32 if hidden_size == 12 else 64,
            time_fourier_bands=4 if hidden_size == 12 else 16,
            gradient_checkpointing=gradient_checkpointing,
            precision="float32" if hidden_size == 12 else "bf16",
            cycle_noise_std=0.01,
            cycle_dropout=0.05,
        ),
        (action_spec or make_action_spec(),),
    )
    return UnifiedCorrectiveForesightPolicy(
        online_state_encoder=online,
        ema_state_target=ema,
        condition_encoder=condition_encoder,
        world_action_model=model,
        objective=CorrectiveForesightObjective(model),
        language_cache=language_cache,
    )


def make_batch(
    policy: UnifiedCorrectiveForesightPolicy,
    *,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> TrajectoryBatch:
    torch.manual_seed(211)
    time_steps = 9
    action_steps = time_steps - 1
    action = torch.randn(batch_size, action_steps, 2, device=device) * 0.1
    batch = TrajectoryBatch(
        rgb=torch.randn(
            batch_size, time_steps, 1, 3, 8, 16, device=device
        ),
        camera_mask=torch.ones(
            batch_size, time_steps, 1, dtype=torch.bool, device=device
        ),
        proprio=torch.randn(batch_size, time_steps, 3, device=device),
        proprio_mask=torch.ones(
            batch_size, time_steps, 3, dtype=torch.bool, device=device
        ),
        action=action,
        action_dimension_mask=torch.ones_like(action, dtype=torch.bool),
        observation_valid_mask=torch.ones(
            batch_size, time_steps, dtype=torch.bool, device=device
        ),
        action_valid_mask=torch.ones(
            batch_size, action_steps, dtype=torch.bool, device=device
        ),
        transition_valid_mask=torch.ones(
            batch_size, action_steps, dtype=torch.bool, device=device
        ),
        delta_time=torch.full(
            (batch_size, action_steps), 0.1, device=device
        ),
        context_index=0,
        task_text=(None,) * batch_size,
        condition_ids=make_condition_ids(
            policy.condition_encoder.vocabulary, batch_size, device
        ),
        dataset_id="synthetic.dataset.v1",
        action_spec_id="test.ee_delta.v1",
    )
    batch.validate(expected_action_spec_id="test.ee_delta.v1")
    return batch


def observation_from_batch(batch: TrajectoryBatch) -> PolicyObservation:
    return PolicyObservation(
        rgb=batch.rgb[:, :1],
        camera_mask=batch.camera_mask[:, :1],
        proprio=batch.proprio[:, :1],
        proprio_mask=batch.proprio_mask[:, :1],
        observation_valid_mask=batch.observation_valid_mask[:, :1],
        task_text=batch.task_text,
        condition_ids=batch.condition_ids,
    )
