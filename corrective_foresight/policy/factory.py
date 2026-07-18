from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from corrective_foresight.conditioning.encoder import ConditionEncoder
from corrective_foresight.conditioning.language_cache import LanguageEmbeddingCache
from corrective_foresight.conditioning.vocabulary import ConditionVocabulary
from corrective_foresight.config.schema import ActionSpec
from corrective_foresight.model.dinov3_backbone import DinoV3PatchBackbone
from corrective_foresight.model.ema import EMAStateTarget
from corrective_foresight.model.objectives import CorrectiveForesightObjective
from corrective_foresight.model.spatial_resampler import AnchoredSpatialResampler
from corrective_foresight.model.state_encoder import OnlineStateEncoder, StateAdapter
from corrective_foresight.model.world_action_transformer import (
    WorldActionConfig,
    WorldActionTransformer,
)
from corrective_foresight.policy.unified_policy import (
    UnifiedCorrectiveForesightPolicy,
)


PRODUCTION_WORLD_ACTION_CONFIG = WorldActionConfig(
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    mlp_ratio=4,
    dropout=0.1,
    state_tokens=9,
    action_horizon=8,
    max_time_steps=64,
    time_fourier_bands=16,
    gradient_checkpointing=True,
    precision="bf16",
    cycle_noise_std=0.01,
    cycle_dropout=0.05,
)


def assemble_unified_policy(
    *,
    backbone: nn.Module,
    action_specs: Sequence[ActionSpec],
    vocabulary: ConditionVocabulary,
    world_action_config: WorldActionConfig,
    language_cache: LanguageEmbeddingCache | None,
    language_embedding_dim: int,
    proprio_dimension: int,
    max_cameras: int,
    ema_tau: float,
    device: torch.device | str,
) -> UnifiedCorrectiveForesightPolicy:
    target_device = torch.device(device)
    if getattr(backbone, "hidden_size", None) != world_action_config.hidden_size:
        raise ValueError("backbone hidden dimension does not match world-action config")
    if not action_specs:
        raise ValueError("policy assembly requires at least one ActionSpec")
    if type(language_embedding_dim) is not int or language_embedding_dim <= 0:
        raise ValueError("language_embedding_dim must be a positive integer")
    if language_cache is not None and language_cache.dimension != language_embedding_dim:
        raise ValueError("language cache dimension does not match assembly config")
    if type(proprio_dimension) is not int or proprio_dimension <= 0:
        raise ValueError("proprio_dimension must be a positive integer")
    if type(max_cameras) is not int or max_cameras <= 0:
        raise ValueError("max_cameras must be a positive integer")

    backbone = backbone.to(target_device)
    backbone.requires_grad_(False)
    backbone.eval()
    state_adapter = StateAdapter(
        spatial_resampler=AnchoredSpatialResampler(
            hidden_size=world_action_config.hidden_size,
            max_cameras=max_cameras,
            num_attention_heads=world_action_config.num_attention_heads,
        ),
        proprio_dimension=proprio_dimension,
        hidden_size=world_action_config.hidden_size,
    ).to(target_device)
    online_state_encoder = OnlineStateEncoder(backbone, state_adapter).to(target_device)
    ema_state_target = EMAStateTarget(online_state_encoder, tau=ema_tau).to(
        target_device
    )
    condition_encoder = ConditionEncoder(
        vocabulary,
        hidden_dim=world_action_config.hidden_size,
        language_embedding_dim=language_embedding_dim,
    ).to(target_device)
    world_action_model = WorldActionTransformer(
        world_action_config,
        tuple(action_specs),
    ).to(target_device)
    policy = UnifiedCorrectiveForesightPolicy(
        online_state_encoder=online_state_encoder,
        ema_state_target=ema_state_target,
        condition_encoder=condition_encoder,
        world_action_model=world_action_model,
        objective=CorrectiveForesightObjective(world_action_model),
        language_cache=language_cache,
    ).to(target_device)
    return policy


def assemble_production_policy(
    *,
    snapshot_dir: str | Path,
    action_specs: Sequence[ActionSpec],
    vocabulary: ConditionVocabulary,
    language_cache: LanguageEmbeddingCache | None,
    proprio_dimension: int,
    max_cameras: int,
    ema_tau: float,
    device: torch.device | str,
) -> UnifiedCorrectiveForesightPolicy:
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("production policy assembly requires CUDA")
    backbone = DinoV3PatchBackbone.from_pretrained_exact(
        Path(snapshot_dir),
        device=target_device,
    )
    return assemble_unified_policy(
        backbone=backbone,
        action_specs=action_specs,
        vocabulary=vocabulary,
        world_action_config=PRODUCTION_WORLD_ACTION_CONFIG,
        language_cache=language_cache,
        language_embedding_dim=512,
        proprio_dimension=proprio_dimension,
        max_cameras=max_cameras,
        ema_tau=ema_tau,
        device=target_device,
    )
