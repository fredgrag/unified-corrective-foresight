from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from corrective_foresight.config.schema import ActionSpec
from corrective_foresight.model.action_adapters import ActionAdapterRegistry
from corrective_foresight.model.outputs import (
    CyclePrediction,
    DeltaPrediction,
    InversePrediction,
    PolicyVelocityPrediction,
)
from corrective_foresight.model.token_types import TokenRole, TokenView
from corrective_foresight.model.token_views import (
    build_cycle_view,
    build_forward_view,
    build_inverse_view,
    build_policy_view,
)
from corrective_foresight.model.transformer import CausalTokenTransformer


@dataclass(frozen=True, slots=True)
class WorldActionConfig:
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    mlp_ratio: int
    dropout: float
    state_tokens: int
    action_horizon: int
    max_time_steps: int
    time_fourier_bands: int
    gradient_checkpointing: bool
    precision: str
    cycle_noise_std: float
    cycle_dropout: float

    def __post_init__(self) -> None:
        if min(
            self.hidden_size,
            self.num_attention_heads,
            self.mlp_ratio,
            self.max_time_steps,
            self.time_fourier_bands,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by attention heads")
        if self.num_layers < 2:
            raise ValueError("WorldActionTransformer requires at least two layers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if self.state_tokens != 9:
            raise ValueError("state_tokens must be exactly 9")
        if self.action_horizon != 8:
            raise ValueError("action_horizon must be exactly 8")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be bool")
        if self.precision not in {"float32", "bf16"}:
            raise ValueError("precision must be float32 or bf16")
        if not math.isfinite(self.cycle_noise_std) or self.cycle_noise_std < 0:
            raise ValueError("cycle_noise_std must be finite and nonnegative")
        if not 0.0 <= self.cycle_dropout < 1.0:
            raise ValueError("cycle_dropout must satisfy 0 <= value < 1")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> WorldActionConfig:
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != expected:
            raise ValueError(f"model config fields must be exactly {sorted(expected)}")
        return cls(**value)


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, hidden_size: int, bands: int) -> None:
        super().__init__()
        frequencies = 2.0 * math.pi * torch.pow(2.0, torch.arange(bands).float())
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.projection = nn.Sequential(
            nn.Linear(2 * bands, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, time: Tensor) -> Tensor:
        if not time.is_floating_point() or not torch.isfinite(time).all().item():
            raise ValueError("continuous time must be finite floating point")
        if (time < 0).any().item():
            raise ValueError("continuous time must be nonnegative")
        phase = time[..., None] * self.frequencies.to(time)
        return self.projection(torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1))


class WorldActionTransformer(nn.Module):
    def __init__(self, config: WorldActionConfig, action_specs: Sequence[ActionSpec]) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.action_adapters = ActionAdapterRegistry(action_specs, hidden_size)
        self.transformer = CausalTokenTransformer(
            hidden_size=hidden_size,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        self.delta_projection = nn.Linear(hidden_size, hidden_size)
        self.delta_norm = nn.LayerNorm(hidden_size)
        self.role_embedding = nn.Embedding(len(TokenRole), hidden_size)
        self.modality_embedding = nn.Embedding(5, hidden_size)
        self.timestep_embedding = nn.Embedding(config.max_time_steps + 1, hidden_size)
        self.query_position_embedding = nn.Embedding(
            max(config.state_tokens, config.action_horizon), hidden_size
        )
        self.delta_queries = nn.Parameter(torch.empty(config.state_tokens, hidden_size))
        self.action_query = nn.Parameter(torch.empty(1, hidden_size))
        self.policy_queries = nn.Parameter(torch.empty(config.action_horizon, hidden_size))
        self.policy_horizon_embedding = nn.Embedding(config.action_horizon, hidden_size)
        self.continuous_time_embedding = ContinuousTimeEmbedding(
            hidden_size, config.time_fourier_bands
        )
        nn.init.normal_(self.delta_queries, std=0.02)
        nn.init.normal_(self.action_query, std=0.02)
        nn.init.normal_(self.policy_queries, std=0.02)

    def predict_delta(
        self,
        condition_tokens: Tensor,
        initial_state: Tensor,
        normalized_actions: Tensor,
        delta_time: Tensor,
        action_spec_ids: Sequence[str],
        *,
        start_time: int = 0,
    ) -> DeltaPrediction:
        batch_size = self._validate_conditions(condition_tokens, action_spec_ids)
        self._validate_state(initial_state, batch_size, dimensions=3)
        adapter = self.action_adapters.resolve_batch(action_spec_ids)
        if normalized_actions.ndim != 3 or normalized_actions.shape[:2] != delta_time.shape:
            raise ValueError("actions [B,N,Da] and delta_time [B,N] must align")
        if normalized_actions.shape[0] != batch_size:
            raise ValueError("action batch must match conditions")
        if normalized_actions.shape[1] == 0:
            raise ValueError("dynamics rollout requires at least one transition")
        if type(start_time) is not int or start_time < 0:
            raise ValueError("start_time must be a nonnegative integer")
        if delta_time.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("delta_time must be floating point")
        if not torch.isfinite(delta_time).all().item() or (delta_time <= 0).any().item():
            raise ValueError("delta_time must be finite and positive")
        adapter._require_action(normalized_actions)

        current_state = initial_state
        states = [current_state]
        predicted_deltas: list[Tensor] = []
        for step in range(normalized_actions.shape[1]):
            time_embedding = self.continuous_time_embedding(delta_time[:, step])
            action_token = adapter.project_action(normalized_actions[:, step : step + 1])
            action_token = action_token + time_embedding[:, None, :]
            delta_queries = self.delta_queries[None].expand(batch_size, -1, -1)
            delta_queries = delta_queries + time_embedding[:, None, :]
            view = build_forward_view(
                condition_tokens,
                current_state,
                action_token,
                delta_queries,
                semantic_time=start_time + step,
            )
            hidden = self.transformer(self._embed_view(view))
            query_indices = view.indices(TokenRole.DELTA_QUERY)
            predicted_delta = self.delta_projection(hidden[:, query_indices])
            predicted_deltas.append(predicted_delta)
            current_state = current_state + predicted_delta
            states.append(current_state)
        return DeltaPrediction(
            delta=torch.stack(predicted_deltas, dim=1),
            states=torch.stack(states, dim=1),
        )

    def predict_inverse(
        self,
        condition_tokens: Tensor,
        state_tokens: Tensor,
        target_delta: Tensor,
        action_spec_ids: Sequence[str],
    ) -> InversePrediction:
        mean, log_variance = self._predict_inverse_like(
            condition_tokens,
            state_tokens,
            self.delta_norm(target_delta),
            action_spec_ids,
            cycle=False,
        )
        return InversePrediction(mean=mean, log_variance=log_variance)

    def predict_cycle(
        self,
        condition_tokens: Tensor,
        state_tokens: Tensor,
        predicted_delta: Tensor,
        action_spec_ids: Sequence[str],
    ) -> CyclePrediction:
        prepared_delta = self.delta_norm(predicted_delta)
        if self.training and self.config.cycle_noise_std:
            prepared_delta = prepared_delta + torch.randn_like(prepared_delta) * self.config.cycle_noise_std
        if self.training and self.config.cycle_dropout:
            prepared_delta = F.dropout(
                prepared_delta, p=self.config.cycle_dropout, training=True
            )
        mean, log_variance = self._predict_inverse_like(
            condition_tokens,
            state_tokens,
            prepared_delta,
            action_spec_ids,
            cycle=True,
        )
        return CyclePrediction(mean=mean, log_variance=log_variance)

    def predict_policy_velocity(
        self,
        condition_tokens: Tensor,
        observed_states: Tensor,
        noisy_flow_actions: Tensor,
        flow_time: Tensor,
        action_spec_ids: Sequence[str],
        *,
        observed_state_valid_mask: Tensor | None = None,
    ) -> PolicyVelocityPrediction:
        batch_size = self._validate_conditions(condition_tokens, action_spec_ids)
        self._validate_state(observed_states, batch_size, dimensions=4)
        adapter = self.action_adapters.resolve_batch(action_spec_ids)
        adapter._require_action(noisy_flow_actions)
        if noisy_flow_actions.shape[:2] != (batch_size, self.config.action_horizon):
            raise ValueError("policy noisy actions must have shape [B,8,Da]")
        if flow_time.numel() != batch_size:
            raise ValueError("flow_time must contain one scalar per batch item")
        if not flow_time.is_floating_point() or not torch.isfinite(flow_time).all().item():
            raise ValueError("flow_time must be finite floating point within [0, 1]")
        if (flow_time < 0).any().item() or (flow_time > 1).any().item():
            raise ValueError("flow_time must be within [0, 1]")
        time_tokens = self.continuous_time_embedding(flow_time.reshape(batch_size))
        time_tokens = time_tokens[:, None, :].expand(-1, self.config.action_horizon, -1)
        flow_state_tokens = adapter.project_flow_state(noisy_flow_actions)
        flow_state_tokens = flow_state_tokens + self.policy_queries[None]
        horizon_ids = torch.arange(self.config.action_horizon, device=condition_tokens.device)
        horizon_tokens = self.policy_horizon_embedding(horizon_ids)[None].expand(
            batch_size, -1, -1
        )
        observed_state_valid = None
        if observed_state_valid_mask is not None:
            if (
                observed_state_valid_mask.dtype is not torch.bool
                or observed_state_valid_mask.device != observed_states.device
                or observed_state_valid_mask.shape != observed_states.shape[:2]
            ):
                raise ValueError(
                    "observed_state_valid_mask must be bool with shape [B,T]"
                )
            observed_state_valid = observed_state_valid_mask[..., None].expand(
                *observed_states.shape[:3]
            )
        view = build_policy_view(
            condition_tokens,
            observed_states,
            flow_state_tokens,
            time_tokens,
            horizon_tokens,
            observed_state_valid=observed_state_valid,
        )
        hidden = self.transformer(self._embed_view(view))
        query_indices = view.indices(TokenRole.POLICY_QUERY)
        velocity = adapter.predict_flow_velocity(hidden[:, query_indices])
        return PolicyVelocityPrediction(velocity=velocity)

    def _predict_inverse_like(
        self,
        condition_tokens: Tensor,
        state_tokens: Tensor,
        delta_tokens: Tensor,
        action_spec_ids: Sequence[str],
        *,
        cycle: bool,
    ) -> tuple[Tensor, Tensor]:
        batch_size = self._validate_conditions(condition_tokens, action_spec_ids)
        self._validate_state(state_tokens, batch_size, dimensions=4)
        if delta_tokens.shape != state_tokens.shape:
            raise ValueError("delta tokens must match state token shape [B,N,9,H]")
        adapter = self.action_adapters.resolve_batch(action_spec_ids)
        transitions = state_tokens.shape[1]
        repeated_conditions = condition_tokens[:, None].expand(
            batch_size, transitions, *condition_tokens.shape[1:]
        ).reshape(batch_size * transitions, *condition_tokens.shape[1:])
        flattened_states = state_tokens.reshape(
            batch_size * transitions, self.config.state_tokens, self.config.hidden_size
        )
        flattened_delta = delta_tokens.reshape_as(flattened_states)
        action_query = self.action_query[None].expand(
            batch_size * transitions, -1, -1
        )
        builder = build_cycle_view if cycle else build_inverse_view
        view = builder(
            repeated_conditions,
            flattened_states,
            flattened_delta,
            action_query,
        )
        transition_offsets = torch.arange(
            transitions, device=condition_tokens.device
        )[None].expand(batch_size, -1).reshape(-1)
        hidden = self.transformer(
            self._embed_view(view, time_offsets=transition_offsets)
        )
        query_index = view.indices(TokenRole.ACTION_QUERY)[0]
        mean, raw_log_variance = adapter.predict_inverse(hidden[:, query_index : query_index + 1])
        action_dimension = adapter.spec.dimension
        mean = mean.reshape(batch_size, transitions, action_dimension)
        log_variance = raw_log_variance.reshape(
            batch_size, transitions, action_dimension
        ).clamp(-10.0, 2.0)
        return mean, log_variance

    def _embed_view(
        self, view: TokenView, *, time_offsets: Tensor | None = None
    ) -> TokenView:
        device = view.tokens.device
        role_ids = torch.tensor(
            [list(TokenRole).index(item.role) for item in view.metadata], device=device
        )
        modality_ids = torch.tensor(
            [_modality_id(item.role) for item in view.metadata], device=device
        )
        base_time_ids = torch.tensor(
            [0 if item.is_condition else item.semantic_time + 1 for item in view.metadata],
            device=device,
        )
        if time_offsets is None:
            time_offsets = torch.zeros(
                view.tokens.shape[0], dtype=torch.long, device=device
            )
        elif (
            time_offsets.dtype is not torch.long
            or time_offsets.device != device
            or time_offsets.shape != (view.tokens.shape[0],)
            or (time_offsets < 0).any().item()
        ):
            raise ValueError("time_offsets must be nonnegative int64 with shape [B]")
        non_condition = torch.tensor(
            [not item.is_condition for item in view.metadata], device=device
        )
        time_ids = base_time_ids[None].expand(view.tokens.shape[0], -1).clone()
        time_ids[:, non_condition] += time_offsets[:, None]
        if time_ids.max().item() > self.config.max_time_steps:
            raise ValueError("token semantic time exceeds max_time_steps")
        embedded = (
            view.tokens
            + self.role_embedding(role_ids)[None]
            + self.modality_embedding(modality_ids)[None]
            + self.timestep_embedding(time_ids)
        )
        query_positions = torch.zeros_like(embedded)
        counters: dict[tuple[TokenRole, int], int] = {}
        for index, item in enumerate(view.metadata):
            if item.role not in {
                TokenRole.DELTA_QUERY,
                TokenRole.ACTION_QUERY,
                TokenRole.POLICY_QUERY,
            }:
                continue
            key = (item.role, item.block_id)
            position = counters.get(key, 0)
            counters[key] = position + 1
            query_positions[:, index] = self.query_position_embedding.weight[position]
        return TokenView(
            tokens=embedded + query_positions,
            metadata=view.metadata,
            key_padding_mask=view.key_padding_mask,
        )

    def _validate_conditions(
        self, condition_tokens: Tensor, action_spec_ids: Sequence[str]
    ) -> int:
        if (
            condition_tokens.ndim != 3
            or condition_tokens.shape[-1] != self.config.hidden_size
            or not condition_tokens.is_floating_point()
            or not torch.isfinite(condition_tokens).all().item()
        ):
            raise ValueError("condition tokens must be finite [B,C,H]")
        batch_size = condition_tokens.shape[0]
        if len(action_spec_ids) != batch_size:
            raise ValueError("ActionSpec identifiers must match batch size")
        self.action_adapters.resolve_batch(action_spec_ids)
        return batch_size

    def _validate_state(self, state: Tensor, batch_size: int, dimensions: int) -> None:
        if state.ndim != dimensions or state.shape[0] != batch_size:
            raise ValueError("state tensor has an invalid batch or rank")
        if state.shape[-2:] != (self.config.state_tokens, self.config.hidden_size):
            raise ValueError("state tensor must preserve [9,H] tokens")
        if not state.is_floating_point() or not torch.isfinite(state).all().item():
            raise ValueError("state tensor must be finite floating point")


def _modality_id(role: TokenRole) -> int:
    if role is TokenRole.CONDITION:
        return 0
    if role is TokenRole.STATE:
        return 1
    if role is TokenRole.ACTION:
        return 2
    if role in {TokenRole.TARGET_DELTA, TokenRole.PREDICTED_DELTA}:
        return 3
    return 4
