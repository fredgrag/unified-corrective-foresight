from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from corrective_foresight.model.token_types import (
    TokenMetadata,
    TokenRole,
    TokenView,
)


def build_forward_view(
    condition_tokens: Tensor,
    state_tokens: Tensor,
    demonstrated_action_tokens: Tensor,
    delta_query_tokens: Tensor,
    *,
    condition_valid: Tensor | None = None,
    state_valid: Tensor | None = None,
    action_valid: Tensor | None = None,
    query_valid: Tensor | None = None,
    semantic_time: int = 0,
) -> TokenView:
    batch_size, hidden_size = _base_shape(condition_tokens, "condition_tokens")
    _require_tokens(state_tokens, batch_size, hidden_size, "state_tokens")
    _require_tokens(
        demonstrated_action_tokens,
        batch_size,
        hidden_size,
        "demonstrated_action_tokens",
        count=1,
    )
    _require_tokens(
        delta_query_tokens,
        batch_size,
        hidden_size,
        "delta_query_tokens",
        count=state_tokens.shape[1],
    )
    _require_semantic_time(semantic_time)
    return _view(
        blocks=(
            condition_tokens,
            state_tokens,
            demonstrated_action_tokens,
            delta_query_tokens,
        ),
        valid_masks=(condition_valid, state_valid, action_valid, query_valid),
        metadata=(
            _metadata(TokenRole.CONDITION, condition_tokens.shape[1], -1, 0, True),
            _metadata(TokenRole.STATE, state_tokens.shape[1], semantic_time, 1),
            _metadata(TokenRole.ACTION, 1, semantic_time, 2),
            _metadata(
                TokenRole.DELTA_QUERY,
                delta_query_tokens.shape[1],
                semantic_time + 1,
                3,
            ),
        ),
    )


def build_inverse_view(
    condition_tokens: Tensor,
    state_tokens: Tensor,
    target_delta_tokens: Tensor,
    action_query_tokens: Tensor,
    *,
    condition_valid: Tensor | None = None,
    state_valid: Tensor | None = None,
    delta_valid: Tensor | None = None,
    query_valid: Tensor | None = None,
    semantic_time: int = 0,
) -> TokenView:
    return _build_inverse_like(
        condition_tokens=condition_tokens,
        state_tokens=state_tokens,
        delta_tokens=target_delta_tokens,
        action_query_tokens=action_query_tokens,
        delta_role=TokenRole.TARGET_DELTA,
        condition_valid=condition_valid,
        state_valid=state_valid,
        delta_valid=delta_valid,
        query_valid=query_valid,
        semantic_time=semantic_time,
    )


def build_cycle_view(
    condition_tokens: Tensor,
    state_tokens: Tensor,
    predicted_delta_tokens: Tensor,
    action_query_tokens: Tensor,
    *,
    condition_valid: Tensor | None = None,
    state_valid: Tensor | None = None,
    delta_valid: Tensor | None = None,
    query_valid: Tensor | None = None,
    semantic_time: int = 0,
) -> TokenView:
    return _build_inverse_like(
        condition_tokens=condition_tokens,
        state_tokens=state_tokens,
        delta_tokens=predicted_delta_tokens,
        action_query_tokens=action_query_tokens,
        delta_role=TokenRole.PREDICTED_DELTA,
        condition_valid=condition_valid,
        state_valid=state_valid,
        delta_valid=delta_valid,
        query_valid=query_valid,
        semantic_time=semantic_time,
    )


def build_policy_view(
    condition_tokens: Tensor,
    observed_state_tokens: Tensor,
    flow_state_tokens: Tensor,
    flow_time_tokens: Tensor,
    horizon_tokens: Tensor,
    *,
    condition_valid: Tensor | None = None,
    observed_state_valid: Tensor | None = None,
    query_valid: Tensor | None = None,
    start_time: int = 0,
) -> TokenView:
    batch_size, hidden_size = _base_shape(condition_tokens, "condition_tokens")
    if observed_state_tokens.ndim != 4 or min(observed_state_tokens.shape) <= 0:
        raise ValueError("observed_state_tokens must have shape [B,T,S,H]")
    if observed_state_tokens.shape[0] != batch_size or observed_state_tokens.shape[3] != hidden_size:
        raise ValueError("observed_state_tokens batch/hidden dimensions must match")
    for name, component in (
        ("flow_state_tokens", flow_state_tokens),
        ("flow_time_tokens", flow_time_tokens),
        ("horizon_tokens", horizon_tokens),
    ):
        _require_tokens(component, batch_size, hidden_size, name, count=8)
    _require_semantic_time(start_time)
    time_steps, state_count = observed_state_tokens.shape[1:3]
    flattened_states = observed_state_tokens.reshape(
        batch_size, time_steps * state_count, hidden_size
    )
    if observed_state_valid is not None:
        if observed_state_valid.dtype is not torch.bool or observed_state_valid.shape != (
            batch_size,
            time_steps,
            state_count,
        ):
            raise ValueError("observed_state_valid must be bool with shape [B,T,S]")
        flattened_state_valid = observed_state_valid.reshape(
            batch_size, time_steps * state_count
        )
    else:
        flattened_state_valid = None
    policy_queries = flow_state_tokens + flow_time_tokens + horizon_tokens
    state_metadata: list[TokenMetadata] = []
    for offset in range(time_steps):
        state_metadata.extend(
            _metadata(
                TokenRole.STATE,
                state_count,
                start_time + offset,
                1 + offset,
            )
        )
    query_metadata = tuple(
        TokenMetadata(
            role=TokenRole.POLICY_QUERY,
            semantic_time=start_time + time_steps + horizon,
            block_id=1 + time_steps,
            valid=True,
            is_condition=False,
        )
        for horizon in range(8)
    )
    return _view(
        blocks=(condition_tokens, flattened_states, policy_queries),
        valid_masks=(condition_valid, flattened_state_valid, query_valid),
        metadata=(
            _metadata(TokenRole.CONDITION, condition_tokens.shape[1], -1, 0, True),
            tuple(state_metadata),
            query_metadata,
        ),
    )


def _build_inverse_like(
    *,
    condition_tokens: Tensor,
    state_tokens: Tensor,
    delta_tokens: Tensor,
    action_query_tokens: Tensor,
    delta_role: TokenRole,
    condition_valid: Tensor | None,
    state_valid: Tensor | None,
    delta_valid: Tensor | None,
    query_valid: Tensor | None,
    semantic_time: int,
) -> TokenView:
    batch_size, hidden_size = _base_shape(condition_tokens, "condition_tokens")
    _require_tokens(state_tokens, batch_size, hidden_size, "state_tokens")
    _require_tokens(
        delta_tokens,
        batch_size,
        hidden_size,
        "delta_tokens",
        count=state_tokens.shape[1],
    )
    _require_tokens(
        action_query_tokens,
        batch_size,
        hidden_size,
        "action_query_tokens",
        count=1,
    )
    _require_semantic_time(semantic_time)
    return _view(
        blocks=(condition_tokens, state_tokens, delta_tokens, action_query_tokens),
        valid_masks=(condition_valid, state_valid, delta_valid, query_valid),
        metadata=(
            _metadata(TokenRole.CONDITION, condition_tokens.shape[1], -1, 0, True),
            _metadata(TokenRole.STATE, state_tokens.shape[1], semantic_time, 1),
            _metadata(delta_role, delta_tokens.shape[1], semantic_time + 1, 2),
            _metadata(TokenRole.ACTION_QUERY, 1, semantic_time, 3),
        ),
    )


def _view(
    *,
    blocks: Sequence[Tensor],
    valid_masks: Sequence[Tensor | None],
    metadata: Sequence[tuple[TokenMetadata, ...]],
) -> TokenView:
    valid = tuple(
        _valid_mask(block, mask, index)
        for index, (block, mask) in enumerate(zip(blocks, valid_masks, strict=True))
    )
    return TokenView(
        tokens=torch.cat(tuple(blocks), dim=1),
        metadata=tuple(item for group in metadata for item in group),
        key_padding_mask=~torch.cat(valid, dim=1),
    )


def _base_shape(tokens: Tensor, name: str) -> tuple[int, int]:
    if tokens.ndim != 3 or min(tokens.shape) <= 0 or not tokens.is_floating_point():
        raise ValueError(f"{name} must be floating point with shape [B,N,H]")
    return tokens.shape[0], tokens.shape[2]


def _require_tokens(
    tokens: Tensor,
    batch_size: int,
    hidden_size: int,
    name: str,
    count: int | None = None,
) -> None:
    _base_shape(tokens, name)
    if tokens.shape[0] != batch_size or tokens.shape[2] != hidden_size:
        raise ValueError(f"{name} batch/hidden dimensions must match conditions")
    if count is not None and tokens.shape[1] != count:
        qualifier = "exactly eight" if count == 8 else f"exactly {count}"
        raise ValueError(f"{name} must contain {qualifier} tokens")


def _valid_mask(tokens: Tensor, mask: Tensor | None, index: int) -> Tensor:
    expected_shape = tokens.shape[:2]
    if mask is None:
        return torch.ones(expected_shape, dtype=torch.bool, device=tokens.device)
    if mask.dtype is not torch.bool or mask.shape != expected_shape:
        raise ValueError(f"valid mask {index} must be bool with shape {expected_shape}")
    if mask.device != tokens.device:
        raise ValueError(f"valid mask {index} must be on the token device")
    return mask


def _metadata(
    role: TokenRole,
    count: int,
    semantic_time: int,
    block_id: int,
    is_condition: bool = False,
) -> tuple[TokenMetadata, ...]:
    return tuple(
        TokenMetadata(role, semantic_time, block_id, True, is_condition)
        for _ in range(count)
    )


def _require_semantic_time(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("semantic time must be a nonnegative integer")
