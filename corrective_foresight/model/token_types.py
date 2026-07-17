from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class TokenRole(str, Enum):
    CONDITION = "condition"
    STATE = "state"
    ACTION = "action"
    TARGET_DELTA = "target_delta"
    PREDICTED_DELTA = "predicted_delta"
    DELTA_QUERY = "delta_query"
    ACTION_QUERY = "action_query"
    POLICY_QUERY = "policy_query"


@dataclass(frozen=True, slots=True)
class TokenMetadata:
    role: TokenRole
    semantic_time: int
    block_id: int
    valid: bool
    is_condition: bool

    def __post_init__(self) -> None:
        if not isinstance(self.role, TokenRole):
            raise ValueError("token role must be a TokenRole")
        if type(self.semantic_time) is not int:
            raise ValueError("semantic_time must be an integer")
        if type(self.block_id) is not int or self.block_id < 0:
            raise ValueError("block_id must be a nonnegative integer")
        if type(self.valid) is not bool or type(self.is_condition) is not bool:
            raise ValueError("valid and is_condition must be bool")
        if self.is_condition != (self.role is TokenRole.CONDITION):
            raise ValueError("condition flag must match the condition token role")
        if self.is_condition and (self.semantic_time != -1 or self.block_id != 0):
            raise ValueError("condition tokens must use semantic_time -1 and block 0")
        if not self.is_condition and (self.semantic_time < 0 or self.block_id == 0):
            raise ValueError("trajectory tokens require nonnegative time and positive block")


@dataclass(slots=True)
class TokenView:
    tokens: Tensor
    metadata: tuple[TokenMetadata, ...]
    key_padding_mask: Tensor

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3 or min(self.tokens.shape) <= 0:
            raise ValueError("tokens must have shape [B,L,H] with positive dimensions")
        if not self.tokens.is_floating_point():
            raise ValueError("tokens must be floating point")
        batch_size, sequence_length = self.tokens.shape[:2]
        if len(self.metadata) != sequence_length:
            raise ValueError("metadata length must equal token sequence length")
        if not all(isinstance(item, TokenMetadata) for item in self.metadata):
            raise ValueError("metadata entries must be TokenMetadata")
        if self.key_padding_mask.dtype is not torch.bool or self.key_padding_mask.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError("key_padding_mask must be bool with shape [B,L]")
        structural_invalid = torch.tensor(
            [not item.valid for item in self.metadata],
            device=self.key_padding_mask.device,
        )
        if structural_invalid.any().item() and not self.key_padding_mask[
            :, structural_invalid
        ].all().item():
            raise ValueError("structurally invalid tokens must be padded in every sample")
        if self.key_padding_mask.all(dim=1).any().item():
            raise ValueError("every sample requires at least one valid token")
        valid_values = ~self.key_padding_mask[..., None].expand_as(self.tokens)
        if not torch.isfinite(self.tokens[valid_values]).all().item():
            raise ValueError("valid token values must be finite")

    def indices(self, role: TokenRole) -> tuple[int, ...]:
        return tuple(
            index for index, item in enumerate(self.metadata) if item.role is role
        )

    def to(self, device: torch.device | str) -> TokenView:
        return TokenView(
            tokens=self.tokens.to(device),
            metadata=self.metadata,
            key_padding_mask=self.key_padding_mask.to(device),
        )
