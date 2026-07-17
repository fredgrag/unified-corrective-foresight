from __future__ import annotations

import torch
from torch import Tensor

from corrective_foresight.model.token_types import TokenMetadata


def compile_attention_mask(metadata: tuple[TokenMetadata, ...]) -> Tensor:
    if not metadata or not all(isinstance(item, TokenMetadata) for item in metadata):
        raise ValueError("metadata must be a nonempty tuple of TokenMetadata")
    sequence_length = len(metadata)
    allowed = torch.zeros(sequence_length, sequence_length, dtype=torch.bool)
    for query_index, query in enumerate(metadata):
        for key_index, key in enumerate(metadata):
            if not key.valid:
                continue
            if query.is_condition:
                allowed[query_index, key_index] = key.is_condition
            elif key.is_condition:
                allowed[query_index, key_index] = True
            else:
                allowed[query_index, key_index] = key.block_id <= query.block_id
    if (~allowed.any(dim=1)).any().item():
        raise ValueError("every token query requires at least one structurally valid key")
    return allowed
