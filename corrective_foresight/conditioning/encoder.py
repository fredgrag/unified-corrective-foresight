from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from corrective_foresight.conditioning.language_cache import LanguageEmbeddingCache
from corrective_foresight.conditioning.vocabulary import (
    CONDITION_NAMESPACES,
    ConditionVocabulary,
)


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        vocabulary: ConditionVocabulary,
        hidden_dim: int = 768,
        language_embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or language_embedding_dim <= 0:
            raise ValueError("condition dimensions must be positive")
        self.vocabulary = vocabulary
        self.hidden_dim = hidden_dim
        self.language_embedding_dim = language_embedding_dim
        self.discrete_embeddings = nn.ModuleDict(
            {
                namespace: nn.Embedding(vocabulary.size(namespace), hidden_dim)
                for namespace in CONDITION_NAMESPACES
            }
        )
        self.language_projection = nn.Linear(language_embedding_dim, hidden_dim)
        self.null_language = nn.Parameter(torch.empty(hidden_dim))
        self.token_type_embedding = nn.Parameter(
            torch.empty(len(CONDITION_NAMESPACES) + 1, hidden_dim)
        )
        nn.init.normal_(self.null_language, std=0.02)
        nn.init.normal_(self.token_type_embedding, std=0.02)

    @property
    def vocabulary_hash(self) -> str:
        return self.vocabulary.content_hash

    def forward(
        self,
        condition_ids: Mapping[str, Tensor],
        task_text: Sequence[str | None],
        language_cache: LanguageEmbeddingCache | None = None,
    ) -> Tensor:
        if set(condition_ids) != set(CONDITION_NAMESPACES):
            raise ValueError(
                "condition identifier namespaces must be exactly "
                f"{sorted(CONDITION_NAMESPACES)}"
            )
        batch_size = len(task_text)
        if batch_size <= 0:
            raise ValueError("condition batch cannot be empty")

        discrete_tokens: list[Tensor] = []
        device: torch.device | None = None
        for namespace in CONDITION_NAMESPACES:
            identifiers = condition_ids[namespace]
            if identifiers.shape != (batch_size,) or identifiers.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError(f"{namespace} identifiers must be integer shape [B]")
            if (identifiers < 0).any().item() or (
                identifiers >= self.vocabulary.size(namespace)
            ).any().item():
                raise ValueError(f"{namespace} identifier is out of range")
            if device is None:
                device = identifiers.device
            elif identifiers.device != device:
                raise ValueError("all condition identifiers must share one device")
            discrete_tokens.append(self.discrete_embeddings[namespace](identifiers))
        discrete = torch.stack(discrete_tokens, dim=1)

        if any(text is not None for text in task_text):
            if language_cache is None:
                raise ValueError("non-null task text requires a language cache")
            if language_cache.dimension != self.language_embedding_dim:
                raise ValueError(
                    "language cache dimension does not match ConditionEncoder"
                )
        language_tokens: list[Tensor] = []
        for text in task_text:
            if text is None:
                language_tokens.append(self.null_language)
            else:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("task text must be a nonempty string or null")
                embedding = language_cache.lookup(text).to(
                    device=device,
                    dtype=self.language_projection.weight.dtype,
                )
                language_tokens.append(self.language_projection(embedding))
        language = torch.stack(language_tokens, dim=0)[:, None, :]
        tokens = torch.cat((discrete, language), dim=1)
        return tokens + self.token_type_embedding[None, :, :]
