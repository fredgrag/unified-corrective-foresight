from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from corrective_foresight.model.attention_contract import compile_attention_mask
from corrective_foresight.model.token_types import TokenView


class CausalTokenTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if min(hidden_size, num_layers, num_attention_heads, mlp_ratio) <= 0:
            raise ValueError("Transformer dimensions must be positive")
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.hidden_size = hidden_size
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList(
            _CausalTransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, view: TokenView) -> Tensor:
        if view.tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                f"token hidden dimension must be {self.hidden_size}, "
                f"got {view.tokens.shape[-1]}"
            )
        key_padding_mask = view.key_padding_mask.to(view.tokens.device)
        allowed = compile_attention_mask(view.metadata).to(view.tokens.device)
        attention_mask = ~allowed
        hidden = torch.where(
            key_padding_mask[..., None],
            torch.zeros_like(view.tokens),
            view.tokens,
        )
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(
                    layer,
                    hidden,
                    attention_mask,
                    key_padding_mask,
                    use_reentrant=False,
                )
            else:
                hidden = layer(hidden, attention_mask, key_padding_mask)
            hidden = hidden.masked_fill(key_padding_mask[..., None], 0.0)
        hidden = self.output_norm(hidden)
        return hidden.masked_fill(key_padding_mask[..., None], 0.0)


class _CausalTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            query=normalized,
            key=normalized,
            value=normalized,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = hidden + self.attention_dropout(attended)
        return hidden + self.mlp(self.mlp_norm(hidden))
