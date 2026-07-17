from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from corrective_foresight.model.masked_reductions import masked_mse


def dynamics_diagnostics(
    *,
    predicted_states: Tensor,
    target_states: Tensor,
    base_states: Tensor,
    predicted_deltas: Tensor,
    target_deltas: Tensor,
    transition_mask: Tensor,
) -> dict[str, Tensor]:
    expected = predicted_states.shape
    if (
        predicted_states.ndim != 4
        or min(expected) <= 0
        or target_states.shape != expected
        or base_states.shape != expected
        or predicted_deltas.shape != expected
        or target_deltas.shape != expected
    ):
        raise ValueError("dynamics diagnostics require matching [B,N,9,H] tensors")
    if transition_mask.dtype is not torch.bool or transition_mask.shape != expected[:2]:
        raise ValueError("transition_mask must be bool with shape [B,N]")
    feature_mask = transition_mask[..., None, None].expand(expected)
    token_mask = transition_mask[..., None].expand(expected[:-1])
    valid_predicted_states = predicted_states[token_mask].float()
    valid_target_states = target_states[token_mask].float()
    valid_predicted_deltas = predicted_deltas[token_mask].float()
    valid_target_deltas = target_deltas[token_mask].float()
    for name, value in (
        ("predicted states", valid_predicted_states),
        ("target states", valid_target_states),
        ("predicted deltas", valid_predicted_deltas),
        ("target deltas", valid_target_deltas),
    ):
        if value.numel() == 0:
            raise ValueError("dynamics diagnostics have no valid transitions")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} contain non-finite valid values")

    visual_mse = masked_mse(
        predicted_states,
        target_states,
        feature_mask,
        objective_name="visual_mse_loss",
    )
    visual_delta = masked_mse(
        predicted_deltas,
        target_deltas,
        feature_mask,
        objective_name="visual_delta_loss",
    )
    cosine = (
        1.0
        - F.cosine_similarity(
            valid_predicted_states,
            valid_target_states,
            dim=-1,
        )
    ).mean(dtype=torch.float32)
    copy_last = masked_mse(
        base_states,
        target_states,
        feature_mask,
        objective_name="copy_last_mse",
    )
    improvement = (copy_last - visual_mse) / (copy_last + 1e-8)
    return {
        "visual_mse_loss": visual_mse,
        "visual_delta_loss": visual_delta,
        "visual_cosine_loss": cosine,
        "copy_last_mse": copy_last,
        "improvement_vs_copy_last": improvement,
        "pred_token_std": _feature_std(valid_predicted_states),
        "target_token_std": _feature_std(valid_target_states),
        "pred_delta_std": _feature_std(valid_predicted_deltas),
        "target_delta_std": _feature_std(valid_target_deltas),
    }


def _feature_std(valid_tokens: Tensor) -> Tensor:
    return valid_tokens.reshape(-1, valid_tokens.shape[-1]).std(
        dim=0, unbiased=False
    ).mean(dtype=torch.float32)
