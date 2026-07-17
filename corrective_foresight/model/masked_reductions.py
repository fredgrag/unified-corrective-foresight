from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_mean(values: Tensor, mask: Tensor, *, objective_name: str) -> Tensor:
    _validate_value_mask(values, mask, objective_name)
    valid_values = values[mask].float()
    if not torch.isfinite(valid_values).all().item():
        raise ValueError(f"{objective_name} contains non-finite valid values")
    return valid_values.mean(dtype=torch.float32)


def masked_mse(
    predicted: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    objective_name: str,
) -> Tensor:
    safe_predicted, safe_target = _safe_pair(
        predicted, target, mask, objective_name
    )
    squared_error = (safe_predicted.float() - safe_target.float()).square()
    return masked_mean(squared_error, mask, objective_name=objective_name)


def masked_smooth_l1(
    predicted: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    objective_name: str,
) -> Tensor:
    safe_predicted, safe_target = _safe_pair(
        predicted, target, mask, objective_name
    )
    error = F.smooth_l1_loss(
        safe_predicted.float(),
        safe_target.float(),
        reduction="none",
        beta=1.0,
    )
    return masked_mean(error, mask, objective_name=objective_name)


def _safe_pair(
    predicted: Tensor,
    target: Tensor,
    mask: Tensor,
    objective_name: str,
) -> tuple[Tensor, Tensor]:
    if (
        predicted.shape != target.shape
        or predicted.shape != mask.shape
        or predicted.device != target.device
        or predicted.device != mask.device
    ):
        raise ValueError(
            f"{objective_name} predicted, target, and mask must share shape/device"
        )
    if not predicted.is_floating_point() or not target.is_floating_point():
        raise ValueError(f"{objective_name} tensors must be floating point")
    if mask.dtype is not torch.bool:
        raise ValueError(f"{objective_name} mask must be bool")
    if not mask.any().item():
        raise ValueError(f"{objective_name} has no valid elements")
    if not torch.isfinite(predicted[mask]).all().item() or not torch.isfinite(
        target[mask]
    ).all().item():
        raise ValueError(f"{objective_name} contains non-finite valid values")
    return (
        torch.where(mask, predicted, torch.zeros_like(predicted)),
        torch.where(mask, target, torch.zeros_like(target)),
    )


def _validate_value_mask(values: Tensor, mask: Tensor, objective_name: str) -> None:
    if not objective_name:
        raise ValueError("objective_name must be nonempty")
    if (
        values.shape != mask.shape
        or values.device != mask.device
        or mask.dtype is not torch.bool
    ):
        raise ValueError(f"{objective_name} values and bool mask must share shape/device")
    if not values.is_floating_point():
        raise ValueError(f"{objective_name} values must be floating point")
    if not mask.any().item():
        raise ValueError(f"{objective_name} has no valid elements")
