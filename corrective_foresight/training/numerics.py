from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor, nn


class NumericalGuardError(RuntimeError):
    pass


def require_finite_objective(
    value: Tensor,
    *,
    objective_name: str,
    dataset_id: str,
    rank: int,
    global_step: int,
) -> None:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 0
        or not value.is_floating_point()
        or not torch.isfinite(value).item()
    ):
        raise NumericalGuardError(
            _context(
                f"non-finite objective {objective_name}",
                dataset_id,
                rank,
                global_step,
            )
        )


def per_loss_gradient_norms(
    losses: Mapping[str, Tensor],
    parameters: Sequence[nn.Parameter],
    *,
    dataset_id: str,
    rank: int,
    global_step: int,
) -> dict[str, Tensor]:
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not trainable:
        raise ValueError("shared parameter sequence cannot be empty")
    norms: dict[str, Tensor] = {}
    for name, loss in losses.items():
        require_finite_objective(
            loss,
            objective_name=name,
            dataset_id=dataset_id,
            rank=rank,
            global_step=global_step,
        )
        gradients = torch.autograd.grad(
            loss.float(),
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        squared_norm: Tensor | None = None
        for parameter, gradient in zip(trainable, gradients, strict=True):
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all().item():
                raise NumericalGuardError(
                    _context(
                        f"non-finite per-loss gradient {name} for shape "
                        f"{tuple(parameter.shape)}",
                        dataset_id,
                        rank,
                        global_step,
                    )
                )
            value = gradient.float().square().sum()
            squared_norm = value if squared_norm is None else squared_norm + value
        if squared_norm is None:
            raise NumericalGuardError(
                _context(
                    f"objective {name} has no shared-parameter gradient",
                    dataset_id,
                    rank,
                    global_step,
                )
            )
        norms[name] = squared_norm.sqrt().detach()
    return norms


def clip_and_validate_gradients(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    *,
    max_norm: float,
    dataset_id: str,
    rank: int,
    global_step: int,
) -> Tensor:
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    parameters: list[nn.Parameter] = []
    squared_norm: Tensor | None = None
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        if not torch.isfinite(gradient).all().item():
            raise NumericalGuardError(
                _context(
                    f"non-finite gradient in {name}",
                    dataset_id,
                    rank,
                    global_step,
                )
            )
        parameters.append(parameter)
        value = gradient.float().square().sum()
        squared_norm = value if squared_norm is None else squared_norm + value
    if not parameters or squared_norm is None:
        raise NumericalGuardError(
            _context("no gradients before optimizer step", dataset_id, rank, global_step)
        )
    pre_clip_norm = squared_norm.sqrt()
    torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        error_if_nonfinite=True,
    )
    return pre_clip_norm.detach()


def _context(message: str, dataset_id: str, rank: int, global_step: int) -> str:
    return (
        f"{message}; dataset={dataset_id}; rank={rank}; "
        f"step={global_step}"
    )
