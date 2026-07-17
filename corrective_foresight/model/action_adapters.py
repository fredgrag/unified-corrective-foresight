from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

import torch
from torch import Tensor, nn

from corrective_foresight.config.schema import ActionSpec


class ActionAdapter(nn.Module):
    def __init__(self, spec: ActionSpec, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.spec = spec
        self.hidden_size = hidden_size
        self.action_input_projection = nn.Linear(spec.dimension, hidden_size)
        self.flow_state_projection = nn.Linear(spec.dimension, hidden_size)
        self.flow_velocity_head = nn.Linear(hidden_size, spec.dimension)
        self.inverse_mean_head = nn.Linear(hidden_size, spec.dimension)
        self.inverse_logvar_head = nn.Linear(hidden_size, spec.dimension)

    def normalize(self, action: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.spec.normalize(action, mask)

    def denormalize(self, action: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.spec.denormalize(action, mask)

    def project_action(self, normalized_action: Tensor) -> Tensor:
        self._require_action(normalized_action)
        return self.action_input_projection(normalized_action)

    def project_flow_state(self, noisy_action: Tensor) -> Tensor:
        self._require_action(noisy_action)
        return self.flow_state_projection(noisy_action)

    def predict_flow_velocity(self, hidden: Tensor) -> Tensor:
        self._require_hidden(hidden)
        return self.flow_velocity_head(hidden)

    def predict_inverse(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        self._require_hidden(hidden)
        return self.inverse_mean_head(hidden), self.inverse_logvar_head(hidden)

    def _require_action(self, action: Tensor) -> None:
        if (
            action.ndim < 2
            or action.shape[-1] != self.spec.dimension
            or not action.is_floating_point()
        ):
            raise ValueError(
                f"action must be floating point with final dimension {self.spec.dimension}"
            )
        if not torch.isfinite(action).all().item():
            raise ValueError("projected action input must be finite")

    def _require_hidden(self, hidden: Tensor) -> None:
        if (
            hidden.ndim < 2
            or hidden.shape[-1] != self.hidden_size
            or not hidden.is_floating_point()
        ):
            raise ValueError(
                f"hidden input must be floating point with final dimension {self.hidden_size}"
            )
        if not torch.isfinite(hidden).all().item():
            raise ValueError("action head hidden input must be finite")


class ActionAdapterRegistry(nn.Module):
    def __init__(self, specs: Sequence[ActionSpec], hidden_size: int = 768) -> None:
        super().__init__()
        if not specs:
            raise ValueError("ActionAdapterRegistry requires at least one ActionSpec")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        spec_by_id: dict[str, ActionSpec] = {}
        key_by_id: dict[str, str] = {}
        adapters: dict[str, ActionAdapter] = {}
        for spec in specs:
            if not isinstance(spec, ActionSpec):
                raise ValueError("registry entries must be ActionSpec")
            if spec.spec_id in spec_by_id:
                raise ValueError(f"duplicate ActionSpec id: {spec.spec_id}")
            key = spec.content_hash
            spec_by_id[spec.spec_id] = spec
            key_by_id[spec.spec_id] = key
            adapters[key] = ActionAdapter(spec, hidden_size)
        self.hidden_size = hidden_size
        self.adapters = nn.ModuleDict(adapters)
        self.specs = MappingProxyType(spec_by_id)
        self._key_by_id = MappingProxyType(key_by_id)

    def resolve(self, spec_id: str) -> ActionAdapter:
        try:
            key = self._key_by_id[spec_id]
        except KeyError as error:
            raise KeyError(f"unknown ActionSpec: {spec_id}") from error
        return self.adapters[key]

    def resolve_batch(self, spec_ids: Sequence[str]) -> ActionAdapter:
        if not spec_ids:
            raise ValueError("ActionSpec batch identifiers cannot be empty")
        unique = set(spec_ids)
        if len(unique) != 1:
            raise ValueError(f"mixed ActionSpec batch is forbidden: {sorted(unique)}")
        return self.resolve(next(iter(unique)))
