from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from corrective_foresight.policy.unified_policy import (
    UnifiedCorrectiveForesightPolicy,
)


@dataclass(frozen=True, slots=True)
class OptimizerParameterGroups:
    protected: tuple[nn.Parameter, ...]
    action: tuple[nn.Parameter, ...]

    @property
    def names(self) -> tuple[str, str]:
        return ("protected", "action")

    def validate(self, policy: UnifiedCorrectiveForesightPolicy) -> None:
        protected_ids = {id(item) for item in self.protected}
        action_ids = {id(item) for item in self.action}
        if len(protected_ids) != len(self.protected):
            raise ValueError("protected optimizer parameters contain duplicates")
        if len(action_ids) != len(self.action):
            raise ValueError("action optimizer parameters contain duplicates")
        if protected_ids & action_ids:
            raise ValueError("optimizer parameter groups overlap")
        trainable_ids = {
            id(item) for item in policy.parameters() if item.requires_grad
        }
        if protected_ids | action_ids != trainable_ids:
            raise ValueError("optimizer parameter groups are not exhaustive")


def build_optimizer_parameter_groups(
    policy: UnifiedCorrectiveForesightPolicy,
) -> OptimizerParameterGroups:
    if not isinstance(policy, UnifiedCorrectiveForesightPolicy):
        raise ValueError("optimizer groups require UnifiedCorrectiveForesightPolicy")
    model = policy.world_action_model
    protected: list[nn.Parameter] = []
    action: list[nn.Parameter] = []

    _extend(protected, policy.online_state_encoder.adapter)
    _extend(protected, policy.condition_encoder)
    _extend(protected, model.transformer)
    for module in (
        model.role_embedding,
        model.modality_embedding,
        model.timestep_embedding,
        model.query_position_embedding,
        model.continuous_time_embedding,
        model.delta_projection,
        model.delta_norm,
    ):
        _extend(protected, module)
    protected.append(model.delta_queries)

    action.append(model.action_query)
    action.append(model.policy_queries)
    _extend(action, model.policy_horizon_embedding)

    for key in sorted(model.action_adapters.adapters):
        adapter = model.action_adapters.adapters[key]
        _extend(protected, adapter.action_input_projection)
        for module in (
            adapter.flow_state_projection,
            adapter.flow_velocity_head,
            adapter.inverse_mean_head,
            adapter.inverse_logvar_head,
        ):
            _extend(action, module)

    groups = OptimizerParameterGroups(
        protected=tuple(protected),
        action=tuple(action),
    )
    groups.validate(policy)
    return groups


def _extend(destination: list[nn.Parameter], module: nn.Module) -> None:
    destination.extend(
        parameter for parameter in module.parameters() if parameter.requires_grad
    )
