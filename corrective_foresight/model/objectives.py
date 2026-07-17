from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

import torch
from torch import Tensor, nn

from corrective_foresight.model.flow import (
    FlowTrainingSample,
    masked_velocity_mse,
    sample_flow_training,
)
from corrective_foresight.model.masked_reductions import (
    masked_mean,
    masked_mse,
    masked_smooth_l1,
)
from corrective_foresight.model.metrics import dynamics_diagnostics
from corrective_foresight.model.world_action_transformer import WorldActionTransformer


DYNAMICS_HORIZONS = (1, 2, 4, 8)
_RAW_HORIZON_WEIGHTS = (1.0, 0.8, 0.64, 0.512)
DEFAULT_OPTIMIZED_TERMS = frozenset(
    {
        "dynamics_loss",
        "inverse_action_loss",
        "action_cycle_loss",
        "policy_flow_loss",
    }
)
REQUIRED_METRICS = frozenset(
    {
        "train_loss",
        "val_loss",
        "dynamics_loss",
        "visual_loss",
        "visual_mse_loss",
        "visual_delta_loss",
        "visual_cosine_loss",
        "policy_flow_loss",
        "action_loss",
        "action_horizon_loss",
        "inverse_action_loss",
        "inverse_action_mse",
        "self_correction_cycle_loss",
        "copy_last_mse",
        "improvement_vs_copy_last",
        "pred_token_std",
        "target_token_std",
        "pred_delta_std",
        "target_delta_std",
    }
)


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    dynamics_weight: float = 1.0
    inverse_weight: float = 1.0
    action_cycle_weight: float = 0.1
    policy_flow_weight: float = 1.0
    cycle_warmup_steps: int = 5000

    def __post_init__(self) -> None:
        if any(
            type(value) is not float
            for value in (
                self.dynamics_weight,
                self.inverse_weight,
                self.action_cycle_weight,
                self.policy_flow_weight,
            )
        ) or type(self.cycle_warmup_steps) is not int:
            raise ValueError("objective weights must be floats and warmup must be int")
        actual = (
            self.dynamics_weight,
            self.inverse_weight,
            self.action_cycle_weight,
            self.policy_flow_weight,
        )
        if actual != (1.0, 1.0, 0.1, 1.0) or self.cycle_warmup_steps != 5000:
            raise ValueError("default objective weights and warmup are fixed")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ObjectiveConfig:
        expected = {
            "dynamics_weight",
            "inverse_weight",
            "action_cycle_weight",
            "policy_flow_weight",
            "cycle_warmup_steps",
        }
        extras = set(value) - expected
        if extras:
            raise ValueError(f"forbidden objective configuration: {sorted(extras)}")
        if set(value) != expected:
            raise ValueError(f"objective config fields must be exactly {sorted(expected)}")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ObjectiveInputs:
    condition_tokens: Tensor
    online_states: Tensor
    target_states: Tensor
    normalized_actions: Tensor
    observation_valid_mask: Tensor
    transition_valid_mask: Tensor
    action_dimension_mask: Tensor
    delta_time: Tensor
    context_index: int
    action_spec_ids: tuple[str, ...]
    global_step: int


@dataclass(frozen=True, slots=True)
class DynamicsComputation:
    loss: Tensor
    horizon_losses: Mapping[int, Tensor]
    one_step_predicted_states: Tensor
    one_step_predicted_deltas: Tensor
    one_step_valid_mask: Tensor


@dataclass(frozen=True, slots=True)
class PolicyFlowLosses:
    velocity_loss: Tensor
    clean_action_mse: Tensor


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    total_loss: Tensor
    optimized_terms: frozenset[str]
    losses: Mapping[str, Tensor]
    metrics: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if self.optimized_terms != DEFAULT_OPTIMIZED_TERMS:
            raise ValueError("optimized objective whitelist has changed")
        if set(self.losses) != set(DEFAULT_OPTIMIZED_TERMS):
            raise ValueError("loss mapping must contain exactly four optimized terms")
        if set(self.metrics) != set(REQUIRED_METRICS):
            raise ValueError("metric mapping does not match the required logging surface")
        for name, value in (
            ("total_loss", self.total_loss),
            *self.losses.items(),
            *self.metrics.items(),
        ):
            if (
                not isinstance(value, Tensor)
                or value.ndim != 0
                or not value.is_floating_point()
                or not torch.isfinite(value).item()
            ):
                raise ValueError(f"{name} must be a finite floating-point scalar")


class CorrectiveForesightObjective(nn.Module):
    def __init__(
        self,
        model: WorldActionTransformer,
        config: ObjectiveConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, WorldActionTransformer):
            raise ValueError("objective requires WorldActionTransformer")
        if model.config.cycle_noise_std != 0.01 or model.config.cycle_dropout != 0.05:
            raise ValueError("main objective requires cycle noise 0.01 and dropout 0.05")
        self.model = model
        self.config = config or ObjectiveConfig()

    def forward(
        self,
        inputs: ObjectiveInputs,
        *,
        flow_generator: torch.Generator,
    ) -> ObjectiveResult:
        self._validate_inputs(inputs)
        action_mask = (
            inputs.transition_valid_mask[..., None]
            & inputs.action_dimension_mask
        )
        safe_actions = torch.where(
            action_mask, inputs.normalized_actions, torch.zeros_like(inputs.normalized_actions)
        )
        dynamics = compute_recursive_dynamics(
            model=self.model,
            condition_tokens=inputs.condition_tokens,
            online_states=inputs.online_states,
            target_states=inputs.target_states,
            normalized_actions=safe_actions,
            transition_valid_mask=inputs.transition_valid_mask,
            action_dimension_mask=inputs.action_dimension_mask,
            delta_time=inputs.delta_time,
            action_spec_ids=inputs.action_spec_ids,
        )

        transition_state_mask = inputs.transition_valid_mask[..., None, None]
        safe_current_states = torch.where(
            transition_state_mask,
            inputs.online_states[:, :-1],
            torch.zeros_like(inputs.online_states[:, :-1]),
        )
        target_delta = inputs.target_states[:, 1:] - inputs.target_states[:, :-1]
        safe_target_delta = torch.where(
            transition_state_mask,
            target_delta,
            torch.zeros_like(target_delta),
        )
        inverse = self.model.predict_inverse(
            inputs.condition_tokens,
            safe_current_states,
            safe_target_delta,
            inputs.action_spec_ids,
        )
        inverse_loss = diagonal_gaussian_nll(
            inverse.mean,
            inverse.log_variance,
            safe_actions,
            action_mask,
        )
        inverse_mse = masked_mse(
            inverse.mean,
            safe_actions,
            action_mask,
            objective_name="inverse_action_mse",
        )

        cycle = self.model.predict_cycle(
            inputs.condition_tokens,
            safe_current_states,
            dynamics.one_step_predicted_deltas,
            inputs.action_spec_ids,
        )
        cycle_loss = masked_action_cycle_loss(
            cycle.mean,
            safe_actions,
            action_mask,
        )

        start = inputs.context_index
        end = start + self.model.config.action_horizon
        policy_action = safe_actions[:, start:end]
        policy_mask = action_mask[:, start:end]
        flow_sample = sample_flow_training(
            policy_action,
            generator=flow_generator,
            valid_mask=policy_mask,
        )
        observed_state_mask = inputs.observation_valid_mask[:, : start + 1]
        observed_states = torch.where(
            observed_state_mask[..., None, None],
            inputs.online_states[:, : start + 1],
            torch.zeros_like(inputs.online_states[:, : start + 1]),
        )
        policy = self.model.predict_policy_velocity(
            inputs.condition_tokens,
            observed_states,
            flow_sample.x_t,
            flow_sample.t,
            inputs.action_spec_ids,
            observed_state_valid_mask=observed_state_mask,
        )
        policy_losses = compute_policy_flow_losses(
            policy.velocity,
            flow_sample,
            policy_action,
            policy_mask,
        )

        losses = {
            "dynamics_loss": dynamics.loss,
            "inverse_action_loss": inverse_loss,
            "action_cycle_loss": cycle_loss,
            "policy_flow_loss": policy_losses.velocity_loss,
        }
        cycle_weight = action_cycle_weight(inputs.global_step)
        total_loss = (
            losses["dynamics_loss"]
            + losses["inverse_action_loss"]
            + cycle_weight * losses["action_cycle_loss"]
            + losses["policy_flow_loss"]
        )
        one_step_targets = inputs.target_states[:, 1:]
        one_step_bases = inputs.target_states[:, :-1]
        one_step_target_delta = one_step_targets - one_step_bases
        diagnostic_metrics = dynamics_diagnostics(
            predicted_states=dynamics.one_step_predicted_states,
            target_states=one_step_targets,
            base_states=one_step_bases,
            predicted_deltas=dynamics.one_step_predicted_deltas,
            target_deltas=one_step_target_delta,
            transition_mask=dynamics.one_step_valid_mask,
        )
        detached_total = total_loss.detach()
        metrics = {
            "train_loss": detached_total,
            "val_loss": detached_total,
            "dynamics_loss": dynamics.loss.detach(),
            "visual_loss": dynamics.loss.detach(),
            **{name: value.detach() for name, value in diagnostic_metrics.items()},
            "policy_flow_loss": policy_losses.velocity_loss.detach(),
            "action_loss": policy_losses.velocity_loss.detach(),
            "action_horizon_loss": policy_losses.clean_action_mse.detach(),
            "inverse_action_loss": inverse_loss.detach(),
            "inverse_action_mse": inverse_mse.detach(),
            "self_correction_cycle_loss": cycle_loss.detach(),
        }
        return ObjectiveResult(
            total_loss=total_loss,
            optimized_terms=DEFAULT_OPTIMIZED_TERMS,
            losses=MappingProxyType(losses),
            metrics=MappingProxyType(metrics),
        )

    def _validate_inputs(self, inputs: ObjectiveInputs) -> None:
        if not isinstance(inputs, ObjectiveInputs):
            raise ValueError("inputs must be ObjectiveInputs")
        condition = inputs.condition_tokens
        online = inputs.online_states
        target = inputs.target_states
        if (
            condition.ndim != 3
            or condition.shape[-1] != self.model.config.hidden_size
            or not condition.is_floating_point()
            or not torch.isfinite(condition).all().item()
        ):
            raise ValueError("condition_tokens must be finite [B,C,H]")
        batch_size = condition.shape[0]
        if (
            online.ndim != 4
            or online.shape != target.shape
            or online.shape[0] != batch_size
            or online.shape[1] < 9
            or online.shape[2:] != (
                self.model.config.state_tokens,
                self.model.config.hidden_size,
            )
        ):
            raise ValueError("online/target states must share shape [B,T>=9,9,H]")
        if not online.is_floating_point() or not target.is_floating_point():
            raise ValueError("online/target states must be floating point")
        if target.requires_grad:
            raise ValueError("EMA target states must be stop-gradient")
        transitions = online.shape[1] - 1
        expected_action_shape = (
            batch_size,
            transitions,
            self.model.action_adapters.resolve_batch(
                inputs.action_spec_ids
            ).spec.dimension,
        )
        if inputs.normalized_actions.shape != expected_action_shape:
            raise ValueError(f"normalized_actions must have shape {expected_action_shape}")
        if not inputs.normalized_actions.is_floating_point():
            raise ValueError("normalized_actions must be floating point")
        if inputs.action_dimension_mask.shape != expected_action_shape:
            raise ValueError("action_dimension_mask must match normalized_actions")
        if inputs.action_dimension_mask.dtype is not torch.bool:
            raise ValueError("action_dimension_mask must be bool")
        if inputs.observation_valid_mask.shape != (batch_size, transitions + 1):
            raise ValueError("observation_valid_mask must have shape [B,T]")
        if inputs.observation_valid_mask.dtype is not torch.bool:
            raise ValueError("observation_valid_mask must be bool")
        if inputs.transition_valid_mask.shape != (batch_size, transitions):
            raise ValueError("transition_valid_mask must have shape [B,T-1]")
        if inputs.transition_valid_mask.dtype is not torch.bool:
            raise ValueError("transition_valid_mask must be bool")
        endpoints = (
            inputs.observation_valid_mask[:, :-1]
            & inputs.observation_valid_mask[:, 1:]
        )
        if (inputs.transition_valid_mask & ~endpoints).any().item():
            raise ValueError("valid transitions require both observation endpoints")
        if inputs.delta_time.shape != (batch_size, transitions):
            raise ValueError("delta_time must have shape [B,T-1]")
        if not inputs.delta_time.is_floating_point():
            raise ValueError("delta_time must be floating point")
        if (
            not torch.isfinite(inputs.delta_time[inputs.transition_valid_mask]).all().item()
            or (inputs.delta_time[inputs.transition_valid_mask] <= 0).any().item()
        ):
            raise ValueError("valid delta_time must be finite and positive")
        if type(inputs.context_index) is not int or not (
            0 <= inputs.context_index
            and inputs.context_index + self.model.config.action_horizon <= transitions
        ):
            raise ValueError("context_index must own a complete eight-action chunk")
        if type(inputs.global_step) is not int:
            raise ValueError("global_step must be an integer")
        if len(inputs.action_spec_ids) != batch_size:
            raise ValueError("action_spec_ids must match batch size")
        tensors = (
            online,
            target,
            inputs.normalized_actions,
            inputs.observation_valid_mask,
            inputs.transition_valid_mask,
            inputs.action_dimension_mask,
            inputs.delta_time,
        )
        if any(value.device != condition.device for value in tensors):
            raise ValueError("all objective tensors must share a device")
        state_mask = inputs.observation_valid_mask[..., None, None].expand_as(online)
        action_mask = (
            inputs.transition_valid_mask[..., None]
            & inputs.action_dimension_mask
        )
        for name, value, mask in (
            ("online states", online, state_mask),
            ("target states", target, state_mask),
            ("normalized actions", inputs.normalized_actions, action_mask),
        ):
            if not torch.isfinite(value[mask]).all().item():
                raise ValueError(f"{name} contain non-finite valid values")


def normalized_horizon_weights() -> dict[int, float]:
    denominator = sum(_RAW_HORIZON_WEIGHTS)
    return {
        horizon: raw / denominator
        for horizon, raw in zip(
            DYNAMICS_HORIZONS, _RAW_HORIZON_WEIGHTS, strict=True
        )
    }


def combine_horizon_losses(horizon_losses: Mapping[int, Tensor]) -> Tensor:
    if set(horizon_losses) != set(DYNAMICS_HORIZONS):
        raise ValueError("dynamics horizon losses must contain exactly 1, 2, 4, and 8")
    weights = normalized_horizon_weights()
    total: Tensor | None = None
    for horizon in DYNAMICS_HORIZONS:
        loss = horizon_losses[horizon]
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise ValueError(f"dynamics horizon {horizon} loss must be finite scalar")
        weighted = loss.float() * weights[horizon]
        total = weighted if total is None else total + weighted
    if total is None:
        raise RuntimeError("fixed dynamics horizons unexpectedly empty")
    return total


def horizon_chain_mask(transition_mask: Tensor, horizon: int) -> Tensor:
    if transition_mask.dtype is not torch.bool or transition_mask.ndim != 2:
        raise ValueError("transition_mask must be bool [B,N]")
    if type(horizon) is not int or horizon <= 0 or horizon > transition_mask.shape[1]:
        raise ValueError("horizon must fit the transition sequence")
    return transition_mask.unfold(1, horizon, 1).all(dim=-1)


def compute_recursive_dynamics(
    *,
    model,
    condition_tokens: Tensor,
    online_states: Tensor,
    target_states: Tensor,
    normalized_actions: Tensor,
    transition_valid_mask: Tensor,
    action_dimension_mask: Tensor,
    delta_time: Tensor,
    action_spec_ids: Sequence[str],
) -> DynamicsComputation:
    if online_states.ndim != 4 or online_states.shape != target_states.shape:
        raise ValueError("dynamics states must share shape [B,T,9,H]")
    batch_size, time_steps, state_tokens, hidden_size = online_states.shape
    transitions = time_steps - 1
    if transitions < max(DYNAMICS_HORIZONS):
        raise ValueError("dynamics requires at least eight transitions")
    if normalized_actions.shape[:2] != (batch_size, transitions):
        raise ValueError("dynamics actions must align with state transitions")
    if action_dimension_mask.shape != normalized_actions.shape:
        raise ValueError("dynamics action mask must match actions")
    if (
        transition_valid_mask.dtype is not torch.bool
        or transition_valid_mask.shape != (batch_size, transitions)
        or action_dimension_mask.dtype is not torch.bool
    ):
        raise ValueError("dynamics masks have invalid shape or dtype")
    if delta_time.shape != (batch_size, transitions):
        raise ValueError("dynamics delta_time must align with transitions")

    per_horizon_predictions: dict[int, list[Tensor]] = {
        horizon: [] for horizon in DYNAMICS_HORIZONS
    }
    per_horizon_targets: dict[int, list[Tensor]] = {
        horizon: [] for horizon in DYNAMICS_HORIZONS
    }
    per_horizon_valid: dict[int, list[Tensor]] = {
        horizon: [] for horizon in DYNAMICS_HORIZONS
    }
    one_step_states: list[Tensor] = []
    one_step_deltas: list[Tensor] = []
    for origin in range(transitions):
        rollout_length = min(max(DYNAMICS_HORIZONS), transitions - origin)
        rollout_transition_mask = transition_valid_mask[
            :, origin : origin + rollout_length
        ]
        rollout_action_mask = (
            rollout_transition_mask[..., None]
            & action_dimension_mask[:, origin : origin + rollout_length]
        )
        safe_initial = torch.where(
            transition_valid_mask[:, origin, None, None],
            online_states[:, origin],
            torch.zeros_like(online_states[:, origin]),
        )
        rollout_actions = normalized_actions[:, origin : origin + rollout_length]
        safe_actions = torch.where(
            rollout_action_mask, rollout_actions, torch.zeros_like(rollout_actions)
        )
        rollout_delta_time = delta_time[:, origin : origin + rollout_length]
        safe_delta_time = torch.where(
            rollout_transition_mask,
            rollout_delta_time,
            torch.ones_like(rollout_delta_time),
        )
        prediction = model.predict_delta(
            condition_tokens,
            safe_initial,
            safe_actions,
            safe_delta_time,
            action_spec_ids,
            start_time=origin,
        )
        one_step_states.append(prediction.states[:, 1])
        one_step_deltas.append(prediction.delta[:, 0])
        for horizon in DYNAMICS_HORIZONS:
            if horizon > rollout_length:
                continue
            valid_chain = rollout_transition_mask[:, :horizon].all(dim=1)
            per_horizon_predictions[horizon].append(prediction.states[:, horizon])
            per_horizon_targets[horizon].append(target_states[:, origin + horizon])
            per_horizon_valid[horizon].append(valid_chain)

    horizon_losses: dict[int, Tensor] = {}
    for horizon in DYNAMICS_HORIZONS:
        predicted = torch.stack(per_horizon_predictions[horizon], dim=1)
        target = torch.stack(per_horizon_targets[horizon], dim=1)
        valid = torch.stack(per_horizon_valid[horizon], dim=1)
        feature_mask = valid[..., None, None].expand(
            -1, -1, state_tokens, hidden_size
        )
        horizon_losses[horizon] = masked_smooth_l1(
            predicted,
            target,
            feature_mask,
            objective_name=f"dynamics_horizon_{horizon}",
        )
    return DynamicsComputation(
        loss=combine_horizon_losses(horizon_losses),
        horizon_losses=MappingProxyType(horizon_losses),
        one_step_predicted_states=torch.stack(one_step_states, dim=1),
        one_step_predicted_deltas=torch.stack(one_step_deltas, dim=1),
        one_step_valid_mask=transition_valid_mask,
    )


def diagonal_gaussian_nll(
    mean: Tensor,
    log_variance: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    if mean.shape != log_variance.shape or mean.shape != target.shape:
        raise ValueError("inverse Gaussian tensors must share shape")
    if not log_variance.is_floating_point():
        raise ValueError("inverse log variance must be floating point")
    valid_log_variance = log_variance[mask]
    if (
        valid_log_variance.numel() == 0
        or not torch.isfinite(valid_log_variance).all().item()
        or (valid_log_variance < -10.0).any().item()
        or (valid_log_variance > 2.0).any().item()
    ):
        raise ValueError("inverse log variance must be finite within [-10, 2]")
    safe_mean = torch.where(mask, mean, torch.zeros_like(mean)).float()
    safe_target = torch.where(mask, target, torch.zeros_like(target)).float()
    safe_log_variance = torch.where(
        mask, log_variance, torch.zeros_like(log_variance)
    ).float()
    nll = 0.5 * (
        math.log(2.0 * math.pi)
        + safe_log_variance
        + (safe_target - safe_mean).square() * torch.exp(-safe_log_variance)
    )
    return masked_mean(nll, mask, objective_name="inverse_action_loss")


def masked_action_cycle_loss(
    recovered_mean: Tensor,
    demonstrated_action: Tensor,
    mask: Tensor,
) -> Tensor:
    return masked_smooth_l1(
        recovered_mean,
        demonstrated_action,
        mask,
        objective_name="action_cycle_loss",
    )


def action_cycle_weight(global_step: int) -> float:
    if type(global_step) is not int:
        raise ValueError("global_step must be an integer")
    return 0.1 * min(max(global_step, 0) / 5000.0, 1.0)


def compute_policy_flow_losses(
    predicted_velocity: Tensor,
    sample: FlowTrainingSample,
    target_action: Tensor,
    mask: Tensor,
) -> PolicyFlowLosses:
    velocity_loss = masked_velocity_mse(
        predicted_velocity,
        sample.target_velocity,
        mask,
    )
    clean_action_mse = masked_mse(
        sample.clean_estimate(predicted_velocity),
        target_action,
        mask,
        objective_name="action_horizon_loss",
    )
    return PolicyFlowLosses(
        velocity_loss=velocity_loss,
        clean_action_mse=clean_action_mse,
    )
