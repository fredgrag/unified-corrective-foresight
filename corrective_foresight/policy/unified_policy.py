from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor, nn

from corrective_foresight.conditioning.encoder import ConditionEncoder
from corrective_foresight.conditioning.language_cache import LanguageEmbeddingCache
from corrective_foresight.data.batch import TrajectoryBatch
from corrective_foresight.model.ema import EMAStateTarget
from corrective_foresight.model.flow import IntegrationReport, sample_rectified_flow
from corrective_foresight.model.metrics import dynamics_diagnostics
from corrective_foresight.model.objectives import (
    CorrectiveForesightObjective,
    ObjectiveInputs,
    compute_recursive_dynamics,
)
from corrective_foresight.model.state_encoder import OnlineStateEncoder
from corrective_foresight.model.world_action_transformer import WorldActionTransformer
from corrective_foresight.training.stages import (
    TrainingOutput,
    TrainingStage,
)


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    rgb: Tensor
    camera_mask: Tensor
    proprio: Tensor
    proprio_mask: Tensor
    observation_valid_mask: Tensor
    task_text: tuple[str | None, ...]
    condition_ids: Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class ActionChunkPrediction:
    normalized_actions: Tensor
    denormalized_actions: Tensor
    consistency: Tensor
    inverse_variance: Tensor
    solver_report: IntegrationReport

    def __post_init__(self) -> None:
        if (
            self.normalized_actions.ndim != 3
            or self.normalized_actions.shape[1] != 8
            or self.denormalized_actions.shape != self.normalized_actions.shape
            or self.consistency.shape != self.normalized_actions.shape[:2]
            or self.inverse_variance.shape != self.normalized_actions.shape
        ):
            raise ValueError("action chunk prediction shapes are inconsistent")
        for name, value in (
            ("normalized actions", self.normalized_actions),
            ("denormalized actions", self.denormalized_actions),
            ("consistency", self.consistency),
            ("inverse variance", self.inverse_variance),
        ):
            if not value.is_floating_point() or not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite floating point")


class UnifiedCorrectiveForesightPolicy(nn.Module):
    def __init__(
        self,
        *,
        online_state_encoder: OnlineStateEncoder,
        ema_state_target: EMAStateTarget,
        condition_encoder: ConditionEncoder,
        world_action_model: WorldActionTransformer,
        objective: CorrectiveForesightObjective,
        language_cache: LanguageEmbeddingCache | None,
    ) -> None:
        super().__init__()
        hidden_size = world_action_model.config.hidden_size
        if online_state_encoder.adapter.hidden_size != hidden_size:
            raise ValueError("online state encoder hidden size does not match model")
        if condition_encoder.hidden_dim != hidden_size:
            raise ValueError("condition encoder hidden size does not match model")
        if ema_state_target.online is not online_state_encoder:
            raise ValueError("EMA target must track the supplied online encoder")
        if objective.model is not world_action_model:
            raise ValueError("objective and policy must share the world-action model")
        self.online_state_encoder = online_state_encoder
        self.ema_state_target = ema_state_target
        self.condition_encoder = condition_encoder
        self.world_action_model = world_action_model
        object.__setattr__(self, "_objective", objective)
        self.language_cache = language_cache
        self._last_ema_step = -1

    @property
    def objective(self) -> CorrectiveForesightObjective:
        return object.__getattribute__(self, "_objective")

    @property
    def last_ema_step(self) -> int:
        return self._last_ema_step

    def forward(
        self,
        batch: TrajectoryBatch,
        stage: TrainingStage | str,
        global_step: int,
        generator: torch.Generator,
    ) -> TrainingOutput:
        return self.compute_training_objective(
            batch,
            stage,
            global_step,
            generator,
        )

    def compute_training_objective(
        self,
        batch: TrajectoryBatch,
        stage: TrainingStage | str,
        global_step: int,
        generator: torch.Generator,
    ) -> TrainingOutput:
        stage = TrainingStage.parse(stage)
        batch.validate(expected_action_spec_id=batch.action_spec_id)
        batch_size = batch.rgb.shape[0]
        condition_tokens = self.condition_encoder(
            batch.condition_ids,
            batch.task_text,
            self.language_cache,
        )
        safe_rgb, safe_camera_mask, safe_proprio, safe_proprio_mask = (
            self._encoder_inputs(
                batch.rgb,
                batch.camera_mask,
                batch.proprio,
                batch.proprio_mask,
                batch.observation_valid_mask,
            )
        )
        online_states = self.online_state_encoder(
            safe_rgb,
            safe_camera_mask,
            safe_proprio,
            safe_proprio_mask,
        )
        target_states = self.ema_state_target.encode_target(
            safe_rgb,
            safe_camera_mask,
            safe_proprio,
            safe_proprio_mask,
        )
        action_spec_ids = (batch.action_spec_id,) * batch_size
        if stage is TrainingStage.WORLD_PRETRAIN:
            dynamics = compute_recursive_dynamics(
                model=self.world_action_model,
                condition_tokens=condition_tokens,
                online_states=online_states,
                target_states=target_states,
                normalized_actions=batch.action,
                transition_valid_mask=batch.transition_valid_mask,
                action_dimension_mask=batch.action_dimension_mask,
                delta_time=batch.delta_time,
                action_spec_ids=action_spec_ids,
            )
            diagnostics = dynamics_diagnostics(
                predicted_states=dynamics.one_step_predicted_states,
                target_states=target_states[:, 1:],
                base_states=target_states[:, :-1],
                predicted_deltas=dynamics.one_step_predicted_deltas,
                target_deltas=target_states[:, 1:] - target_states[:, :-1],
                transition_mask=batch.transition_valid_mask,
            )
            losses = MappingProxyType({"dynamics_loss": dynamics.loss})
            metrics = MappingProxyType(
                {
                    "dynamics_loss": dynamics.loss.detach(),
                    "visual_loss": dynamics.loss.detach(),
                    **{name: value.detach() for name, value in diagnostics.items()},
                }
            )
            return TrainingOutput(
                stage=stage,
                loss=dynamics.loss,
                optimized_terms=frozenset({"dynamics_loss"}),
                losses=losses,
                metrics=metrics,
            )

        result = self.objective(
            ObjectiveInputs(
                condition_tokens=condition_tokens,
                online_states=online_states,
                target_states=target_states,
                normalized_actions=batch.action,
                observation_valid_mask=batch.observation_valid_mask,
                transition_valid_mask=batch.transition_valid_mask,
                action_dimension_mask=batch.action_dimension_mask,
                delta_time=batch.delta_time,
                context_index=batch.context_index,
                action_spec_ids=action_spec_ids,
                global_step=global_step,
            ),
            flow_generator=generator,
        )
        return TrainingOutput(
            stage=stage,
            loss=result.total_loss,
            optimized_terms=result.optimized_terms,
            losses=result.losses,
            metrics=result.metrics,
        )

    @torch.no_grad()
    def predict_action_chunk(
        self,
        observation: PolicyObservation,
        action_spec_id: str,
        flow_seed: int,
        solver: str,
    ) -> ActionChunkPrediction:
        self._validate_observation(observation)
        was_training = self.training
        self.eval()
        try:
            condition_tokens = self.condition_encoder(
                observation.condition_ids,
                observation.task_text,
                self.language_cache,
            )
            safe_rgb, safe_camera_mask, safe_proprio, safe_proprio_mask = (
                self._encoder_inputs(
                    observation.rgb,
                    observation.camera_mask,
                    observation.proprio,
                    observation.proprio_mask,
                    observation.observation_valid_mask,
                )
            )
            observed_states = self.online_state_encoder(
                safe_rgb,
                safe_camera_mask,
                safe_proprio,
                safe_proprio_mask,
            )
            adapter = self.world_action_model.action_adapters.resolve(action_spec_id)
            batch_size = observed_states.shape[0]
            action_spec_ids = (action_spec_id,) * batch_size

            def velocity_field(noisy_action: Tensor, flow_time: Tensor) -> Tensor:
                return self.world_action_model.predict_policy_velocity(
                    condition_tokens,
                    observed_states,
                    noisy_action,
                    flow_time,
                    action_spec_ids,
                    observed_state_valid_mask=observation.observation_valid_mask,
                ).velocity

            normalized_actions, report = sample_rectified_flow(
                velocity_field,
                (
                    batch_size,
                    self.world_action_model.config.action_horizon,
                    adapter.spec.dimension,
                ),
                solver=solver,
                intervals=10,
                noise_seed=flow_seed,
                device=observed_states.device,
                dtype=observed_states.dtype,
            )
            current_state = observed_states[:, -1]
            delta_time = normalized_actions.new_full(
                normalized_actions.shape[:2],
                1.0 / adapter.spec.frequency_hz,
            )
            rollout = self.world_action_model.predict_delta(
                condition_tokens,
                current_state,
                normalized_actions,
                delta_time,
                action_spec_ids,
            )
            recovered = self.world_action_model.predict_cycle(
                condition_tokens,
                rollout.states[:, :-1],
                rollout.delta,
                action_spec_ids,
            )
            consistency = (
                recovered.mean.float() - normalized_actions.float()
            ).square().mean(dim=-1)
            inverse_variance = recovered.log_variance.float().exp()
            denormalized = adapter.denormalize(normalized_actions)
            denormalized = adapter.spec.clamp_to_bounds(denormalized)
            return ActionChunkPrediction(
                normalized_actions=normalized_actions,
                denormalized_actions=denormalized,
                consistency=consistency,
                inverse_variance=inverse_variance,
                solver_report=report,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def update_ema(self, global_step: int) -> None:
        if type(global_step) is not int or global_step <= self._last_ema_step:
            raise ValueError("EMA global_step must be strictly increasing")
        self.ema_state_target.update()
        self._last_ema_step = global_step

    def restore_ema_step(self, value: int) -> None:
        if type(value) is not int or value < -1:
            raise ValueError("restored EMA step must be an integer >= -1")
        self._last_ema_step = value

    def _validate_observation(self, observation: PolicyObservation) -> None:
        if not isinstance(observation, PolicyObservation):
            raise ValueError("observation must be PolicyObservation")
        if observation.rgb.ndim != 6 or observation.rgb.shape[3] != 3:
            raise ValueError("observation RGB must have shape [B,T,V,3,H,W]")
        batch_size, time_steps = observation.rgb.shape[:2]
        if observation.observation_valid_mask.shape != (batch_size, time_steps):
            raise ValueError("observation_valid_mask must have shape [B,T]")
        if observation.observation_valid_mask.dtype is not torch.bool:
            raise ValueError("observation_valid_mask must be bool")
        if not observation.observation_valid_mask[:, -1].all().item():
            raise ValueError("the latest policy observation must be valid")
        if len(observation.task_text) != batch_size:
            raise ValueError("task_text must match observation batch size")

    @staticmethod
    def _encoder_inputs(
        rgb: Tensor,
        camera_mask: Tensor,
        proprio: Tensor,
        proprio_mask: Tensor,
        observation_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        invalid_observation = ~observation_valid_mask
        safe_rgb = torch.where(
            observation_valid_mask[..., None, None, None, None],
            rgb,
            torch.zeros_like(rgb),
        )
        safe_camera_mask = camera_mask.clone()
        safe_camera_mask[:, :, 0] |= invalid_observation
        safe_proprio = torch.where(
            observation_valid_mask[..., None],
            proprio,
            torch.zeros_like(proprio),
        )
        safe_proprio_mask = (
            proprio_mask & observation_valid_mask[..., None]
        )
        return safe_rgb, safe_camera_mask, safe_proprio, safe_proprio_mask
