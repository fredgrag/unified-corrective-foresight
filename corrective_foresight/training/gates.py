from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from statistics import median
from types import MappingProxyType
import uuid

from corrective_foresight.training.validation import ValidationRecord


_AUDIT_STEPS = tuple(range(260, 501, 10))
_CONFLICT_THRESHOLD = -0.05
_DYNAMICS_RATIO_THRESHOLD = 1.20
_OPTIMIZED_VALIDATION_METRICS = (
    "dynamics_loss",
    "inverse_action_loss",
    "self_correction_cycle_loss",
    "policy_flow_loss",
)
_STOP_CONTEXT_FIELDS = {
    "resolved_config",
    "git_commit",
    "data_spec_hashes",
    "checkpoint_manifest_sha256",
    "stage",
    "checkpoint_step",
}


@dataclass(frozen=True, slots=True)
class AuditMeasurement:
    optimizer_step: int
    minimum_dynamics_cosine: float

    def __post_init__(self) -> None:
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("audit optimizer_step must be nonnegative")
        if (
            isinstance(self.minimum_dynamics_cosine, bool)
            or not isinstance(self.minimum_dynamics_cosine, (int, float))
            or not math.isfinite(float(self.minimum_dynamics_cosine))
            or not -1.0 <= float(self.minimum_dynamics_cosine) <= 1.0
        ):
            raise ValueError("audit cosine must be finite within [-1, 1]")


@dataclass(frozen=True, slots=True)
class AuditDecision:
    enable_pcgrad: bool
    measurement_count: int
    conflicting_measurements: int
    conflict_fraction: float
    conflict_threshold: float
    world_dynamics_loss: float
    audit_dynamics_loss: float
    dynamics_ratio: float
    dynamics_ratio_threshold: float
    optimizer_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.enable_pcgrad) is not bool:
            raise ValueError("audit enable_pcgrad must be bool")
        if (
            type(self.measurement_count) is not int
            or self.measurement_count != 25
            or type(self.conflicting_measurements) is not int
            or not 0 <= self.conflicting_measurements <= self.measurement_count
        ):
            raise ValueError("audit decision counts are invalid")
        if tuple(self.optimizer_steps) != _AUDIT_STEPS:
            raise ValueError("audit decision optimizer steps are invalid")
        expected_fraction = self.conflicting_measurements / self.measurement_count
        if not math.isclose(
            self.conflict_fraction,
            expected_fraction,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("audit decision conflict fraction is invalid")
        if self.conflict_threshold != _CONFLICT_THRESHOLD:
            raise ValueError("audit decision conflict threshold is invalid")
        world = _positive_finite(
            self.world_dynamics_loss,
            "world_dynamics_loss",
        )
        audit = _positive_finite(
            self.audit_dynamics_loss,
            "audit_dynamics_loss",
        )
        expected_ratio = audit / world
        if (
            self.dynamics_ratio_threshold != _DYNAMICS_RATIO_THRESHOLD
            or not math.isclose(
                self.dynamics_ratio,
                expected_ratio,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        ):
            raise ValueError("audit decision dynamics ratio is invalid")
        expected_enable = (
            self.conflicting_measurements >= 8
            and self.dynamics_ratio > _DYNAMICS_RATIO_THRESHOLD
        )
        if self.enable_pcgrad != expected_enable:
            raise ValueError("audit decision PCGrad result is inconsistent")


@dataclass(frozen=True, slots=True)
class PilotStepDecision:
    should_stop: bool
    reason: str | None
    triggering_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.should_stop) is not bool:
            raise ValueError("should_stop must be bool")
        if self.should_stop != (self.reason is not None):
            raise ValueError("stop decision and reason disagree")
        if self.reason is not None and not self.reason:
            raise ValueError("stop reason cannot be empty")
        if any(type(step) is not int or step < 0 for step in self.triggering_steps):
            raise ValueError("triggering steps must be nonnegative integers")
        if self.should_stop and len(self.triggering_steps) != 3:
            raise ValueError("stop decision requires three triggering steps")


@dataclass(frozen=True, slots=True)
class WorldGateResult:
    outcome: str
    accepted: bool
    requires_approval: bool
    improvement_vs_copy_last: float
    target: float
    reason: str

    def __post_init__(self) -> None:
        expected = {
            "accepted": (True, False, "target_met"),
            "marginal": (False, True, "positive_below_target"),
            "rejected": (False, False, "copy_last_not_improved"),
        }
        if self.outcome not in expected:
            raise ValueError("world gate outcome is invalid")
        if (
            (self.accepted, self.requires_approval, self.reason)
            != expected[self.outcome]
            or not math.isfinite(self.improvement_vs_copy_last)
            or not math.isfinite(self.target)
            or self.target <= 0.0
        ):
            raise ValueError("world gate result is inconsistent")
        derived = (
            "accepted"
            if self.improvement_vs_copy_last >= self.target
            else "marginal"
            if self.improvement_vs_copy_last > 0.0
            else "rejected"
        )
        if derived != self.outcome:
            raise ValueError("world gate outcome does not match improvement")


@dataclass(frozen=True, slots=True)
class UnifiedGateResult:
    passed: bool
    reasons: tuple[str, ...]
    success_gain: int
    median_reward_change: float
    final_improvement_vs_copy_last: float
    dynamics_ratio: float

    def __post_init__(self) -> None:
        if type(self.passed) is not bool or self.passed != (not self.reasons):
            raise ValueError("unified gate pass state and reasons disagree")
        if (
            not isinstance(self.reasons, tuple)
            or len(set(self.reasons)) != len(self.reasons)
            or any(not isinstance(reason, str) or not reason for reason in self.reasons)
        ):
            raise ValueError("unified gate reasons are invalid")
        if type(self.success_gain) is not int or not -10 <= self.success_gain <= 10:
            raise ValueError("unified gate success gain is invalid")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in (
                self.median_reward_change,
                self.final_improvement_vs_copy_last,
                self.dynamics_ratio,
            )
        ) or self.dynamics_ratio <= 0.0:
            raise ValueError("unified gate metrics are invalid")


def decide_gradient_audit(
    measurements: Sequence[AuditMeasurement],
    world_dynamics_loss: float,
    audit_dynamics_loss: float,
) -> AuditDecision:
    values = tuple(measurements)
    if len(values) != len(_AUDIT_STEPS):
        raise ValueError("audit requires exactly 25 measurements")
    if any(not isinstance(value, AuditMeasurement) for value in values):
        raise ValueError("audit measurements must be AuditMeasurement values")
    steps = tuple(value.optimizer_step for value in values)
    if steps != _AUDIT_STEPS:
        raise ValueError(f"audit measurement steps must be exactly {_AUDIT_STEPS}")
    world = _positive_finite(world_dynamics_loss, "world_dynamics_loss")
    audit = _positive_finite(audit_dynamics_loss, "audit_dynamics_loss")
    conflicts = sum(
        float(value.minimum_dynamics_cosine) < _CONFLICT_THRESHOLD
        for value in values
    )
    ratio = audit / world
    return AuditDecision(
        enable_pcgrad=(conflicts >= 8 and ratio > _DYNAMICS_RATIO_THRESHOLD),
        measurement_count=len(values),
        conflicting_measurements=conflicts,
        conflict_fraction=conflicts / len(values),
        conflict_threshold=_CONFLICT_THRESHOLD,
        world_dynamics_loss=world,
        audit_dynamics_loss=audit,
        dynamics_ratio=ratio,
        dynamics_ratio_threshold=_DYNAMICS_RATIO_THRESHOLD,
        optimizer_steps=steps,
    )


def negative_nll_deteriorated(current: float, best: float) -> bool:
    current_value = _finite(current, "current")
    best_value = _finite(best, "best")
    return current_value > best_value + 0.20 * max(abs(best_value), 1e-8)


class ValidationStopMonitor:
    def __init__(self, *, world_dynamics_loss: float) -> None:
        self.world_dynamics_loss = _positive_finite(
            world_dynamics_loss,
            "world_dynamics_loss",
        )
        self._last_step = -1
        self._best: dict[str, float] = {}
        self._windows: dict[str, list[int]] = {
            "copy_last_non_improvement": [],
            "world_dynamics_degradation": [],
            "optimized_loss_deterioration": [],
        }
        self._stopped = False

    def observe(self, record: ValidationRecord) -> PilotStepDecision:
        if self._stopped:
            raise RuntimeError("validation stop monitor already stopped")
        if not isinstance(record, ValidationRecord) or record.stage != "unified":
            raise ValueError("stop monitor requires unified ValidationRecord")
        if record.global_step <= self._last_step:
            raise ValueError("validation steps must be strictly increasing")
        missing = {
            "improvement_vs_copy_last",
            *_OPTIMIZED_VALIDATION_METRICS,
        } - set(record.metrics)
        if missing:
            raise ValueError(f"validation record is missing metrics: {sorted(missing)}")
        self._last_step = record.global_step

        deteriorated = sum(
            name in self._best
            and negative_nll_deteriorated(record.metrics[name], self._best[name])
            for name in _OPTIMIZED_VALIDATION_METRICS
        )
        conditions = {
            "copy_last_non_improvement": (
                record.metrics["improvement_vs_copy_last"] <= 0.0
            ),
            "world_dynamics_degradation": (
                record.metrics["dynamics_loss"]
                > _DYNAMICS_RATIO_THRESHOLD * self.world_dynamics_loss
            ),
            "optimized_loss_deterioration": deteriorated >= 2,
        }

        for name in _OPTIMIZED_VALIDATION_METRICS:
            current = record.metrics[name]
            self._best[name] = min(self._best.get(name, current), current)

        if record.global_step < 500:
            return PilotStepDecision(False, None)
        for reason, active in conditions.items():
            windows = self._windows[reason]
            if active:
                windows.append(record.global_step)
                del windows[:-3]
            else:
                windows.clear()
        for reason in (
            "copy_last_non_improvement",
            "world_dynamics_degradation",
            "optimized_loss_deterioration",
        ):
            windows = self._windows[reason]
            if len(windows) == 3:
                self._stopped = True
                return PilotStepDecision(True, reason, tuple(windows))
        return PilotStepDecision(False, None)

    @property
    def best_metrics(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._best))


def assess_world_gate(
    final_validation: ValidationRecord,
    *,
    target: float = 0.05,
) -> WorldGateResult:
    if (
        not isinstance(final_validation, ValidationRecord)
        or final_validation.stage != "world_pretrain"
        or final_validation.global_step != 5000
    ):
        raise ValueError("world gate requires world_pretrain validation at step 5000")
    target_value = _positive_finite(target, "world gate target")
    try:
        improvement = final_validation.metrics["improvement_vs_copy_last"]
    except KeyError as error:
        raise ValueError("world validation lacks improvement_vs_copy_last") from error
    if improvement >= target_value:
        return WorldGateResult(
            "accepted",
            True,
            False,
            improvement,
            target_value,
            "target_met",
        )
    if improvement > 0.0:
        return WorldGateResult(
            "marginal",
            False,
            True,
            improvement,
            target_value,
            "positive_below_target",
        )
    return WorldGateResult(
        "rejected",
        False,
        False,
        improvement,
        target_value,
        "copy_last_not_improved",
    )


def assess_unified_gate(
    *,
    final_validation: ValidationRecord,
    world_dynamics_loss: float,
    stop_rule_fired: bool,
    optimized_metrics_diverged: bool,
    checkpoint_verified: bool,
    resume_verified: bool,
    baseline_successes: int,
    unified_successes: int,
    paired_reward_changes: Sequence[float],
    actions_within_bounds: bool,
    tracking_consistent: bool,
) -> UnifiedGateResult:
    if (
        not isinstance(final_validation, ValidationRecord)
        or final_validation.stage != "unified"
        or final_validation.global_step != 5000
    ):
        raise ValueError("unified gate requires unified validation at step 5000")
    world = _positive_finite(world_dynamics_loss, "world_dynamics_loss")
    for name, value in (
        ("stop_rule_fired", stop_rule_fired),
        ("optimized_metrics_diverged", optimized_metrics_diverged),
        ("checkpoint_verified", checkpoint_verified),
        ("resume_verified", resume_verified),
        ("actions_within_bounds", actions_within_bounds),
        ("tracking_consistent", tracking_consistent),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} must be bool")
    if (
        type(baseline_successes) is not int
        or type(unified_successes) is not int
        or not 0 <= baseline_successes <= 10
        or not 0 <= unified_successes <= 10
    ):
        raise ValueError("success counts must be integers within [0, 10]")
    changes = tuple(_finite(value, "paired reward change") for value in paired_reward_changes)
    if len(changes) != 10:
        raise ValueError("unified gate requires exactly ten paired rewards")
    try:
        improvement = final_validation.metrics["improvement_vs_copy_last"]
        dynamics = final_validation.metrics["dynamics_loss"]
    except KeyError as error:
        raise ValueError("unified validation lacks gate metrics") from error
    ratio = dynamics / world
    success_gain = unified_successes - baseline_successes
    reward_median = float(median(changes))
    reasons: list[str] = []
    if stop_rule_fired:
        reasons.append("automatic_stop_fired")
    if improvement <= 0.0:
        reasons.append("copy_last_not_improved")
    if ratio > _DYNAMICS_RATIO_THRESHOLD:
        reasons.append("world_dynamics_degraded")
    if optimized_metrics_diverged:
        reasons.append("optimized_metrics_diverged")
    if not checkpoint_verified:
        reasons.append("checkpoint_not_verified")
    if not resume_verified:
        reasons.append("resume_not_verified")
    if success_gain < 2:
        reasons.append("success_gain_below_two")
    if reward_median <= 0.0:
        reasons.append("median_reward_not_positive")
    if not actions_within_bounds:
        reasons.append("action_bounds_violated")
    if not tracking_consistent:
        reasons.append("tracking_not_consistent")
    return UnifiedGateResult(
        passed=not reasons,
        reasons=tuple(reasons),
        success_gain=success_gain,
        median_reward_change=reward_median,
        final_improvement_vs_copy_last=improvement,
        dynamics_ratio=ratio,
    )


def write_audit_decision(path: str | Path, decision: AuditDecision) -> None:
    if not isinstance(decision, AuditDecision):
        raise ValueError("audit writer requires AuditDecision")
    _write_atomic_json(path, asdict(decision))


def read_audit_decision(path: str | Path) -> AuditDecision:
    value = _read_json_report(path)
    expected = set(AuditDecision.__dataclass_fields__)
    if set(value) != expected:
        raise ValueError("audit decision fields do not match contract")
    steps = value["optimizer_steps"]
    if not isinstance(steps, list):
        raise ValueError("audit decision optimizer_steps must be a list")
    value["optimizer_steps"] = tuple(steps)
    try:
        return AuditDecision(**value)
    except TypeError as error:
        raise ValueError("audit decision values do not match contract") from error


def write_stop_report(
    path: str | Path,
    decision: PilotStepDecision,
    *,
    context: Mapping[str, object],
) -> None:
    if not isinstance(decision, PilotStepDecision) or not decision.should_stop:
        raise ValueError("stop report requires a stopping decision")
    if not isinstance(context, Mapping) or set(context) != _STOP_CONTEXT_FIELDS:
        raise ValueError(
            f"stop report context fields must be exactly "
            f"{sorted(_STOP_CONTEXT_FIELDS)}"
        )
    resolved_config = context["resolved_config"]
    data_spec_hashes = context["data_spec_hashes"]
    if not isinstance(resolved_config, Mapping) or not resolved_config:
        raise ValueError("stop report resolved_config must be a nonempty mapping")
    if (
        not isinstance(data_spec_hashes, Mapping)
        or not data_spec_hashes
        or any(
            not isinstance(name, str)
            or not name
            or not _is_sha256(value)
            for name, value in data_spec_hashes.items()
        )
    ):
        raise ValueError("stop report data_spec_hashes are invalid")
    if not _is_hex_digest(context["git_commit"], 40):
        raise ValueError("stop report git_commit is invalid")
    if not _is_sha256(context["checkpoint_manifest_sha256"]):
        raise ValueError("stop report checkpoint manifest hash is invalid")
    if context["stage"] != "unified":
        raise ValueError("stop report stage must be unified")
    if context["checkpoint_step"] != decision.triggering_steps[-1]:
        raise ValueError("stop report checkpoint step does not match decision")
    _write_atomic_json(
        path,
        {
            "reason": decision.reason,
            "triggering_steps": list(decision.triggering_steps),
            "context": dict(context),
        },
    )


def write_world_gate_report(path: str | Path, result: WorldGateResult) -> None:
    if not isinstance(result, WorldGateResult):
        raise ValueError("world gate writer requires WorldGateResult")
    _write_atomic_json(path, asdict(result))


def read_world_gate_report(path: str | Path) -> WorldGateResult:
    value = _read_json_report(path)
    expected = set(WorldGateResult.__dataclass_fields__)
    if set(value) != expected:
        raise ValueError("world gate fields do not match contract")
    try:
        return WorldGateResult(**value)
    except TypeError as error:
        raise ValueError("world gate values do not match contract") from error


def write_unified_gate_report(path: str | Path, result: UnifiedGateResult) -> None:
    if not isinstance(result, UnifiedGateResult):
        raise ValueError("unified gate writer requires UnifiedGateResult")
    _write_atomic_json(path, asdict(result))


def read_unified_gate_report(path: str | Path) -> UnifiedGateResult:
    value = _read_json_report(path)
    expected = set(UnifiedGateResult.__dataclass_fields__)
    if set(value) != expected:
        raise ValueError("unified gate fields do not match contract")
    reasons = value["reasons"]
    if not isinstance(reasons, list):
        raise ValueError("unified gate reasons must be a list")
    value["reasons"] = tuple(reasons)
    try:
        return UnifiedGateResult(**value)
    except TypeError as error:
        raise ValueError("unified gate values do not match contract") from error


def _write_atomic_json(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("report parent must be a physical directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"report already exists: {destination}")
    temporary = parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"report already exists: {destination}")
        os.rename(temporary, destination)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _read_json_report(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("gate report must be a physical file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("gate report is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError("gate report root must be a mapping")
    return value


def _positive_finite(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _is_sha256(value: object) -> bool:
    return _is_hex_digest(value, 64)


def _is_hex_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )
