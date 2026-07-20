from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean, median
import uuid

from corrective_foresight.config.schema import ActionSpec
from corrective_foresight.evaluation.records import EvaluationRecord
from corrective_foresight.training.gates import read_unified_gate_report
from corrective_foresight.training.run_manifest import RunManifest
from corrective_foresight.tracking.wandb_tracker import TrackingMetadata


_POLICY_LABELS = {
    "untrained_action_from_world_5000",
    "unified_5000",
    "unified_20000",
}


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    seed: int
    success: bool
    total_reward: float
    episode_length: int
    action_bounds_ok: bool
    protocol_hash: str
    label: str = ""
    checkpoint_manifest_sha256: str = ""
    dataset_spec_hash: str = ""
    action_spec_hash: str = ""
    environment_hash: str = ""
    flow_protocol_hash: str = ""
    video_sha256: str = ""
    record_sha256: str = ""
    consistency_mean: float = 0.0
    inverse_variance_mean: float = 0.0
    total_nfe: int = 0

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 9:
            raise ValueError("episode seed must be within 0 through 9")
        if type(self.success) is not bool or type(self.action_bounds_ok) is not bool:
            raise ValueError("episode boolean fields are invalid")
        if (
            not math.isfinite(self.total_reward)
            or type(self.episode_length) is not int
            or self.episode_length <= 0
            or not math.isfinite(self.consistency_mean)
            or not math.isfinite(self.inverse_variance_mean)
            or type(self.total_nfe) is not int
            or self.total_nfe < 0
        ):
            raise ValueError("episode scalar fields are invalid")
        if not _sha256(self.protocol_hash):
            raise ValueError("episode protocol_hash must be SHA-256")
        for name in (
            "checkpoint_manifest_sha256",
            "dataset_spec_hash",
            "action_spec_hash",
            "environment_hash",
            "flow_protocol_hash",
            "video_sha256",
            "record_sha256",
        ):
            value = getattr(self, name)
            if value and not _sha256(value):
                raise ValueError(f"episode {name} must be SHA-256 when present")
        if self.label:
            validate_policy_label(self.label)


@dataclass(frozen=True, slots=True)
class PairedEvaluationReport:
    baseline_label: str
    candidate_label: str
    baseline_success_count: int
    candidate_success_count: int
    baseline_success_rate: float
    candidate_success_rate: float
    success_gain: int
    paired_reward_changes: tuple[float, ...]
    median_reward_change: float
    baseline_episode_length_mean: float
    candidate_episode_length_mean: float
    baseline_consistency_mean: float
    candidate_consistency_mean: float
    baseline_inverse_variance_mean: float
    candidate_inverse_variance_mean: float
    baseline_nfe_mean: float
    candidate_nfe_mean: float
    action_bounds_ok: bool
    effect_gate_passed: bool
    reasons: tuple[str, ...]
    required_success_gain: int

    def __post_init__(self) -> None:
        if self.baseline_label:
            validate_policy_label(self.baseline_label)
        if self.candidate_label:
            validate_policy_label(self.candidate_label)
        if self.effect_gate_passed != (not self.reasons):
            raise ValueError("effect gate pass state and reasons disagree")
        if len(self.paired_reward_changes) != 10:
            raise ValueError("paired report requires ten reward changes")
        if self.success_gain != (
            self.candidate_success_count - self.baseline_success_count
        ):
            raise ValueError("paired report success gain is inconsistent")


@dataclass(frozen=True, slots=True)
class ValidationCandidate:
    optimizer_step: int
    improvement_vs_copy_last: float
    dynamics_loss: float

    def __post_init__(self) -> None:
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("validation optimizer_step must be nonnegative")
        if not math.isfinite(self.improvement_vs_copy_last):
            raise ValueError("validation copy improvement must be finite")
        if not math.isfinite(self.dynamics_loss) or self.dynamics_loss < 0.0:
            raise ValueError("validation dynamics loss must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    retained: tuple[Path, ...]
    delete: tuple[Path, ...]


def validate_policy_label(label: str) -> str:
    if not isinstance(label, str):
        raise ValueError("policy label must be a string")
    if label.startswith("world_pretrain"):
        raise ValueError("world-pretrain is not a policy result")
    if label not in _POLICY_LABELS:
        raise ValueError(f"unsupported conflict-fix policy label: {label!r}")
    return label


def build_paired_report(
    baseline: Sequence[EpisodeSummary],
    candidate: Sequence[EpisodeSummary],
    *,
    required_success_gain: int = 2,
) -> PairedEvaluationReport:
    if type(required_success_gain) is not int or required_success_gain <= 0:
        raise ValueError("required_success_gain must be a positive integer")
    baseline_by_seed = _index_summaries(baseline, "baseline")
    candidate_by_seed = _index_summaries(candidate, "candidate")
    if tuple(baseline_by_seed) != tuple(range(10)) or tuple(candidate_by_seed) != tuple(
        range(10)
    ):
        raise ValueError("paired evaluation seeds must be exactly 0 through 9")
    for seed in range(10):
        before = baseline_by_seed[seed]
        after = candidate_by_seed[seed]
        if before.protocol_hash != after.protocol_hash:
            raise ValueError(f"paired evaluation protocol drift at seed {seed}")
        for name in (
            "dataset_spec_hash",
            "action_spec_hash",
            "environment_hash",
            "flow_protocol_hash",
        ):
            before_value = getattr(before, name)
            after_value = getattr(after, name)
            if before_value and before_value != after_value:
                raise ValueError(f"paired evaluation {name} drift at seed {seed}")
    for name, values in (
        ("baseline", tuple(baseline_by_seed.values())),
        ("candidate", tuple(candidate_by_seed.values())),
    ):
        for field in (
            "label",
            "checkpoint_manifest_sha256",
            "dataset_spec_hash",
            "action_spec_hash",
            "environment_hash",
            "protocol_hash",
        ):
            distinct = {getattr(item, field) for item in values}
            if len(distinct) != 1:
                raise ValueError(f"{name} evaluation {field} changes across seeds")
    baseline_manifests = {
        item.checkpoint_manifest_sha256
        for item in baseline_by_seed.values()
        if item.checkpoint_manifest_sha256
    }
    candidate_manifests = {
        item.checkpoint_manifest_sha256
        for item in candidate_by_seed.values()
        if item.checkpoint_manifest_sha256
    }
    if baseline_manifests or candidate_manifests:
        if (
            len(baseline_manifests) != 1
            or len(candidate_manifests) != 1
            or baseline_manifests == candidate_manifests
        ):
            raise ValueError("paired checkpoints must have distinct manifest hashes")

    ordered_before = tuple(baseline_by_seed[seed] for seed in range(10))
    ordered_after = tuple(candidate_by_seed[seed] for seed in range(10))
    changes = tuple(
        after.total_reward - before.total_reward
        for before, after in zip(ordered_before, ordered_after, strict=True)
    )
    baseline_successes = sum(item.success for item in ordered_before)
    candidate_successes = sum(item.success for item in ordered_after)
    success_gain = candidate_successes - baseline_successes
    reward_median = float(median(changes))
    bounds_ok = all(
        item.action_bounds_ok for item in (*ordered_before, *ordered_after)
    )
    reasons: list[str] = []
    if success_gain < required_success_gain:
        reasons.append("success_gain_below_required")
    if reward_median <= 0.0:
        reasons.append("median_reward_not_positive")
    if not bounds_ok:
        reasons.append("action_bounds_violated")
    return PairedEvaluationReport(
        baseline_label=ordered_before[0].label,
        candidate_label=ordered_after[0].label,
        baseline_success_count=baseline_successes,
        candidate_success_count=candidate_successes,
        baseline_success_rate=baseline_successes / 10,
        candidate_success_rate=candidate_successes / 10,
        success_gain=success_gain,
        paired_reward_changes=changes,
        median_reward_change=reward_median,
        baseline_episode_length_mean=mean(
            item.episode_length for item in ordered_before
        ),
        candidate_episode_length_mean=mean(
            item.episode_length for item in ordered_after
        ),
        baseline_consistency_mean=mean(
            item.consistency_mean for item in ordered_before
        ),
        candidate_consistency_mean=mean(
            item.consistency_mean for item in ordered_after
        ),
        baseline_inverse_variance_mean=mean(
            item.inverse_variance_mean for item in ordered_before
        ),
        candidate_inverse_variance_mean=mean(
            item.inverse_variance_mean for item in ordered_after
        ),
        baseline_nfe_mean=mean(item.total_nfe for item in ordered_before),
        candidate_nfe_mean=mean(item.total_nfe for item in ordered_after),
        action_bounds_ok=bounds_ok,
        effect_gate_passed=not reasons,
        reasons=tuple(reasons),
        required_success_gain=required_success_gain,
    )


def load_episode_summaries(
    evaluation_root: str | Path,
    label: str,
    *,
    action_spec: ActionSpec,
) -> tuple[EpisodeSummary, ...]:
    validate_policy_label(label)
    if not isinstance(action_spec, ActionSpec):
        raise ValueError("episode loading requires ActionSpec")
    directory = Path(evaluation_root) / label
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("evaluation label directory must be physical")
    expected_names = {f"seed-{seed}.json" for seed in range(10)}
    actual_names = {path.name for path in directory.glob("seed-*.json")}
    if actual_names != expected_names:
        raise ValueError("evaluation records must contain exactly seeds 0 through 9")
    summaries: list[EpisodeSummary] = []
    for seed in range(10):
        path = directory / f"seed-{seed}.json"
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"evaluation seed {seed} is invalid JSON") from error
        record = EvaluationRecord(value)
        payload = record.value
        if payload["tag"] != label or payload["checkpoint"]["label"] != label:
            raise ValueError("evaluation policy label does not match record")
        if payload["environment"]["seed"] != seed:
            raise ValueError("evaluation environment seed does not match filename")
        if payload["action"]["spec_hash"] != action_spec.content_hash:
            raise ValueError("evaluation ActionSpec hash mismatch")
        checkpoint_root = Path(payload["checkpoint"]["directory"])
        checkpoint_manifest = checkpoint_root / "manifest.json"
        if (
            checkpoint_root.is_symlink()
            or not checkpoint_root.is_dir()
            or checkpoint_manifest.is_symlink()
            or not checkpoint_manifest.is_file()
        ):
            raise ValueError("evaluation checkpoint provenance is unavailable")
        if hashlib.sha256(checkpoint_manifest.read_bytes()).hexdigest() != payload[
            "checkpoint"
        ]["manifest_sha256"]:
            raise ValueError("evaluation checkpoint manifest SHA-256 mismatch")
        bounds_ok = _actions_within_bounds(payload["steps"], action_spec)
        if not bounds_ok:
            raise ValueError("evaluation physical action is outside ActionSpec bounds")
        video_path = Path(payload["video"]["path"])
        if video_path.is_symlink() or not video_path.is_file():
            raise ValueError("evaluation video must be a physical file")
        video_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()
        if video_hash != payload["video"]["sha256"]:
            raise ValueError("evaluation video SHA-256 mismatch")
        environment_base = dict(payload["environment"])
        environment_base.pop("seed")
        protocol_material = {
            "environment": environment_base,
            "protocol": payload["protocol"],
            "flow": {
                name: payload["flow"][name]
                for name in (
                    "solver",
                    "time_grid",
                    "intervals",
                    "nfe_per_step",
                )
            },
            "dataset_spec_hash": payload["dataset"]["spec_hash"],
            "action_spec_hash": payload["action"]["spec_hash"],
        }
        summaries.append(
            EpisodeSummary(
                seed=seed,
                success=payload["result"]["success"],
                total_reward=float(payload["result"]["total_reward"]),
                episode_length=payload["result"]["length"],
                action_bounds_ok=True,
                protocol_hash=_canonical_hash(protocol_material),
                label=label,
                checkpoint_manifest_sha256=payload["checkpoint"][
                    "manifest_sha256"
                ],
                dataset_spec_hash=payload["dataset"]["spec_hash"],
                action_spec_hash=payload["action"]["spec_hash"],
                environment_hash=_canonical_hash(environment_base),
                flow_protocol_hash=_canonical_hash(
                    {
                        "solver": payload["flow"]["solver"],
                        "time_grid": payload["flow"]["time_grid"],
                        "intervals": payload["flow"]["intervals"],
                        "nfe_per_step": payload["flow"]["nfe_per_step"],
                        "seeds": payload["flow"]["seeds"],
                    }
                ),
                video_sha256=video_hash,
                record_sha256=hashlib.sha256(raw).hexdigest(),
                consistency_mean=float(
                    payload["diagnostics"]["consistency_mean"]
                ),
                inverse_variance_mean=float(
                    payload["diagnostics"]["inverse_variance_mean"]
                ),
                total_nfe=payload["flow"]["total_nfe"],
            )
        )
    return tuple(summaries)


def select_best_validation(
    records: Sequence[ValidationCandidate],
) -> ValidationCandidate:
    values = tuple(records)
    if not values or any(not isinstance(item, ValidationCandidate) for item in values):
        raise ValueError("validation selection requires candidates")
    if len({item.optimizer_step for item in values}) != len(values):
        raise ValueError("validation candidate steps must be unique")
    eligible = tuple(item for item in values if item.improvement_vs_copy_last > 0.0)
    if not eligible:
        raise ValueError("no validation has positive copy-last improvement")
    return min(eligible, key=lambda item: (item.dynamics_loss, item.optimizer_step))


def write_effect_gate(
    path: str | Path,
    report: PairedEvaluationReport,
) -> None:
    if not isinstance(report, PairedEvaluationReport):
        raise ValueError("effect gate writer requires PairedEvaluationReport")
    _write_atomic_json(path, asdict(report))


def verify_checkpoint_payloads(checkpoint: str | Path) -> str:
    root = Path(checkpoint)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint must be a physical directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("checkpoint manifest must be a physical file")
    manifest = RunManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    expected = {"manifest.json", *manifest.value["files"]}
    entries = tuple(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("checkpoint directory contains non-file entries")
    actual = {path.name for path in entries}
    if actual != expected:
        raise ValueError("checkpoint payload file set does not match manifest")
    for name, metadata in manifest.value["files"].items():
        payload = root / name
        if payload.is_symlink() or not payload.is_file():
            raise ValueError(f"checkpoint payload is not physical: {name}")
        if payload.stat().st_size != metadata["size"]:
            raise ValueError(f"checkpoint payload size mismatch: {name}")
        if hashlib.sha256(payload.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"checkpoint payload SHA-256 mismatch: {name}")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def plan_checkpoint_retention(output_root: str | Path) -> RetentionPlan:
    root = Path(output_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("retention output root must be physical")
    retained: set[Path] = set()
    delete: set[Path] = set()
    for stage, final_step in (("world_pretrain", 5000), ("unified", 5000)):
        checkpoint_root = root / stage / "checkpoints"
        validation_root = root / stage / "validation"
        if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
            raise ValueError(f"retention {stage} checkpoint root is unavailable")
        checkpoints: dict[int, Path] = {}
        for path in checkpoint_root.iterdir():
            if path.is_symlink() or not path.is_dir():
                raise ValueError("retention checkpoint entries must be physical dirs")
            prefix = f"{stage}-"
            if not path.name.startswith(prefix) or not path.name[len(prefix) :].isdigit():
                raise ValueError("retention checkpoint name is invalid")
            checkpoints[int(path.name[len(prefix) :])] = path
        if 0 not in checkpoints or final_step not in checkpoints:
            raise ValueError(f"retention {stage} lacks initial/final checkpoint")
        candidates: list[ValidationCandidate] = []
        for step in sorted(checkpoints):
            validation_path = validation_root / f"{stage}-{step:06d}.json"
            if not validation_path.is_file() or validation_path.is_symlink():
                continue
            value = json.loads(validation_path.read_text(encoding="utf-8"))
            metrics = value.get("metrics") if isinstance(value, Mapping) else None
            if not isinstance(metrics, Mapping):
                raise ValueError("retention validation metrics are invalid")
            candidates.append(
                ValidationCandidate(
                    optimizer_step=step,
                    improvement_vs_copy_last=float(
                        metrics["improvement_vs_copy_last"]
                    ),
                    dynamics_loss=float(metrics["dynamics_loss"]),
                )
            )
        best = select_best_validation(candidates)
        keep_steps = {0, best.optimizer_step, final_step}
        retained.update(checkpoints[step] for step in keep_steps)
        delete.update(
            path for step, path in checkpoints.items() if step not in keep_steps
        )
    audit_checkpoints = root / "gradient_audit/checkpoints"
    if audit_checkpoints.exists():
        if audit_checkpoints.is_symlink() or not audit_checkpoints.is_dir():
            raise ValueError("audit checkpoint root is invalid")
        delete.update(path for path in audit_checkpoints.iterdir())
    return RetentionPlan(
        retained=tuple(sorted(retained)),
        delete=tuple(sorted(delete)),
    )


def apply_verified_retention(output_root: str | Path) -> RetentionPlan:
    root = Path(output_root)
    gate = read_unified_gate_report(root / "unified-gate-report.json")
    if not gate.passed:
        raise ValueError("retention requires passed unified gate")
    resume_marker = root / "unified-resume-verification.json"
    if resume_marker.is_symlink() or not resume_marker.is_file():
        raise ValueError("retention requires resume verification")
    try:
        resume_value = json.loads(resume_marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("retention resume verification is invalid JSON") from error
    audit_decision = root / "gradient-audit-decision.json"
    final_checkpoint = root / "unified/checkpoints/unified-005000"
    if audit_decision.is_symlink() or not audit_decision.is_file():
        raise ValueError("retention requires physical audit decision")
    if (
        not isinstance(resume_value, Mapping)
        or resume_value.get("audit_decision_sha256")
        != hashlib.sha256(audit_decision.read_bytes()).hexdigest()
        or resume_value.get("checkpoint_manifest_sha256")
        != verify_checkpoint_payloads(final_checkpoint)
        or resume_value.get("world_size") != 4
        or resume_value.get("loaded_global_step") != 5000
        or resume_value.get("verified_next_optimizer_step") != 5001
    ):
        raise ValueError("retention resume verification evidence is inconsistent")
    for path, expected_step in (
        (root / "world_pretrain/tracking-metadata.json", 5000),
        (root / "gradient_audit/tracking-metadata.json", 500),
        (root / "unified/tracking-metadata.json", 5000),
        (
            root
            / "evaluations/untrained_action_from_world_5000/tracking-metadata.json",
            0,
        ),
        (root / "evaluations/unified_5000/tracking-metadata.json", 5000),
    ):
        metadata = _read_tracking_metadata(path)
        if not metadata.sync_complete or metadata.last_optimizer_step != expected_step:
            raise ValueError("retention requires synchronized W&B metadata")
    plan = plan_checkpoint_retention(root)
    for checkpoint in (*plan.retained, *plan.delete):
        verify_checkpoint_payloads(checkpoint)
    for path in plan.delete:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("retention deletion target must be a physical directory")
    for path in plan.delete:
        shutil.rmtree(path)
    _write_atomic_json(
        root / "retention-report.json",
        {
            "retained": [str(path) for path in plan.retained],
            "deleted": [str(path) for path in plan.delete],
        },
    )
    return plan


def _index_summaries(
    values: Sequence[EpisodeSummary],
    name: str,
) -> dict[int, EpisodeSummary]:
    items = tuple(values)
    if any(not isinstance(item, EpisodeSummary) for item in items):
        raise ValueError(f"{name} summaries contain invalid values")
    indexed = {item.seed: item for item in items}
    if len(indexed) != len(items):
        raise ValueError(f"{name} evaluation seeds must be unique")
    return dict(sorted(indexed.items()))


def _actions_within_bounds(steps: object, action_spec: ActionSpec) -> bool:
    if not isinstance(steps, Sequence):
        return False
    for step in steps:
        if not isinstance(step, Mapping):
            return False
        action = step.get("physical_action")
        if (
            not isinstance(action, list)
            or len(action) != action_spec.dimension
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < lower
                or float(value) > upper
                for value, lower, upper in zip(
                    action,
                    action_spec.minimum,
                    action_spec.maximum,
                    strict=True,
                )
            )
        ):
            return False
    return True


def _canonical_hash(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_atomic_json(path: str | Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"effect gate already exists: {destination}")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("effect gate parent must be a physical directory")
    temporary = parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True)
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, destination)
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_tracking_metadata(path: Path) -> TrackingMetadata:
    if path.is_symlink() or not path.is_file():
        raise ValueError("tracking metadata must be a physical file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("tracking metadata is invalid JSON") from error
    return TrackingMetadata.from_mapping(value)
