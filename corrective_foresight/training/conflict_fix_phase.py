from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from corrective_foresight.config.conflict_fix import ConflictFixConfig
from corrective_foresight.training.gates import (
    read_audit_decision,
    read_unified_gate_report,
    read_world_gate_report,
)
from corrective_foresight.tracking.wandb_tracker import TrackingMetadata


@dataclass(frozen=True, slots=True)
class ConflictFixPhasePlan:
    phase: str
    stage: str
    experiment: Path
    output_root: Path
    target_step: int
    gradient_mode: str
    job_type: str
    start_mode: str
    checkpoint: Path | None
    audit_decision: Path | None
    audit_decision_sha256: str | None
    save_final_checkpoint: bool
    resume_existing_output: bool


def resolve_conflict_fix_phase(
    config: ConflictFixConfig,
    phase: str,
    *,
    init_checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    audit_decision: str | Path | None = None,
) -> ConflictFixPhasePlan:
    if not isinstance(config, ConflictFixConfig):
        raise ValueError("phase resolution requires ConflictFixConfig")
    if phase not in {
        "world_pretrain",
        "gradient_audit",
        "unified_gate",
        "unified_continue",
    }:
        raise ValueError("unsupported conflict-fix phase")
    root = config.output_root
    if root.is_symlink():
        raise ValueError("conflict-fix output root cannot be a symlink")
    world_output = root / "world_pretrain"
    audit_output = root / "gradient_audit"
    unified_output = root / "unified"
    if config.world_config.output_root != world_output:
        raise ValueError("world experiment output root is not canonical")
    if config.unified_config.output_root != unified_output:
        raise ValueError("unified experiment output root is not canonical")

    if phase == "world_pretrain":
        _require_absent_output(world_output)
        return ConflictFixPhasePlan(
            phase=phase,
            stage="world_pretrain",
            experiment=config.world_experiment,
            output_root=world_output,
            target_step=config.world_steps,
            gradient_mode="ordinary",
            job_type="world_pretrain",
            start_mode="fresh",
            checkpoint=None,
            audit_decision=None,
            audit_decision_sha256=None,
            save_final_checkpoint=True,
            resume_existing_output=False,
        )

    world_gate = read_world_gate_report(root / "world-gate-report.json")
    if not world_gate.accepted:
        raise ValueError("world gate is not accepted")
    canonical_world_checkpoint = (
        world_output / "checkpoints/world_pretrain-005000"
    )

    if phase == "gradient_audit":
        checkpoint = _require_canonical_checkpoint(
            init_checkpoint,
            canonical_world_checkpoint,
        )
        if resume is not None or audit_decision is not None:
            raise ValueError("gradient audit accepts only world warmstart")
        _require_absent_output(audit_output)
        return ConflictFixPhasePlan(
            phase=phase,
            stage="unified",
            experiment=config.unified_experiment,
            output_root=audit_output,
            target_step=config.audit_steps,
            gradient_mode="audit",
            job_type="gradient_audit",
            start_mode="warmstart",
            checkpoint=checkpoint,
            audit_decision=None,
            audit_decision_sha256=None,
            save_final_checkpoint=False,
            resume_existing_output=False,
        )

    decision_path, decision_hash, gradient_mode = _load_canonical_decision(
        root,
        audit_decision,
    )
    if phase == "unified_gate":
        checkpoint = _require_canonical_checkpoint(
            init_checkpoint,
            canonical_world_checkpoint,
        )
        if resume is not None:
            raise ValueError("unified gate must warmstart, not resume")
        _require_absent_output(unified_output)
        return ConflictFixPhasePlan(
            phase=phase,
            stage="unified",
            experiment=config.unified_experiment,
            output_root=unified_output,
            target_step=config.unified_gate_steps,
            gradient_mode=gradient_mode,
            job_type="unified_pilot",
            start_mode="warmstart",
            checkpoint=checkpoint,
            audit_decision=decision_path,
            audit_decision_sha256=decision_hash,
            save_final_checkpoint=True,
            resume_existing_output=False,
        )

    if init_checkpoint is not None:
        raise ValueError("unified continuation accepts only exact resume")
    canonical_resume = unified_output / "checkpoints/unified-005000"
    checkpoint = _require_canonical_checkpoint(resume, canonical_resume)
    if unified_output.is_symlink() or not unified_output.is_dir():
        raise ValueError("unified continuation output must already exist")
    tracking_metadata = unified_output / "tracking-metadata.json"
    if tracking_metadata.is_symlink() or not tracking_metadata.is_file():
        raise ValueError("unified continuation requires tracking metadata")
    try:
        metadata = TrackingMetadata.from_mapping(
            json.loads(tracking_metadata.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("unified continuation tracking metadata is invalid") from error
    if (
        not metadata.sync_complete
        or metadata.last_optimizer_step != 5000
        or metadata.project != config.tracking.project
        or metadata.group != config.tracking.group
    ):
        raise ValueError("unified continuation tracking is not synchronized")
    try:
        unified_gate = read_unified_gate_report(
            root / "unified-gate-report.json"
        )
    except ValueError as error:
        raise ValueError("unified gate report is unavailable or invalid") from error
    if not unified_gate.passed:
        raise ValueError("unified gate did not authorize continuation")
    return ConflictFixPhasePlan(
        phase=phase,
        stage="unified",
        experiment=config.unified_experiment,
        output_root=unified_output,
        target_step=config.unified_total_steps,
        gradient_mode=gradient_mode,
        job_type="unified_pilot",
        start_mode="resume",
        checkpoint=checkpoint,
        audit_decision=decision_path,
        audit_decision_sha256=decision_hash,
        save_final_checkpoint=True,
        resume_existing_output=True,
    )


def _load_canonical_decision(
    root: Path,
    value: str | Path | None,
) -> tuple[Path, str, str]:
    if value is None:
        raise ValueError("unified phase requires audit decision")
    canonical = root / "gradient-audit-decision.json"
    source = Path(value)
    if source.is_symlink() or not source.is_file():
        raise ValueError("audit decision must be a physical file")
    if source.resolve(strict=True) != canonical.resolve(strict=True):
        raise ValueError("audit decision path is not canonical")
    decision = read_audit_decision(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return source.resolve(strict=True), digest, (
        "pcgrad" if decision.enable_pcgrad else "ordinary"
    )


def _require_canonical_checkpoint(
    value: str | Path | None,
    canonical: Path,
) -> Path:
    if value is None:
        raise ValueError("phase requires checkpoint")
    if canonical.is_symlink() or not canonical.is_dir():
        raise ValueError("canonical checkpoint is unavailable")
    source = Path(value)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("checkpoint must be a physical directory")
    if source.resolve(strict=True) != canonical.resolve(strict=True):
        raise ValueError("checkpoint path is not canonical")
    manifest = source / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("canonical checkpoint lacks physical manifest.json")
    return source.resolve(strict=True)


def _require_absent_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"conflict-fix phase output already exists: {path}")
