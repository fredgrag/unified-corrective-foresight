from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from corrective_foresight.config.conflict_fix import load_conflict_fix_config
from corrective_foresight.config.loader import load_action_spec
from corrective_foresight.evaluation.conflict_fix_report import (
    apply_verified_retention,
    build_paired_report,
    load_episode_summaries,
    verify_checkpoint_payloads,
    write_effect_gate,
)
from corrective_foresight.training.gates import (
    ValidationStopMonitor,
    assess_unified_gate,
    write_unified_gate_report,
)
from corrective_foresight.training.validation import ValidationRecord
from corrective_foresight.tracking.wandb_tracker import (
    TrackingMetadata,
    WandbTracker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conflict-fix-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_conflict_fix_config(args.conflict_fix_config)
    root = config.output_root
    action_spec = load_action_spec(config.unified_config.action_specs[0])
    evaluation_root = root / "evaluations"
    baseline = load_episode_summaries(
        evaluation_root,
        "untrained_action_from_world_5000",
        action_spec=action_spec,
    )
    candidate = load_episode_summaries(
        evaluation_root,
        "unified_5000",
        action_spec=action_spec,
    )
    paired = build_paired_report(
        baseline,
        candidate,
        required_success_gain=config.required_success_gain,
    )
    write_effect_gate(root / "paired-effect-report.json", paired)

    world_validation = _read_validation(
        root / "world_pretrain/validation/world_pretrain-005000.json"
    )
    unified_validation = _read_validation(
        root / "unified/validation/unified-005000.json"
    )
    stop_monitor = ValidationStopMonitor(
        world_dynamics_loss=world_validation.metrics["dynamics_loss"]
    )
    stop_rule_fired = False
    optimized_metrics_diverged = False
    validation_root = root / "unified/validation"
    history = tuple(sorted(validation_root.glob("unified-*.json")))
    if not history or history[-1].name != "unified-005000.json":
        raise ValueError("unified validation history is incomplete")
    for path in history:
        decision = stop_monitor.observe(_read_validation(path))
        if decision.should_stop:
            stop_rule_fired = True
            optimized_metrics_diverged = (
                decision.reason == "optimized_loss_deterioration"
            )
            break

    checkpoint = root / "unified/checkpoints/unified-005000"
    manifest_hash = verify_checkpoint_payloads(checkpoint)
    audit_path = root / "gradient-audit-decision.json"
    audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    resume_verified = _verify_resume_marker(
        root / "unified-resume-verification.json",
        checkpoint_manifest_sha256=manifest_hash,
        audit_decision_sha256=audit_hash,
    )
    tracking_consistent = _tracking_consistent(config)
    gate = assess_unified_gate(
        final_validation=unified_validation,
        world_dynamics_loss=world_validation.metrics["dynamics_loss"],
        stop_rule_fired=stop_rule_fired,
        optimized_metrics_diverged=optimized_metrics_diverged,
        checkpoint_verified=True,
        resume_verified=resume_verified,
        baseline_successes=paired.baseline_success_count,
        unified_successes=paired.candidate_success_count,
        paired_reward_changes=paired.paired_reward_changes,
        actions_within_bounds=paired.action_bounds_ok,
        tracking_consistent=tracking_consistent,
    )
    write_unified_gate_report(root / "unified-gate-report.json", gate)

    tracker = WandbTracker.resume(
        config=config.tracking,
        rank=0,
        output_root=root / "unified",
        job_type="unified_pilot",
        sanitized_run_config={
            "pilot_id": config.pilot_id,
            "gate_report_sha256": hashlib.sha256(
                (root / "unified-gate-report.json").read_bytes()
            ).hexdigest(),
        },
    )
    tracker.log_gate(
        {
            "gate/passed": float(gate.passed),
            "gate/success_gain": float(gate.success_gain),
            "gate/median_reward_change": gate.median_reward_change,
            "gate/dynamics_ratio": gate.dynamics_ratio,
        },
        optimizer_step=5000,
    )
    tracker.finish(sync_complete=True)
    if gate.passed:
        apply_verified_retention(root)


def _read_validation(path: Path) -> ValidationRecord:
    if path.is_symlink() or not path.is_file():
        raise ValueError("gate validation record must be physical")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("gate validation record must be a mapping")
    return ValidationRecord(**value)


def _verify_resume_marker(
    path: Path,
    *,
    checkpoint_manifest_sha256: str,
    audit_decision_sha256: str,
) -> bool:
    if path.is_symlink() or not path.is_file():
        raise ValueError("resume verification marker must be physical")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format_version",
        "checkpoint_manifest_sha256",
        "audit_decision_sha256",
        "gradient_mode",
        "world_size",
        "loaded_global_step",
        "verified_next_optimizer_step",
        "accumulation_steps",
        "dataset_ids",
        "metrics",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("resume verification fields do not match contract")
    metrics = value["metrics"]
    if (
        value["format_version"] != 1
        or value["checkpoint_manifest_sha256"] != checkpoint_manifest_sha256
        or value["audit_decision_sha256"] != audit_decision_sha256
        or value["gradient_mode"] not in {"ordinary", "pcgrad"}
        or value["world_size"] != 4
        or value["loaded_global_step"] != 5000
        or value["verified_next_optimizer_step"] != 5001
        or value["accumulation_steps"] != 8
        or not isinstance(value["dataset_ids"], list)
        or len(value["dataset_ids"]) != 8
        or not isinstance(metrics, dict)
        or not metrics
        or any(
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            for metric in metrics.values()
        )
    ):
        raise ValueError("resume verification evidence is invalid")
    return True


def _tracking_consistent(config) -> bool:
    checks = (
        (config.output_root / "world_pretrain/tracking-metadata.json", 5000),
        (config.output_root / "gradient_audit/tracking-metadata.json", 500),
        (config.output_root / "unified/tracking-metadata.json", 5000),
        (
            config.output_root
            / "evaluations/untrained_action_from_world_5000/tracking-metadata.json",
            0,
        ),
        (
            config.output_root
            / "evaluations/unified_5000/tracking-metadata.json",
            5000,
        ),
    )
    for path, step in checks:
        if path.is_symlink() or not path.is_file():
            return False
        value = json.loads(path.read_text(encoding="utf-8"))
        metadata = TrackingMetadata.from_mapping(value)
        if (
            metadata.project != config.tracking.project
            or metadata.group != config.tracking.group
            or metadata.last_optimizer_step != step
            or not metadata.sync_complete
        ):
            return False
    return True


if __name__ == "__main__":
    main()
