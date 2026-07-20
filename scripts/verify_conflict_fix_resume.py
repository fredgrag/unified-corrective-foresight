from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch
import torch.distributed as dist

from corrective_foresight.config.conflict_fix import load_conflict_fix_config
from corrective_foresight.runtime import assemble_runtime_policy, load_production_runtime
from corrective_foresight.training.checkpoint import ExpectedCheckpointContract
from corrective_foresight.training.distributed_checkpoint import (
    load_distributed_checkpoint_strict,
)
from corrective_foresight.training.gates import read_audit_decision
from corrective_foresight.training.trainer import Trainer, TrainerConfig
from train import _build_mixer, _checkpoint_state, _initialize_distributed_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conflict-fix-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit-decision", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("resume verification requires CUDA")
    config = load_conflict_fix_config(args.conflict_fix_config)
    decision_path = args.audit_decision.resolve(strict=True)
    if decision_path != (config.output_root / "gradient-audit-decision.json").resolve(
        strict=True
    ):
        raise ValueError("resume verification audit decision is not canonical")
    decision = read_audit_decision(decision_path)
    gradient_mode = "pcgrad" if decision.enable_pcgrad else "ordinary"
    runtime = load_production_runtime(config.unified_experiment)
    context = _initialize_distributed_context()
    if context.world_size != 4:
        raise RuntimeError("resume verification requires exactly four ranks")
    device = context.device
    torch.manual_seed(runtime.config.seed)
    torch.cuda.manual_seed_all(runtime.config.seed)
    policy = assemble_runtime_policy(runtime, device=device)
    mixer = _build_mixer(runtime, context, split="train")
    training = runtime.config.training
    trainer = Trainer(
        policy,
        TrainerConfig(
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            accumulation_steps=training.accumulation_steps,
            max_grad_norm=training.max_grad_norm,
            warmup_steps=training.warmup_steps,
            total_steps=training.total_steps,
            bf16=training.bf16,
            ddp=True,
            stage="unified",
            protected_lr_multiplier=config.protected_lr_multiplier,
            gradient_mode=gradient_mode,
            gradient_diagnostic_interval=config.conflict_log_interval,
            pcgrad_seed=runtime.config.seed,
        ),
        rank=context.rank,
        distributed_context=context,
    )
    flow_generator = torch.Generator(device=device).manual_seed(
        runtime.config.seed + 300
    )
    state = _checkpoint_state(
        runtime,
        policy,
        trainer,
        mixer,
        flow_generator,
        context,
    )
    checkpoint = args.checkpoint.resolve(strict=True)
    expected_checkpoint = (
        config.output_root / "unified/checkpoints/unified-005000"
    ).resolve(strict=True)
    if checkpoint != expected_checkpoint:
        raise ValueError("resume verification checkpoint is not canonical")
    resume = load_distributed_checkpoint_strict(
        checkpoint,
        ExpectedCheckpointContract.from_state(state),
        context,
    )
    if resume.global_step != 5000:
        raise ValueError("resume verification requires checkpoint step 5000")

    result = None
    batches = []
    while result is None or not result.optimizer_stepped:
        batch = mixer.next_batch().to(device)
        batches.append(batch.dataset_id)
        result = trainer.train_step(
            batch,
            "unified",
            resume.global_step,
            flow_generator,
        )
    metrics = {
        **{
            name: float(value.detach().float().item())
            for name, value in result.output.metrics.items()
        },
        **dict(result.gradient_metrics),
        "pre_clip_gradient_norm": float(
            result.pre_clip_gradient_norm.detach().float().item()
        ),
    }
    if not metrics or any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("resume verification produced non-finite metrics")
    if len(batches) != training.accumulation_steps:
        raise ValueError("resume verification did not cover full accumulation")
    if context.rank == 0:
        manifest_hash = hashlib.sha256(
            (checkpoint / "manifest.json").read_bytes()
        ).hexdigest()
        audit_hash = hashlib.sha256(decision_path.read_bytes()).hexdigest()
        _write_atomic(
            config.output_root / "unified-resume-verification.json",
            {
                "format_version": 1,
                "checkpoint_manifest_sha256": manifest_hash,
                "audit_decision_sha256": audit_hash,
                "gradient_mode": gradient_mode,
                "world_size": context.world_size,
                "loaded_global_step": resume.global_step,
                "verified_next_optimizer_step": resume.global_step + 1,
                "accumulation_steps": len(batches),
                "dataset_ids": batches,
                "metrics": dict(sorted(metrics.items())),
            },
        )
    dist.barrier()
    dist.destroy_process_group()


def _write_atomic(path: Path, value: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"resume verification already exists: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("resume verification parent must be physical")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


if __name__ == "__main__":
    main()
