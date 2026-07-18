from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from corrective_foresight.data.mixer import BalancedLeRobotMixer
from corrective_foresight.data.stateful_sampler import StatefulDistributedBatchSampler
from corrective_foresight.training.checkpoint import (
    CheckpointState,
    ExpectedCheckpointContract,
)
from corrective_foresight.training.distributed import DistributedContext
from corrective_foresight.training.distributed_checkpoint import (
    load_distributed_checkpoint_strict,
    save_distributed_checkpoint_atomic,
)
from corrective_foresight.training.stages import TrainingStage
from corrective_foresight.training.trainer import Trainer, TrainerConfig
from tests.fixtures.fake_trajectory_dataset import FakeTrajectoryDataset
from tests.unit.policy_fakes import make_condition_ids
from tests.unit.test_checkpoint_validation import make_state


class _ConditionedFakeDataset(FakeTrajectoryDataset):
    def __init__(self, vocabulary, *, size: int = 64) -> None:
        super().__init__("test.dataset.v1", "test.ee_delta.v1", size=size)
        self.vocabulary = vocabulary

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        sample = replace(
            sample,
            rgb=torch.zeros(sample.rgb.shape[0], 1, 3, 8, 16),
            proprio=torch.full((sample.proprio.shape[0], 3), float(index)),
            proprio_mask=torch.ones(sample.proprio.shape[0], 3, dtype=torch.bool),
        )
        condition_ids = make_condition_ids(self.vocabulary, 1)
        condition_ids = {
            name: values[0].clone() for name, values in condition_ids.items()
        }
        return replace(sample, task_text=None, condition_ids=condition_ids)


def _training_state(rank: int, context: DistributedContext) -> CheckpointState:
    torch.manual_seed(7001)
    base = make_state()
    base.policy.restore_ema_step(-1)
    trainer = Trainer(
        base.policy,
        TrainerConfig(
            learning_rate=1e-3,
            weight_decay=0.01,
            accumulation_steps=1,
            max_grad_norm=1.0,
            warmup_steps=0,
            total_steps=10,
            bf16=False,
            ddp=True,
        ),
        rank=rank,
        distributed_context=context,
    )
    dataset = _ConditionedFakeDataset(base.policy.condition_encoder.vocabulary)
    sampler = StatefulDistributedBatchSampler(
        dataset_size=len(dataset),
        batch_size=2,
        seed=307,
        rank=rank,
        world_size=context.world_size,
    )
    mixer = BalancedLeRobotMixer(
        datasets={dataset.dataset_id: dataset},
        samplers={dataset.dataset_id: sampler},
        weights={dataset.dataset_id: 1.0},
        generator=torch.Generator().manual_seed(311),
    )
    return replace(
        base,
        trainer=trainer,
        mixer=mixer,
        flow_generator=torch.Generator().manual_seed(313),
        epoch=0,
        global_step=0,
        optimizer_step=0,
        cycle_warmup_step=0,
    )


def _run_steps(state: CheckpointState, start: int, count: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for step in range(start, start + count):
        batch = state.mixer.next_batch()
        result = state.trainer.train_step(
            batch,
            TrainingStage.UNIFIED,
            step,
            state.flow_generator,
        )
        records.append(
            {
                "dataset_id": batch.dataset_id,
                "sample_indices": [
                    float(value) for value in batch.proprio[:, 0, 0].tolist()
                ],
                "metrics": {
                    name: float(value.item())
                    for name, value in sorted(result.output.metrics.items())
                },
            }
        )
    return records


def _hash_tensor_state(value: object) -> str:
    stream = io.BytesIO()
    torch.save(value, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _write_run_result(path: str, state: CheckpointState, records: list[dict[str, object]]) -> None:
    payload = {
        "records": records,
        "model_hash": _hash_tensor_state(state.policy.state_dict()),
        "ema_hash": _hash_tensor_state(state.policy.ema_state_target.adapter.state_dict()),
        "optimizer_hash": _hash_tensor_state(state.trainer.optimizer.state_dict()),
        "scheduler_hash": _hash_tensor_state(state.trainer.scheduler.state_dict()),
        "flow_state": state.flow_generator.get_state().tolist(),
        "mixer_state": state.mixer.state_dict(),
        "optimizer_step": state.trainer.policy.last_ema_step + 1,
        "last_ema_step": state.policy.last_ema_step,
    }
    Path(path).write_text(json.dumps(payload, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def _exact_resume_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    mode: str,
    checkpoint: str,
    result_directory: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        context = DistributedContext.from_initialized_process_group(
            local_rank=rank,
            device="cpu",
        )
        state = _training_state(rank, context)
        if mode == "baseline":
            records = _run_steps(state, 0, 4)
        elif mode == "save":
            records = _run_steps(state, 0, 2)
            save_distributed_checkpoint_atomic(
                checkpoint,
                replace(
                    state,
                    epoch=0,
                    global_step=2,
                    optimizer_step=2,
                    cycle_warmup_step=2,
                ),
                context,
            )
        elif mode == "load":
            resume = load_distributed_checkpoint_strict(
                checkpoint,
                ExpectedCheckpointContract.from_state(state),
                context,
            )
            if resume.global_step != 2:
                raise AssertionError(resume.global_step)
            records = _run_steps(state, resume.global_step, 2)
        else:
            raise ValueError(mode)
        _write_run_result(
            str(Path(result_directory) / f"rank-{rank}.json"), state, records
        )
    finally:
        dist.destroy_process_group()


def _spawn_phase(
    *,
    directory: Path,
    name: str,
    mode: str,
    checkpoint: Path,
) -> list[dict[str, object]]:
    rendezvous = directory / f"{name}.rendezvous"
    result_directory = directory / f"{name}.results"
    result_directory.mkdir()
    mp.spawn(
        _exact_resume_worker,
        args=(2, str(rendezvous), mode, str(checkpoint), str(result_directory)),
        nprocs=2,
        join=True,
    )
    return [
        json.loads(
            (result_directory / f"rank-{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(2)
    ]


class DistributedCheckpointResumeTest(unittest.TestCase):
    def test_fresh_process_resume_matches_uninterrupted_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            baseline = _spawn_phase(
                directory=root,
                name="baseline",
                mode="baseline",
                checkpoint=checkpoint,
            )
            _spawn_phase(
                directory=root,
                name="save",
                mode="save",
                checkpoint=checkpoint,
            )
            resumed = _spawn_phase(
                directory=root,
                name="load",
                mode="load",
                checkpoint=checkpoint,
            )
            self.assertEqual(len(baseline), 2)
            self.assertEqual(len(resumed), 2)
            for baseline_rank, resumed_rank in zip(baseline, resumed, strict=True):
                self.assertEqual(
                    baseline_rank["records"][2:], resumed_rank["records"]
                )
                for key in (
                    "model_hash",
                    "ema_hash",
                    "optimizer_hash",
                    "scheduler_hash",
                    "flow_state",
                    "mixer_state",
                    "optimizer_step",
                    "last_ema_step",
                ):
                    self.assertEqual(baseline_rank[key], resumed_rank[key], key)


if __name__ == "__main__":
    unittest.main()
