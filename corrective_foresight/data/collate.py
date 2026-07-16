from __future__ import annotations

from collections.abc import Sequence

import torch

from corrective_foresight.data.batch import TrajectoryBatch
from corrective_foresight.data.lerobot_adapter import TrajectorySample


def collate_trajectory_samples(
    samples: Sequence[TrajectorySample],
) -> TrajectoryBatch:
    if not samples:
        raise ValueError("cannot collate an empty trajectory sample list")
    first = samples[0]
    if any(sample.action_spec_id != first.action_spec_id for sample in samples):
        raise ValueError("all samples in a batch must use the same ActionSpec")
    if any(sample.dataset_id != first.dataset_id for sample in samples):
        raise ValueError("all samples in a batch must come from the same dataset")
    if any(sample.context_index != first.context_index for sample in samples):
        raise ValueError("all samples in a batch must share context_index")

    tensor_names = (
        "rgb",
        "camera_mask",
        "proprio",
        "proprio_mask",
        "action",
        "action_dimension_mask",
        "observation_valid_mask",
        "action_valid_mask",
        "transition_valid_mask",
        "delta_time",
    )
    expected_shapes = {
        name: tuple(getattr(first, name).shape) for name in tensor_names
    }
    for sample in samples[1:]:
        for name, expected_shape in expected_shapes.items():
            if tuple(getattr(sample, name).shape) != expected_shape:
                raise ValueError(
                    f"sample {name} shape mismatch: expected {expected_shape}, "
                    f"got {tuple(getattr(sample, name).shape)}"
                )

    condition_names = set(first.condition_ids)
    if any(set(sample.condition_ids) != condition_names for sample in samples):
        raise ValueError("all samples must contain identical condition identifier keys")
    condition_ids = {
        name: torch.stack(
            [torch.as_tensor(sample.condition_ids[name]) for sample in samples]
        ).reshape(len(samples))
        for name in sorted(condition_names)
    }
    batch = TrajectoryBatch(
        **{
            name: torch.stack([getattr(sample, name) for sample in samples])
            for name in tensor_names
        },
        context_index=first.context_index,
        task_text=tuple(sample.task_text for sample in samples),
        condition_ids=condition_ids,
        dataset_id=first.dataset_id,
        action_spec_id=first.action_spec_id,
    )
    batch.validate(expected_action_spec_id=first.action_spec_id)
    return batch
