from __future__ import annotations

import torch

from corrective_foresight.data.lerobot_adapter import TrajectorySample


class FakeTrajectoryDataset:
    def __init__(self, dataset_id: str, action_spec_id: str, size: int = 64) -> None:
        self.dataset_id = dataset_id
        self.action_spec_id = action_spec_id
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> TrajectorySample:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        time_steps = 10
        action = torch.zeros(time_steps - 1, 2)
        action[:, 0] = float(index)
        return TrajectorySample(
            rgb=torch.zeros(time_steps, 1, 3, 2, 2),
            camera_mask=torch.ones(time_steps, 1, dtype=torch.bool),
            proprio=torch.full((time_steps, 1), float(index)),
            proprio_mask=torch.ones(time_steps, 1, dtype=torch.bool),
            action=action,
            action_dimension_mask=torch.ones_like(action, dtype=torch.bool),
            observation_valid_mask=torch.ones(time_steps, dtype=torch.bool),
            action_valid_mask=torch.ones(time_steps - 1, dtype=torch.bool),
            transition_valid_mask=torch.ones(time_steps - 1, dtype=torch.bool),
            delta_time=torch.full((time_steps - 1,), 0.1),
            context_index=1,
            task_text=f"{self.dataset_id}:{index}",
            condition_ids={},
            dataset_id=self.dataset_id,
            action_spec_id=self.action_spec_id,
        )
