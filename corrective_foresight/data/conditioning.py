from __future__ import annotations

from dataclasses import replace

import torch

from corrective_foresight.conditioning.vocabulary import (
    CONDITION_NAMESPACES,
    ConditionVocabulary,
)
from corrective_foresight.config.schema import ActionSpec, DatasetSpec
from corrective_foresight.data.lerobot_adapter import TrajectorySample


class ConditionedTrajectoryDataset:
    def __init__(
        self,
        dataset,
        *,
        dataset_spec: DatasetSpec,
        action_spec: ActionSpec,
        vocabulary: ConditionVocabulary,
    ) -> None:
        if dataset_spec.action_spec_id != action_spec.spec_id:
            raise ValueError("DatasetSpec and ActionSpec identifiers do not match")
        if dataset_spec.fps != action_spec.frequency_hz:
            raise ValueError("DatasetSpec and ActionSpec frequencies do not match")
        if len(dataset) <= 0:
            raise ValueError("conditioned dataset cannot be empty")
        self.dataset = dataset
        self.dataset_spec = dataset_spec
        self.action_spec = action_spec
        self.vocabulary = vocabulary
        fixed_values = {
            "dataset": dataset_spec.dataset_id,
            "embodiment": dataset_spec.embodiment_id,
            "action_spec": action_spec.spec_id,
            "control_mode": action_spec.control_mode,
        }
        self._fixed_ids = {
            namespace: self._declared_id(namespace, value)
            for namespace, value in fixed_values.items()
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TrajectorySample:
        sample = self.dataset[index]
        if not isinstance(sample, TrajectorySample):
            raise ValueError("conditioned dataset must return TrajectorySample")
        if sample.dataset_id != self.dataset_spec.dataset_id:
            raise ValueError("sample dataset_id does not match DatasetSpec")
        if sample.action_spec_id != self.action_spec.spec_id:
            raise ValueError("sample action_spec_id does not match ActionSpec")
        if sample.condition_ids:
            raise ValueError("source trajectory sample already contains condition IDs")
        task_text = sample.task_text
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError("trajectory sample task text must be nonempty")
        condition_ids = {
            **{
                namespace: torch.tensor(identifier, dtype=torch.long)
                for namespace, identifier in self._fixed_ids.items()
            },
            "task": torch.tensor(
                self._declared_id("task", task_text),
                dtype=torch.long,
            ),
        }
        if set(condition_ids) != set(CONDITION_NAMESPACES):
            raise RuntimeError("condition binding did not produce every namespace")
        return replace(sample, condition_ids=condition_ids)

    def _declared_id(self, namespace: str, value: str) -> int:
        identifier = self.vocabulary.id_for(namespace, value)
        if self.vocabulary.token_for(namespace, identifier) != value:
            qualifier = "task" if namespace == "task" else f"{namespace} value"
            raise ValueError(f"undeclared {qualifier}: {value}")
        return identifier
