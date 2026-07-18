from __future__ import annotations

import unittest

from corrective_foresight.conditioning.vocabulary import ConditionVocabulary
from corrective_foresight.data.conditioning import ConditionedTrajectoryDataset
from tests.fixtures.fake_trajectory_dataset import FakeTrajectoryDataset
from tests.unit.test_action_spec import make_action_spec
from tests.unit.test_dataset_spec import make_dataset_spec


def make_vocabulary(*tasks: str) -> ConditionVocabulary:
    dataset_spec = make_dataset_spec()
    action_spec = make_action_spec()
    return ConditionVocabulary.build(
        {
            "dataset": (dataset_spec.dataset_id,),
            "task": tasks,
            "embodiment": (dataset_spec.embodiment_id,),
            "action_spec": (action_spec.spec_id,),
            "control_mode": (action_spec.control_mode,),
        }
    )


class ConditionedTrajectoryDatasetTest(unittest.TestCase):
    def test_binds_every_condition_namespace_from_specs_and_task_text(self) -> None:
        dataset_spec = make_dataset_spec()
        action_spec = make_action_spec()
        task = f"{dataset_spec.dataset_id}:0"
        vocabulary = make_vocabulary(task)
        dataset = ConditionedTrajectoryDataset(
            FakeTrajectoryDataset(dataset_spec.dataset_id, action_spec.spec_id, size=1),
            dataset_spec=dataset_spec,
            action_spec=action_spec,
            vocabulary=vocabulary,
        )

        sample = dataset[0]

        expected_values = {
            "dataset": dataset_spec.dataset_id,
            "task": task,
            "embodiment": dataset_spec.embodiment_id,
            "action_spec": action_spec.spec_id,
            "control_mode": action_spec.control_mode,
        }
        self.assertEqual(set(sample.condition_ids), set(expected_values))
        for namespace, value in expected_values.items():
            self.assertEqual(
                sample.condition_ids[namespace].item(),
                vocabulary.id_for(namespace, value),
            )

    def test_rejects_task_text_not_declared_in_vocabulary(self) -> None:
        dataset_spec = make_dataset_spec()
        action_spec = make_action_spec()
        dataset = ConditionedTrajectoryDataset(
            FakeTrajectoryDataset(dataset_spec.dataset_id, action_spec.spec_id, size=2),
            dataset_spec=dataset_spec,
            action_spec=action_spec,
            vocabulary=make_vocabulary(f"{dataset_spec.dataset_id}:0"),
        )

        with self.assertRaisesRegex(ValueError, "undeclared task"):
            dataset[1]


if __name__ == "__main__":
    unittest.main()
