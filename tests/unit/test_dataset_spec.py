from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from corrective_foresight.config.loader import load_dataset_spec
from corrective_foresight.config.schema import DatasetSpec


def make_dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        schema_version=1,
        dataset_id="test.dataset.v1",
        repo_id="org/test-dataset",
        local_root=None,
        revision="0123456789abcdef",
        fps=10.0,
        camera_features={"base": "observation.images.base"},
        optional_camera_roles=(),
        proprio_features=("observation.state",),
        proprio_required=True,
        task_feature="task",
        language_feature=None,
        embodiment_id="test.robot.v1",
        action_feature="action",
        action_spec_id="test.ee_delta.v1",
        sample_weight=1.0,
        split_episode_ids={"train": (0, 1), "validation": (2,), "evaluation": (3,)},
    )


class DatasetSpecTest(unittest.TestCase):
    def test_rejects_both_hub_and_local_location(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one.*repo_id.*local_root"):
            replace(make_dataset_spec(), local_root="/tmp/dataset")

    def test_rejects_relative_local_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_root.*absolute"):
            replace(make_dataset_spec(), repo_id=None, local_root="relative/path")

    def test_rejects_overlapping_episode_splits(self) -> None:
        with self.assertRaisesRegex(ValueError, "episode splits.*disjoint"):
            replace(
                make_dataset_spec(),
                split_episode_ids={"train": (0, 1), "validation": (1,)},
            )

    def test_rejects_unknown_optional_camera_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "optional_camera_roles"):
            replace(make_dataset_spec(), optional_camera_roles=("wrist",))

    def test_rejects_nonstring_feature_names_with_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera_features.*nonempty strings"):
            replace(
                make_dataset_spec(),
                camera_features={"base": 123},
            )

    def test_loader_rejects_unknown_fields(self) -> None:
        text = """
schema_version: 1
dataset_id: broken
repo_id: org/repo
local_root: null
revision: abc
fps: 10
camera_features: {base: observation.images.base}
optional_camera_roles: []
proprio_features: [observation.state]
proprio_required: true
task_feature: task
language_feature: null
embodiment_id: robot
action_feature: action
action_spec_id: action.v1
sample_weight: 1
split_episode_ids: {train: [0]}
unexpected_field: true
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.yaml"
            path.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown fields.*unexpected_field"):
                load_dataset_spec(path)

    def test_loader_converts_yaml_lists_to_immutable_tuples(self) -> None:
        text = """
schema_version: 1
dataset_id: local.dataset.v1
repo_id: null
local_root: /tmp/local-dataset
revision: local-sha256
fps: 20
camera_features: {base: observation.images.base}
optional_camera_roles: []
proprio_features: [observation.state]
proprio_required: true
task_feature: task
language_feature: null
embodiment_id: robot.v1
action_feature: action
action_spec_id: action.v1
sample_weight: 2
split_episode_ids: {train: [1, 2], validation: [3]}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.yaml"
            path.write_text(text, encoding="utf-8")

            spec = load_dataset_spec(path)

        self.assertEqual(spec.proprio_features, ("observation.state",))
        self.assertEqual(spec.split_episode_ids["train"], (1, 2))
        self.assertEqual(spec.to_dict()["camera_features"], {"base": "observation.images.base"})


if __name__ == "__main__":
    unittest.main()
