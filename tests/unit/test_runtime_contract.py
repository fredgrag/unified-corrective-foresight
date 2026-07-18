from __future__ import annotations

import unittest

import torch

from corrective_foresight.runtime import (
    build_condition_vocabulary,
    encode_condition_ids,
    production_backbone_provenance,
)
from tests.unit.test_action_spec import make_action_spec
from tests.unit.test_dataset_spec import make_dataset_spec


class RuntimeContractTest(unittest.TestCase):
    def test_builds_frozen_vocabulary_and_online_ids_from_specs(self) -> None:
        dataset_spec = make_dataset_spec()
        action_spec = make_action_spec()
        task = "Move to the target."
        vocabulary = build_condition_vocabulary(
            (dataset_spec,),
            (action_spec,),
            {dataset_spec.dataset_id: (task,)},
        )

        identifiers = encode_condition_ids(
            vocabulary,
            dataset_spec=dataset_spec,
            action_spec=action_spec,
            task_text=task,
            device="cpu",
        )

        self.assertEqual(
            set(identifiers),
            {"dataset", "task", "embodiment", "action_spec", "control_mode"},
        )
        self.assertTrue(all(value.dtype is torch.long for value in identifiers.values()))
        self.assertTrue(all(value.shape == (1,) for value in identifiers.values()))

    def test_online_ids_reject_undeclared_task(self) -> None:
        dataset_spec = make_dataset_spec()
        action_spec = make_action_spec()
        vocabulary = build_condition_vocabulary(
            (dataset_spec,),
            (action_spec,),
            {dataset_spec.dataset_id: ("Declared task",)},
        )

        with self.assertRaisesRegex(ValueError, "undeclared task"):
            encode_condition_ids(
                vocabulary,
                dataset_spec=dataset_spec,
                action_spec=action_spec,
                task_text="Different task",
                device="cpu",
            )

    def test_backbone_provenance_is_exact_modelscope_delivery(self) -> None:
        self.assertEqual(
            production_backbone_provenance(),
            {
                "canonical_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "canonical_revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
                "delivery_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "delivery_revision": "23d0280ae6ee4ced592a3459674ad027d3c18906",
                "weights_sha256": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
            },
        )


if __name__ == "__main__":
    unittest.main()
