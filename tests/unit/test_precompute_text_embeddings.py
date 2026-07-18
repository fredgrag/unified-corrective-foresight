from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from scripts.precompute_text_embeddings import (
    normalize_legacy_clip_state_dict,
    load_frozen_clip_model,
)


class PrecomputeTextEmbeddingsTest(unittest.TestCase):
    @patch("scripts.precompute_text_embeddings.verify_clip_snapshot")
    @patch("scripts.precompute_text_embeddings.torch.load")
    @patch("scripts.precompute_text_embeddings.CLIPConfig.from_pretrained")
    @patch("scripts.precompute_text_embeddings.CLIPModel")
    def test_loader_verifies_and_uses_only_the_pinned_local_snapshot(
        self, clip_model, from_pretrained, torch_load, verify_snapshot
    ) -> None:
        model = clip_model.return_value
        model.to.return_value = model
        snapshot = Path("/verified/clip-snapshot")
        state_dict = {
            "projection": torch.ones(1),
            "text_model.embeddings.position_ids": torch.arange(77).reshape(1, 77),
            "vision_model.embeddings.position_ids": torch.arange(50).reshape(1, 50),
        }
        torch_load.return_value = state_dict

        result = load_frozen_clip_model(
            torch.device("cpu"), snapshot_dir=snapshot
        )

        self.assertIs(result, model)
        verify_snapshot.assert_called_once_with(snapshot)
        from_pretrained.assert_called_once_with(
            str(snapshot), local_files_only=True
        )
        clip_model.assert_called_once_with(from_pretrained.return_value)
        torch_load.assert_called_once_with(
            snapshot / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict.assert_called_once()
        loaded_state, = model.load_state_dict.call_args.args
        self.assertEqual(set(loaded_state), {"projection"})
        self.assertTrue(torch.equal(loaded_state["projection"], torch.ones(1)))
        self.assertEqual(model.load_state_dict.call_args.kwargs, {"strict": True})
        model.to.assert_called_once_with(torch.device("cpu"))
        model.eval.assert_called_once_with()
        model.requires_grad_.assert_called_once_with(False)

    def test_legacy_position_ids_must_be_exact_reconstructible_indices(self) -> None:
        state_dict = {
            "text_model.embeddings.position_ids": torch.zeros(1, 77),
            "vision_model.embeddings.position_ids": torch.arange(50).reshape(1, 50),
        }

        with self.assertRaisesRegex(ValueError, "position_ids"):
            normalize_legacy_clip_state_dict(state_dict)


if __name__ == "__main__":
    unittest.main()
