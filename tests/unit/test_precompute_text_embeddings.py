from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from scripts.precompute_text_embeddings import (
    MODEL_ID,
    MODEL_REVISION,
    load_frozen_clip_text_model,
)


class PrecomputeTextEmbeddingsTest(unittest.TestCase):
    @patch("scripts.precompute_text_embeddings.CLIPTextModelWithProjection.from_pretrained")
    def test_loader_keeps_revision_but_does_not_force_upstream_weight_format(
        self, from_pretrained
    ) -> None:
        model = from_pretrained.return_value
        model.to.return_value = model

        result = load_frozen_clip_text_model(torch.device("cpu"))

        self.assertIs(result, model)
        from_pretrained.assert_called_once_with(MODEL_ID, revision=MODEL_REVISION)
        model.to.assert_called_once_with(torch.device("cpu"))
        model.eval.assert_called_once_with()
        model.requires_grad_.assert_called_once_with(False)


if __name__ == "__main__":
    unittest.main()
