from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import Tensor, nn

from corrective_foresight.model.dinov3_artifact import DinoV3ArtifactError
from corrective_foresight.model.dinov3_backbone import (
    DinoV3PatchBackbone,
    _validate_production_config,
)


class IdentityProcessor:
    def __call__(self, *, images: Tensor, **kwargs: object) -> dict[str, Tensor]:
        if kwargs != {"return_tensors": "pt", "do_rescale": False}:
            raise AssertionError(f"unexpected processor arguments: {kwargs}")
        return {"pixel_values": images}


class TinyDinoModel(nn.Module):
    def __init__(self, sequence_adjustment: int = 0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            patch_size=2,
            num_register_tokens=2,
            hidden_size=4,
        )
        self.sequence_adjustment = sequence_adjustment

    def forward(self, pixel_values: Tensor) -> SimpleNamespace:
        batch_size, _, height, width = pixel_values.shape
        patch_count = (height // 2) * (width // 2) + self.sequence_adjustment
        prefix = torch.full((batch_size, 3, 4), -100.0, device=pixel_values.device)
        patches = torch.arange(
            batch_size * patch_count * 4,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        ).reshape(batch_size, patch_count, 4)
        return SimpleNamespace(last_hidden_state=torch.cat((prefix, patches), dim=1))


def production_model() -> TinyDinoModel:
    model = TinyDinoModel()
    model.config = SimpleNamespace(
        architectures=["DINOv3ViTModel"],
        model_type="dinov3_vit",
        hidden_size=768,
        patch_size=16,
        num_register_tokens=4,
        num_hidden_layers=12,
        num_attention_heads=12,
    )
    return model


class DinoV3PatchBackboneTest(unittest.TestCase):
    def test_separates_prefix_tokens_and_preserves_row_major_patch_grid(self) -> None:
        model = TinyDinoModel()
        backbone = DinoV3PatchBackbone(model=model, processor=IdentityProcessor())
        rgb = torch.randn(2, 3, 4, 6, requires_grad=True)

        backbone.train()
        grid = backbone(rgb)

        self.assertFalse(backbone.training)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(grid.tokens.shape, (2, 2, 3, 4))
        torch.testing.assert_close(grid.tokens[0, 0, 0], torch.arange(4.0))
        torch.testing.assert_close(grid.tokens[0, 0, 1], torch.arange(4.0, 8.0))
        self.assertFalse(grid.tokens.requires_grad)

    def test_rejects_sequence_that_disagrees_with_processed_patch_grid(self) -> None:
        backbone = DinoV3PatchBackbone(
            model=TinyDinoModel(sequence_adjustment=-1),
            processor=IdentityProcessor(),
        )

        with self.assertRaisesRegex(ValueError, "patch-token count"):
            backbone(torch.zeros(1, 3, 4, 6))

    @patch(
        "corrective_foresight.model.dinov3_backbone."
        "DINOv3ViTImageProcessorFast.from_pretrained"
    )
    @patch(
        "corrective_foresight.model.dinov3_backbone."
        "DINOv3ViTModel.from_pretrained"
    )
    @patch("corrective_foresight.model.dinov3_backbone.verify_snapshot")
    def test_exact_loader_verifies_and_uses_local_files_only(
        self, verify, load_model, load_processor
    ) -> None:
        model = production_model()
        load_model.return_value = model
        load_processor.return_value = IdentityProcessor()
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)

            result = DinoV3PatchBackbone.from_pretrained_exact(snapshot, device="cpu")

        self.assertIs(result.model, model)
        resolved = snapshot.resolve()
        verify.assert_called_once_with(resolved)
        load_model.assert_called_once_with(str(resolved), local_files_only=True)
        load_processor.assert_called_once_with(str(resolved), local_files_only=True)

    @patch(
        "corrective_foresight.model.dinov3_backbone."
        "DINOv3ViTImageProcessorFast.from_pretrained"
    )
    @patch(
        "corrective_foresight.model.dinov3_backbone."
        "DINOv3ViTModel.from_pretrained"
    )
    @patch("corrective_foresight.model.dinov3_backbone.verify_snapshot")
    def test_integrity_failure_happens_before_transformers_loading(
        self, verify, load_model, load_processor
    ) -> None:
        verify.side_effect = DinoV3ArtifactError("model.safetensors digest mismatch")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DinoV3ArtifactError, "digest mismatch"):
                DinoV3PatchBackbone.from_pretrained_exact(Path(directory))

        load_model.assert_not_called()
        load_processor.assert_not_called()

    def test_production_config_must_match_frozen_vit_base_contract(self) -> None:
        valid = vars(production_model().config)
        mutations = {
            "architectures": ["DINOv3ViTForImageClassification"],
            "model_type": "vit",
            "hidden_size": 1024,
            "patch_size": 14,
            "num_register_tokens": 0,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
        }
        for name, invalid_value in mutations.items():
            with self.subTest(name=name):
                candidate = dict(valid)
                candidate[name] = invalid_value
                with self.assertRaisesRegex(DinoV3ArtifactError, name):
                    _validate_production_config(SimpleNamespace(**candidate))

        _validate_production_config(SimpleNamespace(**valid))


if __name__ == "__main__":
    unittest.main()
