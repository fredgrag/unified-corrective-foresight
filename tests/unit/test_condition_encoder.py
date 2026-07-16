from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from corrective_foresight.conditioning.encoder import ConditionEncoder
from corrective_foresight.conditioning.language_cache import LanguageEmbeddingCache
from corrective_foresight.conditioning.vocabulary import ConditionVocabulary


def make_vocabulary() -> ConditionVocabulary:
    return ConditionVocabulary.build(
        {
            "dataset": ["dataset.v1"],
            "task": ["pick", "place"],
            "embodiment": ["panda"],
            "action_spec": ["ee_delta.v1"],
            "control_mode": ["pd_ee_delta_pose"],
        }
    )


def write_cache(directory: str) -> LanguageEmbeddingCache:
    return LanguageEmbeddingCache.write(
        tensor_path=Path(directory) / "language.safetensors",
        metadata_path=Path(directory) / "language.json",
        embeddings={
            "pick the cube": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "place the cube": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        },
        model_id="openai/clip-vit-base-patch32",
        revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    )


def condition_ids(vocabulary: ConditionVocabulary) -> dict[str, torch.Tensor]:
    return {
        "dataset": torch.tensor([vocabulary.id_for("dataset", "dataset.v1")] * 2),
        "task": torch.tensor(
            [vocabulary.id_for("task", "pick"), vocabulary.id_for("task", "place")]
        ),
        "embodiment": torch.tensor([vocabulary.id_for("embodiment", "panda")] * 2),
        "action_spec": torch.tensor(
            [vocabulary.id_for("action_spec", "ee_delta.v1")] * 2
        ),
        "control_mode": torch.tensor(
            [vocabulary.id_for("control_mode", "pd_ee_delta_pose")] * 2
        ),
    }


class ConditionEncoderTest(unittest.TestCase):
    def test_language_cache_round_trip_is_normalized_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = write_cache(directory)
            restored = LanguageEmbeddingCache.load(
                Path(directory) / "language.safetensors",
                Path(directory) / "language.json",
            )

        self.assertEqual(restored.content_hash, cache.content_hash)
        self.assertEqual(restored.dimension, 4)
        torch.testing.assert_close(
            restored.lookup("pick the cube"),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )

    def test_missing_language_embedding_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = write_cache(directory)

            with self.assertRaisesRegex(KeyError, "missing CLIP embedding"):
                cache.lookup("unknown instruction")

    def test_produces_six_distinct_condition_tokens_and_gradients(self) -> None:
        torch.manual_seed(4)
        vocabulary = make_vocabulary()
        encoder = ConditionEncoder(
            vocabulary=vocabulary,
            hidden_dim=8,
            language_embedding_dim=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = write_cache(directory)
            output = encoder(
                condition_ids(vocabulary),
                task_text=("pick the cube", None),
                language_cache=cache,
            )

        self.assertEqual(output.shape, (2, 6, 8))
        self.assertFalse(torch.allclose(output[:, 0], output[:, 1]))
        output.sum().backward()
        self.assertIsNotNone(encoder.language_projection.weight.grad)
        self.assertIsNotNone(encoder.null_language.grad)
        self.assertEqual(encoder.vocabulary_hash, vocabulary.content_hash)

    def test_text_requires_cache_but_all_null_language_does_not(self) -> None:
        vocabulary = make_vocabulary()
        encoder = ConditionEncoder(vocabulary, hidden_dim=8, language_embedding_dim=4)

        with self.assertRaisesRegex(ValueError, "language cache"):
            encoder(
                condition_ids(vocabulary),
                task_text=("pick the cube", None),
                language_cache=None,
            )

        output = encoder(
            condition_ids(vocabulary),
            task_text=(None, None),
            language_cache=None,
        )
        self.assertEqual(output.shape, (2, 6, 8))

    def test_rejects_missing_condition_namespace_and_out_of_range_id(self) -> None:
        vocabulary = make_vocabulary()
        encoder = ConditionEncoder(vocabulary, hidden_dim=8, language_embedding_dim=4)
        identifiers = condition_ids(vocabulary)
        del identifiers["task"]
        with self.assertRaisesRegex(ValueError, "condition identifier namespaces"):
            encoder(identifiers, task_text=(None, None))

        identifiers = condition_ids(vocabulary)
        identifiers["task"][0] = vocabulary.size("task")
        with self.assertRaisesRegex(ValueError, "task.*out of range"):
            encoder(identifiers, task_text=(None, None))


if __name__ == "__main__":
    unittest.main()
