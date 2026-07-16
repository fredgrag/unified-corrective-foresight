from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from corrective_foresight.conditioning.vocabulary import (
    NULL_LANGUAGE_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    ConditionVocabulary,
)


def declarations() -> dict[str, list[str]]:
    return {
        "dataset": ["zeta", "alpha"],
        "task": ["pick", "place"],
        "embodiment": ["panda"],
        "action_spec": ["ee_delta.v1"],
        "control_mode": ["pd_ee_delta_pose"],
    }


class ConditionVocabularyTest(unittest.TestCase):
    def test_assigns_reserved_and_sorted_explicit_ids(self) -> None:
        vocabulary = ConditionVocabulary.build(declarations())

        self.assertEqual(vocabulary.id_for("dataset", PAD_TOKEN), 0)
        self.assertEqual(vocabulary.id_for("dataset", UNK_TOKEN), 1)
        self.assertEqual(vocabulary.id_for("dataset", "alpha"), 2)
        self.assertEqual(vocabulary.id_for("dataset", "zeta"), 3)
        self.assertEqual(vocabulary.null_language_token, NULL_LANGUAGE_TOKEN)

    def test_order_of_declarations_does_not_change_hash(self) -> None:
        first = ConditionVocabulary.build(declarations())
        reversed_values = {
            key: list(reversed(values)) for key, values in declarations().items()
        }
        second = ConditionVocabulary.build(reversed_values)

        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_unknown_maps_only_to_unk_and_is_counted(self) -> None:
        vocabulary = ConditionVocabulary.build(declarations())

        self.assertEqual(vocabulary.id_for("task", "unseen task"), 1)
        self.assertEqual(vocabulary.id_for("task", "unseen task"), 1)
        self.assertEqual(vocabulary.unknown_counts, {"task": 2})

    def test_rejects_duplicate_or_reserved_declarations(self) -> None:
        duplicate = declarations()
        duplicate["task"] = ["pick", "pick"]
        with self.assertRaisesRegex(ValueError, "duplicate.*task"):
            ConditionVocabulary.build(duplicate)

        reserved = declarations()
        reserved["task"] = [PAD_TOKEN]
        with self.assertRaisesRegex(ValueError, "reserved token"):
            ConditionVocabulary.build(reserved)

    def test_requires_every_condition_namespace(self) -> None:
        missing = declarations()
        del missing["embodiment"]

        with self.assertRaisesRegex(ValueError, "condition namespaces"):
            ConditionVocabulary.build(missing)

    def test_json_round_trip_preserves_content_hash(self) -> None:
        vocabulary = ConditionVocabulary.build(declarations())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditions.json"
            vocabulary.save(path)
            restored = ConditionVocabulary.load(path)

        self.assertEqual(restored.to_dict(), vocabulary.to_dict())
        self.assertEqual(restored.content_hash, vocabulary.content_hash)


if __name__ == "__main__":
    unittest.main()
