from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
NULL_LANGUAGE_TOKEN = "<NULL_LANGUAGE>"
CONDITION_NAMESPACES = (
    "dataset",
    "task",
    "embodiment",
    "action_spec",
    "control_mode",
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ConditionVocabulary:
    def __init__(self, namespace_tokens: Mapping[str, tuple[str, ...]]) -> None:
        if set(namespace_tokens) != set(CONDITION_NAMESPACES):
            raise ValueError(
                "condition namespaces must be exactly "
                f"{sorted(CONDITION_NAMESPACES)}"
            )
        tokens = {name: tuple(namespace_tokens[name]) for name in CONDITION_NAMESPACES}
        for namespace, values in tokens.items():
            if len(values) < 2 or values[:2] != (PAD_TOKEN, UNK_TOKEN):
                raise ValueError(
                    f"{namespace} must start with reserved PAD and UNK tokens"
                )
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate tokens in namespace {namespace}")
            if values[2:] != tuple(sorted(values[2:])):
                raise ValueError(f"declared tokens in {namespace} must be sorted")
        self._tokens = MappingProxyType(tokens)
        self._ids = {
            namespace: MappingProxyType(
                {token: index for index, token in enumerate(values)}
            )
            for namespace, values in tokens.items()
        }
        self._unknown_counts: dict[str, int] = {}

    @classmethod
    def build(
        cls, declarations: Mapping[str, Iterable[str]]
    ) -> ConditionVocabulary:
        if set(declarations) != set(CONDITION_NAMESPACES):
            raise ValueError(
                "condition namespaces must be exactly "
                f"{sorted(CONDITION_NAMESPACES)}"
            )
        reserved = {PAD_TOKEN, UNK_TOKEN, NULL_LANGUAGE_TOKEN}
        tokens: dict[str, tuple[str, ...]] = {}
        for namespace in CONDITION_NAMESPACES:
            values = list(declarations[namespace])
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{namespace} declarations must be nonempty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate declaration in namespace {namespace}")
            collision = reserved.intersection(values)
            if collision:
                raise ValueError(
                    f"reserved token cannot be declared in {namespace}: "
                    f"{sorted(collision)}"
                )
            tokens[namespace] = (PAD_TOKEN, UNK_TOKEN, *sorted(values))
        return cls(tokens)

    @property
    def null_language_token(self) -> str:
        return NULL_LANGUAGE_TOKEN

    @property
    def unknown_counts(self) -> dict[str, int]:
        return dict(self._unknown_counts)

    @property
    def content_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    def size(self, namespace: str) -> int:
        self._require_namespace(namespace)
        return len(self._tokens[namespace])

    def id_for(self, namespace: str, value: str) -> int:
        self._require_namespace(namespace)
        identifier = self._ids[namespace].get(value)
        if identifier is not None:
            return identifier
        self._unknown_counts[namespace] = self._unknown_counts.get(namespace, 0) + 1
        return self._ids[namespace][UNK_TOKEN]

    def token_for(self, namespace: str, identifier: int) -> str:
        self._require_namespace(namespace)
        if identifier < 0 or identifier >= self.size(namespace):
            raise ValueError(f"{namespace} identifier is out of range: {identifier}")
        return self._tokens[namespace][identifier]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "special_tokens": {
                "pad": PAD_TOKEN,
                "unk": UNK_TOKEN,
                "null_language": NULL_LANGUAGE_TOKEN,
            },
            "namespaces": {
                namespace: list(self._tokens[namespace])
                for namespace in CONDITION_NAMESPACES
            },
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["content_hash"] = self.content_hash
        destination.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ConditionVocabulary:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        required = {"format_version", "special_tokens", "namespaces", "content_hash"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("condition vocabulary JSON fields are invalid")
        content_hash = payload.pop("content_hash")
        if payload.get("format_version") != 1:
            raise ValueError("unsupported condition vocabulary format_version")
        expected_special = {
            "pad": PAD_TOKEN,
            "unk": UNK_TOKEN,
            "null_language": NULL_LANGUAGE_TOKEN,
        }
        if payload.get("special_tokens") != expected_special:
            raise ValueError("condition vocabulary special tokens are invalid")
        namespace_tokens = {
            str(namespace): tuple(values)
            for namespace, values in payload["namespaces"].items()
        }
        vocabulary = cls(namespace_tokens)
        if content_hash != vocabulary.content_hash:
            raise ValueError("condition vocabulary content_hash mismatch")
        return vocabulary

    def _require_namespace(self, namespace: str) -> None:
        if namespace not in self._tokens:
            raise ValueError(f"unknown condition namespace: {namespace}")
