from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
import torch.nn.functional as F


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_hash(tensor: Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class LanguageEmbeddingCache:
    model_id: str
    revision: str
    dimension: int
    content_hash: str
    _embeddings: Mapping[str, Tensor]

    @classmethod
    def write(
        cls,
        tensor_path: str | Path,
        metadata_path: str | Path,
        embeddings: Mapping[str, Tensor],
        model_id: str,
        revision: str,
    ) -> LanguageEmbeddingCache:
        if not model_id.strip() or not revision.strip():
            raise ValueError("language cache model_id and revision must be nonempty")
        if not embeddings:
            raise ValueError("language cache embeddings cannot be empty")
        normalized: dict[str, Tensor] = {}
        dimension: int | None = None
        entries: list[dict[str, str]] = []
        tensor_values: dict[str, Tensor] = {}
        for text in sorted(embeddings):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("language cache texts must be nonempty strings")
            value = torch.as_tensor(embeddings[text], dtype=torch.float32).reshape(-1)
            if dimension is None:
                dimension = value.numel()
            if value.numel() != dimension:
                raise ValueError("language cache embeddings must share one dimension")
            if not torch.isfinite(value).all().item() or value.norm().item() <= 0:
                raise ValueError(f"invalid language embedding for text: {text}")
            value = F.normalize(value, dim=0).cpu().contiguous()
            tensor_key = "text_" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            normalized[text] = value
            tensor_values[tensor_key] = value
            entries.append(
                {
                    "text": text,
                    "tensor_key": tensor_key,
                    "tensor_sha256": _tensor_hash(value),
                }
            )
        metadata = {
            "format_version": 1,
            "model_id": model_id,
            "revision": revision,
            "dimension": dimension,
            "entries": entries,
        }
        content_hash = _canonical_hash(metadata)
        tensor_destination = Path(tensor_path)
        metadata_destination = Path(metadata_path)
        tensor_destination.parent.mkdir(parents=True, exist_ok=True)
        metadata_destination.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensor_values, str(tensor_destination))
        metadata["content_hash"] = content_hash
        metadata_destination.write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return cls(
            model_id=model_id,
            revision=revision,
            dimension=int(dimension),
            content_hash=content_hash,
            _embeddings=MappingProxyType(normalized),
        )

    @classmethod
    def load(
        cls, tensor_path: str | Path, metadata_path: str | Path
    ) -> LanguageEmbeddingCache:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        required = {
            "format_version",
            "model_id",
            "revision",
            "dimension",
            "entries",
            "content_hash",
        }
        if not isinstance(metadata, dict) or set(metadata) != required:
            raise ValueError("language cache metadata fields are invalid")
        content_hash = metadata.pop("content_hash")
        if metadata["format_version"] != 1:
            raise ValueError("unsupported language cache format_version")
        if content_hash != _canonical_hash(metadata):
            raise ValueError("language cache content_hash mismatch")
        dimension = metadata["dimension"]
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("language cache dimension must be a positive integer")
        tensors = load_file(str(tensor_path), device="cpu")
        expected_keys = {entry["tensor_key"] for entry in metadata["entries"]}
        if set(tensors) != expected_keys:
            raise ValueError("language cache tensor keys do not match metadata")

        embeddings: dict[str, Tensor] = {}
        for entry in metadata["entries"]:
            if set(entry) != {"text", "tensor_key", "tensor_sha256"}:
                raise ValueError("language cache entry fields are invalid")
            text = entry["text"]
            tensor = tensors[entry["tensor_key"]].to(torch.float32).reshape(-1)
            if tensor.numel() != dimension:
                raise ValueError(f"language embedding dimension mismatch for: {text}")
            if not torch.isfinite(tensor).all().item() or not torch.isclose(
                tensor.norm(), torch.tensor(1.0), atol=1e-5, rtol=1e-5
            ).item():
                raise ValueError(f"language embedding is not finite/unit-normalized: {text}")
            if _tensor_hash(tensor) != entry["tensor_sha256"]:
                raise ValueError(f"language embedding hash mismatch for: {text}")
            if text in embeddings:
                raise ValueError(f"duplicate text in language cache: {text}")
            embeddings[text] = tensor.contiguous()
        return cls(
            model_id=metadata["model_id"],
            revision=metadata["revision"],
            dimension=dimension,
            content_hash=content_hash,
            _embeddings=MappingProxyType(embeddings),
        )

    def lookup(self, text: str) -> Tensor:
        if text not in self._embeddings:
            raise KeyError(f"missing CLIP embedding for task text: {text}")
        return self._embeddings[text].detach().clone()
