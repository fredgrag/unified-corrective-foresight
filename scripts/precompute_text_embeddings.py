#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPConfig, CLIPModel

from corrective_foresight.conditioning.language_cache import LanguageEmbeddingCache
from corrective_foresight.conditioning.clip_artifact import (
    CLIP_MODEL_ID,
    CLIP_REVISION,
    verify_clip_snapshot,
)


MODEL_ID = CLIP_MODEL_ID
MODEL_REVISION = CLIP_REVISION

_LEGACY_POSITION_ID_LENGTHS = {
    "text_model.embeddings.position_ids": 77,
    "vision_model.embeddings.position_ids": 50,
}


def normalize_legacy_clip_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized = dict(state_dict)
    for key, length in _LEGACY_POSITION_ID_LENGTHS.items():
        if key not in normalized:
            raise ValueError(f"pinned CLIP checkpoint is missing legacy {key}")
        value = normalized.pop(key)
        expected = torch.arange(length, dtype=torch.long).reshape(1, length)
        if value.dtype != torch.long or not torch.equal(value.cpu(), expected):
            raise ValueError(f"legacy {key} is not the reconstructible index buffer")
    return normalized


def load_frozen_clip_model(
    device: torch.device,
    *,
    snapshot_dir: Path,
) -> CLIPModel:
    verify_clip_snapshot(snapshot_dir)
    config = CLIPConfig.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
    )
    model = CLIPModel(config)
    state_dict = torch.load(
        snapshot_dir / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(
        normalize_legacy_clip_state_dict(state_dict),
        strict=True,
    )
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_tasks(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("tasks JSON must be a nonempty list")
    if any(not isinstance(text, str) or not text.strip() for text in value):
        raise ValueError("every task must be a nonempty string")
    if len(set(value)) != len(value):
        raise ValueError("tasks JSON contains duplicate strings")
    return sorted(value)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    tasks = load_tasks(args.tasks_json)
    device = torch.device(args.device)
    model = load_frozen_clip_model(device, snapshot_dir=args.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.snapshot_dir),
        local_files_only=True,
    )

    embeddings: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start : start + args.batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            outputs = model.get_text_features(**inputs)
            values = F.normalize(outputs.pooler_output.float(), dim=-1).cpu()
            embeddings.update(zip(batch, values, strict=True))

    cache = LanguageEmbeddingCache.write(
        tensor_path=args.tensor_output,
        metadata_path=args.metadata_output,
        embeddings=embeddings,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
    )
    print(
        f"Cached {len(tasks)} task embeddings at revision {MODEL_REVISION}; "
        f"content hash {cache.content_hash}"
    )


if __name__ == "__main__":
    main()
