#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from corrective_foresight.conditioning.language_cache import LanguageEmbeddingCache


MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def load_frozen_clip_text_model(
    device: torch.device,
) -> CLIPTextModelWithProjection:
    model = CLIPTextModelWithProjection.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
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
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    model = load_frozen_clip_text_model(device)

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
            outputs = model(**inputs)
            values = F.normalize(outputs.text_embeds.float(), dim=-1).cpu()
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
