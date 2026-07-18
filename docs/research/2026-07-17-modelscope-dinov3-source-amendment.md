# ModelScope DINOv3 Source Amendment

Last updated: 2026-07-17

Status: approved source amendment to the Unified Corrective Foresight final
design. This document changes artifact delivery only.

## Purpose

The frozen DINOv3 ViT-B/16 backbone is delivered from a verified ModelScope
mirror instead of the gated Hugging Face download path. The model architecture,
tensor contract, training objective, and evaluation protocol do not change.

## Frozen Model Identity

```text
architecture                    DINOv3 ViT-B/16
canonical model id              facebook/dinov3-vitb16-pretrain-lvd1689m
canonical Hugging Face revision 5931719e67bbdb9737e363e781fb0c67687896bc
ModelScope model id              facebook/dinov3-vitb16-pretrain-lvd1689m
ModelScope revision              23d0280ae6ee4ced592a3459674ad027d3c18906
hidden size                      768
patch size                       16
register tokens                  4
parameter count                  85,660,416
```

The ModelScope ViT-L/16 repository is not used because its 1024-dimensional
hidden state would change the approved architecture.

## Equivalence Evidence

| File | Immutable SHA256 | Bytes | Cross-hub result |
| --- | --- | ---: | --- |
| `model.safetensors` | `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b` | 342,662,192 | exact SHA256 |
| `config.json` | `3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec` | 744 | exact Git blob `9f0c03c12a21df9678269d96dac7b59819349792` |
| `preprocessor_config.json` | `960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7` | 585 | exact Git blob `0126173c1ab7b75bbd9102e37de8a56ffe0a012b` |
| `LICENSE.md` | `25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e` | 7,503 | exact licensed source file |

The configuration remains `DINOv3ViTModel` with hidden size 768, 12 layers,
12 attention heads, patch size 16, and 4 register tokens.

## Artifact Layout

The model is stored as physical files inside the isolated project:

```text
artifacts/models/facebook/dinov3-vitb16-pretrain-lvd1689m/
  23d0280ae6ee4ced592a3459674ad027d3c18906/
    config.json
    preprocessor_config.json
    model.safetensors
    LICENSE.md
    source-manifest.json
```

No symlink is permitted. `artifacts/` remains excluded from Git; the immutable
expected manifest is tracked with source code.

## Download And Verification

The downloader has no `modelscope` Python dependency and must:

1. Resolve every URL against the full ModelScope commit, never `master`.
2. Stream each file to a `.partial` path on the destination filesystem.
3. Verify exact byte size and SHA256 before promotion.
4. Use `fsync` and atomic rename to publish a complete file.
5. Write `source-manifest.json` only after all model files pass.
6. Re-verify existing files before reusing them.

Interrupted, truncated, substituted, mismatched, or symlinked artifacts fail
closed and are never accepted by the runtime loader.

## Runtime Loading

`DinoV3PatchBackbone.from_pretrained_exact` accepts only the verified local
snapshot directory. Before Transformers construction it validates the local
manifest, exact file set, every digest and size, both provenance revisions,
and the frozen 768/16/4 architecture metadata.

The processor and model are loaded from that directory with
`local_files_only=True`. Production initialization cannot contact either hub
or silently select another revision. The canonical Hugging Face revision
remains in checkpoint provenance even though ModelScope delivers the bytes.

The frozen-backbone, prefix separation, 2-by-4 anchors, `[B,T,9,768]` states,
online adapter, EMA target, four-loss objective, and all evaluation contracts
remain unchanged.

## Error Handling And Verification

A missing file, wrong digest, wrong byte size, invalid manifest, unexpected
architecture, mutable revision, or symlink raises an artifact-integrity error.
There is no fallback to ViT-L/16, cached `timm` weights, Hugging Face, or
ModelScope `master`.

The non-skippable real gate downloads the exact snapshot, runs two distinct
RGB frames on CUDA, proves nonconstant patches and `[1,2,9,768]` states, runs
backward, and proves every DINO parameter gradient remains absent. The full
warning-as-error suite, compile check, shell syntax check, diff check, and
no-symlink gate must also pass.
