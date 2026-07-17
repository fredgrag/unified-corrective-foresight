# ModelScope DINOv3 Source Amendment

Date: 2026-07-17

## Purpose

Replace the gated Hugging Face download path for the frozen DINOv3 ViT-B/16
backbone with a verified ModelScope mirror. This amendment changes artifact
delivery only. It does not change the model architecture, tensor contract,
training objective, or evaluation protocol.

## Frozen Model Identity

The production backbone remains:

```text
architecture: DINOv3 ViT-B/16
canonical model id: facebook/dinov3-vitb16-pretrain-lvd1689m
canonical Hugging Face revision: 5931719e67bbdb9737e363e781fb0c67687896bc
ModelScope model id: facebook/dinov3-vitb16-pretrain-lvd1689m
ModelScope revision: 23d0280ae6ee4ced592a3459674ad027d3c18906
hidden size: 768
patch size: 16
register tokens: 4
parameter count: 85,660,416
```

The originally suggested ModelScope ViT-L/16 repository is not used because it
has a 1024-dimensional hidden state and would alter the approved architecture.

## Equivalence Evidence

The ModelScope repository was compared with the canonical Hugging Face
revision before this amendment was approved.

| File | Immutable SHA256 | Bytes | Cross-hub result |
| --- | --- | ---: | --- |
| `model.safetensors` | `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b` | 342,662,192 | exact SHA256 |
| `config.json` | `3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec` | 744 | exact Git blob `9f0c03c12a21df9678269d96dac7b59819349792` |
| `preprocessor_config.json` | `960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7` | 585 | exact Git blob `0126173c1ab7b75bbd9102e37de8a56ffe0a012b` |
| `LICENSE.md` | `25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e` | 7,503 | exact licensed source file |

The ModelScope configuration declares `DINOv3ViTModel`, hidden size 768,
12 layers, 12 attention heads, patch size 16, and 4 register tokens. These are
the existing state-encoder assumptions.

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

No symlink is permitted. The large artifact directory is excluded from Git;
the immutable expected manifest remains tracked with the source code.

## Download And Verification Flow

The project provides a dedicated downloader with no `modelscope` Python
dependency:

1. Resolve every URL against the full ModelScope commit, never `master`.
2. Download each file to a temporary `.partial` path in the destination
   filesystem.
3. Verify byte size and SHA256 against the tracked immutable manifest.
4. Atomically rename the complete file into place only after verification.
5. Write `source-manifest.json` last, including both source revisions and every
   verified digest.
6. On a rerun, reuse a file only after re-verifying its size and digest.

Interrupted, truncated, substituted, or mismatched files fail closed and are
never promoted to the production artifact path.

## Runtime Loading

`DinoV3PatchBackbone.from_pretrained_exact` accepts the verified local snapshot
directory. Before constructing Transformers objects it validates:

- the local manifest and full file set;
- every required size and SHA256;
- model type, hidden size, patch size, and register token count;
- the ModelScope and canonical Hugging Face provenance fields.

It then loads `DINOv3ViTImageProcessorFast` and `DINOv3ViTModel` from the local
directory with `local_files_only=True`. Production model initialization does
not contact either hub and cannot silently fall back to another revision.

The existing frozen-backbone, patch-prefix separation, 2-by-4 anchor,
`[B,T,9,768]`, online-adapter, and EMA contracts remain unchanged.

## Error Handling

The implementation raises a specific artifact-integrity error for a missing
file, wrong digest, wrong byte size, invalid manifest, unexpected architecture,
or mutable revision. It never falls back to ViT-L/16, the cached `timm` model,
Hugging Face, or an unpinned ModelScope branch.

## Verification

TDD adds the following gates:

- downloader URLs contain the exact ModelScope commit;
- a valid fixture is installed atomically;
- corrupted and truncated artifacts fail before Transformers loading;
- the loader uses only the verified local directory and
  `local_files_only=True`;
- model metadata must be exactly 768/16/4;
- the real ModelScope snapshot loads on CUDA, processes two distinct RGB
  frames, produces nonconstant patches and `[1,2,9,768]` states, and leaves all
  DINO parameters without gradients after backward;
- the complete warning-as-error test suite, compile check, shell syntax check,
  diff check, and no-symlink gate pass before the Task 6 commit.

## Documentation Contract

The root engineering rule, final research design, implementation plan, and
documentation index must record this source amendment. The original canonical
Hugging Face revision remains part of checkpoint provenance even though the
bytes are delivered by ModelScope.
