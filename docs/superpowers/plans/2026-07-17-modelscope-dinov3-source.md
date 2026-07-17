# ModelScope DINOv3 Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved frozen DINOv3 ViT-B/16 bytes from an immutable ModelScope snapshot, verify them before use, and complete the existing Task 6 CUDA gate without changing the model architecture.

**Architecture:** A tracked source manifest owns both hub identities and every required file digest. A standard-library downloader installs physical files atomically under the ignored project `artifacts/models/` tree; the production backbone verifies that snapshot and loads Transformers with `local_files_only=True`. The existing 768-dimensional state adapter and EMA target remain unchanged.

**Tech Stack:** Python 3.12 standard library, PyTorch 2.8.0, Transformers 5.3.0, safetensors, `unittest`, ModelScope pinned resolve URLs.

## Global Constraints

- Use SSH `root@8.130.172.181` port `1010` for every remote command.
- Keep `facebook/dinov3-vitb16-pretrain-lvd1689m`; ViT-L/16 is forbidden.
- Canonical Hugging Face revision is `5931719e67bbdb9737e363e781fb0c67687896bc`.
- ModelScope revision is `23d0280ae6ee4ced592a3459674ad027d3c18906`; `master` is forbidden.
- The weight SHA256 is `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.
- Production metadata must be hidden size 768, patch size 16, and 4 register tokens.
- Store physical files inside the isolated project; do not create symlinks.
- Do not add the `modelscope` Python package or contact a hub during model loading.
- Preserve exactly four optimized losses and every previously frozen model contract.
- Follow warning-as-error RED/GREEN TDD; the real CUDA test cannot skip.

---

### Task 1: Immutable Artifact Contract And Atomic Downloader

**Files:**
- Create: `corrective_foresight/model/manifests/dinov3-vitb16-lvd1689m.json`
- Create: `corrective_foresight/model/dinov3_artifact.py`
- Create: `scripts/download_dinov3.py`
- Create: `tests/unit/test_dinov3_artifact.py`
- Verify: `.gitignore` already contains `/artifacts/`; do not weaken it

**Interfaces:**
- Consumes: an artifact root `Path`, the tracked JSON manifest, and HTTPS responses from pinned ModelScope resolve URLs.
- Produces: `load_expected_manifest() -> DinoV3SourceManifest`, `install_snapshot(artifact_root: Path) -> Path`, `verify_snapshot(snapshot_dir: Path) -> None`, and `default_snapshot_path() -> Path`.

- [ ] **Step 1: Write the strict manifest and RED tests**

The tracked JSON must contain these exact identities and files:

```json
{
  "schema_version": 1,
  "canonical": {
    "provider": "huggingface",
    "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "revision": "5931719e67bbdb9737e363e781fb0c67687896bc"
  },
  "source": {
    "provider": "modelscope",
    "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "revision": "23d0280ae6ee4ced592a3459674ad027d3c18906"
  },
  "files": {
    "LICENSE.md": {"size": 7503, "sha256": "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"},
    "config.json": {"size": 744, "sha256": "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec"},
    "model.safetensors": {"size": 342662192, "sha256": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"},
    "preprocessor_config.json": {"size": 585, "sha256": "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7"}
  }
}
```

Tests must use a tiny injected manifest and in-memory response objects. Assert:

```python
def test_installs_only_from_full_revision_and_promotes_atomically(self) -> None:
    payloads = {"config.json": b"config", "model.safetensors": b"weights"}
    manifest = tiny_manifest(payloads)
    snapshot = install_snapshot(self.root, manifest=manifest, open_url=fake_open(payloads))
    self.assertEqual({path.name for path in snapshot.iterdir()}, {
        "config.json", "model.safetensors", "source-manifest.json"
    })
    self.assertTrue(all(manifest.source_revision in url for url in requested_urls))
    self.assertFalse(any("master" in url for url in requested_urls))

def test_corruption_and_symlink_fail_closed(self) -> None:
    snapshot = install_tiny_valid_snapshot(self.root)
    (snapshot / "config.json").write_bytes(b"corrupt")
    with self.assertRaisesRegex(DinoV3ArtifactError, "config.json.*digest"):
        verify_snapshot(snapshot, manifest=tiny_manifest)
```

Also prove wrong byte size, missing file, altered local manifest, extra file, and a symlink are rejected.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONWARNINGS=error /mnt/workspace/wwl/.venvs/unified-corrective-foresight/bin/python \
  -m unittest tests.unit.test_dinov3_artifact -v
```

Expected: import failure because `corrective_foresight.model.dinov3_artifact` does not exist.

- [ ] **Step 3: Implement strict parsing, streaming hashes, and atomic promotion**

Use immutable dataclasses and an injectable network boundary:

```python
@dataclass(frozen=True, slots=True)
class ArtifactFile:
    size: int
    sha256: str

@dataclass(frozen=True, slots=True)
class DinoV3SourceManifest:
    canonical_model_id: str
    canonical_revision: str
    source_model_id: str
    source_revision: str
    files: Mapping[str, ArtifactFile]

def install_snapshot(
    artifact_root: Path,
    manifest: DinoV3SourceManifest | None = None,
    open_url: Callable[[str], ContextManager[BinaryIO]] = urlopen,
) -> Path:
    selected = manifest or load_expected_manifest()
    snapshot = artifact_root / selected.source_model_id / selected.source_revision
    # Stream each pinned URL to `<name>.partial`, hash while writing, fsync,
    # validate size/digest, then os.replace. Write source-manifest.json last.
    verify_snapshot(snapshot, manifest=selected)
    return snapshot
```

Reject path traversal in manifest names, non-hex digests, nonpositive sizes,
unknown providers, mutable revisions, symlinks, and nonregular files. Delete a
failed `.partial` file before raising `DinoV3ArtifactError`.

The CLI is only a typed wrapper:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "artifacts/models")
    snapshot = install_snapshot(parser.parse_args().artifact_root)
    print(snapshot)
```

- [ ] **Step 4: Run GREEN and boundary checks**

Run the Task 1 unit test with `PYTHONWARNINGS=error`, then:

```bash
python -m compileall -q corrective_foresight/model/dinov3_artifact.py \
  scripts/download_dinov3.py tests/unit/test_dinov3_artifact.py
git diff --check
```

Expected: all artifact tests pass; compile and diff checks emit no output.

- [ ] **Step 5: Commit the independent artifact boundary**

```bash
git add corrective_foresight/model/manifests/dinov3-vitb16-lvd1689m.json \
  corrective_foresight/model/dinov3_artifact.py scripts/download_dinov3.py \
  tests/unit/test_dinov3_artifact.py
git commit -m "feat: verify pinned ModelScope DINOv3 artifacts"
```

---

### Task 2: Verified Local-Only DINOv3 Loader

**Files:**
- Modify: `corrective_foresight/model/dinov3_backbone.py`
- Modify: `corrective_foresight/model/__init__.py`
- Modify: `tests/unit/test_dinov3_backbone.py`
- Modify: `tests/integration/test_real_dinov3.py`

**Interfaces:**
- Consumes: `DinoV3PatchBackbone.from_pretrained_exact(snapshot_dir: Path, device: torch.device | str)`, where `snapshot_dir` has passed `verify_snapshot`.
- Produces: the existing frozen `DinoV3PatchBackbone.forward(rgb) -> PatchGrid` without network access or architecture changes.

- [ ] **Step 1: Replace the hub-loader test with RED local-only contracts**

Update the mocked loader test to assert:

```python
result = DinoV3PatchBackbone.from_pretrained_exact(snapshot_dir, device="cpu")
verify_snapshot.assert_called_once_with(snapshot_dir)
load_processor.assert_called_once_with(str(snapshot_dir), local_files_only=True)
load_model.assert_called_once_with(str(snapshot_dir), local_files_only=True)
```

Add a test where `verify_snapshot` raises `DinoV3ArtifactError`; neither
Transformers loader may be called. Add table-driven config failures for hidden
size other than 768, patch size other than 16, register count other than 4,
architecture other than `DINOv3ViTModel`, and model type other than
`dinov3_vit`.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONWARNINGS=error /mnt/workspace/wwl/.venvs/unified-corrective-foresight/bin/python \
  -m unittest tests.unit.test_dinov3_backbone -v
```

Expected: failures because the current method accepts no snapshot path and
still calls the Hugging Face model ID and revision.

- [ ] **Step 3: Implement verified local loading**

Keep canonical provenance constants but add ModelScope constants from the
manifest. The public loader must follow this order:

```python
@classmethod
def from_pretrained_exact(
    cls,
    snapshot_dir: Path,
    device: torch.device | str = "cpu",
) -> DinoV3PatchBackbone:
    snapshot_dir = snapshot_dir.resolve(strict=True)
    verify_snapshot(snapshot_dir)
    processor = DINOv3ViTImageProcessorFast.from_pretrained(
        str(snapshot_dir), local_files_only=True
    )
    model = DINOv3ViTModel.from_pretrained(
        str(snapshot_dir), local_files_only=True
    )
    _validate_production_config(model.config)
    return cls(model=model.to(device), processor=processor)
```

`_validate_production_config` must require the exact architecture, model type,
768 hidden width, 16 patch size, 4 registers, 12 layers, and 12 heads before
the model reaches CUDA.

Update the real test to take `default_snapshot_path()` and replace the old
authentication failure message with a ModelScope artifact-integrity failure.

- [ ] **Step 4: Run focused GREEN**

Run the backbone, spatial resampler, state encoder, and EMA unit modules with
warnings as errors. Expected: all focused unit tests pass and no network call
occurs.

Do not commit this step separately because these files belong to the existing
uncommitted Task 6 state-encoder change and must pass its real gate first.

---

### Task 3: Synchronize Governing Documentation

**Files:**
- Create: `../docs/research/2026-07-17-modelscope-dinov3-source-amendment.md`
- Modify: `../AGENTS.md`
- Modify: `../docs/README.md`
- Modify: `../docs/engineering/unified-model-implementation-rule.md`
- Modify: `../docs/research/2026-07-16-unified-corrective-foresight-final-design.md`
- Modify: `../docs/implementation/2026-07-16-unified-corrective-foresight-core-implementation-plan.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the approved source specification and the exact manifest from Task 1.
- Produces: one unambiguous source-precedence chain for future agents and users.

- [ ] **Step 1: Copy the approved amendment into root research docs**

The root amendment must preserve the complete identity, hash, layout, loading,
failure, and verification sections from
`docs/superpowers/specs/2026-07-17-modelscope-dinov3-source-design.md`.

- [ ] **Step 2: Update source precedence and frozen defaults**

Add the amendment after the final design and before the implementation plan in
`AGENTS.md` and `docs/README.md`. Replace the obsolete statement that runtime
requires Hugging Face authentication with:

```text
DINOv3 remains the canonical Hugging Face ViT-B/16 revision, but production
bytes are delivered from the content-equivalent pinned ModelScope commit and
verified locally before offline Transformers loading.
```

Task 6 must list the artifact module, manifest, downloader, local-only loader,
and non-skippable ModelScope CUDA gate. Do not alter the state block or losses.

- [ ] **Step 3: Add the operator command to the isolated README**

Document exactly:

```bash
python scripts/download_dinov3.py --artifact-root artifacts/models
python -m unittest tests.integration.test_real_dinov3 -v
```

State that `artifacts/` is ignored, physical, locally verified, and required
before Task 6 or training.

- [ ] **Step 4: Audit documentation consistency**

Run searches for `DINOv3`, `Hugging Face`, `ModelScope`, both revisions, and
the four loss names across all governing files. Expected: no statement still
requires gated Hugging Face download and no model is changed to ViT-L/16.

---

### Task 4: Real Snapshot, CUDA Gate, Regression, And Task 6 Commit

**Files:**
- Materialize, untracked: `artifacts/models/facebook/dinov3-vitb16-pretrain-lvd1689m/23d0280ae6ee4ced592a3459674ad027d3c18906/`
- Commit the existing Task 6 files: `corrective_foresight/model/dinov3_backbone.py`, `spatial_resampler.py`, `state_encoder.py`, `ema.py`, `model/__init__.py`, and Task 6 tests
- Commit: `README.md`

**Interfaces:**
- Consumes: the ModelScope artifact installer, verified local loader, RGB/camera/proprio tensors, and CUDA device 0.
- Produces: verified `[1,2,9,768]` state blocks and detached EMA state targets from the exact frozen DINOv3 ViT-B/16.

- [ ] **Step 1: Download and independently verify the real snapshot**

Run:

```bash
/mnt/workspace/wwl/.venvs/unified-corrective-foresight/bin/python \
  scripts/download_dinov3.py --artifact-root artifacts/models
sha256sum artifacts/models/facebook/dinov3-vitb16-pretrain-lvd1689m/\
23d0280ae6ee4ced592a3459674ad027d3c18906/model.safetensors
```

Expected SHA256:
`9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.

- [ ] **Step 2: Run the non-skippable real CUDA gate**

Run:

```bash
PYTHONWARNINGS=error /mnt/workspace/wwl/.venvs/unified-corrective-foresight/bin/python \
  -m unittest tests.integration.test_real_dinov3 -v
```

Expected: one PASS after loading two distinct RGB frames; patches are
nonconstant, states are `[1,2,9,768]`, adapter parameters receive gradients,
and every DINO parameter gradient remains `None`.

- [ ] **Step 3: Run complete verification**

Run:

```bash
PYTHONWARNINGS=error /mnt/workspace/wwl/.venvs/unified-corrective-foresight/bin/python \
  -m unittest discover -s tests -v
/mnt/workspace/wwl/.venvs/unified-corrective-foresight/bin/python \
  -m compileall -q corrective_foresight tests scripts
bash -n scripts/*.sh
git diff --check
test -z "$(find . -path ./.git -prune -o -type l -print -quit)"
scripts/verify_environment.sh
```

Expected: all tests and environment checks pass, no warning is emitted, and no
symlink or diff error is found. Confirm `git status --short` does not list any
file under `artifacts/`.

- [ ] **Step 4: Commit the completed Task 6**

```bash
git add README.md corrective_foresight/model tests/unit/fakes.py \
  tests/unit/test_dinov3_backbone.py tests/unit/test_spatial_resampler.py \
  tests/unit/test_state_encoder.py tests/unit/test_ema_state_target.py \
  tests/integration/test_real_dinov3.py
git commit -m "feat: add anchored DINOv3 state encoder and EMA targets"
```

- [ ] **Step 5: Record the checkpoint**

Run `git status --short`, `git log -3 --oneline`, and recompute the root
implementation-plan SHA256. Expected: the isolated Git worktree is clean,
Task 6 is marked complete, and Task 7 is the only in-progress task.
