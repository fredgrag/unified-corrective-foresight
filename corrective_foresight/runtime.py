from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from corrective_foresight.conditioning.clip_artifact import (
    CLIP_MODEL_ID,
    CLIP_REVISION,
)
from corrective_foresight.conditioning.language_cache import LanguageEmbeddingCache
from corrective_foresight.conditioning.vocabulary import (
    CONDITION_NAMESPACES,
    ConditionVocabulary,
)
from corrective_foresight.config.experiment import (
    ExperimentConfig,
    load_experiment_config,
)
from corrective_foresight.config.loader import load_action_spec, load_dataset_spec
from corrective_foresight.config.schema import ActionSpec, DatasetSpec
from corrective_foresight.model.dinov3_artifact import (
    load_expected_manifest,
    verify_snapshot,
)
from corrective_foresight.policy.factory import (
    PRODUCTION_WORLD_ACTION_CONFIG,
    assemble_production_policy,
)
from corrective_foresight.policy.unified_policy import UnifiedCorrectiveForesightPolicy
from corrective_foresight.training.checkpoint import (
    ExpectedPolicyCheckpointContract,
)


@dataclass(frozen=True, slots=True)
class ProductionRuntime:
    config: ExperimentConfig
    dataset_specs: tuple[DatasetSpec, ...]
    action_specs: tuple[ActionSpec, ...]
    vocabulary: ConditionVocabulary
    language_cache: LanguageEmbeddingCache
    proprio_dimension: int
    max_cameras: int
    backbone_provenance: Mapping[str, str]


def build_condition_vocabulary(
    dataset_specs: Sequence[DatasetSpec],
    action_specs: Sequence[ActionSpec],
    task_texts: Mapping[str, Sequence[str]],
) -> ConditionVocabulary:
    if not dataset_specs or not action_specs:
        raise ValueError("condition vocabulary requires dataset and action specs")
    dataset_ids = {spec.dataset_id for spec in dataset_specs}
    if set(task_texts) != dataset_ids:
        raise ValueError("task text declarations must match DatasetSpec identifiers")
    action_by_id = {spec.spec_id: spec for spec in action_specs}
    if len(action_by_id) != len(action_specs):
        raise ValueError("ActionSpec identifiers must be unique")
    for dataset_spec in dataset_specs:
        if dataset_spec.action_spec_id not in action_by_id:
            raise ValueError("DatasetSpec references an undeclared ActionSpec")
    tasks = tuple(text for values in task_texts.values() for text in values)
    if not tasks:
        raise ValueError("condition vocabulary requires at least one task")
    return ConditionVocabulary.build(
        {
            "dataset": dataset_ids,
            "task": tasks,
            "embodiment": {spec.embodiment_id for spec in dataset_specs},
            "action_spec": set(action_by_id),
            "control_mode": {spec.control_mode for spec in action_specs},
        }
    )


def encode_condition_ids(
    vocabulary: ConditionVocabulary,
    *,
    dataset_spec: DatasetSpec,
    action_spec: ActionSpec,
    task_text: str,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    if dataset_spec.action_spec_id != action_spec.spec_id:
        raise ValueError("DatasetSpec and ActionSpec identifiers do not match")
    values = {
        "dataset": dataset_spec.dataset_id,
        "task": task_text,
        "embodiment": dataset_spec.embodiment_id,
        "action_spec": action_spec.spec_id,
        "control_mode": action_spec.control_mode,
    }
    identifiers: dict[str, torch.Tensor] = {}
    for namespace in CONDITION_NAMESPACES:
        value = values[namespace]
        identifier = vocabulary.id_for(namespace, value)
        if vocabulary.token_for(namespace, identifier) != value:
            qualifier = "task" if namespace == "task" else f"{namespace} value"
            raise ValueError(f"undeclared {qualifier}: {value}")
        identifiers[namespace] = torch.tensor(
            [identifier], dtype=torch.long, device=device
        )
    return identifiers


def production_backbone_provenance() -> dict[str, str]:
    manifest = load_expected_manifest()
    weights = manifest.files.get("model.safetensors")
    if weights is None:
        raise ValueError("DINOv3 source manifest does not declare model.safetensors")
    return {
        "canonical_model_id": manifest.canonical_model_id,
        "canonical_revision": manifest.canonical_revision,
        "delivery_model_id": manifest.source_model_id,
        "delivery_revision": manifest.source_revision,
        "weights_sha256": weights.sha256,
    }


def load_production_runtime(path: str | Path) -> ProductionRuntime:
    config = load_experiment_config(path)
    if config.model != PRODUCTION_WORLD_ACTION_CONFIG:
        raise ValueError("experiment model config is not the frozen production model")
    verify_snapshot(config.artifacts.dinov3_snapshot)
    for name, artifact in (
        ("language tensor", config.artifacts.language_tensor),
        ("language metadata", config.artifacts.language_metadata),
    ):
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"{name} must be a physical file: {artifact}")
    language_cache = LanguageEmbeddingCache.load(
        config.artifacts.language_tensor,
        config.artifacts.language_metadata,
    )
    if (
        language_cache.model_id != CLIP_MODEL_ID
        or language_cache.revision != CLIP_REVISION
        or language_cache.dimension != 512
    ):
        raise ValueError("language cache is not the frozen CLIP B/32 contract")
    dataset_specs = tuple(load_dataset_spec(item) for item in config.dataset_specs)
    action_specs = tuple(load_action_spec(item) for item in config.action_specs)
    vocabulary = build_condition_vocabulary(
        dataset_specs,
        action_specs,
        config.task_texts,
    )
    for values in config.task_texts.values():
        for task_text in values:
            language_cache.lookup(task_text)
    proprio_dimensions = {
        _dataset_proprio_dimension(spec) for spec in dataset_specs
    }
    if len(proprio_dimensions) != 1:
        raise ValueError(
            "current production state adapter requires one shared proprio dimension"
        )
    return ProductionRuntime(
        config=config,
        dataset_specs=dataset_specs,
        action_specs=action_specs,
        vocabulary=vocabulary,
        language_cache=language_cache,
        proprio_dimension=next(iter(proprio_dimensions)),
        max_cameras=max(len(spec.camera_features) for spec in dataset_specs),
        backbone_provenance=production_backbone_provenance(),
    )


def assemble_runtime_policy(
    runtime: ProductionRuntime,
    *,
    device: torch.device | str,
) -> UnifiedCorrectiveForesightPolicy:
    return assemble_production_policy(
        snapshot_dir=runtime.config.artifacts.dinov3_snapshot,
        action_specs=runtime.action_specs,
        vocabulary=runtime.vocabulary,
        language_cache=runtime.language_cache,
        proprio_dimension=runtime.proprio_dimension,
        max_cameras=runtime.max_cameras,
        ema_tau=runtime.config.training.ema_tau,
        device=device,
    )


def expected_policy_checkpoint_contract(
    runtime: ProductionRuntime,
    policy: UnifiedCorrectiveForesightPolicy,
) -> ExpectedPolicyCheckpointContract:
    return ExpectedPolicyCheckpointContract(
        policy=policy,
        condition_vocabulary_hash=runtime.vocabulary.content_hash,
        dataset_spec_hashes={
            spec.dataset_id: spec.content_hash for spec in runtime.dataset_specs
        },
        action_spec_hashes={
            spec.spec_id: spec.content_hash for spec in runtime.action_specs
        },
        lerobot_version="0.5.1",
        backbone_provenance=runtime.backbone_provenance,
        dataset_revisions={
            spec.dataset_id: spec.revision for spec in runtime.dataset_specs
        },
        model_config=asdict(runtime.config.model),
        loss_config=asdict(policy.objective.config),
    )


def _dataset_proprio_dimension(dataset_spec: DatasetSpec) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(
        repo_id=dataset_spec.repo_id or dataset_spec.dataset_id,
        root=dataset_spec.local_root,
        revision=dataset_spec.revision if dataset_spec.repo_id is not None else None,
    )
    dimension = 0
    for feature in dataset_spec.proprio_features:
        if feature not in metadata.features:
            raise ValueError(f"missing declared proprio feature: {feature}")
        shape = metadata.features[feature].get("shape")
        if not isinstance(shape, (tuple, list)) or not shape:
            raise ValueError(f"invalid proprio feature shape: {feature}")
        feature_dimension = math.prod(shape)
        if feature_dimension <= 0:
            raise ValueError(f"invalid proprio feature dimension: {feature}")
        dimension += feature_dimension
    if dimension <= 0:
        raise ValueError("production runtime requires positive proprio dimension")
    return dimension
