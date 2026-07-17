from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


MANIFEST_FIELDS = {
    "format_version",
    "epoch",
    "steps",
    "git",
    "runtime",
    "backbone_provenance",
    "dataset_revisions",
    "model_config",
    "loss_config",
    "condition_vocabulary",
    "dataset_specs",
    "action_specs",
    "files",
}


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class RunManifest:
    value: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = dict(self.value)
        if set(value) != MANIFEST_FIELDS:
            raise ValueError(
                f"manifest fields must be exactly {sorted(MANIFEST_FIELDS)}"
            )
        if value["format_version"] != 1:
            raise ValueError("unsupported checkpoint manifest format_version")
        if type(value["epoch"]) is not int or value["epoch"] < 0:
            raise ValueError("manifest epoch must be nonnegative")
        _require_exact_int_mapping(
            "steps",
            value["steps"],
            {"global_step", "optimizer_step", "cycle_warmup_step"},
        )
        git = value["git"]
        if (
            not isinstance(git, Mapping)
            or set(git) != {"commit", "dirty"}
            or not isinstance(git["commit"], str)
            or len(git["commit"]) != 40
            or any(character not in "0123456789abcdef" for character in git["commit"])
            or type(git["dirty"]) is not bool
        ):
            raise ValueError("manifest git provenance is invalid")
        runtime = value["runtime"]
        if (
            not isinstance(runtime, Mapping)
            or set(runtime) != {"lerobot_version"}
            or not isinstance(runtime["lerobot_version"], str)
            or not runtime["lerobot_version"]
        ):
            raise ValueError("manifest runtime provenance is invalid")
        backbone = value["backbone_provenance"]
        backbone_fields = {
            "canonical_model_id",
            "canonical_revision",
            "delivery_model_id",
            "delivery_revision",
            "weights_sha256",
        }
        if (
            not isinstance(backbone, Mapping)
            or set(backbone) != backbone_fields
            or any(not isinstance(item, str) or not item for item in backbone.values())
            or len(backbone["canonical_revision"]) != 40
            or len(backbone["delivery_revision"]) != 40
            or len(backbone["weights_sha256"]) != 64
        ):
            raise ValueError("manifest backbone provenance is invalid")
        revisions = value["dataset_revisions"]
        if (
            not isinstance(revisions, Mapping)
            or not revisions
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(revision, str)
                or not revision
                for name, revision in revisions.items()
            )
        ):
            raise ValueError("manifest dataset_revisions are invalid")
        for name in (
            "model_config",
            "loss_config",
        ):
            if not isinstance(value[name], Mapping) or not value[name]:
                raise ValueError(f"manifest {name} must be a nonempty mapping")
        _validate_hashed_content(
            "condition_vocabulary", value["condition_vocabulary"], single=True
        )
        _validate_hashed_content("dataset_specs", value["dataset_specs"])
        _validate_hashed_content("action_specs", value["action_specs"])
        files = value["files"]
        if not isinstance(files, Mapping) or not files:
            raise ValueError("manifest files must be a nonempty mapping")
        for name, metadata in files.items():
            if (
                not isinstance(name, str)
                or not isinstance(metadata, Mapping)
                or set(metadata) != {"sha256", "size"}
                or not isinstance(metadata["sha256"], str)
                or len(metadata["sha256"]) != 64
                or type(metadata["size"]) is not int
                or metadata["size"] <= 0
            ):
                raise ValueError(f"manifest file metadata is invalid for {name}")
        object.__setattr__(self, "value", MappingProxyType(value))

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("checkpoint manifest is not valid JSON") from error
        if not isinstance(value, Mapping):
            raise ValueError("checkpoint manifest must be a JSON object")
        return cls(value)

    def to_json(self) -> str:
        return json.dumps(
            dict(self.value),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ) + "\n"


def _require_exact_int_mapping(
    name: str, value: object, fields: set[str]
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or any(type(item) is not int or item < 0 for item in value.values())
    ):
        raise ValueError(f"manifest {name} fields are invalid")


def _validate_hashed_content(
    name: str, value: object, *, single: bool = False
) -> None:
    entries = {"value": value} if single else value
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError(f"manifest {name} must be a nonempty mapping")
    for key, entry in entries.items():
        if (
            not isinstance(key, str)
            or not isinstance(entry, Mapping)
            or set(entry) != {"content", "content_hash"}
            or not isinstance(entry["content"], Mapping)
            or entry["content_hash"] != canonical_hash(entry["content"])
        ):
            raise ValueError(f"manifest {name} hash/content is invalid for {key}")
