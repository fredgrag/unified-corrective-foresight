"""Strict configuration contracts."""

from corrective_foresight.config.loader import load_action_spec, load_dataset_spec
from corrective_foresight.config.schema import ActionSpec, DatasetSpec

__all__ = [
    "ActionSpec",
    "DatasetSpec",
    "load_action_spec",
    "load_dataset_spec",
]
