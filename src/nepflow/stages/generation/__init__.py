"""Canonical structure-generation stage and its typed boundaries."""

from .models import GenerationManifest, GenerationRequest, GenerationResult
from .generators.composition import CompositionGrid
from .provenance import (
    annotate_base_structures,
    assign_seed_ids,
    deduplicate_base_structures,
    merge_provenance,
)
from .stage import GenerationStage

__all__ = [
    "GenerationManifest",
    "GenerationRequest",
    "GenerationResult",
    "GenerationStage",
    "CompositionGrid",
    "annotate_base_structures",
    "assign_seed_ids",
    "deduplicate_base_structures",
    "merge_provenance",
]
