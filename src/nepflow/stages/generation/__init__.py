"""Canonical structure-generation stage and typed boundaries."""

from __future__ import annotations

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


def __getattr__(name: str):
    if name in {"GenerationManifest", "GenerationRequest", "GenerationResult"}:
        from .models import GenerationManifest, GenerationRequest, GenerationResult

        return locals()[name]
    if name == "GenerationStage":
        from .stage import GenerationStage

        return GenerationStage
    if name == "CompositionGrid":
        from .generators.composition import CompositionGrid

        return CompositionGrid
    if name in {
        "annotate_base_structures",
        "assign_seed_ids",
        "deduplicate_base_structures",
        "merge_provenance",
    }:
        from . import provenance

        return getattr(provenance, name)
    raise AttributeError(name)
