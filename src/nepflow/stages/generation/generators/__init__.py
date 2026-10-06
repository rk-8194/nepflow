"""Canonical generation-generator boundaries with lazy optional backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ConfigurationalGenerator
    from .composition import CompositionGrid
    from .composition_primitives import (
        CompositionRealization,
        allocate_crystal_quota,
        assign_composition,
        calculate_composition_realization,
        composition_label,
        measure_composition_realization,
        realize_composition,
        realized_composition,
        record_composition_metadata,
    )
    from .materials_project import MaterialsProjectFetcher, build_materials_project_fetcher
    from .materials_project_generator import MaterialsProjectGenerator
    from .random_solution import RandomSolidSolutionGenerator
    from .segregated import SegregatedGenerator
    from .sqs import IcetSQSBackend, SQSGenerationError, SQSGenerator

__all__ = [
    "ConfigurationalGenerator",
    "CompositionGrid",
    "CompositionRealization",
    "allocate_crystal_quota",
    "assign_composition",
    "calculate_composition_realization",
    "composition_label",
    "measure_composition_realization",
    "realize_composition",
    "realized_composition",
    "record_composition_metadata",
    "IcetSQSBackend",
    "RandomSolidSolutionGenerator",
    "SegregatedGenerator",
    "SQSGenerationError",
    "SQSGenerator",
    "MaterialsProjectFetcher",
    "MaterialsProjectGenerator",
    "build_materials_project_fetcher",
]

_OWNERS = {
    "ConfigurationalGenerator": (".base", "ConfigurationalGenerator"),
    "CompositionGrid": (".composition", "CompositionGrid"),
    "CompositionRealization": (".composition_primitives", "CompositionRealization"),
    "allocate_crystal_quota": (".composition_primitives", "allocate_crystal_quota"),
    "assign_composition": (".composition_primitives", "assign_composition"),
    "calculate_composition_realization": (
        ".composition_primitives",
        "calculate_composition_realization",
    ),
    "composition_label": (".composition_primitives", "composition_label"),
    "measure_composition_realization": (
        ".composition_primitives",
        "measure_composition_realization",
    ),
    "realize_composition": (".composition_primitives", "realize_composition"),
    "realized_composition": (".composition_primitives", "realized_composition"),
    "record_composition_metadata": (".composition_primitives", "record_composition_metadata"),
    "IcetSQSBackend": (".sqs", "IcetSQSBackend"),
    "RandomSolidSolutionGenerator": (".random_solution", "RandomSolidSolutionGenerator"),
    "SegregatedGenerator": (".segregated", "SegregatedGenerator"),
    "SQSGenerationError": (".sqs", "SQSGenerationError"),
    "SQSGenerator": (".sqs", "SQSGenerator"),
    "MaterialsProjectFetcher": (".materials_project", "MaterialsProjectFetcher"),
    "build_materials_project_fetcher": (".materials_project", "build_materials_project_fetcher"),
    "MaterialsProjectGenerator": (".materials_project_generator", "MaterialsProjectGenerator"),
}


def __getattr__(name: str):
    owner = _OWNERS.get(name)
    if owner is None:
        raise AttributeError(name)
    module = __import__(f"{__name__}{owner[0]}", fromlist=[owner[1]])
    return getattr(module, owner[1])
