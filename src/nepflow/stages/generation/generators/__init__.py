"""Canonical generation-generator boundaries with lazy optional backends."""

from __future__ import annotations

__all__ = [
    "ConfigurationalGenerator",
    "CompositionGrid",
    "allocate_crystal_quota",
    "assign_composition",
    "composition_label",
    "realized_composition",
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
    "allocate_crystal_quota": (".composition_primitives", "allocate_crystal_quota"),
    "assign_composition": (".composition_primitives", "assign_composition"),
    "composition_label": (".composition_primitives", "composition_label"),
    "realized_composition": (".composition_primitives", "realized_composition"),
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
