"""Focused perturbation families and their typed coordinator."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coordinator import (
        PerturbationCoordinator,
        PerturbationTaskError,
        family_applies_to_base,
    )
from .defects import (
    antisites,
    gas_in_vacancy,
    gas_interstitials,
    interstitials,
    substitutions,
    vacancies,
    vacancy_interstitial,
)
from .elasticity import (
    ElasticRecord,
    fit_elastic_tensor,
    strain_matrix_to_voigt,
)
from .grain_boundaries import GrainBoundaryConstructionError, grain_boundaries
from .models import (
    PerturbationCounts,
    PerturbationRejection,
    PerturbationSettings,
    PerturbationTask,
    PerturbationTaskResult,
    derive_child_seed,
)
from .surfaces import SurfaceConstructionError, surfaces

__all__ = [
    "PerturbationCoordinator",
    "PerturbationCounts",
    "PerturbationRejection",
    "PerturbationSettings",
    "PerturbationTask",
    "PerturbationTaskError",
    "PerturbationTaskResult",
    "derive_child_seed",
    "family_applies_to_base",
    "antisites",
    "gas_in_vacancy",
    "gas_interstitials",
    "interstitials",
    "substitutions",
    "vacancies",
    "vacancy_interstitial",
    "SurfaceConstructionError",
    "surfaces",
    "GrainBoundaryConstructionError",
    "grain_boundaries",
    "ElasticRecord",
    "fit_elastic_tensor",
    "strain_matrix_to_voigt",
]


def __getattr__(name: str):
    """Defer coordinator imports until validation has finished importing surfaces."""

    if name in {"PerturbationCoordinator", "PerturbationTaskError", "family_applies_to_base"}:
        from . import coordinator

        return getattr(coordinator, name)
    raise AttributeError(name)
