"""Internal peer-algorithm dispatch for complete-candidate selection."""

from __future__ import annotations

from collections.abc import Mapping

from .base import SelectionAlgorithm, SelectionAlgorithmRequest, SelectionAlgorithmResult
from .fps import FPSSelectionAlgorithm
from .information_entropy import InformationEntropyConfig, InformationEntropySelectionAlgorithm

AlgorithmRegistry = Mapping[str, SelectionAlgorithm]


def dispatch_selection_algorithm(
    request: SelectionAlgorithmRequest,
    *,
    registry: AlgorithmRegistry | None = None,
) -> SelectionAlgorithmResult:
    """Dispatch exactly once to the explicitly requested algorithm."""

    algorithms = (
        {
            FPSSelectionAlgorithm.algorithm_id: FPSSelectionAlgorithm(),
            InformationEntropySelectionAlgorithm.algorithm_id: (
                InformationEntropySelectionAlgorithm()
            ),
        }
        if registry is None
        else registry
    )
    try:
        algorithm = algorithms[request.algorithm_id]
    except KeyError as exc:
        raise ValueError(f"Unknown selection algorithm: {request.algorithm_id!r}") from exc
    return algorithm.select(request)


__all__ = [
    "AlgorithmRegistry",
    "FPSSelectionAlgorithm",
    "InformationEntropyConfig",
    "InformationEntropySelectionAlgorithm",
    "SelectionAlgorithm",
    "SelectionAlgorithmRequest",
    "SelectionAlgorithmResult",
    "dispatch_selection_algorithm",
]
