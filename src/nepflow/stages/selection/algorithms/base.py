"""Common typed boundary for complete-candidate selection algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class SelectionAlgorithmRequest:
    """Inputs shared by one complete-candidate selection algorithm."""

    candidate_ids: tuple[str, ...]
    candidate_structures: Sequence[Any]
    representations: np.ndarray
    target_count: int
    anchor_indices: tuple[int, ...]
    anchor_ids: tuple[str, ...]
    algorithm_id: str
    algorithm_version: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.candidate_ids) != len(self.candidate_structures):
            raise ValueError("candidate IDs and candidate structures must have equal length")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if self.target_count < 0:
            raise ValueError("target_count must be non-negative")
        if any(index < 0 or index >= len(self.candidate_ids) for index in self.anchor_indices):
            raise ValueError("anchor index is outside the candidate pool")
        expected_anchor_ids = tuple(self.candidate_ids[index] for index in self.anchor_indices)
        if tuple(self.anchor_ids) != expected_anchor_ids:
            raise ValueError("anchor IDs must match anchor indices in candidate order")


@dataclass(frozen=True, slots=True)
class SelectionAlgorithmResult:
    """Identity-safe result returned by one selection algorithm."""

    algorithm_id: str
    algorithm_version: str
    selected_indices: tuple[int, ...]
    selected_candidate_ids: tuple[str, ...]
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    minimum_distance: float = 0.0

    def __post_init__(self) -> None:
        if len(self.selected_indices) != len(self.selected_candidate_ids):
            raise ValueError("selected indices and candidate IDs must have equal length")


class SelectionAlgorithm(Protocol):
    """Protocol implemented by one complete-candidate selection algorithm."""

    algorithm_id: str
    algorithm_version: str

    def select(self, request: SelectionAlgorithmRequest) -> SelectionAlgorithmResult:
        """Select complete candidates for the supplied request."""

