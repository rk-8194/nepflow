"""Typed records for the canonical selection stage boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .algorithms.information_entropy.diagnostics import EntropyScientificDiagnostics


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Complete in-memory result produced by one selection run."""

    descriptors: np.ndarray
    train_indices: list[int]
    train_min_dist: float | None
    train_seed_count: int
    train_single_element_elastic_count: int
    train_elastic_count: int
    train_anchor_count: int
    train_fps_count: int
    test_indices: list[int]
    test_min_dist: float | None
    min_train_test_dist: float | None
    mean_train_test_dist: float | None
    seed_indices: list[int]
    single_element_elastic_indices: list[int]
    elastic_indices: list[int]
    algorithm_id: str = "fps"
    algorithm_version: str = "selection-request-v1"
    train_acquisition_order: list[str] = field(default_factory=list)
    train_entropy_objective: float | None = None
    train_entropy_cross_entropy: float | None = None
    train_entropy_forward_kl: float | None = None
    train_entropy_pool_fingerprint: str | None = None
    train_entropy_graph_fingerprint: str | None = None
    train_entropy_kernel_operator_fingerprint: str | None = None
    train_entropy_contributions_fingerprint: str | None = None
    train_entropy_state_fingerprint: str | None = None
    train_entropy_history: list[dict[str, Any]] = field(default_factory=list)
    train_entropy_provenance: Mapping[str, Any] = field(default_factory=dict)
    entropy_diagnostics: EntropyScientificDiagnostics | None = None
    train_min_dist_applicable: bool = True
    train_fps_count_applicable: bool = True
    test_selection_policy: str = "candidate_mean_fps_legacy"
    test_selection_version: str = "candidate-mean-fps-v1"
    test_selection_provenance: Mapping[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        """Return the legacy result shape at the artifact boundary."""

        return {
            "descriptors": self.descriptors,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "train_indices": self.train_indices,
            "train_min_dist": self.train_min_dist,
            "train_seed_count": self.train_seed_count,
            "train_single_element_elastic_count": self.train_single_element_elastic_count,
            "train_elastic_count": self.train_elastic_count,
            "train_anchor_count": self.train_anchor_count,
            "train_fps_count": self.train_fps_count,
            "train_min_dist_applicable": self.train_min_dist_applicable,
            "train_fps_count_applicable": self.train_fps_count_applicable,
            "train_acquisition_order": self.train_acquisition_order,
            "train_entropy_objective": self.train_entropy_objective,
            "train_entropy_cross_entropy": self.train_entropy_cross_entropy,
            "train_entropy_forward_kl": self.train_entropy_forward_kl,
            "train_entropy_pool_fingerprint": self.train_entropy_pool_fingerprint,
            "train_entropy_graph_fingerprint": self.train_entropy_graph_fingerprint,
            "train_entropy_kernel_operator_fingerprint": (
                self.train_entropy_kernel_operator_fingerprint
            ),
            "train_entropy_contributions_fingerprint": self.train_entropy_contributions_fingerprint,
            "train_entropy_state_fingerprint": self.train_entropy_state_fingerprint,
            "train_entropy_history": self.train_entropy_history,
            "train_entropy_provenance": self.train_entropy_provenance,
            "entropy_diagnostics": (
                None if self.entropy_diagnostics is None else self.entropy_diagnostics.to_manifest()
            ),
            "test_indices": self.test_indices,
            "test_min_dist": self.test_min_dist,
            "min_train_test_dist": self.min_train_test_dist,
            "mean_train_test_dist": self.mean_train_test_dist,
            "seed_indices": self.seed_indices,
            "single_element_elastic_indices": self.single_element_elastic_indices,
            "elastic_indices": self.elastic_indices,
            "test_selection_policy": self.test_selection_policy,
            "test_selection_version": self.test_selection_version,
            "test_selection_provenance": self.test_selection_provenance,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow read-only mapping access while callers migrate to fields."""

        return self.as_mapping()[key]

    @property
    def train_entropy_diagnostics(self) -> EntropyScientificDiagnostics | None:
        """Compatibility spelling for the structured entropy closure record."""

        return self.entropy_diagnostics


__all__ = ["SelectionResult"]
