"""Deterministic complete-candidate information-entropy selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from nepflow.stages.selection.representations import LocalEnvironmentRepresentation

from ..base import SelectionAlgorithmRequest, SelectionAlgorithmResult
from .models import InformationEntropyConfig


class InformationEntropySelectionAlgorithm:
    """Greedy finite-pool coverage over local-environment information.

    The local representation is supplied through the request options.  The
    algorithm acquires complete candidates; local rows are only the reference
    population used to calculate the objective.
    """

    algorithm_id = "information_entropy"
    algorithm_version = "information-entropy-v1"

    @staticmethod
    def _configuration(options: Mapping[str, Any]) -> InformationEntropyConfig:
        config = options.get("information_entropy_config")
        if isinstance(config, InformationEntropyConfig):
            return config
        if isinstance(config, Mapping):
            return InformationEntropyConfig(**dict(config))
        return InformationEntropyConfig(
            background_mass=float(options.get("background_mass", 1.0e-12)),
            kernel_scale=float(options.get("kernel_scale", 1.0)),
        )

    @staticmethod
    def _local_rows(request: SelectionAlgorithmRequest) -> tuple[np.ndarray, tuple[int, ...]]:
        local = request.options.get("local_representation")
        if not isinstance(local, LocalEnvironmentRepresentation):
            raise ValueError(
                "information-entropy selection requires a local environment representation"
            )
        row_values = np.asarray(local.descriptors, dtype=np.float64)
        row_candidates = tuple(request.candidate_ids.index(row.candidate_id) for row in local.rows)
        return row_values, row_candidates

    def select(self, request: SelectionAlgorithmRequest) -> SelectionAlgorithmResult:
        if request.algorithm_id != self.algorithm_id:
            raise ValueError(
                f"Request algorithm {request.algorithm_id!r} does not match information entropy"
            )
        candidate_count = len(request.candidate_ids)
        anchors = tuple(sorted(set(request.anchor_indices)))
        if len(anchors) > request.target_count:
            raise ValueError("anchor count exceeds information-entropy target count")
        if candidate_count == 0:
            return SelectionAlgorithmResult(
                self.algorithm_id,
                self.algorithm_version,
                (),
                (),
            )
        if request.target_count >= candidate_count:
            selected = tuple(range(candidate_count))
            return SelectionAlgorithmResult(
                self.algorithm_id,
                self.algorithm_version,
                selected,
                tuple(request.candidate_ids[index] for index in selected),
            )

        config = self._configuration(request.options)
        rows, row_candidates = self._local_rows(request)
        if rows.ndim != 2 or rows.shape[0] == 0 or not np.all(np.isfinite(rows)):
            raise ValueError("information-entropy local rows must be finite and non-empty")
        row_counts = np.bincount(row_candidates, minlength=candidate_count).astype(np.float64)
        if np.any(row_counts == 0.0):
            raise ValueError("every candidate must own at least one local environment")
        target_weights = np.asarray(
            [1.0 / (candidate_count * row_counts[candidate]) for candidate in row_candidates],
            dtype=np.float64,
        )
        distances = np.sum((rows[:, None, :] - rows[None, :, :]) ** 2, axis=2)
        similarities = np.exp(-distances / (2.0 * config.kernel_scale**2))
        candidate_kernels = np.empty((candidate_count, rows.shape[0]), dtype=np.float64)
        for candidate in range(candidate_count):
            source = np.flatnonzero(np.asarray(row_candidates) == candidate)
            kernel = np.mean(similarities[:, source], axis=1)
            normalization = float(np.sum(kernel * target_weights))
            candidate_kernels[candidate] = kernel / max(normalization, config.background_mass)

        support = np.full(rows.shape[0], config.background_mass, dtype=np.float64)
        selected = list(anchors)
        for candidate in anchors:
            support += candidate_kernels[candidate]
        remaining = set(range(candidate_count)) - set(selected)
        objective_history: list[float] = [float(np.sum(target_weights * np.log(support)))]
        while len(selected) < request.target_count and remaining:
            best_candidate = min(remaining, key=lambda index: request.candidate_ids[index])
            best_gain = -np.inf
            best_objective = -np.inf
            for candidate in sorted(remaining, key=lambda index: request.candidate_ids[index]):
                trial_support = support + candidate_kernels[candidate]
                objective = float(np.sum(target_weights * np.log(trial_support)))
                gain = objective - objective_history[-1]
                if gain > best_gain:
                    best_candidate = candidate
                    best_gain = gain
                    best_objective = objective
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            support += candidate_kernels[best_candidate]
            objective_history.append(best_objective)

        selected = tuple(sorted(selected))
        return SelectionAlgorithmResult(
            algorithm_id=self.algorithm_id,
            algorithm_version=self.algorithm_version,
            selected_indices=selected,
            selected_candidate_ids=tuple(request.candidate_ids[index] for index in selected),
            diagnostics=(
                {
                    "objective_history": tuple(objective_history),
                    "local_row_count": int(rows.shape[0]),
                    "background_mass": config.background_mass,
                },
            ),
        )


__all__ = ["InformationEntropySelectionAlgorithm"]
