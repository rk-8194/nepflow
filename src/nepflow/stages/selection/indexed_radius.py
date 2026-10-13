"""Dependency-neutral exact indexed radius counts.

The entropy reference measure and the entropy neighbour backends need the same
low-level operation: find candidate locations with ``cKDTree`` and certify
the final answer with the canonical float64 NumPy distance expression.  This
module intentionally imports neither entropy models nor bandwidth code so it
can be used during representation construction without an import cycle.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

import numpy as np

from nepflow.resources.budget import ResourceBudgetService, ResourceCapacityError

logger = logging.getLogger(__name__)


def _validate_descriptors(descriptors: np.ndarray) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("indexed radius descriptors must be a non-empty 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("indexed radius descriptors must be finite")
    return np.ascontiguousarray(values, dtype=np.float64)


def _validate_positive_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_chunk_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError("reference_measure_chunk_size must be a positive integer")
    return int(value)


def group_descriptor_locations(
    descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic unique locations, representatives, inverse IDs and counts."""

    values = _validate_descriptors(descriptors)
    locations, first_indices, row_to_location = np.unique(
        values,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    inverse = np.asarray(row_to_location, dtype=np.int64)
    counts = np.bincount(inverse, minlength=int(locations.shape[0])).astype(np.int64, copy=False)
    return (
        np.ascontiguousarray(locations, dtype=np.float64),
        np.asarray(first_indices, dtype=np.int64),
        inverse,
        counts,
    )


def _radius_guard(values: np.ndarray, radius: float) -> float:
    """Return a conservative distance guard for the cKDTree candidate query.

    The canonical oracle is ``sum((x-y)**2, dtype=float64) <= radius**2``.
    Subtraction, multiplication, summation and the tree's norm calculation
    can round independently.  The bound below is deliberately conservative:
    it covers the coordinate scale, the dimension-dependent summation error,
    and one outward representable step.  Candidates in the guard annulus are
    canonically rechecked, so the guard changes only operational candidate
    discovery and never the scientific inclusion rule.
    """

    dimension = max(1, int(values.shape[1]))
    coordinate_scale = max(1.0, float(np.max(np.abs(values))))
    norm_scale = max(radius, math.sqrt(float(dimension)) * coordinate_scale)
    eps = np.finfo(np.float64).eps
    # gamma_n = n*eps/(1-n*eps), with a safe linear upper bound here.
    rounding = 32.0 * (dimension + 1.0) * eps * norm_scale
    return max(
        float(np.nextafter(radius, np.inf) - radius),
        rounding,
        1.0e-15 * max(1.0, radius),
    )


class IndexedRadiusCountService:
    """Count inclusive radius neighbours while preserving row multiplicity."""

    def __init__(
        self,
        descriptors: np.ndarray,
        *,
        chunk_size: int = 1024,
        resource_budget: ResourceBudgetService | None = None,
    ) -> None:
        self.descriptors = _validate_descriptors(descriptors)
        self.chunk_size = _validate_chunk_size(chunk_size)
        self.resource_budget = resource_budget
        logger.info(
            "Indexed occupancy index build started: rows=%d, dimensions=%d",
            self.descriptors.shape[0],
            self.descriptors.shape[1],
        )
        started = time.perf_counter()
        (
            self.unique_locations,
            self.location_representatives,
            self.row_to_location,
            self.location_multiplicity,
        ) = group_descriptor_locations(self.descriptors)
        self.grouping_seconds = max(0.0, time.perf_counter() - started)
        try:
            from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]
        except ImportError as exc:
            raise ValueError(
                "indexed reference-measure occupancy requires scipy.spatial.cKDTree"
            ) from exc
        tree_started = time.perf_counter()
        # The count-only tree stores original rows, not just unique locations:
        # cKDTree's return_length is then already multiplicity-weighted.  The
        # unique-location grouping remains the deterministic identity and
        # diagnostic layer, while duplicate rows never require candidate-list
        # materialisation on the ordinary path.
        self._tree: Any = cKDTree(self.descriptors, copy_data=True)
        self.tree_seconds = max(0.0, time.perf_counter() - tree_started)
        self.worker_count = self._resolve_worker_count()
        self.ambiguous_count = 0
        self.outer_candidate_count = 0
        logger.info(
            "Indexed occupancy index build completed: rows=%d, unique_locations=%d, "
            "duplicate_rows=%d, grouping_elapsed=%.3fs, tree_elapsed=%.3fs, workers=%d",
            self.descriptors.shape[0],
            self.unique_locations.shape[0],
            self.descriptors.shape[0] - self.unique_locations.shape[0],
            self.grouping_seconds,
            self.tree_seconds,
            self.worker_count,
        )

    def _resolve_worker_count(self) -> int:
        available = max(1, int(os.cpu_count() or 1))
        limits = [available]
        if self.resource_budget is not None and self.resource_budget.budget.worker_cap is not None:
            limits.append(int(self.resource_budget.budget.worker_cap))
        for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
            raw = os.environ.get(name, "").strip()
            if raw.isdigit() and int(raw) > 0:
                limits.append(int(raw))
        return max(1, min(limits))

    def _query_counts(self, queries: np.ndarray, radius: float) -> np.ndarray:
        counts: list[np.ndarray] = []
        for start in range(0, int(queries.shape[0]), self.chunk_size):
            stop = min(start + self.chunk_size, int(queries.shape[0]))
            batch = queries[start:stop]
            try:
                raw = self._tree.query_ball_point(
                    batch,
                    radius,
                    eps=0.0,
                    return_length=True,
                    workers=self.worker_count,
                )
            except TypeError as exc:
                raise ValueError(
                    "indexed reference-measure occupancy requires cKDTree "
                    "query_ball_point(return_length=True)"
                ) from exc
            values = np.asarray(raw, dtype=np.int64).reshape(-1)
            if values.shape[0] != batch.shape[0]:
                raise ValueError("indexed radius count query returned an invalid batch shape")
            counts.append(values)
        return np.concatenate(counts) if counts else np.empty(0, dtype=np.int64)

    def _canonical_count(self, source_index: int, candidates: np.ndarray, radius: float) -> int:
        query = self.unique_locations[source_index]
        target_values = self.descriptors[np.asarray(candidates, dtype=np.int64)]
        radius_squared = radius * radius
        count = 0
        # Keep ambiguous verification bounded even when an annulus is dense.
        tile = max(1, min(self.chunk_size, 4096))
        for start in range(0, target_values.shape[0], tile):
            target = target_values[start : start + tile]
            with np.errstate(over="ignore", invalid="ignore"):
                difference = target - query
                squared = np.sum(difference * difference, axis=1, dtype=np.float64)
            admitted = squared <= radius_squared
            if np.any(admitted):
                count += int(np.count_nonzero(admitted))
        return count

    def _boundary_workspace(self, candidate_count: int) -> int:
        return int(4096 + 16 * max(0, int(candidate_count)))

    def _boundary_lease(self, source_index: int, radius: float, candidate_count: int):
        requested = self._boundary_workspace(candidate_count)
        if self.resource_budget is None:
            return None
        remaining = self.resource_budget.remaining_managed_budget
        if remaining is None:
            raise ResourceCapacityError(
                "indexed occupancy boundary certification has unknown memory capacity: "
                f"source_index={source_index}, radius={radius}, "
                f"candidate_count={candidate_count}; configure an explicit runtime memory budget",
                operation="indexed occupancy boundary certification",
                requested_bytes=requested,
                available_bytes=None,
            )
        if requested > remaining:
            raise ResourceCapacityError(
                "indexed occupancy boundary certification exceeds runtime memory capacity: "
                f"source_index={source_index}, radius={radius}, "
                f"inner/outer candidate_count={candidate_count}, requested_bytes={requested}, "
                f"available_bytes={remaining}",
                operation="indexed occupancy boundary certification",
                requested_bytes=requested,
                available_bytes=remaining,
            )
        return self.resource_budget.acquire(
            "indexed occupancy boundary certification",
            requested,
        )

    def count(self, radius: float) -> tuple[np.ndarray, dict[str, Any]]:
        """Return exact original-row occupancies and operational diagnostics."""

        tolerance = _validate_positive_float(radius, "reference_measure_tolerance")
        guard = _radius_guard(self.descriptors, tolerance)
        inner_radius = max(0.0, tolerance - guard)
        outer_radius = tolerance + guard
        queries = self.unique_locations
        started = time.perf_counter()
        logger.info(
            "Indexed occupancy radius queries started: source_locations=%d, rows=%d, radius=%g, "
            "batch=%d, workers=%d",
            queries.shape[0],
            self.descriptors.shape[0],
            tolerance,
            self.chunk_size,
            self.worker_count,
        )
        inner = self._query_counts(queries, inner_radius)
        outer = self._query_counts(queries, outer_radius)
        if inner.shape != outer.shape:
            raise ValueError("indexed radius inner/outer query shapes do not align")
        ambiguous = np.flatnonzero(inner != outer)
        self.ambiguous_count = int(ambiguous.size)
        self.outer_candidate_count = int(np.sum(outer, dtype=np.int64))
        location_counts = np.asarray(inner, dtype=np.int64)
        if ambiguous.size:
            logger.info(
                "Indexed occupancy boundary certification started: ambiguous_sources=%d/%d",
                ambiguous.size,
                queries.shape[0],
            )
        for source_index in ambiguous.tolist():
            candidate_count = int(outer[source_index])
            # The candidate list is created only for an ambiguous source.  It
            # is immediately canonical-verified and never retained globally.
            lease = self._boundary_lease(int(source_index), tolerance, candidate_count)
            try:
                try:
                    raw_candidates = self._tree.query_ball_point(
                        queries[source_index],
                        outer_radius,
                        eps=0.0,
                        workers=1,
                    )
                except TypeError as exc:
                    raise ValueError(
                        "indexed reference-measure occupancy cannot certify a boundary "
                        "without exact cKDTree radius queries"
                    ) from exc
                candidates = np.asarray(raw_candidates, dtype=np.int64)
                if candidates.size != candidate_count:
                    raise ValueError(
                        "indexed radius candidate preflight disagreed with retrieval: "
                        f"source_index={source_index}, radius={tolerance}, "
                        f"expected={candidate_count}, actual={candidates.size}"
                    )
                location_counts[source_index] = self._canonical_count(
                    int(source_index), candidates, tolerance
                )
            finally:
                if lease is not None:
                    lease.close()
        if np.any(location_counts < 1):
            bad = int(np.flatnonzero(location_counts < 1)[0])
            raise ValueError(
                "indexed reference-measure occupancy omitted self: "
                f"source_index={bad}, radius={tolerance}, "
                f"inner_count={int(inner[bad])}, outer_count={int(outer[bad])}"
            )
        counts = location_counts[self.row_to_location]
        elapsed = max(0.0, time.perf_counter() - started)
        logger.info(
            "Indexed occupancy source rows processed: %d/%d, elapsed=%.3fs, "
            "ambiguous_boundary_sources=%d",
            self.descriptors.shape[0],
            self.descriptors.shape[0],
            elapsed,
            self.ambiguous_count,
        )
        counts = np.asarray(counts, dtype=np.int64)
        diagnostics = self.diagnostics(counts, elapsed=elapsed, guard=guard)
        return counts, diagnostics

    def diagnostics(self, counts: np.ndarray, *, elapsed: float, guard: float) -> dict[str, Any]:
        values = np.asarray(counts, dtype=np.int64)
        singleton = int(np.count_nonzero(values == 1))
        multi = int(np.count_nonzero(values > 1))
        quantiles = np.quantile(values.astype(np.float64), [0.0, 0.25, 0.5, 0.75, 1.0])
        return {
            "implementation": "indexed-cKDTree-exact-v1",
            "unique_location_count": int(self.unique_locations.shape[0]),
            "duplicate_row_count": int(self.descriptors.shape[0] - self.unique_locations.shape[0]),
            "duplicate_row_fraction": float(
                (self.descriptors.shape[0] - self.unique_locations.shape[0])
                / self.descriptors.shape[0]
            ),
            "singleton_row_count": singleton,
            "singleton_row_fraction": float(singleton / values.size),
            "multi_neighbour_row_count": multi,
            "multi_neighbour_row_fraction": float(multi / values.size),
            "occupancy_quantiles": {
                "minimum": int(quantiles[0]),
                "q25": float(quantiles[1]),
                "median": float(quantiles[2]),
                "q75": float(quantiles[3]),
                "maximum": int(quantiles[4]),
            },
            "indexed_grouping_seconds": float(self.grouping_seconds),
            "indexed_tree_build_seconds": float(self.tree_seconds),
            "indexed_query_seconds": float(elapsed),
            "indexed_boundary_guard": float(guard),
            "indexed_ambiguous_source_count": int(self.ambiguous_count),
            "indexed_outer_candidate_count": int(self.outer_candidate_count),
            "indexed_worker_count": int(self.worker_count),
        }


__all__ = ["IndexedRadiusCountService", "group_descriptor_locations"]
