"""Deterministic, bounded-memory exact neighbours for entropy calibration."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json
from nepflow.resources.budget import (
    ResourceBudgetService,
    ResourceCapacityError,
    build_resource_budget,
)

from ...indexed_radius import group_descriptor_locations
from .checkpointing import (
    CheckpointValidationError,
    IncrementalCheckpointStore,
    VersionedCheckpointStore,
)
from .models import (
    DEFAULT_NEIGHBOUR_BACKEND_ID,
    DEFAULT_RADIUS_QUERY_BYTES,
    GPU_NEIGHBOUR_BACKEND_ALIASES,
    GPU_NEIGHBOUR_BACKEND_ID,
    GPU_NEIGHBOUR_BACKEND_VERSION,
    INDEXED_NEIGHBOUR_BACKEND_ID,
    INDEXED_NEIGHBOUR_BACKEND_VERSION,
    KERNEL_BANDWIDTH_RULE_SYMMETRIC_MAX,
    NEIGHBOUR_BACKEND_ID,
    NEIGHBOUR_BACKEND_VERSION,
    NEIGHBOUR_DISTINCT_POLICY,
    NEIGHBOUR_METRIC,
    TILED_NEIGHBOUR_BACKEND_ID,
    TILED_NEIGHBOUR_BACKEND_VERSION,
    normalize_kernel_bandwidth_rule,
)

logger = logging.getLogger(__name__)


class RadiusQueryCapacityError(MemoryError):
    """Raised before an exact radius query would exceed its workspace budget."""


CalibrationCapacityError = RadiusQueryCapacityError


# ``None`` means that the cache is planned from the shared runtime budget.
# There is deliberately no machine-independent byte default: a cache is
# optional, while neighbour workspaces remain required and fail fast.
DEFAULT_NEIGHBOUR_CACHE_BYTES: int | None = None


def _validate_memory_limit(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _resolve_neighbour_cache_budget(
    *,
    remaining_bytes: int,
    index_bytes: int,
    radius_query_bytes: int,
    user_cap: int | None,
) -> int:
    """Plan optional cache headroom without changing required-work limits."""

    available = max(0, int(remaining_bytes) - int(index_bytes) - int(radius_query_bytes))
    if user_cap is None:
        return available
    return min(_validate_memory_limit(user_cap, "max_neighbour_cache_bytes"), available)


def _estimate_radius_workspace_bytes(
    candidate_count: int,
    support_count: int,
    dimension: int,
    row_count: int = 0,
) -> int:
    """Conservatively estimate one exact support query's live working set."""

    candidate_count = max(0, int(candidate_count))
    support_count = max(0, int(support_count))
    dimension = max(1, int(dimension))
    row_count = max(0, int(row_count))
    candidate_list = 1024 + 80 * candidate_count
    candidate_distance = 24 * dimension * candidate_count + 24 * candidate_count
    support_expansion = 2048 + 16 * dimension * support_count + 64 * support_count
    persistent_reduction = 8 * row_count
    return int(candidate_list + candidate_distance + support_expansion + persistent_reduction)


def _process_memory_metrics() -> dict[str, int | None]:
    """Read cheap process/cgroup metrics without adding a runtime dependency."""

    peak_rss: int | None = None
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    peak_rss = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError):
        pass
    cgroup_current: int | None = None
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(path, encoding="utf-8") as handle:
                cgroup_current = int(handle.read().strip())
            break
        except (OSError, ValueError):
            continue
    return {"process_peak_rss_bytes": peak_rss, "cgroup_memory_current_bytes": cgroup_current}


@dataclass(frozen=True, slots=True)
class NeighbourBackendCapabilities:
    """Scientific and operational capabilities exposed by a neighbour backend."""

    backend: str
    version: str
    metric: str
    exact: bool
    supports_radius_support: bool
    supports_reusable_index: bool
    max_dimension: int | None = None


_BACKEND_CAPABILITIES = {
    NEIGHBOUR_BACKEND_ID: NeighbourBackendCapabilities(
        backend=NEIGHBOUR_BACKEND_ID,
        version=NEIGHBOUR_BACKEND_VERSION,
        metric=NEIGHBOUR_METRIC,
        exact=True,
        supports_radius_support=False,
        supports_reusable_index=False,
    ),
    INDEXED_NEIGHBOUR_BACKEND_ID: NeighbourBackendCapabilities(
        backend=INDEXED_NEIGHBOUR_BACKEND_ID,
        version=INDEXED_NEIGHBOUR_BACKEND_VERSION,
        metric=NEIGHBOUR_METRIC,
        exact=True,
        supports_radius_support=True,
        supports_reusable_index=True,
    ),
    TILED_NEIGHBOUR_BACKEND_ID: NeighbourBackendCapabilities(
        backend=TILED_NEIGHBOUR_BACKEND_ID,
        version=TILED_NEIGHBOUR_BACKEND_VERSION,
        metric=NEIGHBOUR_METRIC,
        exact=True,
        supports_radius_support=True,
        supports_reusable_index=True,
    ),
    GPU_NEIGHBOUR_BACKEND_ID: NeighbourBackendCapabilities(
        backend=GPU_NEIGHBOUR_BACKEND_ID,
        version=GPU_NEIGHBOUR_BACKEND_VERSION,
        metric=NEIGHBOUR_METRIC,
        exact=True,
        supports_radius_support=True,
        supports_reusable_index=True,
    ),
}


def neighbour_backend_capabilities(backend: str) -> NeighbourBackendCapabilities:
    """Return capabilities for a configured backend without fallback."""

    if backend in GPU_NEIGHBOUR_BACKEND_ALIASES:
        backend = GPU_NEIGHBOUR_BACKEND_ID
    try:
        return _BACKEND_CAPABILITIES[backend]
    except KeyError as exc:
        raise ValueError(f"configured neighbour backend is unavailable: {backend!r}") from exc


def _validate_descriptors(descriptors: Any) -> np.ndarray:
    values = np.asarray(descriptors)
    if values.dtype != np.dtype(np.float64):
        raise ValueError("transformed descriptors must have dtype float64")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("transformed descriptors must be a non-empty 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("transformed descriptors must contain only finite values")
    return np.ascontiguousarray(values, dtype=np.float64)


def _validate_k(k: Any) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be a positive integer")
    value = int(k)
    if value < 1:
        raise ValueError("k must be a positive integer")
    return value


def _validate_chunk_size(chunk_size: Any) -> int:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise ValueError("chunk_size must be a positive integer")
    value = int(chunk_size)
    if value < 1:
        raise ValueError("chunk_size must be a positive integer")
    return value


def _array_fingerprint(values: np.ndarray) -> str:
    payload = (
        str(values.dtype),
        tuple(int(item) for item in values.shape),
        hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
    )
    return sha256_canonical_json(payload)


def _readonly_array(values: Any, *, dtype: np.dtype[Any] | type, ndim: int) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError("neighbour result array has an unexpected dimensionality")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ExactNeighbourResult:
    """Exact kth-distinct-location neighbours and stable identity metadata."""

    radii: np.ndarray
    neighbour_indices: np.ndarray
    neighbour_distances: np.ndarray
    row_to_location: np.ndarray
    unique_locations: np.ndarray
    location_representatives: np.ndarray
    k: int
    representation_fingerprint: str
    row_ids: tuple[str, ...]
    fingerprint: str
    backend: str = NEIGHBOUR_BACKEND_ID
    backend_version: str = NEIGHBOUR_BACKEND_VERSION
    metric: str = NEIGHBOUR_METRIC
    distinct_location_policy: str = NEIGHBOUR_DISTINCT_POLICY
    backend_fingerprint: str = ""

    def __post_init__(self) -> None:
        radii = _readonly_array(self.radii, dtype=np.float64, ndim=1)
        neighbour_indices = _readonly_array(self.neighbour_indices, dtype=np.int64, ndim=2)
        neighbour_distances = _readonly_array(
            self.neighbour_distances,
            dtype=np.float64,
            ndim=2,
        )
        row_to_location = _readonly_array(self.row_to_location, dtype=np.int64, ndim=1)
        unique_locations = _readonly_array(self.unique_locations, dtype=np.float64, ndim=2)
        representatives = _readonly_array(
            self.location_representatives,
            dtype=np.int64,
            ndim=1,
        )
        if neighbour_indices.shape != neighbour_distances.shape:
            raise ValueError("neighbour indices and distances must have equal shape")
        if (
            radii.shape[0] != neighbour_indices.shape[0]
            or radii.shape[0] != row_to_location.shape[0]
        ):
            raise ValueError("neighbour rows must align with radii and row ownership")
        if neighbour_indices.shape[1] != int(self.k):
            raise ValueError("neighbour result width must equal k")
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("neighbour radii must be finite and positive")
        if not np.all(np.isfinite(neighbour_distances)) or np.any(neighbour_distances <= 0.0):
            raise ValueError("neighbour distances must be finite and positive")
        if unique_locations.shape[0] != representatives.shape[0]:
            raise ValueError("unique locations and representatives must align")
        if row_to_location.size and np.max(row_to_location) >= unique_locations.shape[0]:
            raise ValueError("row-to-location mapping is outside the unique-location table")
        if not self.representation_fingerprint or not self.fingerprint:
            raise ValueError("neighbour fingerprints must not be blank")
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "neighbour_indices", neighbour_indices)
        object.__setattr__(self, "neighbour_distances", neighbour_distances)
        object.__setattr__(self, "row_to_location", row_to_location)
        object.__setattr__(self, "unique_locations", unique_locations)
        object.__setattr__(self, "location_representatives", representatives)
        object.__setattr__(self, "k", int(self.k))

    @property
    def distances(self) -> np.ndarray:
        """Compatibility alias for the selected distinct-location distances."""

        return self.neighbour_distances

    @property
    def indices(self) -> np.ndarray:
        """Compatibility alias for representative neighbour row indices."""

        return self.neighbour_indices


class ExactKNNPrefixSupport:
    """Derive strict ``c=1`` supports from an exact canonical kNN result.

    The result stores one representative row per distinct neighbour location.
    This provider expands only the requested source batch back to original
    rows, preserving the canonical neighbour distance for every duplicate row
    and the existing source-major CSR contract.
    """

    def __init__(
        self,
        descriptors: np.ndarray,
        neighbours: ExactNeighbourResult,
        *,
        grouping_source: Any | None = None,
        expected_row_ids: Sequence[str] | None = None,
        max_radius_query_bytes: int | None = None,
        max_calibration_work_bytes: int | None = None,
        resource_budget: ResourceBudgetService | None = None,
        capacity_context: str | None = None,
    ) -> None:
        self.descriptors = _validate_descriptors(descriptors)
        if not isinstance(neighbours, ExactNeighbourResult):
            raise TypeError("knn support requires an ExactNeighbourResult")
        self.neighbours = neighbours
        self.resource_budget = resource_budget or build_resource_budget()
        self.capacity_context = capacity_context
        remaining = self.resource_budget.remaining_managed_budget
        if remaining is None:
            raise ResourceCapacityError(
                self.resource_budget.unknown_memory_message("exact kNN prefix support"),
                operation="exact kNN prefix support",
            )
        if remaining < 1:
            raise ResourceCapacityError(
                "exact kNN prefix support has no allocatable runtime memory headroom",
                operation="exact kNN prefix support",
                requested_bytes=1,
                available_bytes=remaining,
                reserved_headroom_bytes=self.resource_budget.budget.reserved_headroom_bytes,
            )
        radius_limit = (
            max(1, remaining // 2)
            if max_radius_query_bytes is None
            else min(
                _validate_memory_limit(max_radius_query_bytes, "max_radius_query_bytes"),
                remaining,
            )
        )
        work_limit = (
            remaining
            if max_calibration_work_bytes is None
            else min(
                _validate_memory_limit(
                    max_calibration_work_bytes,
                    "max_calibration_work_bytes",
                ),
                remaining,
            )
        )
        self.max_workspace_bytes = min(radius_limit, work_limit)
        self._validate_neighbour_identity(expected_row_ids)

        self._grouping_build_seconds = 0.0
        self._owns_grouping = False
        if grouping_source is not None:
            source_row_to_location = getattr(grouping_source, "row_to_location", None)
            source_rows_by_location = getattr(grouping_source, "rows_by_location", None)
            source_location_indptr = getattr(grouping_source, "location_indptr", None)
            if (
                source_row_to_location is None
                or source_rows_by_location is None
                or source_location_indptr is None
            ):
                raise ValueError("kNN grouping source does not expose the exact row grouping")
            if not np.array_equal(source_row_to_location, neighbours.row_to_location):
                raise ValueError("kNN grouping source row identity does not match neighbours")
            self.rows_by_location = np.asarray(source_rows_by_location, dtype=np.int64)
            self.location_indptr = np.asarray(source_location_indptr, dtype=np.int64)
            self._validate_grouping()
        else:
            self._build_grouping()
        self.grouping_bytes = int(self.rows_by_location.nbytes + self.location_indptr.nbytes)
        self.derived_sources = 0
        self.derived_entries = 0
        self.elapsed_seconds = 0.0
        self.query_workspace_peak_bytes = 0

    def _validate_neighbour_identity(self, expected_row_ids: Sequence[str] | None) -> None:
        neighbours = self.neighbours
        row_count = int(self.descriptors.shape[0])
        location_count = int(neighbours.unique_locations.shape[0])
        if neighbours.radii.shape != (row_count,):
            raise ValueError("kNN prefix radii do not align with the descriptor rows")
        if neighbours.row_to_location.shape != (row_count,):
            raise ValueError("kNN prefix row ownership does not align with descriptors")
        if neighbours.unique_locations.shape[1] != self.descriptors.shape[1]:
            raise ValueError("kNN prefix descriptor dimensionality does not match")
        if neighbours.location_representatives.shape != (location_count,):
            raise ValueError("kNN prefix representatives do not align with locations")
        if neighbours.row_to_location.size and (
            np.min(neighbours.row_to_location) < 0
            or np.max(neighbours.row_to_location) >= location_count
        ):
            raise ValueError("kNN prefix row ownership is outside the location table")
        if len(neighbours.row_ids) != row_count or any(
            not isinstance(value, str) or not value.strip() for value in neighbours.row_ids
        ):
            raise ValueError("kNN prefix row identity is missing or misaligned")
        if expected_row_ids is not None:
            expected_ids = tuple(str(value) for value in expected_row_ids)
            if expected_ids != tuple(neighbours.row_ids):
                raise ValueError("kNN prefix row identity does not match the evaluation pool")
        if neighbours.k < 1 or neighbours.neighbour_indices.shape != (
            row_count,
            neighbours.k,
        ):
            raise ValueError("kNN prefix neighbour dimensions are incompatible")
        if neighbours.k >= location_count:
            raise ValueError("kNN prefix k must leave at least one non-neighbour location")
        if not np.array_equal(neighbours.radii, neighbours.neighbour_distances[:, -1]):
            raise ValueError("kNN prefix radii are not the canonical kth distances")
        representatives = neighbours.location_representatives
        if np.any(representatives < 0) or np.any(representatives >= row_count):
            raise ValueError("kNN prefix representative rows are outside the descriptor pool")

        # Validate the scientific identity in bounded chunks.  Avoid creating
        # a global N x d reconstruction of the descriptor matrix.
        for start in range(0, row_count, 4096):
            stop = min(start + 4096, row_count)
            locations = neighbours.row_to_location[start:stop]
            if not np.array_equal(
                self.descriptors[start:stop], neighbours.unique_locations[locations]
            ):
                raise ValueError("kNN prefix locations do not match descriptors")
            neighbour_rows = neighbours.neighbour_indices[start:stop]
            if np.any(neighbour_rows < 0) or np.any(neighbour_rows >= row_count):
                raise ValueError("kNN prefix neighbour rows are outside the descriptor pool")
            neighbour_locations = neighbours.row_to_location[neighbour_rows]
            own_locations = locations[:, None]
            if np.any(neighbour_locations == own_locations):
                raise ValueError("kNN prefix contains the source location as a neighbour")
            if neighbours.k > 1:
                sorted_locations = np.sort(neighbour_locations, axis=1)
                if np.any(np.diff(sorted_locations, axis=1) == 0):
                    raise ValueError("kNN prefix contains duplicate neighbour locations")
            if np.any(
                representatives[neighbour_locations] != neighbour_rows
            ):
                raise ValueError("kNN prefix neighbour rows are not location representatives")
            distances = neighbours.neighbour_distances[start:stop]
            if np.any(np.diff(distances, axis=1) < 0.0):
                raise ValueError("kNN prefix distances are not in canonical order")

    def _build_grouping(self) -> None:
        row_to_location = self.neighbours.row_to_location
        location_count = int(self.neighbours.unique_locations.shape[0])
        grouping_estimate = int(16 * self.descriptors.shape[0] + 16 * location_count + 8192)
        if grouping_estimate > self.max_workspace_bytes:
            raise RadiusQueryCapacityError(
                "exact kNN prefix grouping exceeds its bounded support workspace: "
                f"requested_bytes={grouping_estimate}, "
                f"max_radius_query_bytes={self.max_workspace_bytes}"
            )
        started = time.perf_counter()
        with self.resource_budget.acquire(
            self.capacity_context or "exact kNN prefix grouping",
            grouping_estimate,
        ):
            self.rows_by_location = np.asarray(
                np.argsort(row_to_location, kind="stable"),
                dtype=np.int64,
            )
            counts = np.bincount(row_to_location, minlength=location_count).astype(
                np.int64,
                copy=False,
            )
            self.location_indptr = np.empty(location_count + 1, dtype=np.int64)
            self.location_indptr[0] = 0
            np.cumsum(counts, dtype=np.int64, out=self.location_indptr[1:])
        self.rows_by_location.setflags(write=False)
        self.location_indptr.setflags(write=False)
        self._owns_grouping = True
        self._grouping_build_seconds = max(0.0, time.perf_counter() - started)
        self._validate_grouping()

    def _validate_grouping(self) -> None:
        row_count = int(self.descriptors.shape[0])
        location_count = int(self.neighbours.unique_locations.shape[0])
        if self.rows_by_location.shape != (row_count,):
            raise ValueError("kNN prefix rows-by-location grouping has the wrong shape")
        if self.location_indptr.shape != (location_count + 1,):
            raise ValueError("kNN prefix location grouping has the wrong shape")
        if (
            self.location_indptr[0] != 0
            or self.location_indptr[-1] != row_count
            or np.any(np.diff(self.location_indptr) < 1)
            or np.any(self.rows_by_location < 0)
            or np.any(self.rows_by_location >= row_count)
        ):
            raise ValueError("kNN prefix location grouping is invalid")
        if not np.array_equal(np.sort(self.rows_by_location), np.arange(row_count)):
            raise ValueError("kNN prefix grouping does not cover every source row once")
        for location in range(location_count):
            start = int(self.location_indptr[location])
            stop = int(self.location_indptr[location + 1])
            rows = self.rows_by_location[start:stop]
            if not np.all(self.neighbours.row_to_location[rows] == location):
                raise ValueError("kNN prefix grouping assigns a row to the wrong location")
            if int(self.neighbours.location_representatives[location]) != int(rows[0]):
                raise ValueError("kNN prefix representative is not the earliest source row")

    def eligible(
        self,
        source_indices: np.ndarray,
        radii: np.ndarray,
    ) -> bool:
        expected = self.neighbours.radii[source_indices]
        if radii.ndim == 1:
            return np.array_equal(radii, expected)
        return np.array_equal(radii, expected[None, :])

    def support_many(
        self,
        source_indices: Sequence[int] | np.ndarray,
        radii: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sources = np.asarray(source_indices)
        if sources.ndim != 1 or sources.size == 0:
            raise ValueError("source_indices must be a non-empty one-dimensional sequence")
        if sources.dtype.kind not in "iu" or sources.dtype.kind == "b":
            raise ValueError("source_indices must contain integer row indices")
        source_rows = np.asarray(sources, dtype=np.int64)
        if np.any(source_rows < 0) or np.any(source_rows >= self.descriptors.shape[0]):
            raise IndexError("source_indices contain a row outside the descriptor rows")
        requested = np.asarray(radii, dtype=np.float64)
        if requested.ndim not in (1, 2) or requested.shape[-1] != source_rows.size:
            raise ValueError(
                "radii must have shape (source_count,) or (candidate_count, source_count)"
            )
        if requested.ndim == 2 and requested.shape[0] == 0:
            raise ValueError("radii must contain at least one candidate")
        if not np.all(np.isfinite(requested)) or np.any(requested <= 0.0):
            raise ValueError("radii must be finite and strictly positive")
        if not self.eligible(source_rows, requested):
            raise ValueError("exact kNN prefix support requires radii equal to canonical kNN radii")
        source_radii = (
            requested if requested.ndim == 1 else requested[0]
        )
        neighbour_locations = self.neighbours.row_to_location[
            self.neighbours.neighbour_indices[source_rows]
        ]
        neighbour_distances = self.neighbours.neighbour_distances[source_rows]
        counts = np.empty(source_rows.size, dtype=np.int64)
        for local_index, source in enumerate(source_rows):
            own_location = int(self.neighbours.row_to_location[source])
            eligible_locations = neighbour_locations[local_index][
                neighbour_distances[local_index] < source_radii[local_index]
            ]
            locations = np.concatenate(
                (np.asarray([own_location], dtype=np.int64), eligible_locations)
            )
            counts[local_index] = int(
                np.sum(
                    self.location_indptr[locations + 1] - self.location_indptr[locations],
                    dtype=np.int64,
                )
            )
        total_support = int(np.sum(counts, dtype=np.int64))
        # Admit the returned CSR arrays plus the per-source concatenation and
        # stable-sort temporaries that coexist with them during expansion.
        workspace_bytes = int(
            (source_rows.size + 1) * np.dtype(np.int64).itemsize
            + total_support * 40
            + 4096
        )
        if workspace_bytes > self.max_workspace_bytes:
            details = (
                f"requested_bytes={workspace_bytes}, max_support_workspace_bytes="
                f"{self.max_workspace_bytes}, source_count={source_rows.size}, "
                f"support_entries={total_support}"
            )
            if self.capacity_context:
                details += f", {self.capacity_context}"
            raise RadiusQueryCapacityError(
                "exact kNN prefix support exceeds memory capacity: " + details
            )

        started = time.perf_counter()
        with self.resource_budget.acquire(
            self.capacity_context or "exact kNN prefix support materialisation",
            workspace_bytes,
        ):
            offsets = np.zeros(source_rows.size + 1, dtype=np.int64)
            np.cumsum(counts, dtype=np.int64, out=offsets[1:])
            targets = np.empty(total_support, dtype=np.int64)
            distances = np.empty(total_support, dtype=np.float64)
            cursor = 0
            for local_index, source in enumerate(source_rows):
                own_location = int(self.neighbours.row_to_location[source])
                eligible = neighbour_distances[local_index] < source_radii[local_index]
                locations = np.concatenate(
                    (
                        np.asarray([own_location], dtype=np.int64),
                        neighbour_locations[local_index, eligible],
                    )
                )
                location_distances = np.concatenate(
                    (
                        np.asarray([0.0], dtype=np.float64),
                        neighbour_distances[local_index, eligible],
                    )
                )
                row_parts = [
                    self.rows_by_location[
                        self.location_indptr[int(location)] : self.location_indptr[
                            int(location) + 1
                        ]
                    ]
                    for location in locations
                ]
                rows = np.concatenate(row_parts).astype(np.int64, copy=False)
                row_distances = np.repeat(
                    location_distances,
                    [int(part.size) for part in row_parts],
                )
                order = np.argsort(rows, kind="stable")
                stop = cursor + int(rows.size)
                targets[cursor:stop] = rows[order]
                distances[cursor:stop] = row_distances[order]
                cursor = stop
        if cursor != total_support or np.any(np.diff(offsets) < 1):
            raise ValueError("exact kNN prefix support materialisation is incomplete")
        offsets.setflags(write=False)
        targets.setflags(write=False)
        distances.setflags(write=False)
        self.derived_sources += int(source_rows.size)
        self.derived_entries += total_support
        self.elapsed_seconds += max(0.0, time.perf_counter() - started)
        self.query_workspace_peak_bytes = max(self.query_workspace_peak_bytes, workspace_bytes)
        return offsets, targets, distances

    def operational_metrics(self) -> dict[str, int | float]:
        return {
            "knn_support_derived_sources": int(self.derived_sources),
            "knn_support_derived_entries": int(self.derived_entries),
            "knn_support_grouping_bytes": int(self.grouping_bytes),
            "knn_support_grouping_build_seconds": float(self._grouping_build_seconds),
            "knn_support_elapsed_seconds": float(self.elapsed_seconds),
            "knn_support_query_workspace_peak_bytes": int(self.query_workspace_peak_bytes),
            "knn_support_gpu_fp64_evaluations": 0,
            "knn_support_gpu_tile_scans": 0,
        }


class ExactSymmetricKNNUnionSupport(ExactKNNPrefixSupport):
    """Exact symmetric-max support derived from one canonical kNN prefix.

    For the c=1 bandwidth vector, every directed location edge is admitted
    only when its canonical kNN distance is strictly below the source kth
    radius.  The symmetric support is the union of those directed edges and
    their reverse adjacency.  The reverse index is keyed by unique location,
    while materialisation expands back to every original row through the
    grouping inherited from :class:`ExactKNNPrefixSupport`.
    """

    backend = "symmetric_knn_union"
    backend_version = "symmetric-knn-union-v1"

    def __init__(
        self,
        descriptors: np.ndarray,
        neighbours: ExactNeighbourResult,
        *,
        effective_radii: Sequence[float] | np.ndarray | None = None,
        declared_floor: float | None = None,
        floor: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Construct the exact union and, when required, its floor augmentation."""

        if floor is not None:
            if declared_floor is not None and float(floor) != float(declared_floor):
                raise ValueError("floor and declared_floor must agree")
            declared_floor = float(floor)
        super().__init__(descriptors, neighbours, **kwargs)
        neighbours = self.neighbours
        if not neighbours.backend_fingerprint:
            raise ValueError("symmetric kNN union requires a persisted kNN backend fingerprint")

        raw_radii = np.asarray(neighbours.radii, dtype=np.float64)
        if effective_radii is None:
            effective = raw_radii.copy()
            if declared_floor is not None:
                raise ValueError(
                    "declared_floor requires the complete effective_radii vector"
                )
            declared_floor_value: float | None = None
        else:
            effective = np.asarray(effective_radii)
            if effective.dtype != np.dtype(np.float64) or effective.shape != raw_radii.shape:
                raise ValueError("effective_radii must be a full-pool float64 vector")
            if not np.all(np.isfinite(effective)) or np.any(effective <= 0.0):
                raise ValueError("effective_radii must be finite and strictly positive")
            if declared_floor is None:
                raise ValueError("effective_radii requires a declared floor")
            declared_floor_value = float(declared_floor)
            if not np.isfinite(declared_floor_value) or declared_floor_value <= 0.0:
                raise ValueError("declared_floor must be finite and strictly positive")
            expected_effective = np.maximum(raw_radii, declared_floor_value)
            if not np.array_equal(effective, expected_effective):
                raise ValueError(
                    "effective_radii is not the canonical max(raw_radii, declared_floor) vector"
                )
            effective = np.array(effective, dtype=np.float64, copy=True)
        effective.setflags(write=False)
        self.effective_radii = effective
        self.declared_floor = declared_floor_value
        self.subfloor_row_count = int(
            np.count_nonzero(raw_radii < declared_floor_value)
            if declared_floor_value is not None
            else 0
        )
        self.subfloor_location_count = 0
        self.floor_augmented_support_pairs = 0
        self.floor_distance_evaluations = 0
        self.floor_index: IndexedCPUNeighbourIndex | None = None
        self.floor_extra_indptr = np.zeros(
            neighbours.unique_locations.shape[0] + 1, dtype=np.int64
        )
        self.floor_extra_locations = np.empty(0, dtype=np.int64)
        self.floor_extra_distances = np.empty(0, dtype=np.float64)
        location_count = int(neighbours.unique_locations.shape[0])
        representative_rows = np.asarray(neighbours.location_representatives, dtype=np.int64)
        representative_locations = neighbours.row_to_location[representative_rows]
        if not np.array_equal(
            representative_locations, np.arange(location_count, dtype=np.int64)
        ):
            raise ValueError("kNN prefix representatives are not in canonical location order")

        # Duplicate atomic rows must share the exact same canonical prefix and
        # kth radius.  This is an identity check, not an approximation.
        for location in range(location_count):
            start = int(self.location_indptr[location])
            stop = int(self.location_indptr[location + 1])
            rows = self.rows_by_location[start:stop]
            if not np.all(neighbours.radii[rows] == neighbours.radii[rows[0]]):
                raise ValueError("duplicate rows at one location have different kNN radii")
            if not np.all(
                neighbours.neighbour_indices[rows] == neighbours.neighbour_indices[rows[0]]
            ) or not np.all(
                neighbours.neighbour_distances[rows]
                == neighbours.neighbour_distances[rows[0]]
            ):
                raise ValueError("duplicate rows at one location have different kNN prefixes")

        neighbour_locations = np.asarray(
            neighbours.row_to_location[
                neighbours.neighbour_indices[representative_rows]
            ],
            dtype=np.int64,
        )
        neighbour_distances = np.asarray(
            neighbours.neighbour_distances[representative_rows],
            dtype=np.float64,
        )
        if np.any(
            neighbour_locations == np.arange(location_count, dtype=np.int64)[:, None]
        ):
            raise ValueError("kNN prefix contains a source-location self edge")
        if np.any(np.diff(neighbour_distances, axis=1) < 0.0):
            raise ValueError("kNN prefix distances are not canonically ordered")

        edge_count = int(location_count * neighbours.k)
        reverse_bytes = int(
            edge_count
            * (
                2 * np.dtype(np.int64).itemsize
                + 2 * np.dtype(np.float64).itemsize
            )
            + (location_count + 1) * np.dtype(np.int64).itemsize
            + 4096
        )
        reverse_and_grouping_bytes = reverse_bytes + int(self.grouping_bytes)
        if reverse_and_grouping_bytes > self.max_workspace_bytes:
            raise RadiusQueryCapacityError(
                "symmetric kNN reverse adjacency exceeds its bounded support workspace: "
                f"requested_bytes={reverse_and_grouping_bytes}, max_support_workspace_bytes="
                f"{self.max_workspace_bytes}, locations={location_count}, k={neighbours.k}"
            )
        source_locations = np.repeat(
            np.arange(location_count, dtype=np.int64), neighbours.k
        )
        target_locations = neighbour_locations.reshape(-1)
        distances = neighbour_distances.reshape(-1)
        strict = distances < neighbours.radii[representative_rows][source_locations]
        source_locations = source_locations[strict]
        target_locations = target_locations[strict]
        distances = distances[strict]
        order = np.lexsort((source_locations, target_locations))
        source_locations = source_locations[order]
        target_locations = target_locations[order]
        distances = distances[order]
        counts = np.bincount(target_locations, minlength=location_count).astype(
            np.int64, copy=False
        )
        reverse_indptr = np.empty(location_count + 1, dtype=np.int64)
        reverse_indptr[0] = 0
        np.cumsum(counts, dtype=np.int64, out=reverse_indptr[1:])
        reverse_indptr.setflags(write=False)
        source_locations.setflags(write=False)
        distances.setflags(write=False)
        self.reverse_indptr = reverse_indptr
        self.reverse_source_locations = source_locations
        self.reverse_distances = distances
        self.reverse_index_bytes = reverse_bytes
        self.reused_knn_locations = location_count
        self.new_distance_evaluations = 0
        if declared_floor_value is not None:
            self._build_floor_augmentation(
                raw_radii,
                declared_floor_value,
            )
            if self.subfloor_location_count:
                self.backend = "symmetric_knn_floor_augmented"
                self.backend_version = "symmetric-knn-floor-augmented-v1"
            if self.subfloor_location_count == 0:
                logger.info(
                    "Exact symmetric kNN floor augmentation is a no-op: zero raw "
                    "sub-floor radii; raw kNN support reuse is exact"
                )

    def _build_floor_augmentation(
        self,
        raw_radii: np.ndarray,
        declared_floor: float,
    ) -> None:
        """Build only the strict floor edges absent from the original union."""

        neighbours = self.neighbours
        location_count = int(neighbours.unique_locations.shape[0])
        representative_rows = np.asarray(neighbours.location_representatives, dtype=np.int64)
        subfloor_locations = np.flatnonzero(
            raw_radii[representative_rows] < declared_floor
        ).astype(np.int64, copy=False)
        self.subfloor_location_count = int(subfloor_locations.size)
        if subfloor_locations.size < 2:
            # A one-location subset has no cross-location pair to add.
            return

        subset_descriptor_bytes = int(
            subfloor_locations.size * neighbours.unique_locations.shape[1] * np.dtype(np.float64).itemsize
        )
        subset_workspace = int(
            subset_descriptor_bytes + 32 * subfloor_locations.size + 8192
        )
        if subset_workspace > self.max_workspace_bytes:
            raise RadiusQueryCapacityError(
                "floor-augmentation subset index exceeds its bounded support workspace: "
                f"requested_bytes={subset_workspace}, "
                f"max_support_workspace_bytes={self.max_workspace_bytes}, "
                f"subfloor_locations={subfloor_locations.size}"
            )
        subset_descriptors = np.ascontiguousarray(
            neighbours.unique_locations[subfloor_locations], dtype=np.float64
        )
        subset_index = IndexedCPUNeighbourIndex(
            subset_descriptors,
            max_index_bytes=self.max_workspace_bytes,
            max_radius_query_bytes=self.max_workspace_bytes,
            resource_budget=self.resource_budget,
        )
        self.floor_index = subset_index
        subset_sources = np.arange(subfloor_locations.size, dtype=np.int64)
        floor_radii = np.full(subfloor_locations.size, declared_floor, dtype=np.float64)
        offsets, targets, distances = subset_index.radius_support_many(
            subset_sources,
            floor_radii,
            chunk_size=min(1024, max(1, subfloor_locations.size)),
            context=self.capacity_context or "exact symmetric kNN floor augmentation",
        )
        self.floor_distance_evaluations = int(subset_index.unique_distance_evaluations)
        self.new_distance_evaluations = self.floor_distance_evaluations

        # Build a compact location-level view of the original union.  This is
        # used only once to remove overlaps; duplicate atomic rows are expanded
        # from the inherited grouping during each support query.
        # Only below-floor source locations need an overlap lookup.  Build it
        # from the existing outgoing and reverse CSR arrays instead of growing
        # a Q-sized Python/list-of-arrays structure with repeated appends.
        original_by_source: dict[int, np.ndarray] = {}
        for global_source in subfloor_locations:
            representative = int(neighbours.location_representatives[int(global_source)])
            outgoing_distances = neighbours.neighbour_distances[representative]
            outgoing = neighbours.row_to_location[
                neighbours.neighbour_indices[representative]
            ][outgoing_distances < raw_radii[representative]]
            incoming_start = int(self.reverse_indptr[int(global_source)])
            incoming_stop = int(self.reverse_indptr[int(global_source) + 1])
            incoming = self.reverse_source_locations[incoming_start:incoming_stop]
            original_by_source[int(global_source)] = np.unique(
                np.concatenate((outgoing, incoming))
            )

        extra_by_source: dict[int, np.ndarray] = {}
        distance_by_source: dict[int, np.ndarray] = {}
        empty_locations = np.empty(0, dtype=np.int64)
        empty_distances = np.empty(0, dtype=np.float64)
        subset_to_location = subfloor_locations
        for subset_source, global_source in enumerate(subfloor_locations):
            begin = int(offsets[subset_source])
            end = int(offsets[subset_source + 1])
            global_targets = subset_to_location[targets[begin:end]]
            candidate_distances = distances[begin:end]
            nonself = global_targets != int(global_source)
            global_targets = global_targets[nonself]
            candidate_distances = candidate_distances[nonself]
            existing = original_by_source[int(global_source)]
            is_new = (
                ~np.isin(global_targets, existing, assume_unique=True)
                if existing.size
                else np.ones(global_targets.shape, dtype=bool)
            )
            new_targets = np.asarray(global_targets[is_new], dtype=np.int64)
            new_distances = np.asarray(candidate_distances[is_new], dtype=np.float64)
            order = np.argsort(new_targets, kind="stable")
            extra_by_source[int(global_source)] = new_targets[order]
            distance_by_source[int(global_source)] = new_distances[order]

        extra_total = sum(int(values.size) for values in extra_by_source.values())
        extra_workspace = int(
            8 * (location_count + 1)
            + 16 * extra_total
            + 16 * subfloor_locations.size
            + 4096
        )
        if extra_workspace > self.max_workspace_bytes:
            raise RadiusQueryCapacityError(
                "floor-augmentation support materialisation exceeds its bounded workspace: "
                f"requested_bytes={extra_workspace}, "
                f"max_support_workspace_bytes={self.max_workspace_bytes}, "
                f"additional_pairs={extra_total}"
            )
        extra_indptr = np.zeros(location_count + 1, dtype=np.int64)
        for source in range(location_count):
            extra_indptr[source + 1] = extra_indptr[source] + extra_by_source.get(
                source, empty_locations
            ).size
        extra_count = int(extra_indptr[-1])
        extra_locations = np.empty(extra_count, dtype=np.int64)
        extra_distances = np.empty(extra_count, dtype=np.float64)
        cursor = 0
        for source in range(location_count):
            source_locations = extra_by_source.get(source, empty_locations)
            source_distances = distance_by_source.get(source, empty_distances)
            stop = cursor + int(source_locations.size)
            extra_locations[cursor:stop] = source_locations
            extra_distances[cursor:stop] = source_distances
            cursor = stop
        extra_indptr.setflags(write=False)
        extra_locations.setflags(write=False)
        extra_distances.setflags(write=False)
        self.floor_extra_indptr = extra_indptr
        self.floor_extra_locations = extra_locations
        self.floor_extra_distances = extra_distances
        self.floor_augmented_support_pairs = extra_count
        logger.info(
            "Exact symmetric kNN floor augmentation completed: subfloor_rows=%d, "
            "subfloor_locations=%d, additional_pairs=%d, distance_evaluations=%d",
            self.subfloor_row_count,
            self.subfloor_location_count,
            self.floor_augmented_support_pairs,
            self.floor_distance_evaluations,
        )

    def eligible(
        self,
        source_indices: np.ndarray,
        radii: np.ndarray,
        target_bandwidths: np.ndarray | None = None,
    ) -> bool:
        expected = self.effective_radii[source_indices]
        if radii.ndim == 1:
            source_matches = np.array_equal(radii, expected)
        else:
            source_matches = bool(
                np.all(np.asarray(radii, dtype=np.float64) == expected[None, :])
            )
        if target_bandwidths is None:
            return source_matches
        targets = np.asarray(target_bandwidths, dtype=np.float64)
        if targets.ndim == 1:
            return source_matches and np.array_equal(targets, self.effective_radii)
        return source_matches and bool(np.all(targets == self.effective_radii[None, :]))

    def support_many(
        self,
        source_indices: Sequence[int] | np.ndarray,
        radii: Sequence[float] | np.ndarray,
        *,
        target_bandwidths: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if target_bandwidths is not None:
            targets = np.asarray(target_bandwidths)
            if targets.dtype != np.dtype(np.float64) or targets.ndim not in (1, 2):
                raise ValueError("symmetric kNN target bandwidths must be float64 vectors")
            if targets.shape[-1] != self.descriptors.shape[0]:
                raise ValueError("symmetric kNN target bandwidths must cover the full row pool")
            if not np.all(np.isfinite(targets)) or np.any(targets <= 0.0):
                raise ValueError("symmetric kNN target bandwidths must be positive")
            # The reverse-union proof is specific to the c=1 canonical kth
            # radii.  Reject any other target bandwidth vector rather than
            # silently returning an incomplete support.
            expected = self.effective_radii
            if targets.ndim == 1:
                target_matches = np.array_equal(targets, expected)
            else:
                target_matches = bool(np.all(targets == expected[None, :]))
            if not target_matches:
                raise ValueError(
                    "symmetric kNN union requires target bandwidths equal to the canonical "
                    "kNN radii"
                )
        sources = np.asarray(source_indices)
        requested = np.asarray(radii, dtype=np.float64)
        if sources.ndim != 1 or sources.size == 0 or sources.dtype.kind not in "iu":
            raise ValueError("source_indices must be a non-empty integer vector")
        source_rows = np.asarray(sources, dtype=np.int64)
        if np.any(source_rows < 0) or np.any(source_rows >= self.descriptors.shape[0]):
            raise IndexError("source_indices contain a row outside the descriptor rows")
        if requested.ndim not in (1, 2) or requested.shape[-1] != source_rows.size:
            raise ValueError("radii must align with source_indices")
        if not np.all(np.isfinite(requested)) or np.any(requested <= 0.0):
            raise ValueError("symmetric kNN radii must be finite and positive")
        if not self.eligible(source_rows, requested, target_bandwidths):
            raise ValueError(
                "symmetric kNN union requires the canonical effective c=1 bandwidth vector"
            )

        counts = np.empty(source_rows.size, dtype=np.int64)
        location_parts: list[np.ndarray] = []
        distance_parts: list[np.ndarray] = []
        for local_index, source in enumerate(source_rows):
            source_location = int(self.neighbours.row_to_location[source])
            outgoing = self.neighbours.row_to_location[
                self.neighbours.neighbour_indices[
                    int(self.neighbours.location_representatives[source_location])
                ]
            ]
            outgoing_distances = self.neighbours.neighbour_distances[
                int(self.neighbours.location_representatives[source_location])
            ]
            outgoing = outgoing[outgoing_distances < self.neighbours.radii[source]]
            outgoing_distances = outgoing_distances[
                outgoing_distances < self.neighbours.radii[source]
            ]
            incoming_start = int(self.reverse_indptr[source_location])
            incoming_stop = int(self.reverse_indptr[source_location + 1])
            incoming = self.reverse_source_locations[incoming_start:incoming_stop]
            incoming_distances = self.reverse_distances[incoming_start:incoming_stop]
            locations = np.unique(
                np.concatenate(
                    (
                        np.asarray([source_location], dtype=np.int64),
                        outgoing,
                        incoming,
                        self.floor_extra_locations[
                            self.floor_extra_indptr[source_location] : self.floor_extra_indptr[
                                source_location + 1
                            ]
                        ],
                    )
                )
            )
            distance_by_location: dict[int, float] = {source_location: 0.0}
            distance_by_location.update(
                (int(location), float(distance))
                for location, distance in zip(outgoing, outgoing_distances, strict=True)
            )
            distance_by_location.update(
                (int(location), float(distance))
                for location, distance in zip(incoming, incoming_distances, strict=True)
            )
            extra_start = int(self.floor_extra_indptr[source_location])
            extra_stop = int(self.floor_extra_indptr[source_location + 1])
            if extra_stop > extra_start:
                distance_by_location.update(
                    (int(location), float(distance))
                    for location, distance in zip(
                        self.floor_extra_locations[extra_start:extra_stop],
                        self.floor_extra_distances[extra_start:extra_stop],
                        strict=True,
                    )
                )
            location_distances = np.asarray(
                [distance_by_location[int(location)] for location in locations],
                dtype=np.float64,
            )
            row_count = sum(
                int(self.location_indptr[int(location) + 1] - self.location_indptr[int(location)])
                for location in locations
            )
            counts[local_index] = row_count
            location_parts.append(locations)
            distance_parts.append(location_distances)

        total_support = int(np.sum(counts, dtype=np.int64))
        workspace_bytes = int(
            (source_rows.size + 1) * np.dtype(np.int64).itemsize
            + total_support * (np.dtype(np.int64).itemsize + np.dtype(np.float64).itemsize)
            + 64 * max(1, int(np.max(counts)))
            + 4096
        )
        if workspace_bytes > self.max_workspace_bytes:
            raise RadiusQueryCapacityError(
                "symmetric kNN support materialisation exceeds memory capacity: "
                f"requested_bytes={workspace_bytes}, max_support_workspace_bytes="
                f"{self.max_workspace_bytes}, support_entries={total_support}"
            )
        with self.resource_budget.acquire(
            self.capacity_context or "symmetric kNN support materialisation",
            max(1, workspace_bytes),
        ):
            offsets = np.zeros(source_rows.size + 1, dtype=np.int64)
            np.cumsum(counts, dtype=np.int64, out=offsets[1:])
            targets = np.empty(total_support, dtype=np.int64)
            output_distances = np.empty(total_support, dtype=np.float64)
            cursor = 0
            for locations, location_distances in zip(location_parts, distance_parts, strict=True):
                row_parts = [
                    self.rows_by_location[
                        self.location_indptr[int(location)] : self.location_indptr[
                            int(location) + 1
                        ]
                    ]
                    for location in locations
                ]
                rows = np.concatenate(row_parts).astype(np.int64, copy=False)
                row_distances = np.repeat(
                    location_distances,
                    [int(part.size) for part in row_parts],
                )
                order = np.argsort(rows, kind="stable")
                stop = cursor + int(rows.size)
                targets[cursor:stop] = rows[order]
                output_distances[cursor:stop] = row_distances[order]
                cursor = stop
        offsets.setflags(write=False)
        targets.setflags(write=False)
        output_distances.setflags(write=False)
        self.derived_sources += int(source_rows.size)
        self.derived_entries += total_support
        self.query_workspace_peak_bytes = max(self.query_workspace_peak_bytes, workspace_bytes)
        return offsets, targets, output_distances

    def operational_metrics(self) -> dict[str, Any]:
        metrics = dict(super().operational_metrics())
        floor_index_metrics = (
            self.floor_index.memory_metrics() if self.floor_index is not None else {}
        )
        floor_query_seconds = sum(
            float(floor_index_metrics.get(key, 0.0) or 0.0)
            for key in (
                "radius_preflight_count_seconds",
                "radius_retrieval_seconds",
                "canonical_distance_verification_seconds",
            )
        )
        metrics.update(
            {
                "support_provider": self.backend,
                "support_provider_version": self.backend_version,
                "kernel_bandwidth_rule": KERNEL_BANDWIDTH_RULE_SYMMETRIC_MAX,
                "reverse_index_memory_bytes": int(self.reverse_index_bytes),
                "reused_knn_locations": int(self.reused_knn_locations),
                "new_distance_evaluations": int(self.new_distance_evaluations),
                "support_provider_new_distance_evaluations": int(
                    self.new_distance_evaluations
                ),
                "floor_declared": self.declared_floor,
                "floor_subfloor_row_count": int(self.subfloor_row_count),
                "floor_subfloor_location_count": int(self.subfloor_location_count),
                "floor_augmented_support_pairs": int(self.floor_augmented_support_pairs),
                "floor_new_canonical_distance_evaluations": int(
                    self.floor_distance_evaluations
                ),
                "floor_subset_index_build_seconds": float(
                    floor_index_metrics.get("tree_build_seconds", 0.0) or 0.0
                ),
                "floor_subset_query_seconds": floor_query_seconds,
                "floor_subset_radius_queries": int(
                    floor_index_metrics.get("radius_queries", 0) or 0
                ),
                "floor_augmentation_noop": bool(self.floor_augmented_support_pairs == 0),
                "floor_statement": (
                    "zero sub-floor radii means floor augmentation is a no-op and raw "
                    "kNN reuse is exact"
                    if self.subfloor_location_count == 0
                    else (
                        "floor augmentation produced no new strict sub-floor pairs"
                        if self.floor_augmented_support_pairs == 0
                        else "floor augmentation added the exact strict sub-floor support subset"
                    )
                ),
            }
        )
        if self.subfloor_location_count:
            metrics["support_provider"] = "symmetric_knn_floor_augmented"
            metrics["support_provider_version"] = "symmetric-knn-floor-augmented-v1"
        return metrics


class ExactTiledCPUSupport:
    """Bounded exact CPU support for source/target pair bandwidths.

    The provider performs a count pass followed by a fill pass.  That makes
    the CSR capacity check happen before the output arrays are allocated and
    keeps the temporary distance workspace bounded by one source/target tile.
    """

    backend = "symmetric_tiled_cpu"
    backend_version = "symmetric-tiled-cpu-v1"

    def __init__(
        self,
        descriptors: np.ndarray,
        *,
        tile_size: int = 256,
        max_radius_query_bytes: int | None = None,
        max_calibration_work_bytes: int | None = None,
        resource_budget: ResourceBudgetService | None = None,
        capacity_context: str | None = None,
    ) -> None:
        self.descriptors = _validate_descriptors(descriptors)
        self.tile_size = _validate_chunk_size(tile_size)
        self.resource_budget = resource_budget or build_resource_budget()
        self.capacity_context = capacity_context
        remaining = self.resource_budget.remaining_managed_budget
        if remaining is None:
            raise ResourceCapacityError(
                self.resource_budget.unknown_memory_message("symmetric CPU support"),
                operation="symmetric CPU support",
            )
        limits = [int(remaining)]
        if max_radius_query_bytes is not None:
            limits.append(_validate_memory_limit(max_radius_query_bytes, "max_radius_query_bytes"))
        if max_calibration_work_bytes is not None:
            limits.append(
                _validate_memory_limit(
                    max_calibration_work_bytes, "max_calibration_work_bytes"
                )
            )
        self.max_workspace_bytes = min(limits)
        dimension = int(self.descriptors.shape[1])
        per_target = max(64, 16 * dimension + 64)
        self.target_tile_size = min(
            self.tile_size,
            max(1, (self.max_workspace_bytes - 16 * self.tile_size - 8192) // per_target),
        )
        if self.max_workspace_bytes < per_target + 8192:
            details = (
                "one symmetric CPU support tile exceeds its bounded workspace: "
                f"N={self.descriptors.shape[0]}, d'={self.descriptors.shape[1]}, "
                f"requested_bytes={per_target + 8192}, "
                f"max_support_workspace_bytes={self.max_workspace_bytes}"
            )
            if self.capacity_context:
                details += f", {self.capacity_context}"
            raise RadiusQueryCapacityError(
                details
            )
        self.distance_evaluations = 0
        self.new_distance_evaluations = 0
        self.support_query_count = 0
        self.support_entries = 0
        self.support_query_seconds = 0.0
        self.tile_count = 0
        self.query_workspace_peak_bytes = 0
        self.support_count_total = 0
        self.support_count_samples = 0
        self.min_support_count: int | None = None
        self.max_support_count = 0

    def _distance_tile(self, source: np.ndarray, targets: np.ndarray) -> np.ndarray:
        with np.errstate(over="raise", invalid="raise"):
            try:
                distances = np.sqrt(
                    np.sum((targets - source) ** 2, axis=1, dtype=np.float64)
                )
            except FloatingPointError as exc:
                raise ValueError(
                    "symmetric CPU support distance overflowed or became invalid"
                ) from exc
        if not np.all(np.isfinite(distances)):
            raise ValueError("symmetric CPU support distances must be finite")
        return np.asarray(distances, dtype=np.float64)

    @staticmethod
    def _envelope(values: np.ndarray, *, name: str, expected: int) -> tuple[np.ndarray, int]:
        array = np.asarray(values)
        if array.dtype != np.dtype(np.float64) or array.ndim not in (1, 2):
            raise ValueError(f"{name} must be a float64 vector or candidate matrix")
        if array.shape[-1] != expected or array.shape[0] == 0:
            raise ValueError(f"{name} does not align with the requested rows")
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError(f"{name} must be finite and strictly positive")
        return (
            np.ascontiguousarray(array if array.ndim == 1 else np.max(array, axis=0)),
            1 if array.ndim == 1 else int(array.shape[0]),
        )

    def support_many(
        self,
        source_indices: Sequence[int] | np.ndarray,
        source_bandwidths: Sequence[float] | np.ndarray,
        *,
        target_bandwidths: Sequence[float] | np.ndarray,
        kernel_bandwidth_rule: str = KERNEL_BANDWIDTH_RULE_SYMMETRIC_MAX,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            normalize_kernel_bandwidth_rule(kernel_bandwidth_rule)
            != KERNEL_BANDWIDTH_RULE_SYMMETRIC_MAX
        ):
            raise ValueError("ExactTiledCPUSupport only implements symmetric_max")
        sources = np.asarray(source_indices)
        if sources.ndim != 1 or sources.size == 0 or sources.dtype.kind not in "iu":
            raise ValueError("source_indices must be a non-empty integer vector")
        source_rows = np.asarray(sources, dtype=np.int64)
        if np.any(source_rows < 0) or np.any(source_rows >= self.descriptors.shape[0]):
            raise IndexError("source_indices contain a row outside the descriptor rows")
        source_env, candidate_count = self._envelope(
            np.asarray(source_bandwidths), name="source_bandwidths", expected=source_rows.size
        )
        target_array = np.asarray(target_bandwidths)
        if target_array.dtype != np.dtype(np.float64) or target_array.ndim not in (1, 2):
            raise ValueError("target_bandwidths must be a full-pool float64 vector or matrix")
        if target_array.shape[-1] != self.descriptors.shape[0]:
            raise ValueError("target_bandwidths must align with the full original row pool")
        target_env, target_candidates = self._envelope(
            target_array, name="target_bandwidths", expected=self.descriptors.shape[0]
        )
        if target_candidates not in {1, candidate_count} and target_array.ndim == 2:
            raise ValueError("source and target bandwidth candidate counts do not align")

        row_count = int(self.descriptors.shape[0])
        counts = np.zeros(source_rows.size, dtype=np.int64)

        def scan(pass_name: str) -> None:
            for local_source, source_index in enumerate(source_rows):
                source_values = self.descriptors[int(source_index)]
                source_h = float(source_env[local_source])
                for start in range(0, row_count, self.target_tile_size):
                    stop = min(start + self.target_tile_size, row_count)
                    target_indices = np.arange(start, stop, dtype=np.int64)
                    distances = self._distance_tile(
                        source_values, self.descriptors[start:stop]
                    )
                    self.distance_evaluations += int(target_indices.size)
                    self.new_distance_evaluations += int(target_indices.size)
                    self.tile_count += 1
                    effective = np.maximum(source_h, target_env[start:stop])
                    keep = distances < effective
                    if pass_name == "count":
                        counts[local_source] += int(np.count_nonzero(keep))

        started = time.perf_counter()
        scan("count")
        total_support = int(np.sum(counts, dtype=np.int64))
        workspace_bytes = int(
            (source_rows.size + 1) * np.dtype(np.int64).itemsize
            + total_support * 16
            + self.target_tile_size * (16 * self.descriptors.shape[1] + 64)
            + 8192
        )
        self.query_workspace_peak_bytes = max(self.query_workspace_peak_bytes, workspace_bytes)
        if workspace_bytes > self.max_workspace_bytes:
            detail = (
                f"requested_bytes={workspace_bytes}, max_support_workspace_bytes="
                f"{self.max_workspace_bytes}, source_count={source_rows.size}, "
                f"support_entries={total_support}"
            )
            if self.capacity_context:
                detail += f", {self.capacity_context}"
            raise RadiusQueryCapacityError(
                "symmetric CPU support exceeds memory capacity: " + detail
            )
        with self.resource_budget.acquire(
            self.capacity_context or "symmetric CPU support materialisation",
            max(1, workspace_bytes),
        ):
            offsets = np.zeros(source_rows.size + 1, dtype=np.int64)
            np.cumsum(counts, dtype=np.int64, out=offsets[1:])
            targets = np.empty(total_support, dtype=np.int64)
            distances_out = np.empty(total_support, dtype=np.float64)
            cursor_by_source = offsets[:-1].copy()
            for local_source, source_index in enumerate(source_rows):
                source_values = self.descriptors[int(source_index)]
                source_h = float(source_env[local_source])
                cursor = int(cursor_by_source[local_source])
                for start in range(0, row_count, self.target_tile_size):
                    stop = min(start + self.target_tile_size, row_count)
                    target_indices = np.arange(start, stop, dtype=np.int64)
                    distances = self._distance_tile(
                        source_values, self.descriptors[start:stop]
                    )
                    self.distance_evaluations += int(target_indices.size)
                    self.new_distance_evaluations += int(target_indices.size)
                    self.tile_count += 1
                    effective = np.maximum(source_h, target_env[start:stop])
                    keep = distances < effective
                    kept = int(np.count_nonzero(keep))
                    if kept:
                        targets[cursor : cursor + kept] = target_indices[keep]
                        distances_out[cursor : cursor + kept] = distances[keep]
                        cursor += kept
                if cursor != int(offsets[local_source + 1]):
                    raise RuntimeError("symmetric CPU support fill pass disagreed with count pass")
        for count in counts:
            support_count = int(count)
            self.support_count_total += support_count
            self.support_count_samples += 1
            self.max_support_count = max(self.max_support_count, support_count)
            self.min_support_count = (
                support_count
                if self.min_support_count is None
                else min(self.min_support_count, support_count)
            )
        self.support_query_count += int(source_rows.size)
        self.support_entries += total_support
        self.support_query_seconds += max(0.0, time.perf_counter() - started)
        offsets.setflags(write=False)
        targets.setflags(write=False)
        distances_out.setflags(write=False)
        return offsets, targets, distances_out

    def operational_metrics(self) -> dict[str, int | float | str]:
        return {
            "support_provider": self.backend,
            "support_provider_version": self.backend_version,
            "kernel_bandwidth_rule": KERNEL_BANDWIDTH_RULE_SYMMETRIC_MAX,
            "support_query_count": int(self.support_query_count),
            "support_entries": int(self.support_entries),
            "support_query_seconds": float(self.support_query_seconds),
            "support_count_min": self.min_support_count,
            "support_count_mean": (
                self.support_count_total / self.support_count_samples
                if self.support_count_samples
                else None
            ),
            "support_count_max": int(self.max_support_count),
            "distance_evaluations": int(self.distance_evaluations),
            "new_distance_evaluations": int(self.new_distance_evaluations),
            "support_provider_new_distance_evaluations": int(self.new_distance_evaluations),
            "cpu_support_tile_count": int(self.tile_count),
            "query_workspace_peak_bytes": int(self.query_workspace_peak_bytes),
        }


SymmetricKNNPrefixSupport = ExactSymmetricKNNUnionSupport
SymmetricTiledCPUSupport = ExactTiledCPUSupport


def _validate_row_ids(row_ids: Sequence[str] | None, n_rows: int) -> tuple[str, ...]:
    if row_ids is None:
        return tuple(str(index) for index in range(n_rows))
    ordered = tuple(str(value) for value in row_ids)
    if len(ordered) != n_rows or any(not value.strip() for value in ordered):
        raise ValueError("row_ids must be non-blank and align with descriptor rows")
    return ordered


class IndexedCPUNeighbourIndex:
    """Exact Euclidean index over unique descriptor locations.

    The tree is used only to find a conservative candidate set.  Every
    candidate is re-evaluated with the shared float64 Euclidean primitive and
    sorted by ``(distance, earliest_original_row)`` before it is returned.
    This makes cKDTree traversal order and query batching scientifically
    irrelevant while allowing sparse radius supports to avoid full-pool scans.
    """

    backend = INDEXED_NEIGHBOUR_BACKEND_ID
    backend_version = INDEXED_NEIGHBOUR_BACKEND_VERSION
    metric = NEIGHBOUR_METRIC
    distinct_location_policy = NEIGHBOUR_DISTINCT_POLICY

    def __init__(
        self,
        descriptors: np.ndarray,
        *,
        max_index_bytes: int | None = None,
        max_radius_query_bytes: int | None = None,
        max_neighbour_cache_bytes: int | None = DEFAULT_NEIGHBOUR_CACHE_BYTES,
        resource_budget: ResourceBudgetService | None = None,
    ) -> None:
        values = _validate_descriptors(descriptors)
        service = resource_budget or build_resource_budget()
        remaining = service.remaining_managed_budget
        if remaining is None:
            raise ResourceCapacityError(
                service.unknown_memory_message("indexed neighbour construction"),
                operation="indexed neighbour construction",
            )
        if remaining < 1:
            raise ResourceCapacityError(
                "indexed neighbour construction has no allocatable runtime memory headroom",
                operation="indexed neighbour construction",
                requested_bytes=1,
                available_bytes=remaining,
                reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
            )
        explicit_index_limit = (
            None
            if max_index_bytes is None
            else _validate_memory_limit(max_index_bytes, "max_index_bytes")
        )
        effective_index_limit = (
            remaining if explicit_index_limit is None else min(explicit_index_limit, remaining)
        )
        # Preflight the worst-case unique-location/grouping/tree footprint before
        # np.unique and cKDTree can materialize their temporary arrays.  The
        # post-build estimate below remains the tighter check for duplicate rows.
        estimated_peak_index_bytes = int(2 * values.nbytes + 160 * values.shape[0] + 8192)
        if estimated_peak_index_bytes > effective_index_limit:
            if (
                explicit_index_limit is not None
                and estimated_peak_index_bytes > explicit_index_limit
            ):
                raise ValueError(
                    "indexed exact neighbour backend exceeds max_index_bytes before allocation: "
                    f"N={values.shape[0]}, estimated_peak_bytes={estimated_peak_index_bytes}, "
                    f"max_index_bytes={explicit_index_limit}"
                )
            raise ResourceCapacityError(
                "indexed exact neighbour backend cannot reserve its pre-build footprint: "
                f"N={values.shape[0]}, estimated_peak_bytes={estimated_peak_index_bytes}, "
                f"available_bytes={remaining}",
                operation="indexed neighbour construction",
                requested_bytes=estimated_peak_index_bytes,
                available_bytes=remaining,
                reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
            )
        radius_limit = (
            max(1, remaining // 2)
            if max_radius_query_bytes is None
            else min(
                _validate_memory_limit(max_radius_query_bytes, "max_radius_query_bytes"),
                remaining,
            )
        )
        self.resource_budget = service
        logger.info(
            "Indexed neighbour grouping started: N=%d, d'=%d",
            values.shape[0],
            values.shape[1],
        )
        locations, first_indices, row_to_location, _location_multiplicity = (
            group_descriptor_locations(values)
        )
        self.descriptors = values
        self.unique_locations = locations
        self.location_representatives = first_indices
        self.row_to_location = row_to_location
        location_count = int(self.unique_locations.shape[0])
        grouping_started = time.perf_counter()
        self.rows_by_location = np.asarray(
            np.argsort(self.row_to_location, kind="stable"),
            dtype=np.int64,
        )
        location_counts = np.bincount(self.row_to_location, minlength=location_count).astype(
            np.int64,
            copy=False,
        )
        self.location_indptr = np.empty(location_count + 1, dtype=np.int64)
        self.location_indptr[0] = 0
        np.cumsum(location_counts, dtype=np.int64, out=self.location_indptr[1:])
        self.rows_by_location.setflags(write=False)
        self.location_indptr.setflags(write=False)
        self.grouping_build_seconds = max(0.0, time.perf_counter() - grouping_started)
        self.grouping_bytes = int(self.rows_by_location.nbytes + self.location_indptr.nbytes)
        logger.info(
            "Indexed neighbour grouping completed: N=%d, distinct locations=%d, "
            "grouping_bytes=%d, elapsed=%.3fs",
            values.shape[0],
            location_count,
            self.grouping_bytes,
            self.grouping_build_seconds,
        )
        self.max_radius_query_bytes = radius_limit
        self.index_queries = 0
        self.radius_queries = 0
        self.radius_query_preflights = 0
        self.knn_fast_path_certified = 0
        self.knn_fast_path_ambiguous = 0
        self.knn_fast_path_fallback = 0
        self.distance_evaluations = 0
        self.unique_distance_evaluations = 0
        self.atomic_distance_evaluations = 0
        self.current_support_count = 0
        self.min_support_count: int | None = None
        self.support_count_total = 0
        self.support_count_samples = 0
        self.max_support_count = 0
        self.pilot_support_queries = 0
        self.pilot_support_count_total = 0
        self.pilot_support_samples = 0
        self.query_workspace_peak_bytes = 0
        # These are operational-only counters.  They are deliberately updated
        # at chunk/query boundaries rather than inside the per-point loops so
        # diagnostics cannot change the scientific ordering or add log noise.
        self._phase_seconds: dict[str, float] = {}
        self._phase_counts: dict[str, int] = {}
        self._phase_bytes: dict[str, int] = {}
        self.index_bytes = self._estimate_index_bytes()
        if self.index_bytes > effective_index_limit:
            if explicit_index_limit is not None and self.index_bytes > explicit_index_limit:
                raise ValueError(
                    "indexed exact neighbour backend exceeds max_index_bytes: "
                    f"N={values.shape[0]}, Q={self.unique_locations.shape[0]}, "
                    f"estimated_peak_bytes={self.index_bytes}, "
                    f"max_index_bytes={explicit_index_limit}"
                )
            raise ResourceCapacityError(
                "indexed exact neighbour backend exceeds its runtime memory allowance: "
                f"N={values.shape[0]}, Q={self.unique_locations.shape[0]}, "
                f"estimated_peak_bytes={self.index_bytes}, "
                f"max_index_bytes={effective_index_limit}",
                operation="indexed neighbour construction",
                requested_bytes=self.index_bytes,
                available_bytes=remaining,
                reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
            )
        self.max_neighbour_cache_bytes = _resolve_neighbour_cache_budget(
            remaining_bytes=remaining,
            index_bytes=self.index_bytes,
            radius_query_bytes=self.max_radius_query_bytes,
            user_cap=max_neighbour_cache_bytes,
        )
        self.cache_budget_bytes = int(self.max_neighbour_cache_bytes)
        try:
            from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]
        except ImportError as exc:
            raise ValueError(
                f"neighbour backend {self.backend!r} is unavailable: scipy.spatial.cKDTree"
            ) from exc
        logger.info(
            "Indexed neighbour tree build started: distinct locations=%d, estimated_bytes=%d",
            location_count,
            self.index_bytes,
        )
        tree_started = time.perf_counter()
        self._tree: Any = cKDTree(self.unique_locations, copy_data=True)
        self.tree_build_seconds = max(0.0, time.perf_counter() - tree_started)
        logger.info(
            "Indexed neighbour tree completed: distinct locations=%d, tree_bytes=%d, elapsed=%.3fs",
            location_count,
            self.index_bytes,
            self.tree_build_seconds,
        )
        self._neighbour_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._neighbour_cache_lease: Any | None = None
        self._neighbour_cache_key: str | None = None
        self._cache_lock = threading.RLock()
        self.neighbour_cache_bytes = 0
        self.cache_hit = 0
        self.cache_miss = 0
        self.cache_eviction = 0
        self.cache_admission_rejected = 0
        self.cache_peak_bytes = 0
        self.cache_time_saved_seconds = 0.0
        self._last_neighbour_compute_seconds = 0.0
        self.fingerprint = sha256_canonical_json(
            {
                "backend": self.backend,
                "version": self.backend_version,
                "metric": self.metric,
                "distinct_location_policy": self.distinct_location_policy,
                "tie_policy": "distance-then-earliest-original-representative-v1",
                "radius_policy": "strict-float64-euclidean-less-than-v1",
                "locations": _array_fingerprint(self.unique_locations),
                "row_to_location": _array_fingerprint(self.row_to_location),
            }
        )
        owner = weakref.ref(self)

        def evict() -> int:
            instance = owner()
            return 0 if instance is None else instance._evict_neighbour_cache()

        self._cache_eviction_token = service.register_optional_eviction(evict)

    def close(self) -> None:
        """Release optional cache bytes and its runtime-budget registration."""

        self.resource_budget.unregister_optional_eviction(self._cache_eviction_token)
        self._evict_neighbour_cache()

    def __del__(self) -> None:  # pragma: no cover - best-effort lifecycle cleanup
        try:
            self.close()
        except Exception:
            pass

    @property
    def worker_count(self) -> int:
        """Return the bounded SciPy worker count for indexed queries."""

        configured = self.resource_budget.budget.worker_cap
        available = os.cpu_count() or 1
        if configured is None:
            return max(1, available)
        return max(1, min(int(configured), available))

    def _estimate_index_bytes(self) -> int:
        n_locations = int(self.unique_locations.shape[0])
        # Include NumPy storage, compact row grouping, and a conservative cKDTree /
        # Python-container allowance.  This is an operational upper estimate,
        # not a claim about the allocator's exact resident-set accounting.
        return int(
            self.unique_locations.nbytes
            + self.location_representatives.nbytes
            + self.row_to_location.nbytes
            + self.rows_by_location.nbytes
            + self.location_indptr.nbytes
            + 128 * n_locations
            + 4096
        )

    def _record_phase(
        self,
        name: str,
        elapsed: float,
        *,
        count: int = 1,
        bytes_written: int = 0,
    ) -> None:
        """Accumulate deterministic operational phase evidence."""

        self._phase_seconds[name] = self._phase_seconds.get(name, 0.0) + max(0.0, float(elapsed))
        self._phase_counts[name] = self._phase_counts.get(name, 0) + int(count)
        if bytes_written:
            self._phase_bytes[name] = self._phase_bytes.get(name, 0) + int(bytes_written)

    def memory_metrics(self) -> dict[str, Any]:
        """Return operational memory counters for progress diagnostics."""

        metrics: dict[str, Any] = {
            "support_cache_bytes": 0,
            "support_cache_peak_bytes": 0,
            "neighbour_cache_bytes": int(self.neighbour_cache_bytes),
            "cache_bytes": int(self.neighbour_cache_bytes),
            "cache_peak_bytes": int(self.cache_peak_bytes),
            "cache_hit": int(self.cache_hit),
            "cache_miss": int(self.cache_miss),
            "cache_eviction": int(self.cache_eviction),
            "cache_admission_rejected": int(self.cache_admission_rejected),
            "cache_time_saved_seconds": float(self.cache_time_saved_seconds),
            "cache_budget_bytes": int(self.cache_budget_bytes),
            "index_bytes": int(self.index_bytes),
            "worker_count": int(self.worker_count),
            "grouping_bytes": int(self.grouping_bytes),
            "grouping_build_seconds": float(self.grouping_build_seconds),
            "tree_build_seconds": float(self.tree_build_seconds),
            "query_workspace_peak_bytes": int(self.query_workspace_peak_bytes),
            "current_support_count": int(self.current_support_count),
            "min_support_count": self.min_support_count,
            "mean_support_count": (
                self.support_count_total / self.support_count_samples
                if self.support_count_samples
                else None
            ),
            "max_support_count": int(self.max_support_count),
            "pilot_support_queries": int(self.pilot_support_queries),
            "pilot_support_count_total": int(self.pilot_support_count_total),
            "pilot_support_samples": int(self.pilot_support_samples),
            "index_queries": int(self.index_queries),
            "radius_queries": int(self.radius_queries),
            "radius_query_preflights": int(self.radius_query_preflights),
            "knn_fast_path_certified": int(self.knn_fast_path_certified),
            "knn_fast_path_ambiguous": int(self.knn_fast_path_ambiguous),
            "knn_fast_path_certified_count": int(self.knn_fast_path_certified),
            "knn_fast_path_ambiguous_count": int(self.knn_fast_path_ambiguous),
            "knn_fast_path_fallback_count": int(self.knn_fast_path_fallback),
            "distance_evaluations": int(self.distance_evaluations),
            "unique_distance_evaluations": int(self.unique_distance_evaluations),
            "atomic_distance_evaluations": int(self.atomic_distance_evaluations),
            **_process_memory_metrics(),
        }
        for phase, elapsed in self._phase_seconds.items():
            metrics[f"{phase}_seconds"] = float(elapsed)
        for phase, count in self._phase_counts.items():
            metrics[f"{phase}_count"] = int(count)
            if phase == "radius_preflight_count":
                # Keep the historical flattened key above, but expose a name
                # that cannot be mistaken for a count-of-counts counter.
                metrics["radius_preflight_calls"] = int(count)
        for phase, byte_count in self._phase_bytes.items():
            metrics[f"{phase}_bytes"] = int(byte_count)
        aliases = {
            "tree_query": "knn_tree_query",
            "preflight": "radius_preflight_count",
            "canonical_distance": "canonical_distance_verification",
            "checkpoint_fsync": "checkpoint_serialization",
        }
        for alias, phase in aliases.items():
            if f"{alias}_seconds" not in metrics and f"{phase}_seconds" in metrics:
                metrics[f"{alias}_seconds"] = metrics[f"{phase}_seconds"]
            if f"{alias}_count" not in metrics and f"{phase}_count" in metrics:
                metrics[f"{alias}_count"] = metrics[f"{phase}_count"]
            if f"{alias}_bytes" not in metrics and f"{phase}_bytes" in metrics:
                metrics[f"{alias}_bytes"] = metrics[f"{phase}_bytes"]
        # These phases are optional: certified kNN prefixes can bypass every
        # radius query, and checkpointing is caller-controlled.  Keep their
        # documented counters stable so diagnostics consumers do not need to
        # distinguish an absent phase from a phase that executed zero times.
        for phase in (
            "knn_tree_query",
            "knn_boundary_certification",
            "radius_preflight_count",
            "radius_retrieval",
            "canonical_distance_verification",
            "checkpoint_serialization",
            "checkpoint_fsync",
            "checkpoint_digest",
            "checkpoint_read",
            "query_neighbours_many_projection",
        ):
            metrics.setdefault(f"{phase}_seconds", 0.0)
            metrics.setdefault(f"{phase}_count", 0)
            metrics.setdefault(f"{phase}_bytes", 0)
        metrics["radius_preflight_count"] = int(
            metrics.get("radius_preflight_count_count", 0)
        )
        metrics.setdefault("radius_preflight_calls", metrics["radius_preflight_count"])
        return metrics

    def query_nearest_distances(
        self,
        queries: np.ndarray,
        *,
        chunk_size: int = 1024,
        context: str = "indexed nearest-neighbour query",
    ) -> np.ndarray:
        """Return exact nearest distances for external query rows in bounded chunks.

        The query rows are not inserted into the index and no query-by-index
        distance matrix is materialized.  This is used by the independent
        extrapolative holdout policy to compare candidate-owned local rows with
        training-owned local rows.
        """

        values = _validate_descriptors(queries)
        block_size = _validate_chunk_size(chunk_size)
        distances = np.empty(values.shape[0], dtype=np.float64)
        for start in range(0, values.shape[0], block_size):
            stop = min(start + block_size, values.shape[0])
            block = values[start:stop]
            estimated_bytes = _estimate_radius_workspace_bytes(
                int(block.shape[0]),
                1,
                int(block.shape[1]),
            )
            if estimated_bytes > self.max_radius_query_bytes:
                raise self._capacity_error(
                    source_index=None,
                    radius=None,
                    requested_bytes=estimated_bytes,
                    context=context,
                )
            with self.resource_budget.acquire(context, max(1, estimated_bytes)):
                _tree_distances, tree_indices = self._tree.query(
                    block,
                    k=1,
                    workers=self.worker_count,
                )
                nearest = self.unique_locations[np.asarray(tree_indices, dtype=np.int64)]
                delta = nearest - block
                distances[start:stop] = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float64))
        if not np.all(np.isfinite(distances)):
            raise ValueError(f"{context} produced non-finite distances")
        return distances

    @staticmethod
    def _distance_values(
        query: np.ndarray,
        targets: np.ndarray,
        *,
        max_workspace_bytes: int | None = None,
        persistent_workspace_bytes: int = 0,
    ) -> np.ndarray:
        count = int(targets.shape[0])
        distances = np.empty(count, dtype=np.float64)
        if count == 0:
            return distances
        if max_workspace_bytes is None:
            block = count
        else:
            available = (
                int(max_workspace_bytes)
                - int(persistent_workspace_bytes)
                - int(targets.nbytes)
                - int(distances.nbytes)
                - 256
            )
            bytes_per_row = max(8, 16 * int(targets.shape[1]) + 16)
            block = available // bytes_per_row
            if block < 1:
                raise RadiusQueryCapacityError(
                    "indexed neighbour distance workspace exceeds max_radius_query_bytes"
                )
            block = min(count, int(block))
        with np.errstate(over="raise", invalid="raise"):
            try:
                for start in range(0, count, block):
                    stop = min(start + block, count)
                    delta = targets[start:stop] - query
                    squared = delta**2
                    distances[start:stop] = np.sqrt(np.sum(squared, axis=1, dtype=np.float64))
            except FloatingPointError as exc:
                raise ValueError("indexed neighbour distance overflowed or became invalid") from exc
        if not np.all(np.isfinite(distances)):
            raise ValueError("indexed neighbour distances must be finite")
        return distances

    def _distance_values_many_workspace(
        self,
        queries: np.ndarray,
        candidate_lists: Sequence[np.ndarray],
        *,
        persistent_workspace_bytes: int = 0,
    ) -> tuple[int, int]:
        """Return fixed and per-candidate temporary bytes for ragged distances."""

        counts = np.asarray(
            [int(np.asarray(candidates, dtype=np.int64).size) for candidates in candidate_lists],
            dtype=np.int64,
        )
        total = int(np.sum(counts, dtype=np.int64))
        dimension = int(self.unique_locations.shape[1])
        # counts, CSR offsets, flattened candidate IDs, owner IDs, and the
        # returned distance vector all remain live while a distance tile runs.
        fixed = int(
            persistent_workspace_bytes
            + np.asarray(queries).nbytes
            + counts.nbytes
            + (counts.size + 1) * np.dtype(np.int64).itemsize
            + counts.size * np.dtype(np.int64).itemsize
            + 24 * total
            + 256
        )
        return fixed, max(8, 32 * dimension + 16)

    def _distance_values_many(
        self,
        queries: np.ndarray,
        candidate_lists: Sequence[np.ndarray],
        *,
        max_workspace_bytes: int | None = None,
        persistent_workspace_bytes: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recompute canonical distances for a bounded ragged query batch.

        ``cKDTree`` only supplies a conservative candidate set.  The returned
        CSR offsets and distances are therefore always based on the canonical
        float64 ``sqrt(sum(delta**2))`` operation, never on tree distances.
        Flattening the admitted candidate lists lets NumPy evaluate several
        sources per block without constructing a source-by-pool matrix.
        """

        query_values = np.asarray(queries, dtype=np.float64)
        if query_values.ndim != 2 or query_values.shape[1] != self.unique_locations.shape[1]:
            raise ValueError("batched radius queries have an incompatible shape")
        query_count = int(query_values.shape[0])
        if len(candidate_lists) != query_count:
            raise ValueError("batched radius candidates do not align with queries")
        counts = np.asarray(
            [int(np.asarray(candidates, dtype=np.int64).size) for candidates in candidate_lists],
            dtype=np.int64,
        )
        offsets = np.empty(query_count + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, dtype=np.int64, out=offsets[1:])
        total = int(offsets[-1])
        if total == 0:
            return offsets, np.empty(0, dtype=np.float64)

        if max_workspace_bytes is None:
            block = total
        else:
            fixed_bytes, bytes_per_row = self._distance_values_many_workspace(
                query_values,
                candidate_lists,
                persistent_workspace_bytes=persistent_workspace_bytes,
            )
            available = int(max_workspace_bytes) - fixed_bytes
            block = available // bytes_per_row
            if block < 1:
                raise RadiusQueryCapacityError(
                    "indexed neighbour batched distance workspace exceeds max_radius_query_bytes"
                )
            block = min(total, int(block))

        # Admit the fixed ragged vectors before materialising them.  The
        # caller's persistent charge covers the candidate lists that are
        # already live; this allocation adds only the flattened IDs, owners,
        # and returned distances accounted for by the planner above.
        distances = np.empty(total, dtype=np.float64)
        flat_candidates = np.concatenate(
            [np.asarray(candidates, dtype=np.int64) for candidates in candidate_lists]
        )
        owners = np.repeat(np.arange(query_count, dtype=np.int64), counts)

        with np.errstate(over="raise", invalid="raise"):
            try:
                for start in range(0, total, block):
                    stop = min(start + block, total)
                    targets = self.unique_locations[flat_candidates[start:stop]]
                    delta = targets - query_values[owners[start:stop]]
                    squared = delta**2
                    distances[start:stop] = np.sqrt(np.sum(squared, axis=1, dtype=np.float64))
            except FloatingPointError as exc:
                raise ValueError(
                    "indexed neighbour batched distance evaluation overflowed or became invalid"
                ) from exc
        if not np.all(np.isfinite(distances)):
            raise ValueError("indexed neighbour batched distances must be finite")
        return offsets, distances

    def _distance_values_matrix(
        self,
        queries: np.ndarray,
        candidate_matrix: np.ndarray,
        *,
        max_workspace_bytes: int | None = None,
        persistent_workspace_bytes: int = 0,
    ) -> np.ndarray:
        """Recompute canonical distances for a bounded dense candidate block."""

        query_values = np.asarray(queries, dtype=np.float64)
        candidates = np.asarray(candidate_matrix, dtype=np.int64)
        if query_values.ndim != 2 or candidates.ndim != 2:
            raise ValueError("batched neighbour distances require two-dimensional inputs")
        if query_values.shape[0] != candidates.shape[0]:
            raise ValueError("batched neighbour queries and candidates do not align")
        row_count, candidate_count = (int(value) for value in candidates.shape)
        distances = np.empty((row_count, candidate_count), dtype=np.float64)
        if row_count == 0 or candidate_count == 0:
            return distances
        if max_workspace_bytes is None:
            row_block = row_count
        else:
            _fixed_bytes, _tile_bytes, row_block = self._matrix_distance_workspace(
                query_values,
                candidates,
                max_workspace_bytes=int(max_workspace_bytes),
                persistent_workspace_bytes=persistent_workspace_bytes,
            )
            if row_block < 1:
                raise RadiusQueryCapacityError(
                    "indexed neighbour matrix distance workspace exceeds max_radius_query_bytes"
                )
            row_block = min(row_count, int(row_block))
        with np.errstate(over="raise", invalid="raise"):
            try:
                for start in range(0, row_count, row_block):
                    stop = min(start + row_block, row_count)
                    targets = self.unique_locations[candidates[start:stop]]
                    delta = targets - query_values[start:stop, None, :]
                    squared = delta**2
                    distances[start:stop] = np.sqrt(np.sum(squared, axis=2, dtype=np.float64))
            except FloatingPointError as exc:
                raise ValueError(
                    "indexed neighbour matrix distance evaluation overflowed or became invalid"
                ) from exc
        if not np.all(np.isfinite(distances)):
            raise ValueError("indexed neighbour matrix distances must be finite")
        return distances

    def _matrix_distance_workspace(
        self,
        queries: np.ndarray,
        candidates: np.ndarray,
        *,
        max_workspace_bytes: int,
        persistent_workspace_bytes: int = 0,
    ) -> tuple[int, int, int]:
        """Plan matrix-distance temporaries and the largest admissible row tile."""

        row_count, candidate_count = (int(value) for value in candidates.shape)
        dimension = int(self.unique_locations.shape[1])
        fixed_bytes = int(
            persistent_workspace_bytes
            + np.asarray(queries).nbytes
            + candidates.nbytes
            + 8 * row_count * candidate_count
            + 256
        )
        tile_bytes = max(8, 32 * dimension * candidate_count + 16 * candidate_count)
        available = int(max_workspace_bytes) - fixed_bytes
        row_block = available // tile_bytes
        return fixed_bytes, tile_bytes, min(row_count, max(0, int(row_block)))

    def _capacity_error(
        self,
        *,
        source_index: int | None,
        radius: float | None,
        requested_bytes: int,
        context: str | None = None,
    ) -> RadiusQueryCapacityError:
        details = [
            f"N={self.descriptors.shape[0]}",
            f"d'={self.descriptors.shape[1]}",
            f"source_row={source_index if source_index is not None else 'n/a'}",
            f"radius={radius if radius is not None else 'n/a'}",
            f"requested_bytes={int(requested_bytes)}",
            f"max_radius_query_bytes={self.max_radius_query_bytes}",
        ]
        if context:
            details.append(context)
        return RadiusQueryCapacityError(
            "exact radius query exceeds memory capacity: " + ", ".join(details)
        )

    def _query_location_candidates(
        self,
        query: np.ndarray,
        radius: float,
        *,
        source_index: int | None,
        row_count: int = 0,
        context: str | None = None,
    ) -> np.ndarray:
        expanded_radius = radius + max(16.0 * float(np.spacing(radius)), 1.0e-14)
        preflight_started = time.perf_counter()
        self.radius_query_preflights += 1
        try:
            candidate_count = int(
                self._tree.query_ball_point(
                    query,
                    expanded_radius,
                    return_length=True,
                    workers=self.worker_count,
                )
            )
        except TypeError as exc:
            raise self._capacity_error(
                source_index=source_index,
                radius=radius,
                requested_bytes=self.max_radius_query_bytes + 1,
                context="SciPy cKDTree does not support return_length preflight",
            ) from exc
        self._record_phase("radius_preflight_count", time.perf_counter() - preflight_started)
        candidate_estimate = _estimate_radius_workspace_bytes(
            candidate_count,
            0,
            self.descriptors.shape[1],
            row_count,
        )
        self.query_workspace_peak_bytes = max(self.query_workspace_peak_bytes, candidate_estimate)
        if candidate_estimate > self.max_radius_query_bytes:
            raise self._capacity_error(
                source_index=source_index,
                radius=radius,
                requested_bytes=candidate_estimate,
                context=context,
            )
        retrieval_started = time.perf_counter()
        try:
            with self.resource_budget.acquire(
                context or "indexed exact radius candidate retrieval",
                max(1, candidate_estimate),
            ):
                location_indices = np.asarray(
                    self._tree.query_ball_point(query, expanded_radius, workers=self.worker_count),
                    dtype=np.int64,
                )
        except ResourceCapacityError as exc:
            raise self._capacity_error(
                source_index=source_index,
                radius=radius,
                requested_bytes=candidate_estimate,
                context=context,
            ) from exc
        self._record_phase("radius_retrieval", time.perf_counter() - retrieval_started)
        self.radius_queries += 1
        if location_indices.size != candidate_count:
            raise ValueError("cKDTree radius preflight disagreed with its candidate query")
        return location_indices

    def _cache_key(self, k: int) -> str:
        return sha256_canonical_json(
            {
                "schema": "exact-neighbour-cache-v1",
                "representation": self.fingerprint,
                "backend": self.backend,
                "version": self.backend_version,
                "k_max": int(k),
            }
        )

    def _evict_neighbour_cache(self) -> int:
        with self._cache_lock:
            released = int(self.neighbour_cache_bytes)
            if not self._neighbour_cache:
                return 0
            self._neighbour_cache.clear()
            self._neighbour_cache_key = None
            self.neighbour_cache_bytes = 0
            lease = self._neighbour_cache_lease
            self._neighbour_cache_lease = None
            self.cache_eviction += 1
            if lease is not None:
                lease.close()
            logger.info("Evicted optional exact neighbour cache: bytes=%d", released)
            return released

    def _store_neighbour_cache(
        self,
        k: int,
        neighbour_indices: np.ndarray,
        neighbour_distances: np.ndarray,
    ) -> None:
        entry_bytes = int(neighbour_indices.nbytes + neighbour_distances.nbytes)
        with self._cache_lock:
            if entry_bytes > self.max_neighbour_cache_bytes:
                self.cache_admission_rejected += 1
                logger.info(
                    "Rejected optional exact neighbour cache admission: k=%d, "
                    "cache_bytes=%d, budget_bytes=%d, reason=budget",
                    k,
                    entry_bytes,
                    self.max_neighbour_cache_bytes,
                )
                return
            previous_key = self._neighbour_cache_key
        try:
            lease = self.resource_budget.acquire(
                "optional exact neighbour cache", max(1, entry_bytes)
            )
        except ResourceCapacityError:
            with self._cache_lock:
                self.cache_admission_rejected += 1
            logger.info(
                "Rejected optional exact neighbour cache admission: k=%d, "
                "cache_bytes=%d, reason=active-resource-pressure",
                k,
                entry_bytes,
            )
            return
        with self._cache_lock:
            if self._neighbour_cache_key != previous_key:
                lease.close()
                self.cache_admission_rejected += 1
                logger.info(
                    "Rejected optional exact neighbour cache admission: k=%d, "
                    "cache_bytes=%d, reason=concurrent-cache-update",
                    k,
                    entry_bytes,
                )
                return
            # Keep one canonical widest result.  A replacement is admitted
            # while the old entry is still resident; if admission succeeds,
            # the old lease is released immediately afterward.
            old_lease = self._neighbour_cache_lease
            if self._neighbour_cache:
                self.cache_eviction += 1
            if old_lease is not None:
                old_lease.close()
            neighbour_indices.setflags(write=False)
            neighbour_distances.setflags(write=False)
            self._neighbour_cache = {int(k): (neighbour_indices, neighbour_distances)}
            self._neighbour_cache_key = self._cache_key(k)
            self._neighbour_cache_lease = lease
            self.neighbour_cache_bytes = entry_bytes
            self.cache_peak_bytes = max(self.cache_peak_bytes, entry_bytes)
            logger.info(
                "Admitted optional exact neighbour cache: k=%d, cache_bytes=%d, "
                "cache_budget_bytes=%d",
                k,
                entry_bytes,
                self.max_neighbour_cache_bytes,
            )

    def _cached_neighbour_prefix(self, k: int) -> tuple[np.ndarray, np.ndarray] | None:
        with self._cache_lock:
            for cached_k, cached in self._neighbour_cache.items():
                if cached_k >= k and self._neighbour_cache_key == self._cache_key(cached_k):
                    self.cache_hit += 1
                    self.cache_time_saved_seconds += self._last_neighbour_compute_seconds
                    logger.info(
                        "Exact neighbour cache hit: requested_k=%d, cached_k=%d, "
                        "time_saved_seconds=%.6f",
                        k,
                        cached_k,
                        self._last_neighbour_compute_seconds,
                    )
                    return cached[0][:, :k], cached[1][:, :k]
            self.cache_miss += 1
            logger.info("Exact neighbour cache miss: requested_k=%d", k)
            return None

    @staticmethod
    def _euclidean_relative_error_bound(
        query: np.ndarray,
        target: np.ndarray,
        *,
        dimension: int,
    ) -> float | None:
        """Return a certified relative error bound for one FP64 norm.

        For one distance evaluation, ``3*d`` covers the componentwise
        subtraction, square, and conservative sequential accumulation.  The
        fixed allowance covers the square root, scalar result handling, and
        the cKDTree scalar comparison/return path.
        This same bound is applied independently to the canonical NumPy
        distance and to the cKDTree-reported distance; the ``eps=0`` tree
        contract supplies the ordering guarantee for locations omitted from
        the returned prefix.  The bound is evaluated as ``gamma_n`` rather
        than by subtracting two nearly equal powers.  ``nextafter`` is used
        only to round an already-positive bound outward.  Any non-finite,
        subnormal, underflow-prone, or overflow-prone operation is rejected
        so the caller must use complete exact radius verification.
        """

        if dimension < 1 or query.shape != target.shape or query.size != dimension:
            return None
        with np.errstate(over="raise", under="raise", invalid="raise"):
            try:
                delta = np.asarray(target, dtype=np.float64) - np.asarray(query, dtype=np.float64)
                squared = delta * delta
                squared_sum = np.sum(squared, dtype=np.float64)
            except FloatingPointError:
                return None
        if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(squared)):
            return None
        tiny = np.finfo(np.float64).tiny
        nonzero = delta != 0.0
        if np.any(nonzero & (np.abs(delta) < np.sqrt(tiny))):
            return None
        if np.any((squared != 0.0) & (squared < tiny)):
            return None
        if not np.isfinite(squared_sum) or float(squared_sum) <= 0.0:
            return None
        unit_roundoff = np.float64(np.finfo(np.float64).eps / 2.0)
        operation_count = 3 * int(dimension) + 8
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            operation_product = np.float64(operation_count) * unit_roundoff
        if (
            not np.isfinite(operation_product)
            or operation_product <= 0.0
            or operation_product >= 1.0
        ):
            return None
        # Higham's gamma_n = n*u/(1-n*u) is stable here because the numerator
        # and denominator are formed directly; no (1+u)**n - 1 cancellation is
        # involved.  The final nextafter only makes this valid bound outward.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            relative = operation_product / (np.float64(1.0) - operation_product)
        if not np.isfinite(relative) or relative <= 0.0 or relative >= 1.0:
            return None
        return float(np.nextafter(relative, np.inf))

    @staticmethod
    def _distance_interval(value: float, relative_error: float) -> tuple[float, float] | None:
        measured = float(value)
        if (
            not np.isfinite(measured)
            or measured <= 0.0
            or not np.isfinite(relative_error)
            or relative_error < 0.0
            or relative_error >= 1.0
        ):
            return None
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            lower = np.float64(measured) / np.float64(1.0 + relative_error)
            upper = np.float64(measured) / np.float64(1.0 - relative_error)
        if not np.isfinite(lower) or not np.isfinite(upper):
            return None
        return (
            float(np.nextafter(np.float64(lower), np.float64(-np.inf))),
            float(np.nextafter(np.float64(upper), np.float64(np.inf))),
        )

    @classmethod
    def _certified_knn_boundary(
        cls,
        query: np.ndarray,
        kth_target: np.ndarray,
        excluded_target: np.ndarray,
        tree_kth_target: np.ndarray,
        tree_excluded_target: np.ndarray,
        kth_distance: float,
        excluded_distance: float,
        tree_kth_distance: float,
        tree_excluded_distance: float,
        *,
        dimension: int,
    ) -> bool:
        """Certify a kNN prefix using outward-rounded FP64 distance bounds.

        ``cKDTree.query(eps=0)`` promises an exact Euclidean nearest-neighbour
        result.  Its returned boundary is nevertheless checked independently:
        the returned tree distances and the canonical NumPy distances must
        each have a non-overlapping k-th/excluded interval under the forward
        error bound above.  Thus a numerically reordered tree boundary, a
        canonical near-tie, and every underflow/overflow trap are rejected and
        routed through the complete radius enumeration path.
        """

        kth_error = cls._euclidean_relative_error_bound(
            query,
            kth_target,
            dimension=dimension,
        )
        excluded_error = cls._euclidean_relative_error_bound(
            query,
            excluded_target,
            dimension=dimension,
        )
        tree_kth_error = cls._euclidean_relative_error_bound(
            query,
            tree_kth_target,
            dimension=dimension,
        )
        tree_excluded_error = cls._euclidean_relative_error_bound(
            query,
            tree_excluded_target,
            dimension=dimension,
        )
        if (
            kth_error is None
            or excluded_error is None
            or tree_kth_error is None
            or tree_excluded_error is None
        ):
            return False
        canonical_kth = cls._distance_interval(kth_distance, kth_error)
        canonical_excluded = cls._distance_interval(excluded_distance, excluded_error)
        tree_kth = cls._distance_interval(tree_kth_distance, tree_kth_error)
        tree_excluded = cls._distance_interval(tree_excluded_distance, tree_excluded_error)
        if (
            canonical_kth is None
            or canonical_excluded is None
            or tree_kth is None
            or tree_excluded is None
        ):
            return False
        return bool(canonical_kth[1] < canonical_excluded[0] and tree_kth[1] < tree_excluded[0])

    def _compute_location_neighbours(
        self,
        k: int,
        *,
        chunk_size: int = 1024,
        checkpoint_store: VersionedCheckpointStore | IncrementalCheckpointStore | None = None,
        checkpoint_name: str = "knn",
        checkpoint_identity: dict[str, Any] | None = None,
        checkpoint_every_locations: int = 1,
        resume: bool = True,
        preserve_checkpoint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute one canonical widest result and derive narrower prefixes."""

        order_k = _validate_k(k)
        block_size = _validate_chunk_size(chunk_size)
        cached = self._cached_neighbour_prefix(order_k)
        if cached is not None:
            return cached
        n_locations = self.unique_locations.shape[0]
        if n_locations - 1 < order_k:
            raise ValueError(
                f"k={order_k} requires at least {order_k} other distinct descriptor locations; "
                f"the pool contains only {n_locations - 1}"
            )
        started = time.perf_counter()
        logger.info(
            "Indexed exact neighbour search started: N=%d, distinct locations=%d, "
            "k_max=%d, workers=%d, chunk_size=%d",
            self.descriptors.shape[0],
            n_locations,
            order_k,
            self.worker_count,
            min(block_size, n_locations),
        )
        # The cKDTree calls are vectorized by deterministic location chunk.  A
        # single call lets SciPy use its bounded internal worker pool; all
        # exact distance/tie work below is merged in location order.
        chunk_size = min(block_size, n_locations)
        neighbour_indices = np.empty((n_locations, order_k), dtype=np.int64)
        neighbour_distances = np.empty((n_locations, order_k), dtype=np.float64)
        source_start = 0
        incremental_store: IncrementalCheckpointStore | None = None
        legacy_store: VersionedCheckpointStore | None = None
        loaded_legacy = False
        if checkpoint_store is not None:
            if checkpoint_identity is None:
                raise ValueError("kNN checkpoint identity is required")
            if (
                isinstance(checkpoint_every_locations, bool)
                or not isinstance(checkpoint_every_locations, (int, np.integer))
                or int(checkpoint_every_locations) < 1
            ):
                raise ValueError("checkpoint_every_locations must be a positive integer")
            if isinstance(checkpoint_store, IncrementalCheckpointStore):
                incremental_store = checkpoint_store
                legacy_store = VersionedCheckpointStore(checkpoint_store.root)
            elif isinstance(checkpoint_store, VersionedCheckpointStore):
                incremental_store = IncrementalCheckpointStore(
                    checkpoint_store.root,
                    resource_budget=self.resource_budget,
                )
                legacy_store = checkpoint_store
            else:
                raise TypeError("checkpoint_store must be a supported checkpoint store")
            if not resume:
                incremental_store.clear(checkpoint_name)
                legacy_store.clear(checkpoint_name)
            else:
                # Remove only unreferenced partial chunk/manifest artifacts
                # left by a crash before reading the authoritative pointer.
                incremental_store.clean_stale()
            if resume and incremental_store.exists(checkpoint_name):
                checkpoint_started = time.perf_counter()
                _identity, checkpoint_state, checkpoint_arrays = incremental_store.load(
                    checkpoint_name,
                    expected_identity=checkpoint_identity,
                    k_max=order_k,
                    location_count=n_locations,
                )
                source_start = int(checkpoint_state.get("cursor", -1))
                saved_neighbours = checkpoint_arrays.get("neighbour_locations")
                saved_distances = checkpoint_arrays.get("neighbour_distances")
                self._record_phase(
                    "checkpoint_read",
                    time.perf_counter() - checkpoint_started,
                    count=int(checkpoint_state.get("chunk_count", 0)),
                    bytes_written=int(incremental_store.metrics().get("bytes_read", 0)),
                )
                self._phase_bytes["checkpoint_retained"] = int(
                    incremental_store.metrics().get("retained_bytes", 0)
                )
            elif resume and legacy_store.exists(checkpoint_name):
                _identity, checkpoint_state, checkpoint_arrays = legacy_store.load(
                    checkpoint_name,
                    expected_identity=checkpoint_identity,
                )
                loaded_legacy = True
                source_start = int(checkpoint_state.get("cursor", -1))
                saved_neighbours = checkpoint_arrays.get("neighbour_locations")
                saved_distances = checkpoint_arrays.get("neighbour_distances")
                if (
                    source_start < 1
                    or source_start > n_locations
                    or saved_neighbours is None
                    or saved_distances is None
                    or saved_neighbours.shape != (source_start, order_k)
                    or saved_distances.shape != (source_start, order_k)
                    or saved_neighbours.dtype != np.dtype(np.int64)
                    or saved_distances.dtype != np.dtype(np.float64)
                ):
                    raise CheckpointValidationError(
                        "kNN checkpoint cursor or arrays are incompatible"
                    )
                neighbour_indices[:source_start] = saved_neighbours
                neighbour_distances[:source_start] = saved_distances
            if resume and incremental_store.exists(checkpoint_name):
                if (
                    source_start < 0
                    or source_start > n_locations
                    or saved_neighbours is None
                    or saved_distances is None
                    or saved_neighbours.shape != (source_start, order_k)
                    or saved_distances.shape != (source_start, order_k)
                ):
                    raise CheckpointValidationError(
                        "kNN checkpoint cursor or arrays are incompatible"
                    )
                neighbour_indices[:source_start] = saved_neighbours
                neighbour_distances[:source_start] = saved_distances
            if loaded_legacy:
                # Migrate a verified v1 full-prefix checkpoint once.  The
                # old generation remains authoritative until this new first
                # incremental manifest is promoted successfully.
                assert incremental_store is not None
                migration_before = incremental_store.metrics()
                migration = incremental_store.write_chunk(
                    checkpoint_name,
                    identity=checkpoint_identity,
                    start=0,
                    neighbour_locations=neighbour_indices[:source_start],
                    neighbour_distances=neighbour_distances[:source_start],
                    k_max=order_k,
                    location_count=n_locations,
                )
                migration_after = incremental_store.metrics()
                self._record_phase(
                    "checkpoint_serialization",
                    float(migration.get("write_seconds", 0.0)),
                    bytes_written=int(migration.get("bytes_written", 0)),
                )
                self._record_phase(
                    "checkpoint_fsync",
                    float(migration_after.get("fsync_seconds", 0.0))
                    - float(migration_before.get("fsync_seconds", 0.0)),
                )
                self._phase_bytes["checkpoint_retained"] = int(
                    migration_after.get("retained_bytes", 0)
                )
        next_percent = 1
        last_log = started
        start = source_start
        checkpoint_cursor = source_start

        def finish_chunk(completed_stop: int) -> None:
            nonlocal checkpoint_cursor, last_log, next_percent
            if (
                checkpoint_store is not None
                and checkpoint_identity is not None
                and (
                    completed_stop - checkpoint_cursor >= int(checkpoint_every_locations)
                    or completed_stop == n_locations
                )
            ):
                checkpoint_started = time.perf_counter()
                assert incremental_store is not None
                metrics_before = incremental_store.metrics()
                write_metrics = incremental_store.write_chunk(
                    checkpoint_name,
                    identity=checkpoint_identity,
                    start=checkpoint_cursor,
                    neighbour_locations=neighbour_indices[checkpoint_cursor:completed_stop],
                    neighbour_distances=neighbour_distances[checkpoint_cursor:completed_stop],
                    k_max=order_k,
                    location_count=n_locations,
                )
                checkpoint_cursor = completed_stop
                metrics_after = incremental_store.metrics()
                self._record_phase(
                    "checkpoint_serialization",
                    float(
                        write_metrics.get("write_seconds", time.perf_counter() - checkpoint_started)
                    ),
                    bytes_written=int(
                        write_metrics.get(
                            "physical_bytes_written",
                            write_metrics.get("bytes_written", 0),
                        )
                    ),
                )
                self._record_phase(
                    "checkpoint_fsync",
                    float(metrics_after.get("fsync_seconds", 0.0))
                    - float(metrics_before.get("fsync_seconds", 0.0)),
                )
                self._record_phase(
                    "checkpoint_digest",
                    float(metrics_after.get("digest_seconds", 0.0))
                    - float(metrics_before.get("digest_seconds", 0.0)),
                )
                self._phase_bytes["checkpoint_retained"] = int(
                    metrics_after.get("retained_bytes", 0)
                )

            now = time.perf_counter()
            if (
                n_locations < 100
                or completed_stop * 100 >= next_percent * n_locations
                or now - last_log >= 10.0
                or completed_stop == n_locations
            ):
                elapsed = max(0.0, now - started)
                rate = completed_stop / elapsed if elapsed > 0.0 else 0.0
                eta = (n_locations - completed_stop) / rate if rate > 0.0 else 0.0
                process_metrics = _process_memory_metrics()
                logger.info(
                    "Indexed exact neighbour search progress: %d/%d (%.1f%%), "
                    "throughput=%.2f locations/s, elapsed=%.3fs, estimated remaining=%.3fs, "
                    "k_max=%d, index_queries=%d, radius_queries=%d, "
                    "unique_distance_evaluations=%d, index_bytes=%d, "
                    "query_workspace_peak_bytes=%d, process_peak_rss_bytes=%s",
                    completed_stop,
                    n_locations,
                    100.0 * completed_stop / n_locations,
                    rate,
                    elapsed,
                    max(0.0, eta),
                    order_k,
                    self.index_queries,
                    self.radius_queries,
                    self.unique_distance_evaluations,
                    self.index_bytes,
                    self.query_workspace_peak_bytes,
                    process_metrics.get("process_peak_rss_bytes"),
                )
                last_log = now
                while next_percent <= 100 and completed_stop * 100 >= next_percent * n_locations:
                    next_percent += 1

        while start < n_locations:
            stop = min(start + chunk_size, n_locations)
            queries = self.unique_locations[start:stop]
            query_count = stop - start
            tree_workspace = int(
                queries.nbytes
                + 16 * query_count * min(n_locations, order_k + 2)
                + 8 * query_count * order_k
                + 16 * query_count * min(n_locations, order_k + 2)
                + 24 * query_count
                + 8192
            )
            self.query_workspace_peak_bytes = max(
                self.query_workspace_peak_bytes,
                tree_workspace,
            )
            if tree_workspace > self.max_radius_query_bytes:
                if query_count <= 1:
                    raise self._capacity_error(
                        source_index=start,
                        radius=None,
                        requested_bytes=tree_workspace,
                        context=f"k={order_k}, location_chunk=[{start}:{stop})",
                    )
                chunk_size = max(1, query_count // 2)
                logger.info(
                    "Indexed exact kNN chunk reduced before tree allocation: "
                    "location_start=%d, new_chunk_size=%d",
                    start,
                    chunk_size,
                )
                continue
            tree_lease: Any | None = None
            try:
                tree_started = time.perf_counter()
                tree_lease = self.resource_budget.acquire(
                    f"indexed kNN tree candidate tile [{start}:{stop})",
                    max(1, tree_workspace),
                )
                tree_query_count = min(n_locations, order_k + 2)
                # eps=0 is part of the certification proof: cKDTree's exact
                # query contract makes the first excluded location a boundary
                # for every omitted location.  The returned scalar distances
                # are still checked with the independent FP64 intervals below.
                _tree_distances, tree_indices = self._tree.query(
                    queries,
                    k=tree_query_count,
                    eps=0.0,
                    workers=self.worker_count,
                )
                self._record_phase("knn_tree_query", time.perf_counter() - tree_started)
                self.index_queries += stop - start
                tree_distance_matrix = np.asarray(_tree_distances, dtype=np.float64)
                if tree_distance_matrix.ndim == 1:
                    tree_distance_matrix = tree_distance_matrix.reshape(
                        query_count, tree_query_count
                    )
                del _tree_distances
                candidate_matrix = np.asarray(tree_indices, dtype=np.int64)
                if candidate_matrix.ndim == 1:
                    candidate_matrix = candidate_matrix.reshape(query_count, tree_query_count)
                del tree_indices
                initial_candidates = np.empty((query_count, order_k), dtype=np.int64)
                excluded_candidates = np.empty((query_count, 1), dtype=np.int64)
                tree_kth_distances = np.empty(query_count, dtype=np.float64)
                tree_excluded_distances = np.empty(query_count, dtype=np.float64)
                has_excluded_candidate = np.ones(query_count, dtype=bool)
                for offset, query_location in enumerate(range(start, stop)):
                    candidates = candidate_matrix[offset]
                    valid = (
                        (candidates >= 0)
                        & (candidates < n_locations)
                        & (candidates != query_location)
                    )
                    candidates = candidates[valid]
                    reported_distances = tree_distance_matrix[offset][valid]
                    if candidates.shape[0] < order_k:
                        raise ValueError(
                            f"k={order_k} has insufficient distinct neighbours for location "
                            f"{query_location}"
                        )
                    initial_candidates[offset] = candidates[:order_k]
                    tree_kth_distances[offset] = reported_distances[order_k - 1]
                    if candidates.shape[0] > order_k:
                        excluded_candidates[offset, 0] = candidates[order_k]
                        tree_excluded_distances[offset] = reported_distances[order_k]
                    else:
                        # There is no excluded location when the pool has
                        # exactly k other locations.  Keep the matrix shape
                        # bounded and force the conservative guard to fall
                        # back to complete radius verification.
                        excluded_candidates[offset, 0] = query_location
                        tree_excluded_distances[offset] = 0.0
                        has_excluded_candidate[offset] = False
                verification_started = time.perf_counter()
                initial_fixed, initial_tile, initial_rows = self._matrix_distance_workspace(
                    queries,
                    initial_candidates,
                    max_workspace_bytes=self.max_radius_query_bytes,
                    persistent_workspace_bytes=(
                        candidate_matrix.nbytes
                        + initial_candidates.nbytes
                        + excluded_candidates.nbytes
                        + tree_distance_matrix.nbytes
                        + tree_kth_distances.nbytes
                        + tree_excluded_distances.nbytes
                    ),
                )
                if initial_rows < 1:
                    raise RadiusQueryCapacityError(
                        "indexed kNN canonical distance workspace exceeds max_radius_query_bytes"
                    )
                initial_workspace = int(initial_fixed + initial_tile * initial_rows)
                self.query_workspace_peak_bytes = max(
                    self.query_workspace_peak_bytes,
                    initial_workspace,
                )
                tree_lease.resize(max(1, initial_workspace))
                initial_distances = self._distance_values_matrix(
                    queries,
                    initial_candidates,
                    max_workspace_bytes=self.max_radius_query_bytes,
                    persistent_workspace_bytes=(
                        candidate_matrix.nbytes
                        + initial_candidates.nbytes
                        + excluded_candidates.nbytes
                        + tree_distance_matrix.nbytes
                        + tree_kth_distances.nbytes
                        + tree_excluded_distances.nbytes
                    ),
                )
                excluded_fixed, excluded_tile, excluded_rows = self._matrix_distance_workspace(
                    queries,
                    excluded_candidates,
                    max_workspace_bytes=self.max_radius_query_bytes,
                    persistent_workspace_bytes=(
                        candidate_matrix.nbytes
                        + initial_candidates.nbytes
                        + excluded_candidates.nbytes
                        + tree_distance_matrix.nbytes
                        + tree_kth_distances.nbytes
                        + tree_excluded_distances.nbytes
                        + initial_distances.nbytes
                    ),
                )
                if excluded_rows < 1:
                    raise RadiusQueryCapacityError(
                        "indexed kNN boundary certification workspace exceeds "
                        "max_radius_query_bytes"
                    )
                excluded_workspace = int(excluded_fixed + excluded_tile * excluded_rows)
                self.query_workspace_peak_bytes = max(
                    self.query_workspace_peak_bytes,
                    excluded_workspace,
                )
                tree_lease.resize(max(1, initial_workspace, excluded_workspace))
                excluded_distances = self._distance_values_matrix(
                    queries,
                    excluded_candidates,
                    max_workspace_bytes=self.max_radius_query_bytes,
                    persistent_workspace_bytes=(
                        candidate_matrix.nbytes
                        + initial_candidates.nbytes
                        + excluded_candidates.nbytes
                        + tree_distance_matrix.nbytes
                        + tree_kth_distances.nbytes
                        + tree_excluded_distances.nbytes
                        + initial_distances.nbytes
                    ),
                )
            except (ResourceCapacityError, RadiusQueryCapacityError):
                if query_count <= 1:
                    raise
                chunk_size = max(1, query_count // 2)
                logger.info(
                    "Indexed exact kNN chunk reduced for canonical distance memory budget: "
                    "location_start=%d, new_chunk_size=%d",
                    start,
                    chunk_size,
                )
                continue
            finally:
                if tree_lease is not None:
                    tree_lease.close()
            self.distance_evaluations += int(initial_distances.size)
            self.unique_distance_evaluations += int(initial_distances.size)
            self.distance_evaluations += int(excluded_distances.size)
            self.unique_distance_evaluations += int(excluded_distances.size)
            self._record_phase(
                "canonical_distance_verification",
                time.perf_counter() - verification_started,
                count=stop - start,
            )

            certification_started = time.perf_counter()
            chunk_certified = True
            boundary_certified = np.ones(query_count, dtype=bool)
            sorted_candidates: list[np.ndarray] = []
            sorted_distances: list[np.ndarray] = []
            for offset in range(query_count):
                candidates = initial_candidates[offset]
                distances = initial_distances[offset]
                order = np.lexsort(
                    (
                        self.location_representatives[candidates],
                        distances,
                    )
                )
                ordered_distances = distances[order]
                sorted_candidates.append(candidates[order])
                sorted_distances.append(ordered_distances)
                if (
                    not self._certified_knn_boundary(
                        queries[offset],
                        self.unique_locations[sorted_candidates[offset][-1]],
                        self.unique_locations[excluded_candidates[offset, 0]],
                        self.unique_locations[initial_candidates[offset, -1]],
                        self.unique_locations[excluded_candidates[offset, 0]],
                        ordered_distances[-1],
                        excluded_distances[offset, 0],
                        tree_kth_distances[offset],
                        tree_excluded_distances[offset],
                        dimension=int(self.unique_locations.shape[1]),
                    )
                    or not has_excluded_candidate[offset]
                ):
                    boundary_certified[offset] = False
                    chunk_certified = False
            self._record_phase(
                "knn_boundary_certification",
                time.perf_counter() - certification_started,
                count=query_count,
            )
            if chunk_certified:
                for offset, query_location in enumerate(range(start, stop)):
                    neighbour_indices[query_location] = sorted_candidates[offset]
                    neighbour_distances[query_location] = sorted_distances[offset]
                self.knn_fast_path_certified += query_count
                del (
                    candidate_matrix,
                    initial_candidates,
                    initial_distances,
                    excluded_candidates,
                    excluded_distances,
                    tree_distance_matrix,
                    tree_kth_distances,
                    tree_excluded_distances,
                )
                finish_chunk(stop)
                start = stop
                continue
            self.knn_fast_path_ambiguous += int(np.count_nonzero(~boundary_certified))
            self.knn_fast_path_fallback += query_count

            boundary_prepare_workspace = int(initial_distances.nbytes + 24 * query_count + 1024)
            self.query_workspace_peak_bytes = max(
                self.query_workspace_peak_bytes,
                boundary_prepare_workspace,
            )
            try:
                with self.resource_budget.acquire(
                    f"indexed kNN boundary preparation [{start}:{stop})",
                    max(1, boundary_prepare_workspace),
                ):
                    boundaries = np.max(initial_distances, axis=1)
                    expanded = boundaries + np.maximum(
                        16.0 * np.spacing(boundaries),
                        1.0e-14,
                    )
            except ResourceCapacityError:
                if query_count <= 1:
                    raise
                chunk_size = max(1, query_count // 2)
                logger.info(
                    "Indexed exact kNN chunk reduced before boundary preparation: "
                    "location_start=%d, new_chunk_size=%d",
                    start,
                    chunk_size,
                )
                continue
            # The dense candidate matrix and its canonical distances are no
            # longer needed once the validated boundary is known.  Releasing
            # those references before the ragged support query keeps the
            # admission calculation tied to the arrays that are actually
            # live during the next vectorised operation.
            del (
                candidate_matrix,
                initial_candidates,
                initial_distances,
                excluded_candidates,
                excluded_distances,
                tree_distance_matrix,
                tree_kth_distances,
                tree_excluded_distances,
                sorted_candidates,
                sorted_distances,
            )
            preflight_started = time.perf_counter()
            preflight_call_workspace = int(
                8 * self.unique_locations.shape[1] * (stop - start) + 32 * (stop - start) + 8 + 1024
            )
            try:
                with self.resource_budget.acquire(
                    f"indexed kNN radius preflight [{start}:{stop})",
                    max(1, preflight_call_workspace),
                ):
                    expanded = boundaries + np.maximum(
                        16.0 * np.spacing(boundaries),
                        1.0e-14,
                    )
                    candidate_counts = np.atleast_1d(
                        np.asarray(
                            self._tree.query_ball_point(
                                queries,
                                expanded,
                                return_length=True,
                                workers=self.worker_count,
                            ),
                            dtype=np.int64,
                        )
                    )
            except TypeError as exc:
                raise self._capacity_error(
                    source_index=None,
                    radius=float(np.max(boundaries)),
                    requested_bytes=self.max_radius_query_bytes + 1,
                    context=(
                        f"k={order_k}, location_chunk=[{start}:{stop}), "
                        "SciPy cKDTree does not support vectorized preflight"
                    ),
                ) from exc
            except ResourceCapacityError:
                if stop - start <= 1:
                    raise
                chunk_size = max(1, (stop - start) // 2)
                logger.info(
                    "Indexed exact kNN chunk reduced before radius preflight allocation: "
                    "location_start=%d, new_chunk_size=%d",
                    start,
                    chunk_size,
                )
                continue
            self.radius_query_preflights += stop - start
            self._record_phase(
                "radius_preflight_count",
                time.perf_counter() - preflight_started,
                count=stop - start,
            )
            estimates = np.asarray(
                [
                    _estimate_radius_workspace_bytes(
                        int(count),
                        0,
                        self.descriptors.shape[1],
                    )
                    for count in candidate_counts
                ],
                dtype=np.int64,
            )
            # The tree returns a ragged object, but the subsequent canonical
            # verification also retains one int64 candidate array per source,
            # the expanded radii, CSR offsets and one full distance vector.
            # Admit those arrays before asking cKDTree for the ragged lists.
            preflight_workspace = int(
                np.sum(estimates, dtype=np.int64)
                + queries.nbytes
                + expanded.nbytes
                + 8 * (stop - start + 1)
                + 8 * int(np.sum(candidate_counts, dtype=np.int64))
                + 1024
            )
            if np.any(estimates > self.max_radius_query_bytes):
                offending = int(np.flatnonzero(estimates > self.max_radius_query_bytes)[0])
                raise self._capacity_error(
                    source_index=start + offending,
                    radius=float(boundaries[offending]),
                    requested_bytes=int(estimates[offending]),
                    context=f"k={order_k}, location_chunk=[{start}:{stop})",
                )
            total_workspace = max(
                preflight_workspace,
                int(np.sum(estimates, dtype=np.int64)),
            )
            self.query_workspace_peak_bytes = max(
                self.query_workspace_peak_bytes,
                preflight_workspace,
                int(np.max(estimates)),
            )
            retrieval_started = time.perf_counter()
            try:
                with self.resource_budget.acquire(
                    f"indexed kNN location chunk [{start}:{stop})",
                    max(1, total_workspace),
                ):
                    boundary_lists = np.atleast_1d(
                        self._tree.query_ball_point(
                            queries,
                            expanded,
                            workers=self.worker_count,
                        )
                    )
            except ResourceCapacityError:
                if stop - start <= 1:
                    raise
                chunk_size = max(1, (stop - start) // 2)
                logger.info(
                    "Indexed exact neighbour chunk reduced for memory budget: "
                    "location_start=%d, new_chunk_size=%d, requested_workspace_bytes=%d",
                    start,
                    chunk_size,
                    total_workspace,
                )
                continue
            self._record_phase(
                "radius_retrieval",
                time.perf_counter() - retrieval_started,
                count=stop - start,
            )
            self.radius_queries += stop - start
            verification_started = time.perf_counter()
            try:
                with self.resource_budget.acquire(
                    f"indexed kNN boundary candidate materialisation [{start}:{stop})",
                    max(1, preflight_workspace),
                ):
                    boundary_candidates = []
                    for offset, query_location in enumerate(range(start, stop)):
                        boundary_indices = np.asarray(boundary_lists[offset], dtype=np.int64)
                        boundary_candidates.append(
                            boundary_indices[boundary_indices != query_location]
                        )
            except ResourceCapacityError:
                if stop - start <= 1:
                    raise
                chunk_size = max(1, (stop - start) // 2)
                logger.info(
                    "Indexed exact kNN chunk reduced before boundary candidate allocation: "
                    "location_start=%d, new_chunk_size=%d",
                    start,
                    chunk_size,
                )
                continue
            boundary_storage_bytes = int(
                sum(int(candidate.nbytes) for candidate in boundary_candidates)
            )
            boundary_persistent_bytes = int(
                boundary_storage_bytes
                + 8 * int(np.sum(candidate_counts, dtype=np.int64))
                + expanded.nbytes
                + 8 * (stop - start + 1)
                + 1024
            )
            boundary_fixed, boundary_tile = self._distance_values_many_workspace(
                queries,
                boundary_candidates,
                persistent_workspace_bytes=boundary_persistent_bytes,
            )
            boundary_total = int(sum(int(candidate.size) for candidate in boundary_candidates))
            boundary_available = self.max_radius_query_bytes - boundary_fixed
            if boundary_available < boundary_tile:
                if stop - start <= 1:
                    raise self._capacity_error(
                        source_index=start,
                        radius=float(boundaries[0]),
                        requested_bytes=boundary_fixed + boundary_tile,
                        context=f"k={order_k}, location_chunk=[{start}:{stop})",
                    )
                chunk_size = max(1, (stop - start) // 2)
                logger.info(
                    "Indexed exact kNN chunk reduced for boundary distance memory budget: "
                    "location_start=%d, new_chunk_size=%d",
                    start,
                    chunk_size,
                )
                continue
            boundary_tile_count = min(
                boundary_total,
                max(1, boundary_available // boundary_tile),
            )
            boundary_distance_workspace = int(boundary_fixed + boundary_tile_count * boundary_tile)
            boundary_sort_workspace = int(
                max(
                    (
                        _estimate_radius_workspace_bytes(
                            int(count),
                            0,
                            self.descriptors.shape[1],
                        )
                        for count in candidate_counts
                    ),
                    default=0,
                )
                + queries.nbytes
                + expanded.nbytes
                + boundary_storage_bytes
                + 8 * (stop - start + 1)
                + 1024
            )
            boundary_workspace = max(boundary_distance_workspace, boundary_sort_workspace)
            self.query_workspace_peak_bytes = max(
                self.query_workspace_peak_bytes,
                boundary_workspace,
            )
            if boundary_workspace > self.max_radius_query_bytes:
                if stop - start <= 1:
                    raise self._capacity_error(
                        source_index=start,
                        radius=float(boundaries[0]),
                        requested_bytes=boundary_workspace,
                        context=f"k={order_k}, location_chunk=[{start}:{stop})",
                    )
                chunk_size = max(1, (stop - start) // 2)
                logger.info(
                    "Indexed exact kNN chunk reduced for boundary sorting memory budget: "
                    "location_start=%d, new_chunk_size=%d",
                    start,
                    chunk_size,
                )
                continue
            try:
                with self.resource_budget.acquire(
                    f"indexed kNN canonical boundary distance tile [{start}:{stop})",
                    max(1, boundary_workspace),
                ):
                    boundary_offsets, boundary_distances = self._distance_values_many(
                        queries,
                        boundary_candidates,
                        max_workspace_bytes=self.max_radius_query_bytes,
                        persistent_workspace_bytes=boundary_persistent_bytes,
                    )
                    self.distance_evaluations += int(boundary_distances.size)
                    self.unique_distance_evaluations += int(boundary_distances.size)
                    for offset, query_location in enumerate(range(start, stop)):
                        boundary_indices = boundary_candidates[offset]
                        distances = boundary_distances[
                            boundary_offsets[offset] : boundary_offsets[offset + 1]
                        ]
                        positive = distances > 0.0
                        boundary_indices = boundary_indices[positive]
                        distances = distances[positive]
                        if distances.size < order_k:
                            raise ValueError(
                                "indexed neighbour boundary query returned fewer candidates "
                                "than the "
                                f"validated kth distance for location {query_location}"
                            )
                        order = np.lexsort(
                            (
                                self.location_representatives[boundary_indices],
                                distances,
                            )
                        )[:order_k]
                        neighbour_indices[query_location] = boundary_indices[order]
                        neighbour_distances[query_location] = distances[order]
            except (ResourceCapacityError, RadiusQueryCapacityError):
                if stop - start <= 1:
                    raise
                chunk_size = max(1, (stop - start) // 2)
                logger.info(
                    "Indexed exact kNN chunk reduced for boundary distance admission: "
                    "location_start=%d, new_chunk_size=%d",
                    start,
                    chunk_size,
                )
                continue
            self._record_phase(
                "canonical_distance_verification",
                time.perf_counter() - verification_started,
                count=stop - start,
            )

            finish_chunk(stop)
            start = stop

        if checkpoint_store is not None and not preserve_checkpoint:
            assert incremental_store is not None and legacy_store is not None
            incremental_store.clear(checkpoint_name)
            incremental_store.clean_stale()
            legacy_store.clear(checkpoint_name)
            legacy_store.clean_stale()
        neighbour_indices.setflags(write=False)
        neighbour_distances.setflags(write=False)
        self._last_neighbour_compute_seconds = max(0.0, time.perf_counter() - started)
        self._store_neighbour_cache(order_k, neighbour_indices, neighbour_distances)
        phase_metrics = self.memory_metrics()
        logger.info(
            "Indexed exact neighbour search completed: %d/%d (100.0%%), k_max=%d, "
            "elapsed=%.3fs, tree_query=%.6fs, preflight=%.6fs, radius_retrieval=%.6fs, "
            "canonical_verification=%.6fs, checkpoint_fsync=%.6fs, checkpoint_bytes=%d",
            n_locations,
            n_locations,
            order_k,
            self._last_neighbour_compute_seconds,
            float(phase_metrics.get("knn_tree_query_seconds", 0.0) or 0.0),
            float(phase_metrics.get("radius_preflight_count_seconds", 0.0) or 0.0),
            float(phase_metrics.get("radius_retrieval_seconds", 0.0) or 0.0),
            float(phase_metrics.get("canonical_distance_verification_seconds", 0.0) or 0.0),
            float(phase_metrics.get("checkpoint_fsync_seconds", 0.0) or 0.0),
            int(phase_metrics.get("checkpoint_serialization_bytes", 0) or 0),
        )
        return neighbour_indices, neighbour_distances

    def query_neighbours(
        self,
        k: int,
        *,
        chunk_size: int = 1024,
        checkpoint_store: VersionedCheckpointStore | IncrementalCheckpointStore | None = None,
        checkpoint_name: str = "knn",
        checkpoint_identity: dict[str, Any] | None = None,
        checkpoint_every_locations: int = 1,
        resume: bool = True,
        preserve_checkpoint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return radii, representative rows, and distances for all rows."""

        order_k = _validate_k(k)
        location_indices, location_distances = self._compute_location_neighbours(
            order_k,
            chunk_size=chunk_size,
            checkpoint_store=checkpoint_store,
            checkpoint_name=checkpoint_name,
            checkpoint_identity=checkpoint_identity,
            checkpoint_every_locations=checkpoint_every_locations,
            resume=resume,
            preserve_checkpoint=preserve_checkpoint,
        )
        row_locations = self.row_to_location
        rows = np.asarray(
            self.location_representatives[location_indices[row_locations]], dtype=np.int64
        ).copy()
        distances = np.asarray(location_distances[row_locations], dtype=np.float64).copy()
        radii = distances[:, -1].copy()
        return radii, rows, distances

    def query_neighbours_many(
        self,
        ks: Sequence[int],
        *,
        chunk_size: int = 1024,
        checkpoint_store: VersionedCheckpointStore | IncrementalCheckpointStore | None = None,
        checkpoint_name: str = "knn",
        checkpoint_identity: dict[str, Any] | None = None,
        checkpoint_every_locations: int = 1,
        resume: bool = True,
        preserve_checkpoint: bool = False,
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return exact neighbour prefixes from one canonical ``k_max`` pass."""

        query_many_started = time.perf_counter()
        ordered = tuple(_validate_k(value) for value in ks)
        if not ordered:
            raise ValueError("ks must contain at least one positive integer")
        if len(set(ordered)) != len(ordered):
            raise ValueError("ks must contain unique neighbour orders")
        maximum = max(ordered)
        location_indices, location_distances = self._compute_location_neighbours(
            maximum,
            chunk_size=chunk_size,
            checkpoint_store=checkpoint_store,
            checkpoint_name=checkpoint_name,
            checkpoint_identity=checkpoint_identity,
            checkpoint_every_locations=checkpoint_every_locations,
            resume=resume,
            preserve_checkpoint=preserve_checkpoint,
        )
        row_locations = self.row_to_location
        results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for order_k in ordered:
            prefix_indices = location_indices[:, :order_k]
            prefix_distances = location_distances[:, :order_k]
            rows = np.asarray(
                self.location_representatives[prefix_indices[row_locations]], dtype=np.int64
            ).copy()
            distances = np.asarray(prefix_distances[row_locations], dtype=np.float64).copy()
            results[order_k] = (distances[:, -1].copy(), rows, distances)
        self._record_phase(
            "query_neighbours_many_projection",
            time.perf_counter() - query_many_started,
            count=len(ordered),
        )
        return results

    def radius_support_many(
        self,
        source_indices: Sequence[int] | np.ndarray,
        radii: Sequence[float] | np.ndarray,
        *,
        chunk_size: int = 1024,
        context: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact strict-radius supports in bounded CSR form.

        Tree candidates are admitted in source batches, then canonical
        float64 distances are recomputed over the ragged candidate blocks.
        The CSR offsets preserve source order; each source's expanded atomic
        rows are finally sorted by original row index, matching
        :meth:`radius_support` exactly.
        """

        sources = np.asarray(source_indices)
        if sources.ndim != 1 or sources.size == 0:
            raise ValueError("source_indices must be a non-empty one-dimensional sequence")
        if sources.dtype.kind not in "iu" or sources.dtype.kind == "b":
            raise ValueError("source_indices must contain integer row indices")
        source_rows = np.asarray(sources, dtype=np.int64)
        if np.any(source_rows < 0) or np.any(source_rows >= self.descriptors.shape[0]):
            raise IndexError("source_indices contain a row outside the descriptor rows")
        values = np.asarray(radii, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != source_rows.shape[0]:
            raise ValueError("radii must align with source_indices")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("radii must be finite and positive")
        batch_limit = _validate_chunk_size(chunk_size)
        source_count = int(source_rows.size)
        dimension = int(self.descriptors.shape[1])
        offsets = np.zeros(source_count + 1, dtype=np.int64)
        target_parts: list[np.ndarray] = []
        distance_parts: list[np.ndarray] = []
        source_cursor = 0
        while source_cursor < source_count:
            stop = min(source_cursor + batch_limit, source_count)
            # A failed aggregate preflight is handled by reducing the batch;
            # this keeps a dense support complete without ever allocating a
            # source-by-pool matrix.
            while True:
                batch_sources = source_rows[source_cursor:stop]
                batch_radii = values[source_cursor:stop]
                preflight_call_workspace = int(
                    8 * dimension * batch_sources.size + 32 * batch_sources.size + 8 + 1024
                )
                preflight_started = time.perf_counter()
                try:
                    with self.resource_budget.acquire(
                        context or "indexed exact batched radius preflight",
                        max(1, preflight_call_workspace),
                    ):
                        queries = self.descriptors[batch_sources]
                        expanded = batch_radii + np.maximum(
                            16.0 * np.spacing(batch_radii),
                            1.0e-14,
                        )
                        candidate_counts = np.asarray(
                            self._tree.query_ball_point(
                                queries,
                                expanded,
                                return_length=True,
                                workers=self.worker_count,
                            ),
                            dtype=np.int64,
                        ).reshape(-1)
                except TypeError as exc:
                    raise self._capacity_error(
                        source_index=int(batch_sources[0]),
                        radius=float(batch_radii[0]),
                        requested_bytes=self.max_radius_query_bytes + 1,
                        context=(
                            context or "SciPy cKDTree does not support vectorized radius preflight"
                        ),
                    ) from exc
                except ResourceCapacityError:
                    if stop - source_cursor <= 1:
                        raise
                    batch_limit = max(1, (stop - source_cursor) // 2)
                    continue
                if candidate_counts.shape != batch_sources.shape:
                    raise ValueError("cKDTree radius preflight returned an invalid batch shape")
                self.radius_query_preflights += int(batch_sources.size)
                self._record_phase(
                    "radius_preflight_count",
                    time.perf_counter() - preflight_started,
                    count=int(batch_sources.size),
                )
                preflight_workspace = int(
                    sum(
                        _estimate_radius_workspace_bytes(int(count), 0, dimension)
                        for count in candidate_counts
                    )
                    + queries.nbytes
                    + expanded.nbytes
                    + 8 * (batch_sources.size + 1)
                    + 8 * int(np.sum(candidate_counts, dtype=np.int64))
                    + 16 * int(np.sum(candidate_counts, dtype=np.int64))
                    + 1024
                )
                self.query_workspace_peak_bytes = max(
                    self.query_workspace_peak_bytes,
                    preflight_workspace,
                )
                if preflight_workspace > self.max_radius_query_bytes and stop - source_cursor > 1:
                    stop = source_cursor + max(1, (stop - source_cursor) // 2)
                    continue
                if preflight_workspace > self.max_radius_query_bytes:
                    offending = int(np.flatnonzero(candidate_counts)[0])
                    raise self._capacity_error(
                        source_index=int(batch_sources[offending]),
                        radius=float(batch_radii[offending]),
                        requested_bytes=preflight_workspace,
                        context=context,
                    )
                break

            retrieval_started = time.perf_counter()
            try:
                lease = self.resource_budget.acquire(
                    context or "indexed exact batched radius query",
                    max(1, preflight_workspace),
                )
            except (ResourceCapacityError, RadiusQueryCapacityError):
                if stop - source_cursor <= 1:
                    raise
                batch_limit = max(1, (stop - source_cursor) // 2)
                continue
            try:
                raw_candidates = self._tree.query_ball_point(
                    queries,
                    expanded,
                    workers=self.worker_count,
                )
                candidate_lists = [
                    np.asarray(candidate, dtype=np.int64)
                    for candidate in np.atleast_1d(raw_candidates)
                ]
                if len(candidate_lists) != batch_sources.size:
                    raise ValueError("cKDTree radius query returned an invalid batch shape")
                if any(
                    int(candidate.shape[0]) != int(expected)
                    for candidate, expected in zip(candidate_lists, candidate_counts, strict=True)
                ):
                    raise ValueError("cKDTree radius preflight disagreed with its batch query")
                self.radius_queries += int(batch_sources.size)
                self._record_phase(
                    "radius_retrieval",
                    time.perf_counter() - retrieval_started,
                    count=int(batch_sources.size),
                )
                candidate_counts_actual = np.asarray(
                    [int(candidate.size) for candidate in candidate_lists],
                    dtype=np.int64,
                )
                possible_support_counts = [
                    int(
                        np.sum(
                            self.location_indptr[candidate + 1] - self.location_indptr[candidate],
                            dtype=np.int64,
                        )
                    )
                    for candidate in candidate_lists
                ]
                candidate_storage_bytes = int(
                    sum(int(candidate.nbytes) for candidate in candidate_lists)
                )
                # ``expanded`` is retained through canonical verification;
                # spacing/max temporaries and the tree's ragged container are
                # admitted conservatively as well.
                persistent_distance_bytes = int(
                    candidate_storage_bytes + expanded.nbytes + 16 * batch_sources.size + 1024
                )
                distance_fixed_bytes, distance_tile_bytes = self._distance_values_many_workspace(
                    queries,
                    candidate_lists,
                    persistent_workspace_bytes=persistent_distance_bytes,
                )
                distance_total = int(np.sum(candidate_counts_actual, dtype=np.int64))
                available_distance_bytes = self.max_radius_query_bytes - distance_fixed_bytes
                if available_distance_bytes < distance_tile_bytes:
                    raise self._capacity_error(
                        source_index=int(batch_sources[0]),
                        radius=float(batch_radii[0]),
                        requested_bytes=distance_fixed_bytes + distance_tile_bytes,
                        context=context,
                    )
                distance_tile_count = min(
                    distance_total,
                    max(1, available_distance_bytes // distance_tile_bytes),
                )
                distance_workspace = int(
                    distance_fixed_bytes + distance_tile_count * distance_tile_bytes
                )
                support_workspace = int(
                    sum(
                        _estimate_radius_workspace_bytes(
                            int(candidate_count),
                            support_count,
                            dimension,
                        )
                        for candidate_count, support_count in zip(
                            candidate_counts_actual,
                            possible_support_counts,
                            strict=True,
                        )
                    )
                    + queries.nbytes
                    + expanded.nbytes
                    + candidate_storage_bytes
                    + 8 * (batch_sources.size + 1)
                    # The strict filter materialises a repeated radius vector
                    # and a boolean mask alongside the canonical distance
                    # vector.  Include both before those vectorised arrays are
                    # created.
                    + 17 * distance_total
                    + 8 * batch_sources.size
                    + 1024
                    + 64 * sum(possible_support_counts)
                    + 32 * distance_total
                )
                actual_workspace = max(
                    preflight_workspace,
                    distance_workspace,
                    support_workspace,
                )
                self.query_workspace_peak_bytes = max(
                    self.query_workspace_peak_bytes,
                    actual_workspace,
                )
                if actual_workspace > self.max_radius_query_bytes:
                    raise self._capacity_error(
                        source_index=int(batch_sources[0]),
                        radius=float(batch_radii[0]),
                        requested_bytes=actual_workspace,
                        context=context,
                    )
                lease.resize(max(preflight_workspace, actual_workspace))
                verification_started = time.perf_counter()
                distance_offsets, candidate_distances = self._distance_values_many(
                    queries,
                    candidate_lists,
                    max_workspace_bytes=self.max_radius_query_bytes,
                    persistent_workspace_bytes=persistent_distance_bytes,
                )
                self.distance_evaluations += int(candidate_distances.size)
                self.unique_distance_evaluations += int(candidate_distances.size)
                strict_radius_mask = candidate_distances < np.repeat(
                    batch_radii,
                    np.diff(distance_offsets),
                )
                batch_targets: list[np.ndarray] = []
                batch_distances: list[np.ndarray] = []
                batch_support_counts: list[int] = []
                for local_index, source in enumerate(batch_sources):
                    candidate = candidate_lists[local_index]
                    distances = candidate_distances[
                        distance_offsets[local_index] : distance_offsets[local_index + 1]
                    ]
                    included = strict_radius_mask[
                        distance_offsets[local_index] : distance_offsets[local_index + 1]
                    ]
                    included_locations = candidate[included]
                    included_distances = distances[included]
                    if included_locations.size == 0:
                        raise ValueError("positive kernel radius support omitted the source row")
                    location_order = np.argsort(included_locations, kind="stable")
                    ordered_locations = included_locations[location_order]
                    ordered_distances = included_distances[location_order]
                    row_parts = [
                        self.rows_by_location[
                            self.location_indptr[int(location)] : self.location_indptr[
                                int(location) + 1
                            ]
                        ]
                        for location in ordered_locations
                    ]
                    rows = np.concatenate(row_parts).astype(np.int64, copy=False)
                    row_distances = np.repeat(
                        ordered_distances,
                        [int(part.shape[0]) for part in row_parts],
                    )
                    row_order = np.argsort(rows, kind="stable")
                    rows = rows[row_order]
                    row_distances = np.asarray(row_distances[row_order], dtype=np.float64)
                    if not np.any(rows == int(source)):
                        raise ValueError("positive kernel radius support omitted the source row")
                    batch_targets.append(rows)
                    batch_distances.append(row_distances)
                    batch_support_counts.append(int(rows.size))
                self._record_phase(
                    "canonical_distance_verification",
                    time.perf_counter() - verification_started,
                    count=int(batch_sources.size),
                )
                self.distance_evaluations += int(sum(batch_support_counts))
                self.atomic_distance_evaluations += int(sum(batch_support_counts))
                for local_index, support_count in enumerate(batch_support_counts):
                    offsets[source_cursor + local_index + 1] = (
                        offsets[source_cursor + local_index] + support_count
                    )
                    self.current_support_count = support_count
                    if self.min_support_count is None:
                        self.min_support_count = support_count
                    else:
                        self.min_support_count = min(self.min_support_count, support_count)
                    self.support_count_total += support_count
                    self.support_count_samples += 1
                    self.max_support_count = max(self.max_support_count, support_count)
                target_parts.extend(batch_targets)
                distance_parts.extend(batch_distances)
            except (ResourceCapacityError, RadiusQueryCapacityError):
                if stop - source_cursor <= 1:
                    raise
                batch_limit = max(1, (stop - source_cursor) // 2)
                continue
            finally:
                lease.close()
            source_cursor = stop

        total_support = int(offsets[-1])
        retained_part_bytes = int(
            sum(int(part.nbytes) for part in target_parts)
            + sum(int(part.nbytes) for part in distance_parts)
        )
        materialisation_workspace = int(
            offsets.nbytes
            + retained_part_bytes
            + 2 * total_support * np.dtype(np.int64).itemsize
            + 1024
        )
        self.query_workspace_peak_bytes = max(
            self.query_workspace_peak_bytes,
            materialisation_workspace,
        )
        with self.resource_budget.acquire(
            context or "indexed exact batched radius result materialisation",
            max(1, materialisation_workspace),
        ):
            # Fill the final CSR vectors in place.  This preserves source and
            # row order while avoiding np.concatenate's additional full-size
            # temporary alongside the returned arrays.
            targets = np.empty(total_support, dtype=np.int64)
            distances = np.empty(total_support, dtype=np.float64)
            cursor = 0
            for target_part, distance_part in zip(target_parts, distance_parts, strict=True):
                stop = cursor + int(target_part.size)
                targets[cursor:stop] = target_part
                distances[cursor:stop] = distance_part
                cursor = stop
        targets.setflags(write=False)
        distances.setflags(write=False)
        return offsets, targets, distances

    def radius_support(
        self,
        source_index: int,
        radius: float,
        *,
        context: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return all original target rows with exact distance strictly below radius."""

        if isinstance(source_index, bool) or not isinstance(source_index, (int, np.integer)):
            raise ValueError("source_index must be a non-negative integer")
        offsets, targets, distances = self.radius_support_many(
            np.asarray([int(source_index)], dtype=np.int64),
            np.asarray([float(radius)], dtype=np.float64),
            chunk_size=1,
            context=context,
        )
        return targets[offsets[0] : offsets[1]], distances[offsets[0] : offsets[1]]

    query_radius_support = radius_support
    exact_radius_support = radius_support
    query_radius = radius_support


class TiledCPUNeighbourIndex:
    """Exact, dependency-light tiled CPU neighbour backend.

    This backend deliberately performs the complete calculation.  It uses
    deterministic source/target tiles and a bounded top-k reduction, so no
    pairwise ``N x N`` distance matrix is ever allocated.  Unlike the indexed
    backend it does not rely on a spatial tree or an approximate candidate
    set; the explicit backend ID therefore remains inspectable in scientific
    fingerprints and diagnostics.
    """

    backend = TILED_NEIGHBOUR_BACKEND_ID
    backend_version = TILED_NEIGHBOUR_BACKEND_VERSION
    metric = NEIGHBOUR_METRIC
    distinct_location_policy = NEIGHBOUR_DISTINCT_POLICY

    def __init__(
        self,
        descriptors: np.ndarray,
        *,
        tile_size: int = 256,
        max_index_bytes: int | None = None,
        max_radius_query_bytes: int | None = None,
        max_neighbour_cache_bytes: int | None = DEFAULT_NEIGHBOUR_CACHE_BYTES,
        resource_budget: ResourceBudgetService | None = None,
    ) -> None:
        values = _validate_descriptors(descriptors)
        tile = _validate_chunk_size(tile_size)
        service = resource_budget or build_resource_budget()
        remaining = service.remaining_managed_budget
        if remaining is None:
            raise ResourceCapacityError(
                service.unknown_memory_message("tiled neighbour construction"),
                operation="tiled neighbour construction",
            )
        explicit_limit = (
            None
            if max_index_bytes is None
            else _validate_memory_limit(max_index_bytes, "max_index_bytes")
        )
        temporary_peak_bytes = int(
            2 * values.nbytes + 48 * values.shape[0] * np.dtype(np.int64).itemsize + 8192
        )
        if temporary_peak_bytes > remaining:
            raise ResourceCapacityError(
                "tiled exact neighbour grouping exceeds its runtime memory allowance: "
                f"N={values.shape[0]}, d'={values.shape[1]}, "
                f"estimated_peak_bytes={temporary_peak_bytes}, available_bytes={remaining}",
                operation="tiled neighbour construction",
                requested_bytes=temporary_peak_bytes,
                available_bytes=remaining,
                reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
            )
        grouping_started = time.perf_counter()
        locations, first_indices, row_to_location = np.unique(
            values,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        self.resource_budget = service
        self.descriptors = values
        self.unique_locations = np.ascontiguousarray(locations, dtype=np.float64)
        self.location_representatives = np.asarray(first_indices, dtype=np.int64)
        self.row_to_location = np.asarray(row_to_location, dtype=np.int64)
        location_count = int(self.unique_locations.shape[0])
        self.rows_by_location = np.asarray(
            np.argsort(self.row_to_location, kind="stable"), dtype=np.int64
        )
        location_counts = np.bincount(self.row_to_location, minlength=location_count).astype(
            np.int64, copy=False
        )
        self.location_indptr = np.empty(location_count + 1, dtype=np.int64)
        self.location_indptr[0] = 0
        np.cumsum(location_counts, dtype=np.int64, out=self.location_indptr[1:])
        self.rows_by_location.setflags(write=False)
        self.location_indptr.setflags(write=False)
        self.grouping_build_seconds = max(0.0, time.perf_counter() - grouping_started)
        self.tile_size = tile
        self.grouping_bytes = int(self.rows_by_location.nbytes + self.location_indptr.nbytes)
        self.index_bytes = int(
            self.unique_locations.nbytes
            + self.location_representatives.nbytes
            + self.row_to_location.nbytes
            + self.grouping_bytes
            + 4096
        )
        effective_limit = remaining if explicit_limit is None else min(explicit_limit, remaining)
        if self.index_bytes > effective_limit:
            message = (
                "tiled exact neighbour backend exceeds max_index_bytes"
                if explicit_limit is not None and self.index_bytes > explicit_limit
                else "tiled exact neighbour backend exceeds its runtime memory allowance"
            )
            raise ResourceCapacityError(
                f"{message}: N={values.shape[0]}, Q={location_count}, "
                f"estimated_bytes={self.index_bytes}, available_bytes={effective_limit}",
                operation="tiled neighbour construction",
                requested_bytes=self.index_bytes,
                available_bytes=remaining,
                reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
            )
        self.max_radius_query_bytes = (
            max(1, remaining // 2)
            if max_radius_query_bytes is None
            else min(
                _validate_memory_limit(max_radius_query_bytes, "max_radius_query_bytes"),
                remaining,
            )
        )
        self.max_neighbour_cache_bytes = _resolve_neighbour_cache_budget(
            remaining_bytes=remaining,
            index_bytes=self.index_bytes,
            radius_query_bytes=self.max_radius_query_bytes,
            user_cap=max_neighbour_cache_bytes,
        )
        self.cache_budget_bytes = int(self.max_neighbour_cache_bytes)
        self.index_queries = 0
        self.radius_queries = 0
        self.radius_query_preflights = 0
        self.distance_evaluations = 0
        self.unique_distance_evaluations = 0
        self.atomic_distance_evaluations = 0
        self.current_support_count = 0
        self.min_support_count: int | None = None
        self.support_count_total = 0
        self.support_count_samples = 0
        self.max_support_count = 0
        self.query_workspace_peak_bytes = 0
        self._neighbour_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._neighbour_cache_lease: Any | None = None
        self._neighbour_cache_key: str | None = None
        self._cache_lock = threading.RLock()
        self.neighbour_cache_bytes = 0
        self.cache_hit = 0
        self.cache_miss = 0
        self.cache_eviction = 0
        self.cache_admission_rejected = 0
        self.cache_peak_bytes = 0
        self.cache_time_saved_seconds = 0.0
        self._last_neighbour_compute_seconds = 0.0
        self.fingerprint = sha256_canonical_json(
            {
                "backend": self.backend,
                "version": self.backend_version,
                "metric": self.metric,
                "distinct_location_policy": self.distinct_location_policy,
                "tie_policy": "distance-then-earliest-original-representative-v1",
                "radius_policy": "strict-float64-euclidean-less-than-v1",
                "tile_size": self.tile_size,
                "locations": _array_fingerprint(self.unique_locations),
                "row_to_location": _array_fingerprint(self.row_to_location),
            }
        )
        owner = weakref.ref(self)

        def evict() -> int:
            instance = owner()
            return 0 if instance is None else instance._evict_neighbour_cache()

        self._cache_eviction_token = service.register_optional_eviction(evict)

    def close(self) -> None:
        """Release optional cache bytes and its runtime-budget registration."""

        self.resource_budget.unregister_optional_eviction(self._cache_eviction_token)
        self._evict_neighbour_cache()

    def __del__(self) -> None:  # pragma: no cover - best-effort lifecycle cleanup
        try:
            self.close()
        except Exception:
            pass

    @property
    def worker_count(self) -> int:
        configured = self.resource_budget.budget.worker_cap
        available = os.cpu_count() or 1
        return max(1, available if configured is None else min(int(configured), available))

    def memory_metrics(self) -> dict[str, int | float | None]:
        return {
            "support_cache_bytes": 0,
            "support_cache_peak_bytes": 0,
            "neighbour_cache_bytes": int(self.neighbour_cache_bytes),
            "cache_bytes": int(self.neighbour_cache_bytes),
            "cache_peak_bytes": int(self.cache_peak_bytes),
            "cache_hit": int(self.cache_hit),
            "cache_miss": int(self.cache_miss),
            "cache_eviction": int(self.cache_eviction),
            "cache_admission_rejected": int(self.cache_admission_rejected),
            "cache_time_saved_seconds": float(self.cache_time_saved_seconds),
            "cache_budget_bytes": int(self.cache_budget_bytes),
            "index_bytes": self.index_bytes,
            "worker_count": self.worker_count,
            "grouping_bytes": self.grouping_bytes,
            "grouping_build_seconds": 0.0,
            "tree_build_seconds": 0.0,
            "query_workspace_peak_bytes": self.query_workspace_peak_bytes,
            "current_support_count": self.current_support_count,
            "min_support_count": self.min_support_count,
            "mean_support_count": (
                self.support_count_total / self.support_count_samples
                if self.support_count_samples
                else None
            ),
            "max_support_count": self.max_support_count,
            "pilot_support_queries": 0,
            "pilot_support_count_total": 0,
            "pilot_support_samples": 0,
            "index_queries": self.index_queries,
            "radius_queries": self.radius_queries,
            "radius_query_preflights": self.radius_query_preflights,
            "distance_evaluations": self.distance_evaluations,
            "unique_distance_evaluations": self.unique_distance_evaluations,
            "atomic_distance_evaluations": self.atomic_distance_evaluations,
            **_process_memory_metrics(),
        }

    def _distance_values(self, query: np.ndarray, targets: np.ndarray) -> np.ndarray:
        count = int(targets.shape[0])
        if count == 0:
            return np.empty(0, dtype=np.float64)
        bytes_per_target = max(64, 32 * int(targets.shape[1]) + 64)
        fixed_workspace = int(targets.nbytes + 8 * count + 256)
        available_workspace = self.max_radius_query_bytes - fixed_workspace
        if available_workspace < bytes_per_target:
            raise RadiusQueryCapacityError(
                "tiled exact neighbour distance workspace exceeds max_radius_query_bytes"
            )
        tile = min(count, max(1, available_workspace // bytes_per_target))
        workspace = int(fixed_workspace + tile * bytes_per_target)
        self.query_workspace_peak_bytes = max(
            self.query_workspace_peak_bytes,
            workspace,
        )
        distances: np.ndarray | None = None
        with np.errstate(over="raise", invalid="raise"):
            try:
                with self.resource_budget.acquire(
                    "tiled exact neighbour distance tile",
                    max(1, workspace),
                ):
                    distances = np.empty(count, dtype=np.float64)
                    for start in range(0, count, tile):
                        stop = min(start + tile, count)
                        delta = targets[start:stop] - query
                        distances[start:stop] = np.sqrt(
                            np.sum(delta * delta, axis=1, dtype=np.float64)
                        )
            except FloatingPointError as exc:
                raise ValueError("tiled neighbour distance overflowed or became invalid") from exc
        assert distances is not None
        if not np.all(np.isfinite(distances)):
            raise ValueError("tiled neighbour distances must be finite")
        return distances

    def _cache_key(self, k: int) -> str:
        return sha256_canonical_json(
            {
                "schema": "exact-neighbour-cache-v1",
                "representation": self.fingerprint,
                "backend": self.backend,
                "version": self.backend_version,
                "k_max": int(k),
            }
        )

    def _evict_neighbour_cache(self) -> int:
        with self._cache_lock:
            released = int(self.neighbour_cache_bytes)
            if not self._neighbour_cache:
                return 0
            self._neighbour_cache.clear()
            self._neighbour_cache_key = None
            self.neighbour_cache_bytes = 0
            lease = self._neighbour_cache_lease
            self._neighbour_cache_lease = None
            self.cache_eviction += 1
            if lease is not None:
                lease.close()
            logger.info("Evicted optional exact neighbour cache: bytes=%d", released)
            return released

    def _store_neighbour_cache(
        self,
        k: int,
        neighbour_indices: np.ndarray,
        neighbour_distances: np.ndarray,
    ) -> None:
        entry_bytes = int(neighbour_indices.nbytes + neighbour_distances.nbytes)
        with self._cache_lock:
            if entry_bytes > self.max_neighbour_cache_bytes:
                self.cache_admission_rejected += 1
                logger.info(
                    "Rejected optional exact neighbour cache admission: k=%d, "
                    "cache_bytes=%d, budget_bytes=%d, reason=budget",
                    k,
                    entry_bytes,
                    self.max_neighbour_cache_bytes,
                )
                return
            previous_key = self._neighbour_cache_key
        try:
            lease = self.resource_budget.acquire(
                "optional exact neighbour cache", max(1, entry_bytes)
            )
        except ResourceCapacityError:
            with self._cache_lock:
                self.cache_admission_rejected += 1
            logger.info(
                "Rejected optional exact neighbour cache admission: k=%d, "
                "cache_bytes=%d, reason=active-resource-pressure",
                k,
                entry_bytes,
            )
            return
        with self._cache_lock:
            if self._neighbour_cache_key != previous_key:
                lease.close()
                self.cache_admission_rejected += 1
                logger.info(
                    "Rejected optional exact neighbour cache admission: k=%d, "
                    "cache_bytes=%d, reason=concurrent-cache-update",
                    k,
                    entry_bytes,
                )
                return
            old_lease = self._neighbour_cache_lease
            if self._neighbour_cache:
                self.cache_eviction += 1
            if old_lease is not None:
                old_lease.close()
            neighbour_indices.setflags(write=False)
            neighbour_distances.setflags(write=False)
            self._neighbour_cache = {int(k): (neighbour_indices, neighbour_distances)}
            self._neighbour_cache_key = self._cache_key(k)
            self._neighbour_cache_lease = lease
            self.neighbour_cache_bytes = entry_bytes
            self.cache_peak_bytes = max(self.cache_peak_bytes, entry_bytes)
            logger.info(
                "Admitted optional exact neighbour cache: k=%d, cache_bytes=%d, "
                "cache_budget_bytes=%d",
                k,
                entry_bytes,
                self.max_neighbour_cache_bytes,
            )

    def _cached_neighbour_prefix(self, k: int) -> tuple[np.ndarray, np.ndarray] | None:
        with self._cache_lock:
            for cached_k, cached in self._neighbour_cache.items():
                if cached_k >= k and self._neighbour_cache_key == self._cache_key(cached_k):
                    self.cache_hit += 1
                    self.cache_time_saved_seconds += self._last_neighbour_compute_seconds
                    logger.info(
                        "Exact neighbour cache hit: requested_k=%d, cached_k=%d, "
                        "time_saved_seconds=%.6f",
                        k,
                        cached_k,
                        self._last_neighbour_compute_seconds,
                    )
                    return cached[0][:, :k], cached[1][:, :k]
            self.cache_miss += 1
            logger.info("Exact neighbour cache miss: requested_k=%d", k)
            return None

    def _compute_location_neighbours(
        self,
        k: int,
        *,
        chunk_size: int = 1024,
        checkpoint_store: VersionedCheckpointStore | None = None,
        checkpoint_name: str = "knn",
        checkpoint_identity: dict[str, Any] | None = None,
        checkpoint_every_locations: int = 1,
        resume: bool = True,
        preserve_checkpoint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        order_k = _validate_k(k)
        _validate_chunk_size(chunk_size)
        cached = self._cached_neighbour_prefix(order_k)
        if cached is not None:
            return cached
        n_locations = int(self.unique_locations.shape[0])
        if n_locations - 1 < order_k:
            raise ValueError(
                f"k={order_k} requires at least {order_k} other distinct descriptor locations; "
                f"the pool contains only {n_locations - 1}"
            )
        started = time.perf_counter()
        neighbours = np.empty((n_locations, order_k), dtype=np.int64)
        distances_out = np.empty((n_locations, order_k), dtype=np.float64)
        source_start = 0
        if checkpoint_store is not None:
            if checkpoint_identity is None:
                raise ValueError("kNN checkpoint identity is required")
            if (
                isinstance(checkpoint_every_locations, bool)
                or not isinstance(checkpoint_every_locations, (int, np.integer))
                or int(checkpoint_every_locations) < 1
            ):
                raise ValueError("checkpoint_every_locations must be a positive integer")
            if not resume:
                checkpoint_store.clear(checkpoint_name)
            if resume and checkpoint_store.exists(checkpoint_name):
                _identity, checkpoint_state, checkpoint_arrays = checkpoint_store.load(
                    checkpoint_name,
                    expected_identity=checkpoint_identity,
                )
                source_start = int(checkpoint_state.get("cursor", -1))
                saved_neighbours = checkpoint_arrays.get("neighbour_locations")
                saved_distances = checkpoint_arrays.get("neighbour_distances")
                if (
                    source_start < 0
                    or source_start > n_locations
                    or saved_neighbours is None
                    or saved_distances is None
                    or saved_neighbours.shape != (source_start, order_k)
                    or saved_distances.shape != (source_start, order_k)
                ):
                    raise CheckpointValidationError(
                        "kNN checkpoint cursor or arrays are incompatible"
                    )
                neighbours[:source_start] = saved_neighbours
                distances_out[:source_start] = saved_distances
        target_tile = max(
            1,
            min(self.tile_size, _validate_chunk_size(chunk_size), n_locations),
        )
        logger.info(
            "Tiled exact neighbour search started: N=%d, distinct locations=%d, k=%d, "
            "target_tile=%d, worker_cap=%d",
            self.descriptors.shape[0],
            n_locations,
            order_k,
            target_tile,
            self.worker_count,
        )
        for source_location in range(source_start, n_locations):
            best_distances = np.empty(0, dtype=np.float64)
            best_locations = np.empty(0, dtype=np.int64)
            query = self.unique_locations[source_location]
            for start in range(0, n_locations, target_tile):
                stop = min(start + target_tile, n_locations)
                candidates = np.arange(start, stop, dtype=np.int64)
                candidates = candidates[candidates != source_location]
                if candidates.size == 0:
                    continue
                distances = self._distance_values(query, self.unique_locations[candidates])
                self.distance_evaluations += int(candidates.size)
                self.unique_distance_evaluations += int(candidates.size)
                combined_locations = np.concatenate((best_locations, candidates))
                combined_distances = np.concatenate((best_distances, distances))
                order = np.lexsort(
                    (
                        self.location_representatives[combined_locations],
                        combined_distances,
                    )
                )[:order_k]
                best_locations = combined_locations[order]
                best_distances = combined_distances[order]
            if best_locations.size != order_k:
                raise RuntimeError("tiled exact neighbour reduction returned an incomplete row")
            neighbours[source_location] = best_locations
            distances_out[source_location] = best_distances
            self.index_queries += 1
            if (
                checkpoint_store is not None
                and checkpoint_identity is not None
                and (source_location + 1) % int(checkpoint_every_locations) == 0
            ):
                checkpoint_store.write(
                    checkpoint_name,
                    identity=checkpoint_identity,
                    state={"cursor": source_location + 1},
                    arrays={
                        "neighbour_locations": neighbours[: source_location + 1],
                        "neighbour_distances": distances_out[: source_location + 1],
                    },
                )
        if checkpoint_store is not None and not preserve_checkpoint:
            checkpoint_store.clear(checkpoint_name)
            checkpoint_store.clean_stale()
        neighbours.setflags(write=False)
        distances_out.setflags(write=False)
        self._last_neighbour_compute_seconds = max(0.0, time.perf_counter() - started)
        self._store_neighbour_cache(order_k, neighbours, distances_out)
        logger.info(
            "Tiled exact neighbour search completed: %d/%d (100.0%%), k_max=%d, elapsed=%.3fs",
            n_locations,
            n_locations,
            order_k,
            self._last_neighbour_compute_seconds,
        )
        return neighbours, distances_out

    def query_neighbours(
        self,
        k: int,
        *,
        chunk_size: int = 1024,
        checkpoint_store: VersionedCheckpointStore | None = None,
        checkpoint_name: str = "knn",
        checkpoint_identity: dict[str, Any] | None = None,
        checkpoint_every_locations: int = 1,
        resume: bool = True,
        preserve_checkpoint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        location_indices, location_distances = self._compute_location_neighbours(
            k,
            chunk_size=chunk_size,
            checkpoint_store=checkpoint_store,
            checkpoint_name=checkpoint_name,
            checkpoint_identity=checkpoint_identity,
            checkpoint_every_locations=checkpoint_every_locations,
            resume=resume,
            preserve_checkpoint=preserve_checkpoint,
        )
        rows = np.asarray(
            self.location_representatives[location_indices[self.row_to_location]], dtype=np.int64
        )
        distances = np.asarray(location_distances[self.row_to_location], dtype=np.float64)
        return distances[:, -1].copy(), rows.copy(), distances.copy()

    def query_neighbours_many(
        self,
        ks: Sequence[int],
        *,
        chunk_size: int = 1024,
        checkpoint_store: VersionedCheckpointStore | None = None,
        checkpoint_name: str = "knn",
        checkpoint_identity: dict[str, Any] | None = None,
        checkpoint_every_locations: int = 1,
        resume: bool = True,
        preserve_checkpoint: bool = False,
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        ordered = tuple(_validate_k(value) for value in ks)
        if not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("ks must contain unique positive neighbour orders")
        location_indices, location_distances = self._compute_location_neighbours(
            max(ordered),
            chunk_size=chunk_size,
            checkpoint_store=checkpoint_store,
            checkpoint_name=checkpoint_name,
            checkpoint_identity=checkpoint_identity,
            checkpoint_every_locations=checkpoint_every_locations,
            resume=resume,
            preserve_checkpoint=preserve_checkpoint,
        )
        results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for order_k in ordered:
            distances = np.asarray(
                location_distances[self.row_to_location, :order_k], dtype=np.float64
            )
            rows = np.asarray(
                self.location_representatives[location_indices[self.row_to_location, :order_k]],
                dtype=np.int64,
            )
            results[order_k] = (distances[:, -1].copy(), rows.copy(), distances.copy())
        return results

    def radius_support(
        self, source_index: int, radius: float, *, context: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(source_index, bool) or not isinstance(source_index, (int, np.integer)):
            raise ValueError("source_index must be a non-negative integer")
        source = int(source_index)
        if source < 0 or source >= self.descriptors.shape[0]:
            raise IndexError("source_index is outside the descriptor rows")
        value = float(radius)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("radius must be finite and positive")
        self.radius_queries += 1
        self.radius_query_preflights += 1
        source_location = int(self.row_to_location[source])
        included_locations: list[int] = []
        included_distances: list[float] = []
        for start in range(0, self.unique_locations.shape[0], self.tile_size):
            stop = min(start + self.tile_size, self.unique_locations.shape[0])
            locations = np.arange(start, stop, dtype=np.int64)
            distances = self._distance_values(
                self.unique_locations[source_location], self.unique_locations[start:stop]
            )
            self.distance_evaluations += int(locations.size)
            self.unique_distance_evaluations += int(locations.size)
            filter_workspace = int(locations.nbytes + distances.nbytes + distances.size + 1024)
            self.query_workspace_peak_bytes = max(
                self.query_workspace_peak_bytes,
                filter_workspace,
            )
            with self.resource_budget.acquire(
                "tiled exact radius strict filter",
                max(1, filter_workspace),
            ):
                selected = distances < value
                included_locations.extend(int(item) for item in locations[selected])
                included_distances.extend(float(item) for item in distances[selected])
        if not included_locations:
            raise ValueError("positive kernel radius support omitted the source row")
        support_count = sum(
            int(self.location_indptr[location + 1] - self.location_indptr[location])
            for location in included_locations
        )
        estimated_bytes = _estimate_radius_workspace_bytes(
            len(included_locations),
            support_count,
            self.descriptors.shape[1],
            self.descriptors.shape[0],
        )
        self.query_workspace_peak_bytes = max(self.query_workspace_peak_bytes, estimated_bytes)
        if estimated_bytes > self.max_radius_query_bytes:
            detail = (
                f"N={self.descriptors.shape[0]}, d'={self.descriptors.shape[1]}, "
                f"source_row={source}, radius={value}, requested_bytes={estimated_bytes}, "
                f"max_radius_query_bytes={self.max_radius_query_bytes}"
            )
            if context:
                detail += f", {context}"
            raise RadiusQueryCapacityError(
                "exact tiled radius query exceeds memory capacity: " + detail
            )
        rows: list[int] = []
        row_distances: list[float] = []
        for location, distance in zip(included_locations, included_distances, strict=True):
            start = int(self.location_indptr[location])
            stop = int(self.location_indptr[location + 1])
            rows.extend(int(item) for item in self.rows_by_location[start:stop])
            row_distances.extend([distance] * (stop - start))
        order = np.argsort(np.asarray(rows, dtype=np.int64), kind="stable")
        target_indices = np.asarray(rows, dtype=np.int64)[order]
        target_distances = np.asarray(row_distances, dtype=np.float64)[order]
        self.atomic_distance_evaluations += int(target_indices.size)
        self.current_support_count = int(target_indices.size)
        self.min_support_count = (
            self.current_support_count
            if self.min_support_count is None
            else min(self.min_support_count, self.current_support_count)
        )
        self.support_count_total += self.current_support_count
        self.support_count_samples += 1
        self.max_support_count = max(self.max_support_count, self.current_support_count)
        if not np.any(target_indices == source):
            raise ValueError("positive kernel radius support omitted the source row")
        if context:
            logger.debug("Tiled radius support completed: %s", context)
        target_indices.setflags(write=False)
        target_distances.setflags(write=False)
        return target_indices, target_distances

    query_radius_support = radius_support
    exact_radius_support = radius_support
    query_radius = radius_support


def _top_k(
    best_distances: np.ndarray,
    best_representatives: np.ndarray,
    distances: np.ndarray,
    representatives: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if distances.size == 0:
        return best_distances, best_representatives
    order = np.lexsort((representatives, distances))
    take = order[:k]
    combined_distances = np.concatenate((best_distances, distances[take]))
    combined_representatives = np.concatenate((best_representatives, representatives[take]))
    combined_order = np.lexsort((combined_representatives, combined_distances))[:k]
    return combined_distances[combined_order], combined_representatives[combined_order]


def compute_exact_neighbours(
    descriptors: np.ndarray,
    k: int,
    *,
    chunk_size: int = 1024,
    representation_fingerprint: str | None = None,
    row_ids: Sequence[str] | None = None,
    max_neighbour_entries: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    checkpoint_name: str = "knn",
    checkpoint_every_locations: int = 1,
    resume: bool = True,
    checkpoint_metrics: dict[str, Any] | None = None,
    preserve_checkpoint: bool = False,
) -> ExactNeighbourResult:
    """Calculate exact kth-neighbour radii without an ``N x N`` allocation.

    Equal descriptor rows are one distinct location for radius purposes.  The
    first original row at a location is its deterministic representative;
    all original rows remain present in ``row_to_location`` and retain their
    caller-owned probability mass in later calibration.
    """

    values = _validate_descriptors(descriptors)
    order_k = _validate_k(k)
    block = _validate_chunk_size(chunk_size)
    n_rows = values.shape[0]
    if max_neighbour_entries is not None and (
        isinstance(max_neighbour_entries, bool)
        or not isinstance(max_neighbour_entries, (int, np.integer))
        or int(max_neighbour_entries) < 1
    ):
        raise ValueError("max_neighbour_entries must be a positive integer when provided")
    if max_neighbour_entries is not None and n_rows * order_k > int(max_neighbour_entries):
        raise ValueError(
            "exact neighbour result exceeds its explicit bounded-memory limit; "
            "reduce k or configure a scalable neighbour backend"
        )
    service = resource_budget or build_resource_budget()
    remaining = service.remaining_managed_budget
    if remaining is None:
        raise ResourceCapacityError(
            service.unknown_memory_message("exact neighbour calculation"),
            operation="exact neighbour calculation",
        )
    requested_bytes = int(
        values.nbytes
        + 16 * n_rows * order_k
        + 128 * n_rows
        + 32 * min(n_rows, block) * values.shape[1]
    )
    if requested_bytes > remaining:
        raise ResourceCapacityError(
            "exact neighbour result exceeds its runtime memory allowance: "
            f"N={n_rows}, k={order_k}, requested_bytes={requested_bytes}, "
            f"available_bytes={remaining}",
            operation="exact neighbour calculation",
            requested_bytes=requested_bytes,
            available_bytes=remaining,
            reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
        )
    ordered_row_ids = _validate_row_ids(row_ids, n_rows)
    locations, first_indices, row_to_location = np.unique(
        values,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    representatives = np.asarray(first_indices, dtype=np.int64)
    n_locations = locations.shape[0]
    started = time.perf_counter()
    logger.info(
        "Exact neighbour search started: N=%d, distinct locations=%d, k=%d, "
        "backend=%s, chunk_size=%d",
        n_rows,
        n_locations,
        order_k,
        NEIGHBOUR_BACKEND_ID,
        block,
    )
    if n_locations - 1 < order_k:
        raise ValueError(
            f"k={order_k} requires at least {order_k} other distinct descriptor locations; "
            f"the pool contains only {n_locations - 1}"
        )

    radii = np.empty(n_rows, dtype=np.float64)
    neighbour_indices = np.empty((n_rows, order_k), dtype=np.int64)
    neighbour_distances = np.empty((n_rows, order_k), dtype=np.float64)
    checkpoint_store = (
        None if checkpoint_path is None else VersionedCheckpointStore(checkpoint_path)
    )
    checkpoint_identity = {
        "phase": "exact-knn",
        "schema": "exact-knn-checkpoint-v1",
        "backend": NEIGHBOUR_BACKEND_ID,
        "backend_version": NEIGHBOUR_BACKEND_VERSION,
        "descriptors": _array_fingerprint(values),
        "row_ids": list(ordered_row_ids),
        "unique_locations": _array_fingerprint(locations),
        "row_to_location": _array_fingerprint(row_to_location),
        "k": order_k,
        "tie_policy": "distance-then-earliest-original-representative-v1",
    }
    query_start = 0
    if checkpoint_store is not None:
        if (
            isinstance(checkpoint_every_locations, bool)
            or not isinstance(checkpoint_every_locations, (int, np.integer))
            or int(checkpoint_every_locations) < 1
        ):
            raise ValueError("checkpoint_every_locations must be a positive integer")
        if not resume:
            checkpoint_store.clear(checkpoint_name)
        if resume and checkpoint_store.exists(checkpoint_name):
            _identity, checkpoint_state, checkpoint_arrays = checkpoint_store.load(
                checkpoint_name,
                expected_identity=checkpoint_identity,
            )
            query_start = int(checkpoint_state.get("cursor", -1))
            saved_radii = checkpoint_arrays.get("radii")
            saved_indices = checkpoint_arrays.get("neighbour_indices")
            saved_distances = checkpoint_arrays.get("neighbour_distances")
            if (
                query_start < 0
                or query_start > n_rows
                or saved_radii is None
                or saved_indices is None
                or saved_distances is None
                or saved_radii.shape != (query_start,)
                or saved_indices.shape != (query_start, order_k)
                or saved_distances.shape != (query_start, order_k)
            ):
                raise CheckpointValidationError("kNN checkpoint cursor or arrays are incompatible")
            radii[:query_start] = saved_radii
            neighbour_indices[:query_start] = saved_indices
            neighbour_distances[:query_start] = saved_distances
    next_percent = 1

    def report_query_progress(completed: int) -> None:
        nonlocal next_percent
        if n_rows < 100:
            should_report = True
        else:
            should_report = completed * 100 >= next_percent * n_rows
        if not should_report:
            return
        elapsed = max(0.0, time.perf_counter() - started)
        rate = completed / elapsed if completed > 0 and elapsed > 0.0 else 0.0
        remaining = (n_rows - completed) / rate if rate > 0.0 else 0.0
        percentage = 100.0 * completed / n_rows
        logger.info(
            "Exact neighbour search progress: %d/%d (%.1f%%), elapsed=%.3fs, "
            "estimated remaining=%.3fs",
            completed,
            n_rows,
            percentage,
            elapsed,
            max(0.0, remaining),
        )
        if n_rows >= 100:
            while next_percent <= 100 and completed * 100 >= next_percent * n_rows:
                next_percent += 1

    for query_index in range(query_start, n_rows):
        logger.debug("Exact neighbour query %d/%d started", query_index + 1, n_rows)
        query_location = int(row_to_location[query_index])
        query = values[query_index]
        best_distances = np.empty(0, dtype=np.float64)
        best_representatives = np.empty(0, dtype=np.int64)
        for start in range(0, n_locations, block):
            stop = min(start + block, n_locations)
            logger.debug(
                "Exact neighbour query %d/%d processing location block [%d:%d)",
                query_index + 1,
                n_rows,
                start,
                stop,
            )
            location_block = locations[start:stop]
            location_ids = np.arange(start, stop, dtype=np.int64)
            mask = location_ids != query_location
            if not np.any(mask):
                continue
            with np.errstate(over="raise", invalid="raise"):
                try:
                    distances = np.sqrt(np.sum((location_block - query) ** 2, axis=1))
                except FloatingPointError as exc:
                    raise ValueError(
                        "exact neighbour distance overflowed or became invalid"
                    ) from exc
            distances = distances[mask]
            block_representatives = representatives[location_ids[mask]]
            if not np.all(np.isfinite(distances)) or np.any(distances <= 0.0):
                raise ValueError(
                    "distinct descriptor locations did not produce finite positive distances"
                )
            best_distances, best_representatives = _top_k(
                best_distances,
                best_representatives,
                distances,
                block_representatives,
                order_k,
            )
        if best_distances.size != order_k:
            raise ValueError(
                f"k={order_k} has insufficient distinct neighbours for row {query_index}"
            )
        radii[query_index] = best_distances[-1]
        neighbour_indices[query_index] = best_representatives
        neighbour_distances[query_index] = best_distances
        logger.debug("Exact neighbour query %d/%d completed", query_index + 1, n_rows)
        report_query_progress(query_index + 1)
        if (
            checkpoint_store is not None
            and (query_index + 1) % int(checkpoint_every_locations) == 0
        ):
            checkpoint_store.write(
                checkpoint_name,
                identity=checkpoint_identity,
                state={"cursor": query_index + 1},
                arrays={
                    "radii": radii[: query_index + 1],
                    "neighbour_indices": neighbour_indices[: query_index + 1],
                    "neighbour_distances": neighbour_distances[: query_index + 1],
                },
            )
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("exact kth-neighbour radii must be finite and positive")
    if checkpoint_store is not None and not preserve_checkpoint:
        checkpoint_store.clear(checkpoint_name)
        checkpoint_store.clean_stale()
    if checkpoint_metrics is not None:
        checkpoint_metrics.clear()
        if checkpoint_store is not None:
            checkpoint_metrics.update(checkpoint_store.metrics())

    elapsed = max(0.0, time.perf_counter() - started)
    logger.info(
        "Exact neighbour search completed: %d/%d (100.0%%), elapsed=%.3fs", n_rows, n_rows, elapsed
    )

    if representation_fingerprint is None:
        representation_id = _array_fingerprint(values)
    elif not isinstance(representation_fingerprint, str) or not representation_fingerprint.strip():
        raise ValueError("representation_fingerprint must be a non-blank string")
    else:
        representation_id = representation_fingerprint
    identity = {
        "schema": NEIGHBOUR_BACKEND_VERSION,
        "representation": representation_id,
        "row_ids": list(ordered_row_ids),
        "metric": NEIGHBOUR_METRIC,
        "backend": NEIGHBOUR_BACKEND_ID,
        "distinct_location_policy": NEIGHBOUR_DISTINCT_POLICY,
        "k": order_k,
    }
    fingerprint = sha256_canonical_json(identity)
    return ExactNeighbourResult(
        radii=radii,
        neighbour_indices=neighbour_indices,
        neighbour_distances=neighbour_distances,
        row_to_location=np.asarray(row_to_location, dtype=np.int64),
        unique_locations=np.asarray(locations, dtype=np.float64),
        location_representatives=representatives,
        k=order_k,
        representation_fingerprint=representation_id,
        row_ids=ordered_row_ids,
        fingerprint=fingerprint,
        backend_fingerprint=sha256_canonical_json(
            {
                "backend": NEIGHBOUR_BACKEND_ID,
                "version": NEIGHBOUR_BACKEND_VERSION,
                "metric": NEIGHBOUR_METRIC,
                "distinct_location_policy": NEIGHBOUR_DISTINCT_POLICY,
                "tie_policy": "distance-then-earliest-original-representative-v1",
            }
        ),
    )


def compute_indexed_cpu_neighbours(
    descriptors: np.ndarray,
    k: int,
    *,
    index: IndexedCPUNeighbourIndex | None = None,
    chunk_size: int = 1024,
    representation_fingerprint: str | None = None,
    row_ids: Sequence[str] | None = None,
    max_neighbour_entries: int | None = None,
    max_radius_query_bytes: int | None = None,
    max_neighbour_cache_bytes: int | None = DEFAULT_NEIGHBOUR_CACHE_BYTES,
    resource_budget: ResourceBudgetService | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    checkpoint_name: str = "knn",
    checkpoint_every_locations: int = 1,
    resume: bool = True,
    checkpoint_metrics: dict[str, Any] | None = None,
    preserve_checkpoint: bool = False,
) -> ExactNeighbourResult:
    """Calculate exact kth other-location neighbours through the indexed CPU backend."""

    values = _validate_descriptors(descriptors)
    order_k = _validate_k(k)
    _validate_chunk_size(chunk_size)
    if max_neighbour_entries is not None and (
        isinstance(max_neighbour_entries, bool)
        or not isinstance(max_neighbour_entries, (int, np.integer))
        or int(max_neighbour_entries) < 1
    ):
        raise ValueError("max_neighbour_entries must be a positive integer when provided")
    if max_neighbour_entries is not None and values.shape[0] * order_k > int(max_neighbour_entries):
        raise ValueError(
            "indexed exact neighbour result exceeds its explicit bounded-memory limit: "
            f"N={values.shape[0]}, required_entries={values.shape[0] * order_k}, "
            f"max_entries={int(max_neighbour_entries)}"
        )
    active_index = index or IndexedCPUNeighbourIndex(
        values,
        max_radius_query_bytes=max_radius_query_bytes,
        max_neighbour_cache_bytes=max_neighbour_cache_bytes,
        resource_budget=resource_budget,
    )
    if not isinstance(active_index, IndexedCPUNeighbourIndex):
        raise ValueError("indexed neighbour calculation requires an indexed CPU backend")
    if not np.array_equal(active_index.descriptors, values):
        raise ValueError("indexed neighbour index belongs to different descriptors")
    service = (
        resource_budget or getattr(active_index, "resource_budget", None) or build_resource_budget()
    )
    remaining = service.remaining_managed_budget
    if remaining is None:
        raise ResourceCapacityError(
            service.unknown_memory_message("indexed neighbour calculation"),
            operation="indexed neighbour calculation",
        )
    requested_bytes = int(16 * values.shape[0] * order_k + 128 * values.shape[0])
    if requested_bytes > remaining:
        raise ResourceCapacityError(
            "indexed neighbour result exceeds its runtime memory allowance: "
            f"N={values.shape[0]}, k={order_k}, requested_bytes={requested_bytes}, "
            f"available_bytes={remaining}",
            operation="indexed neighbour calculation",
            requested_bytes=requested_bytes,
            available_bytes=remaining,
            reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
        )
    ordered_row_ids = _validate_row_ids(row_ids, values.shape[0])
    checkpoint_store = (
        None
        if checkpoint_path is None
        else IncrementalCheckpointStore(checkpoint_path, resource_budget=service)
    )
    checkpoint_identity = {
        "phase": "exact-knn",
        "schema": "exact-knn-checkpoint-v1",
        "backend": INDEXED_NEIGHBOUR_BACKEND_ID,
        "backend_version": INDEXED_NEIGHBOUR_BACKEND_VERSION,
        "descriptors": _array_fingerprint(values),
        "row_ids": list(ordered_row_ids),
        "unique_locations": _array_fingerprint(active_index.unique_locations),
        "row_to_location": _array_fingerprint(active_index.row_to_location),
        "k": order_k,
        "tie_policy": "distance-then-earliest-original-representative-v1",
    }
    radii, neighbour_indices, neighbour_distances = active_index.query_neighbours(
        order_k,
        chunk_size=chunk_size,
        checkpoint_store=checkpoint_store,
        checkpoint_name=checkpoint_name,
        checkpoint_identity=checkpoint_identity,
        checkpoint_every_locations=checkpoint_every_locations,
        resume=resume,
        preserve_checkpoint=preserve_checkpoint,
    )
    if checkpoint_metrics is not None:
        checkpoint_metrics.clear()
        if checkpoint_store is not None:
            checkpoint_metrics.update(checkpoint_store.metrics())
    representation_id = (
        _array_fingerprint(values)
        if representation_fingerprint is None
        else representation_fingerprint
    )
    if not isinstance(representation_id, str) or not representation_id.strip():
        raise ValueError("representation_fingerprint must be a non-blank string")
    identity = {
        "schema": active_index.backend_version,
        "representation": representation_id,
        "row_ids": list(ordered_row_ids),
        "metric": active_index.metric,
        "backend": active_index.backend,
        "backend_fingerprint": active_index.fingerprint,
        "distinct_location_policy": active_index.distinct_location_policy,
        "tie_policy": "distance-then-earliest-original-representative-v1",
        "k": order_k,
    }
    return ExactNeighbourResult(
        radii=radii,
        neighbour_indices=neighbour_indices,
        neighbour_distances=neighbour_distances,
        row_to_location=active_index.row_to_location,
        unique_locations=active_index.unique_locations,
        location_representatives=active_index.location_representatives,
        k=order_k,
        representation_fingerprint=representation_id,
        row_ids=ordered_row_ids,
        fingerprint=sha256_canonical_json(identity),
        backend=active_index.backend,
        backend_version=active_index.backend_version,
        metric=active_index.metric,
        distinct_location_policy=active_index.distinct_location_policy,
        backend_fingerprint=active_index.fingerprint,
    )


def compute_indexed_cpu_neighbours_many(
    descriptors: np.ndarray,
    ks: Sequence[int],
    *,
    index: IndexedCPUNeighbourIndex | None = None,
    chunk_size: int = 1024,
    representation_fingerprint: str | None = None,
    row_ids: Sequence[str] | None = None,
    max_neighbour_entries: int | None = None,
    max_radius_query_bytes: int | None = None,
    max_neighbour_cache_bytes: int | None = DEFAULT_NEIGHBOUR_CACHE_BYTES,
    resource_budget: ResourceBudgetService | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    checkpoint_name: str = "knn",
    checkpoint_every_locations: int = 1,
    resume: bool = True,
    checkpoint_metrics: dict[str, Any] | None = None,
    preserve_checkpoint: bool = False,
) -> dict[int, ExactNeighbourResult]:
    """Calculate several indexed neighbour orders from one ``k_max`` pass."""

    values = _validate_descriptors(descriptors)
    ordered = tuple(_validate_k(value) for value in ks)
    if not ordered:
        raise ValueError("ks must contain at least one positive integer")
    if len(set(ordered)) != len(ordered):
        raise ValueError("ks must contain unique neighbour orders")
    _validate_chunk_size(chunk_size)
    if max_neighbour_entries is not None and (
        isinstance(max_neighbour_entries, bool)
        or not isinstance(max_neighbour_entries, (int, np.integer))
        or int(max_neighbour_entries) < 1
    ):
        raise ValueError("max_neighbour_entries must be a positive integer when provided")
    maximum = max(ordered)
    if max_neighbour_entries is not None and values.shape[0] * maximum > int(max_neighbour_entries):
        raise ValueError(
            "indexed exact neighbour result exceeds its explicit bounded-memory limit: "
            f"N={values.shape[0]}, required_entries={values.shape[0] * maximum}, "
            f"max_entries={int(max_neighbour_entries)}"
        )
    active_index = index or IndexedCPUNeighbourIndex(
        values,
        max_radius_query_bytes=max_radius_query_bytes,
        max_neighbour_cache_bytes=max_neighbour_cache_bytes,
        resource_budget=resource_budget,
    )
    if not isinstance(active_index, IndexedCPUNeighbourIndex):
        raise ValueError("indexed neighbour calculation requires an indexed CPU backend")
    if not np.array_equal(active_index.descriptors, values):
        raise ValueError("indexed neighbour index belongs to different descriptors")
    service = (
        resource_budget or getattr(active_index, "resource_budget", None) or build_resource_budget()
    )
    remaining = service.remaining_managed_budget
    if remaining is None:
        raise ResourceCapacityError(
            service.unknown_memory_message("indexed neighbour calculation"),
            operation="indexed neighbour calculation",
        )
    requested_bytes = int(16 * values.shape[0] * maximum + 128 * values.shape[0])
    if requested_bytes > remaining:
        raise ResourceCapacityError(
            "indexed neighbour result exceeds its runtime memory allowance: "
            f"N={values.shape[0]}, k={maximum}, requested_bytes={requested_bytes}, "
            f"available_bytes={remaining}",
            operation="indexed neighbour calculation",
            requested_bytes=requested_bytes,
            available_bytes=remaining,
            reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
        )
    ordered_row_ids = _validate_row_ids(row_ids, values.shape[0])
    checkpoint_store = (
        None
        if checkpoint_path is None
        else IncrementalCheckpointStore(checkpoint_path, resource_budget=service)
    )
    checkpoint_identity = {
        "phase": "exact-knn",
        "schema": "exact-knn-checkpoint-v1",
        "backend": INDEXED_NEIGHBOUR_BACKEND_ID,
        "backend_version": INDEXED_NEIGHBOUR_BACKEND_VERSION,
        "descriptors": _array_fingerprint(values),
        "row_ids": list(ordered_row_ids),
        "unique_locations": _array_fingerprint(active_index.unique_locations),
        "row_to_location": _array_fingerprint(active_index.row_to_location),
        "k": maximum,
        "tie_policy": "distance-then-earliest-original-representative-v1",
    }
    queried = active_index.query_neighbours_many(
        ordered,
        chunk_size=chunk_size,
        checkpoint_store=checkpoint_store,
        checkpoint_name=checkpoint_name,
        checkpoint_identity=checkpoint_identity,
        checkpoint_every_locations=checkpoint_every_locations,
        resume=resume,
        preserve_checkpoint=preserve_checkpoint,
    )
    if checkpoint_metrics is not None:
        checkpoint_metrics.clear()
        if checkpoint_store is not None:
            checkpoint_metrics.update(checkpoint_store.metrics())
    representation_id = (
        _array_fingerprint(values)
        if representation_fingerprint is None
        else representation_fingerprint
    )
    if not isinstance(representation_id, str) or not representation_id.strip():
        raise ValueError("representation_fingerprint must be a non-blank string")
    results: dict[int, ExactNeighbourResult] = {}
    for order_k in ordered:
        radii, neighbour_indices, neighbour_distances = queried[order_k]
        identity = {
            "schema": active_index.backend_version,
            "representation": representation_id,
            "row_ids": list(ordered_row_ids),
            "metric": active_index.metric,
            "backend": active_index.backend,
            "backend_fingerprint": active_index.fingerprint,
            "distinct_location_policy": active_index.distinct_location_policy,
            "tie_policy": "distance-then-earliest-original-representative-v1",
            "k": order_k,
        }
        results[order_k] = ExactNeighbourResult(
            radii=radii,
            neighbour_indices=neighbour_indices,
            neighbour_distances=neighbour_distances,
            row_to_location=active_index.row_to_location,
            unique_locations=active_index.unique_locations,
            location_representatives=active_index.location_representatives,
            k=order_k,
            representation_fingerprint=representation_id,
            row_ids=ordered_row_ids,
            fingerprint=sha256_canonical_json(identity),
            backend=active_index.backend,
            backend_version=active_index.backend_version,
            metric=active_index.metric,
            distinct_location_policy=active_index.distinct_location_policy,
            backend_fingerprint=active_index.fingerprint,
        )
    return results


def compute_tiled_cpu_neighbours(
    descriptors: np.ndarray,
    k: int,
    *,
    index: TiledCPUNeighbourIndex | None = None,
    chunk_size: int = 1024,
    representation_fingerprint: str | None = None,
    row_ids: Sequence[str] | None = None,
    max_neighbour_entries: int | None = None,
    max_radius_query_bytes: int | None = None,
    max_neighbour_cache_bytes: int | None = DEFAULT_NEIGHBOUR_CACHE_BYTES,
    resource_budget: ResourceBudgetService | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    checkpoint_name: str = "knn",
    checkpoint_every_locations: int = 1,
    resume: bool = True,
    checkpoint_metrics: dict[str, Any] | None = None,
    preserve_checkpoint: bool = False,
) -> ExactNeighbourResult:
    """Calculate exact neighbours with the explicit blocked CPU backend."""

    values = _validate_descriptors(descriptors)
    order_k = _validate_k(k)
    _validate_chunk_size(chunk_size)
    if max_neighbour_entries is not None and (
        isinstance(max_neighbour_entries, bool)
        or not isinstance(max_neighbour_entries, (int, np.integer))
        or int(max_neighbour_entries) < 1
    ):
        raise ValueError("max_neighbour_entries must be a positive integer when provided")
    required_entries = int(values.shape[0]) * order_k
    if max_neighbour_entries is not None and required_entries > int(max_neighbour_entries):
        raise ValueError(
            "tiled exact neighbour result exceeds its explicit bounded-memory limit: "
            f"N={values.shape[0]}, required_entries={required_entries}, "
            f"max_entries={int(max_neighbour_entries)}"
        )
    service = resource_budget or (
        index.resource_budget if index is not None else build_resource_budget()
    )
    remaining = service.remaining_managed_budget
    requested_bytes = int(values.nbytes + 16 * required_entries + 128 * values.shape[0])
    if remaining is None:
        raise ResourceCapacityError(
            service.unknown_memory_message("tiled exact neighbour calculation"),
            operation="tiled exact neighbour calculation",
        )
    if requested_bytes > remaining:
        raise ResourceCapacityError(
            "tiled exact neighbour result exceeds its runtime memory allowance: "
            f"N={values.shape[0]}, k={order_k}, requested_bytes={requested_bytes}, "
            f"available_bytes={remaining}",
            operation="tiled exact neighbour calculation",
            requested_bytes=requested_bytes,
            available_bytes=remaining,
            reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
        )
    active_index = index or TiledCPUNeighbourIndex(
        values,
        tile_size=chunk_size,
        max_radius_query_bytes=max_radius_query_bytes,
        max_neighbour_cache_bytes=max_neighbour_cache_bytes,
        resource_budget=resource_budget,
    )
    if not isinstance(active_index, TiledCPUNeighbourIndex):
        raise ValueError("tiled neighbour calculation requires the tiled CPU backend")
    if not np.array_equal(active_index.descriptors, values):
        raise ValueError("tiled neighbour index belongs to different descriptors")
    ordered_row_ids = _validate_row_ids(row_ids, values.shape[0])
    checkpoint_store = (
        None if checkpoint_path is None else VersionedCheckpointStore(checkpoint_path)
    )
    checkpoint_identity = {
        "phase": "exact-knn",
        "schema": "exact-knn-checkpoint-v1",
        "backend": TILED_NEIGHBOUR_BACKEND_ID,
        "backend_version": TILED_NEIGHBOUR_BACKEND_VERSION,
        "descriptors": _array_fingerprint(values),
        "row_ids": list(ordered_row_ids),
        "unique_locations": _array_fingerprint(active_index.unique_locations),
        "row_to_location": _array_fingerprint(active_index.row_to_location),
        "k": order_k,
        "tie_policy": "distance-then-earliest-original-representative-v1",
    }
    radii, neighbour_indices, neighbour_distances = active_index.query_neighbours(
        order_k,
        chunk_size=chunk_size,
        checkpoint_store=checkpoint_store,
        checkpoint_name=checkpoint_name,
        checkpoint_identity=checkpoint_identity,
        checkpoint_every_locations=checkpoint_every_locations,
        resume=resume,
        preserve_checkpoint=preserve_checkpoint,
    )
    if checkpoint_metrics is not None:
        checkpoint_metrics.clear()
        if checkpoint_store is not None:
            checkpoint_metrics.update(checkpoint_store.metrics())
    representation_id = (
        _array_fingerprint(values)
        if representation_fingerprint is None
        else representation_fingerprint
    )
    if not isinstance(representation_id, str) or not representation_id.strip():
        raise ValueError("representation_fingerprint must be a non-blank string")
    identity = {
        "schema": active_index.backend_version,
        "representation": representation_id,
        "row_ids": list(ordered_row_ids),
        "metric": active_index.metric,
        "backend": active_index.backend,
        "backend_fingerprint": active_index.fingerprint,
        "distinct_location_policy": active_index.distinct_location_policy,
        "tie_policy": "distance-then-earliest-original-representative-v1",
        "k": order_k,
    }
    return ExactNeighbourResult(
        radii=radii,
        neighbour_indices=neighbour_indices,
        neighbour_distances=neighbour_distances,
        row_to_location=active_index.row_to_location,
        unique_locations=active_index.unique_locations,
        location_representatives=active_index.location_representatives,
        k=order_k,
        representation_fingerprint=representation_id,
        row_ids=ordered_row_ids,
        fingerprint=sha256_canonical_json(identity),
        backend=active_index.backend,
        backend_version=active_index.backend_version,
        metric=active_index.metric,
        distinct_location_policy=active_index.distinct_location_policy,
        backend_fingerprint=active_index.fingerprint,
    )


def compute_tiled_cpu_neighbours_many(
    descriptors: np.ndarray,
    ks: Sequence[int],
    *,
    index: TiledCPUNeighbourIndex | None = None,
    chunk_size: int = 1024,
    representation_fingerprint: str | None = None,
    row_ids: Sequence[str] | None = None,
    max_neighbour_entries: int | None = None,
    max_radius_query_bytes: int | None = None,
    max_neighbour_cache_bytes: int | None = DEFAULT_NEIGHBOUR_CACHE_BYTES,
    resource_budget: ResourceBudgetService | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    checkpoint_name: str = "knn",
    checkpoint_every_locations: int = 1,
    resume: bool = True,
    preserve_checkpoint: bool = False,
) -> dict[int, ExactNeighbourResult]:
    """Calculate several exact tiled neighbour orders in one bounded pass."""

    values = _validate_descriptors(descriptors)
    ordered = tuple(_validate_k(value) for value in ks)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("ks must contain unique positive neighbour orders")
    maximum = max(ordered)
    required_entries = int(values.shape[0]) * maximum
    if max_neighbour_entries is not None and (
        isinstance(max_neighbour_entries, bool)
        or not isinstance(max_neighbour_entries, (int, np.integer))
        or int(max_neighbour_entries) < 1
    ):
        raise ValueError("max_neighbour_entries must be a positive integer when provided")
    if max_neighbour_entries is not None and required_entries > int(max_neighbour_entries):
        raise ValueError(
            "tiled exact neighbour result exceeds its explicit bounded-memory limit: "
            f"N={values.shape[0]}, required_entries={required_entries}, "
            f"max_entries={int(max_neighbour_entries)}"
        )
    service = resource_budget or (
        index.resource_budget if index is not None else build_resource_budget()
    )
    remaining = service.remaining_managed_budget
    requested_bytes = int(values.nbytes + 16 * required_entries + 128 * values.shape[0])
    if remaining is None:
        raise ResourceCapacityError(
            service.unknown_memory_message("tiled exact neighbour calculation"),
            operation="tiled exact neighbour calculation",
        )
    if requested_bytes > remaining:
        raise ResourceCapacityError(
            "tiled exact neighbour result exceeds its runtime memory allowance: "
            f"N={values.shape[0]}, k={maximum}, requested_bytes={requested_bytes}, "
            f"available_bytes={remaining}",
            operation="tiled exact neighbour calculation",
            requested_bytes=requested_bytes,
            available_bytes=remaining,
            reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
        )
    active_index = index or TiledCPUNeighbourIndex(
        values,
        tile_size=chunk_size,
        max_radius_query_bytes=max_radius_query_bytes,
        max_neighbour_cache_bytes=max_neighbour_cache_bytes,
        resource_budget=resource_budget,
    )
    if not isinstance(active_index, TiledCPUNeighbourIndex):
        raise ValueError("tiled neighbour calculation requires the tiled CPU backend")
    if not np.array_equal(active_index.descriptors, values):
        raise ValueError("tiled neighbour index belongs to different descriptors")
    ordered_row_ids = _validate_row_ids(row_ids, values.shape[0])
    checkpoint_store = (
        None if checkpoint_path is None else VersionedCheckpointStore(checkpoint_path)
    )
    checkpoint_identity = {
        "phase": "exact-knn",
        "schema": "exact-knn-checkpoint-v1",
        "backend": TILED_NEIGHBOUR_BACKEND_ID,
        "backend_version": TILED_NEIGHBOUR_BACKEND_VERSION,
        "descriptors": _array_fingerprint(values),
        "row_ids": list(ordered_row_ids),
        "unique_locations": _array_fingerprint(active_index.unique_locations),
        "row_to_location": _array_fingerprint(active_index.row_to_location),
        "k": maximum,
        "tie_policy": "distance-then-earliest-original-representative-v1",
    }
    queried = active_index.query_neighbours_many(
        ordered,
        chunk_size=chunk_size,
        checkpoint_store=checkpoint_store,
        checkpoint_name=checkpoint_name,
        checkpoint_identity=checkpoint_identity,
        checkpoint_every_locations=checkpoint_every_locations,
        resume=resume,
        preserve_checkpoint=preserve_checkpoint,
    )
    representation_id = (
        _array_fingerprint(values)
        if representation_fingerprint is None
        else representation_fingerprint
    )
    if not isinstance(representation_id, str) or not representation_id.strip():
        raise ValueError("representation_fingerprint must be a non-blank string")
    results: dict[int, ExactNeighbourResult] = {}
    for order_k in ordered:
        radii, neighbour_indices, neighbour_distances = queried[order_k]
        identity = {
            "schema": active_index.backend_version,
            "representation": representation_id,
            "row_ids": list(ordered_row_ids),
            "metric": active_index.metric,
            "backend": active_index.backend,
            "backend_fingerprint": active_index.fingerprint,
            "distinct_location_policy": active_index.distinct_location_policy,
            "tie_policy": "distance-then-earliest-original-representative-v1",
            "k": order_k,
        }
        results[order_k] = ExactNeighbourResult(
            radii=radii,
            neighbour_indices=neighbour_indices,
            neighbour_distances=neighbour_distances,
            row_to_location=active_index.row_to_location,
            unique_locations=active_index.unique_locations,
            location_representatives=active_index.location_representatives,
            k=order_k,
            representation_fingerprint=representation_id,
            row_ids=ordered_row_ids,
            fingerprint=sha256_canonical_json(identity),
            backend=active_index.backend,
            backend_version=active_index.backend_version,
            metric=active_index.metric,
            distinct_location_policy=active_index.distinct_location_policy,
            backend_fingerprint=active_index.fingerprint,
        )
    return results


def build_neighbour_index(
    descriptors: np.ndarray,
    *,
    backend: str = DEFAULT_NEIGHBOUR_BACKEND_ID,
    tile_size: int = 256,
    max_index_bytes: int | None = None,
    max_radius_query_bytes: int | None = None,
    max_neighbour_cache_bytes: int | None = DEFAULT_NEIGHBOUR_CACHE_BYTES,
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_gpu_neighbour_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
) -> Any:
    """Build the explicitly requested reusable index, with no fallback."""

    if backend == INDEXED_NEIGHBOUR_BACKEND_ID:
        return IndexedCPUNeighbourIndex(
            descriptors,
            max_index_bytes=max_index_bytes,
            max_radius_query_bytes=max_radius_query_bytes,
            max_neighbour_cache_bytes=max_neighbour_cache_bytes,
            resource_budget=resource_budget,
        )
    if backend == TILED_NEIGHBOUR_BACKEND_ID:
        return TiledCPUNeighbourIndex(
            descriptors,
            tile_size=tile_size,
            max_index_bytes=max_index_bytes,
            max_radius_query_bytes=max_radius_query_bytes,
            max_neighbour_cache_bytes=max_neighbour_cache_bytes,
            resource_budget=resource_budget,
        )
    if backend in GPU_NEIGHBOUR_BACKEND_ALIASES:
        from .gpu_neighbours import ExactTiledGPUNeighbourIndex

        return ExactTiledGPUNeighbourIndex(
            descriptors,
            device=gpu_device,
            gpu_devices=gpu_devices,
            gpu_tile_size=gpu_tile_size,
            max_device_bytes=max_gpu_neighbour_bytes,
            tile_size=tile_size,
            max_index_bytes=max_index_bytes,
            max_radius_query_bytes=max_radius_query_bytes,
            max_neighbour_cache_bytes=max_neighbour_cache_bytes,
            resource_budget=resource_budget,
        )
    if backend == NEIGHBOUR_BACKEND_ID:
        return None
    raise ValueError(f"configured neighbour backend is unavailable: {backend!r}")


def compute_radius_support(
    descriptors: np.ndarray,
    source_index: int,
    radius: float,
    *,
    backend: str = DEFAULT_NEIGHBOUR_BACKEND_ID,
    index: IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None = None,
    chunk_size: int = 1024,
    max_radius_query_bytes: int | None = None,
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_gpu_neighbour_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    context: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strict compact-kernel support for one source row.

    The indexed backend expands unique-location hits to all original rows;
    the reference backend retains its bounded full-pool scan semantics.
    """

    values = _validate_descriptors(descriptors)
    block = _validate_chunk_size(chunk_size)
    if max_radius_query_bytes is None and index is not None:
        limit = index.max_radius_query_bytes
    else:
        service = resource_budget or build_resource_budget()
        remaining = service.remaining_managed_budget
        if remaining is None:
            raise ResourceCapacityError(
                service.unknown_memory_message("exact radius support"),
                operation="exact radius support",
            )
        if remaining < 1:
            raise ResourceCapacityError(
                "exact radius support has no allocatable runtime memory headroom",
                operation="exact radius support",
                requested_bytes=1,
                available_bytes=remaining,
                reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
            )
        limit = (
            max(1, remaining // 2)
            if max_radius_query_bytes is None
            else min(
                _validate_memory_limit(max_radius_query_bytes, "max_radius_query_bytes"),
                remaining,
            )
        )
    if (
        backend
        in {
            INDEXED_NEIGHBOUR_BACKEND_ID,
            TILED_NEIGHBOUR_BACKEND_ID,
        }
        or backend in GPU_NEIGHBOUR_BACKEND_ALIASES
    ):
        active_index = (
            index
            if index is not None
            else build_neighbour_index(
                values,
                backend=backend,
                tile_size=block,
                max_radius_query_bytes=limit,
                gpu_device=gpu_device,
                gpu_devices=gpu_devices,
                gpu_tile_size=gpu_tile_size,
                max_gpu_neighbour_bytes=max_gpu_neighbour_bytes,
                resource_budget=resource_budget,
            )
        )
        if backend in GPU_NEIGHBOUR_BACKEND_ALIASES:
            from .gpu_neighbours import ExactTiledGPUNeighbourIndex

            expected_type = ExactTiledGPUNeighbourIndex
        else:
            expected_type = (
                IndexedCPUNeighbourIndex
                if backend == INDEXED_NEIGHBOUR_BACKEND_ID
                else TiledCPUNeighbourIndex
            )
        if not isinstance(active_index, expected_type) or not np.array_equal(
            active_index.descriptors, values
        ):
            raise ValueError("radius-support index belongs to different descriptors")
        if active_index.max_radius_query_bytes != limit:
            raise ValueError("indexed radius-support limit does not match the active index")
        return active_index.radius_support(source_index, radius, context=context)
    if backend != NEIGHBOUR_BACKEND_ID:
        raise ValueError(f"configured neighbour backend is unavailable: {backend!r}")
    if index is not None:
        raise ValueError("exact_cpu radius support does not accept an indexed index")
    source = int(source_index)
    if source < 0 or source >= values.shape[0]:
        raise IndexError("source_index is outside the descriptor rows")
    value = float(radius)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("radius must be finite and positive")
    scan_estimate = _estimate_radius_workspace_bytes(block, 0, values.shape[1])
    if scan_estimate > limit:
        details = [
            f"N={values.shape[0]}",
            f"d'={values.shape[1]}",
            f"source_row={source}",
            f"radius={value}",
            f"requested_bytes={scan_estimate}",
            f"max_radius_query_bytes={limit}",
        ]
        if context:
            details.append(context)
        raise RadiusQueryCapacityError(
            "exact radius query exceeds memory capacity: " + ", ".join(details)
        )
    included_count = 0
    for start in range(0, values.shape[0], block):
        stop = min(start + block, values.shape[0])
        with np.errstate(over="raise", invalid="raise"):
            try:
                distances = np.sqrt(np.sum((values[start:stop] - values[source]) ** 2, axis=1))
            except FloatingPointError as exc:
                raise ValueError(
                    "exact radius-support distance overflowed or became invalid"
                ) from exc
        if not np.all(np.isfinite(distances)):
            raise ValueError("exact radius-support distances must be finite")
        included = distances < value
        included_count += int(np.count_nonzero(included))
    if included_count == 0:
        raise ValueError("positive kernel radius support omitted the source row")
    estimated_bytes = _estimate_radius_workspace_bytes(
        block,
        included_count,
        values.shape[1],
        values.shape[0],
    )
    if estimated_bytes > limit:
        details = [
            f"N={values.shape[0]}",
            f"d'={values.shape[1]}",
            f"source_row={source}",
            f"radius={value}",
            f"requested_bytes={estimated_bytes}",
            f"max_radius_query_bytes={limit}",
        ]
        if context:
            details.append(context)
        raise RadiusQueryCapacityError(
            "exact radius query exceeds memory capacity: " + ", ".join(details)
        )
    targets = np.empty(included_count, dtype=np.int64)
    distances_out = np.empty(included_count, dtype=np.float64)
    position = 0
    for start in range(0, values.shape[0], block):
        stop = min(start + block, values.shape[0])
        with np.errstate(over="raise", invalid="raise"):
            try:
                distances = np.sqrt(np.sum((values[start:stop] - values[source]) ** 2, axis=1))
            except FloatingPointError as exc:
                raise ValueError(
                    "exact radius-support distance overflowed or became invalid"
                ) from exc
        included = distances < value
        count = int(np.count_nonzero(included))
        if count:
            targets[position : position + count] = np.arange(start, stop, dtype=np.int64)[included]
            distances_out[position : position + count] = distances[included]
            position += count
    if not np.any(targets == source):
        raise ValueError("positive kernel radius support omitted the source row")
    return targets, distances_out


def compute_neighbours(
    descriptors: np.ndarray,
    k: int,
    *,
    backend: str = DEFAULT_NEIGHBOUR_BACKEND_ID,
    index: IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None = None,
    **kwargs: Any,
) -> ExactNeighbourResult:
    """Dispatch to the explicitly configured neighbour backend."""

    if backend == NEIGHBOUR_BACKEND_ID:
        if index is not None:
            raise ValueError("exact_cpu does not accept an indexed neighbour index")
        return compute_exact_neighbours(descriptors, k, **kwargs)
    if backend == INDEXED_NEIGHBOUR_BACKEND_ID:
        if index is not None and not isinstance(index, IndexedCPUNeighbourIndex):
            raise ValueError("exact_indexed_cpu requires an indexed CPU backend")
        return compute_indexed_cpu_neighbours(descriptors, k, index=index, **kwargs)
    if backend == TILED_NEIGHBOUR_BACKEND_ID:
        if index is not None and not isinstance(index, TiledCPUNeighbourIndex):
            raise ValueError("exact_tiled_cpu requires the tiled CPU backend")
        return compute_tiled_cpu_neighbours(descriptors, k, index=index, **kwargs)
    if backend in GPU_NEIGHBOUR_BACKEND_ALIASES:
        from .gpu_neighbours import (
            ExactTiledGPUNeighbourIndex,
            compute_gpu_tiled_neighbours,
        )

        if index is not None and not isinstance(index, ExactTiledGPUNeighbourIndex):
            raise ValueError("exact_tiled_gpu_knn requires the GPU kNN backend")
        return compute_gpu_tiled_neighbours(descriptors, k, index=index, **kwargs)
    raise ValueError(f"configured neighbour backend is unavailable: {backend!r}")


def kth_distinct_neighbour_radii(
    descriptors: np.ndarray,
    k: int,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Return only exact kth-distinct-location radii."""

    return compute_exact_neighbours(descriptors, k, chunk_size=chunk_size).radii


calculate_kth_distinct_radii = kth_distinct_neighbour_radii
exact_neighbour_search = compute_exact_neighbours
compute_kth_distinct_neighbours = compute_exact_neighbours
compute_kth_distinct_neighbour_radii = kth_distinct_neighbour_radii
exact_kth_distinct_neighbour_radii = kth_distinct_neighbour_radii
NeighbourResult = ExactNeighbourResult


def __getattr__(name: str) -> Any:
    """Lazily expose optional GPU kNN symbols without importing CuPy."""

    if name in {
        "ExactTiledGPUNeighbourIndex",
        "compute_gpu_tiled_neighbours",
        "compute_gpu_tiled_neighbours_many",
        "compute_tiled_gpu_neighbours",
        "compute_tiled_gpu_neighbours_many",
    }:
        from . import gpu_neighbours

        return getattr(gpu_neighbours, name)
    raise AttributeError(name)


__all__ = [
    "CalibrationCapacityError",
    "DEFAULT_NEIGHBOUR_CACHE_BYTES",
    "DEFAULT_RADIUS_QUERY_BYTES",
    "ExactKNNPrefixSupport",
    "ExactSymmetricKNNUnionSupport",
    "ExactTiledCPUSupport",
    "SymmetricKNNPrefixSupport",
    "SymmetricTiledCPUSupport",
    "ExactNeighbourResult",
    "IndexedCPUNeighbourIndex",
    "TiledCPUNeighbourIndex",
    "GPU_NEIGHBOUR_BACKEND_ID",
    "GPU_NEIGHBOUR_BACKEND_VERSION",
    "ExactTiledGPUNeighbourIndex",  # noqa: F822 - lazy optional GPU export
    "NeighbourBackendCapabilities",
    "NeighbourResult",
    "RadiusQueryCapacityError",
    "calculate_kth_distinct_radii",
    "neighbour_backend_capabilities",
    "compute_exact_neighbours",
    "compute_indexed_cpu_neighbours",
    "compute_indexed_cpu_neighbours_many",
    "compute_tiled_cpu_neighbours",
    "compute_tiled_cpu_neighbours_many",
    "compute_radius_support",
    "build_neighbour_index",
    "compute_kth_distinct_neighbour_radii",
    "compute_kth_distinct_neighbours",
    "compute_neighbours",
    "compute_gpu_tiled_neighbours",  # noqa: F822 - lazy optional GPU export
    "compute_gpu_tiled_neighbours_many",  # noqa: F822 - lazy optional GPU export
    "compute_tiled_gpu_neighbours",  # noqa: F822 - lazy optional GPU export
    "compute_tiled_gpu_neighbours_many",  # noqa: F822 - lazy optional GPU export
    "exact_neighbour_search",
    "exact_kth_distinct_neighbour_radii",
    "kth_distinct_neighbour_radii",
]
