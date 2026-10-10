"""Deterministic, bounded-memory exact neighbours for entropy calibration."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json

from .models import (
    DEFAULT_NEIGHBOUR_BACKEND_ID,
    INDEXED_NEIGHBOUR_BACKEND_ID,
    INDEXED_NEIGHBOUR_BACKEND_VERSION,
    NEIGHBOUR_BACKEND_ID,
    NEIGHBOUR_BACKEND_VERSION,
    NEIGHBOUR_DISTINCT_POLICY,
    NEIGHBOUR_METRIC,
)

logger = logging.getLogger(__name__)


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
}


def neighbour_backend_capabilities(backend: str) -> NeighbourBackendCapabilities:
    """Return capabilities for a configured backend without fallback."""

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
    ) -> None:
        values = _validate_descriptors(descriptors)
        locations, first_indices, row_to_location = np.unique(
            values,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        self.descriptors = values
        self.unique_locations = np.ascontiguousarray(locations, dtype=np.float64)
        self.location_representatives = np.asarray(first_indices, dtype=np.int64)
        self.row_to_location = np.asarray(row_to_location, dtype=np.int64)
        self._location_rows = tuple(
            np.flatnonzero(self.row_to_location == location).astype(np.int64, copy=False)
            for location in range(self.unique_locations.shape[0])
        )
        try:
            from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]
        except ImportError as exc:
            raise ValueError(
                f"neighbour backend {self.backend!r} is unavailable: scipy.spatial.cKDTree"
            ) from exc
        self._tree: Any = cKDTree(self.unique_locations, copy_data=True)
        self._support_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
        self._neighbour_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.index_queries = 0
        self.radius_queries = 0
        self.distance_evaluations = 0
        self.index_bytes = self._estimate_index_bytes()
        if max_index_bytes is not None:
            if isinstance(max_index_bytes, bool) or not isinstance(
                max_index_bytes, (int, np.integer)
            ):
                raise ValueError("max_index_bytes must be a positive integer")
            if int(max_index_bytes) < 1:
                raise ValueError("max_index_bytes must be a positive integer")
            if self.index_bytes > int(max_index_bytes):
                raise ValueError(
                    "indexed exact neighbour backend exceeds max_index_bytes: "
                    f"N={values.shape[0]}, Q={self.unique_locations.shape[0]}, "
                    f"estimated_peak_bytes={self.index_bytes}, max_index_bytes={int(max_index_bytes)}"
                )
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

    def _estimate_index_bytes(self) -> int:
        n_rows = int(self.descriptors.shape[0])
        n_locations = int(self.unique_locations.shape[0])
        # Include NumPy storage, row buckets, and a conservative cKDTree /
        # Python-container allowance.  This is an operational upper estimate,
        # not a claim about the allocator's exact resident-set accounting.
        return int(
            self.unique_locations.nbytes
            + self.location_representatives.nbytes
            + self.row_to_location.nbytes
            + 8 * n_rows
            + 256 * n_locations
            + 4096
        )

    @staticmethod
    def _distance_values(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
        with np.errstate(over="raise", invalid="raise"):
            try:
                distances = np.sqrt(np.sum((targets - query) ** 2, axis=1))
            except FloatingPointError as exc:
                raise ValueError("indexed neighbour distance overflowed or became invalid") from exc
        if not np.all(np.isfinite(distances)):
            raise ValueError("indexed neighbour distances must be finite")
        return np.asarray(distances, dtype=np.float64)

    def _location_neighbours(self, location_index: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._neighbour_cache.get(k)
        if cached is not None:
            all_indices, all_distances = cached
            return all_indices[location_index], all_distances[location_index]
        n_locations = self.unique_locations.shape[0]
        if n_locations - 1 < k:
            raise ValueError(
                f"k={k} requires at least {k} other distinct descriptor locations; "
                f"the pool contains only {n_locations - 1}"
            )
        neighbour_indices = np.empty((n_locations, k), dtype=np.int64)
        neighbour_distances = np.empty((n_locations, k), dtype=np.float64)
        for query_location in range(n_locations):
            query = self.unique_locations[query_location]
            # The query point itself is one of the returned points.  The
            # (k+1)-th tree distance is therefore a conservative kth-other
            # boundary even when that boundary has many ties.
            self.index_queries += 1
            tree_distances, tree_indices = self._tree.query(query, k=k + 1, workers=1)
            del tree_distances
            candidate_indices = np.asarray(tree_indices, dtype=np.int64).reshape(-1)
            non_self = candidate_indices != query_location
            if not np.any(non_self):
                raise ValueError(f"k={k} has insufficient distinct neighbours")
            initial_indices = candidate_indices[non_self]
            initial_distances = self._distance_values(
                query,
                self.unique_locations[initial_indices],
            )
            self.distance_evaluations += int(initial_indices.shape[0])
            kth_boundary = float(np.sort(initial_distances)[k - 1])
            search_radius = kth_boundary + max(
                16.0 * float(np.spacing(kth_boundary)),
                1.0e-14,
            )
            boundary_indices = np.asarray(
                self._tree.query_ball_point(query, search_radius),
                dtype=np.int64,
            )
            self.radius_queries += 1
            boundary_indices = boundary_indices[boundary_indices != query_location]
            distances = self._distance_values(query, self.unique_locations[boundary_indices])
            self.distance_evaluations += int(boundary_indices.shape[0])
            positive = distances > 0.0
            boundary_indices = boundary_indices[positive]
            distances = distances[positive]
            if distances.size < k:
                raise ValueError(
                    "indexed neighbour boundary query returned fewer candidates than the "
                    "validated kth distance"
                )
            order = np.lexsort(
                (
                    self.location_representatives[boundary_indices],
                    distances,
                )
            )[:k]
            neighbour_indices[query_location] = self.location_representatives[
                boundary_indices[order]
            ]
            neighbour_distances[query_location] = distances[order]
        neighbour_indices.setflags(write=False)
        neighbour_distances.setflags(write=False)
        self._neighbour_cache[k] = (neighbour_indices, neighbour_distances)
        return neighbour_indices[location_index], neighbour_distances[location_index]

    def query_neighbours(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return radii, representative rows, and distances for all rows."""

        order_k = _validate_k(k)
        location_indices, location_distances = self._neighbour_cache.get(order_k, (None, None))
        if location_indices is None or location_distances is None:
            self._location_neighbours(0, order_k)
            location_indices, location_distances = self._neighbour_cache[order_k]
        row_locations = self.row_to_location
        rows = np.asarray(location_indices[row_locations], dtype=np.int64).copy()
        distances = np.asarray(location_distances[row_locations], dtype=np.float64).copy()
        radii = distances[:, -1].copy()
        return radii, rows, distances

    def radius_support(self, source_index: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Return all original target rows with exact distance strictly below radius."""

        if isinstance(source_index, bool) or not isinstance(source_index, (int, np.integer)):
            raise ValueError("source_index must be a non-negative integer")
        source = int(source_index)
        if source < 0 or source >= self.descriptors.shape[0]:
            raise IndexError("source_index is outside the descriptor rows")
        value = float(radius)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("radius must be finite and positive")
        key = (int(self.row_to_location[source]), value)
        cached = self._support_cache.get(key)
        if cached is not None:
            return cached
        query = self.descriptors[source]
        expanded_radius = value + max(16.0 * float(np.spacing(value)), 1.0e-14)
        location_indices = np.asarray(
            self._tree.query_ball_point(query, expanded_radius),
            dtype=np.int64,
        )
        self.radius_queries += 1
        location_distances = self._distance_values(query, self.unique_locations[location_indices])
        self.distance_evaluations += int(location_indices.shape[0])
        included = location_distances < value
        location_indices = location_indices[included]
        location_distances = location_distances[included]
        if location_indices.size == 0:
            raise ValueError("positive kernel radius support omitted the source row")
        row_parts = [self._location_rows[int(location)] for location in location_indices]
        target_indices = np.sort(np.concatenate(row_parts).astype(np.int64, copy=False))
        target_distances = self._distance_values(query, self.descriptors[target_indices])
        self.distance_evaluations += int(target_indices.shape[0])
        strict = target_distances < value
        target_indices = target_indices[strict]
        target_distances = target_distances[strict]
        if not np.any(target_indices == source):
            raise ValueError("positive kernel radius support omitted the source row")
        target_indices.setflags(write=False)
        target_distances.setflags(write=False)
        self._support_cache[key] = (target_indices, target_distances)
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
    max_neighbour_entries: int = 1_000_000,
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
    if isinstance(max_neighbour_entries, bool) or not isinstance(
        max_neighbour_entries, (int, np.integer)
    ):
        raise ValueError("max_neighbour_entries must be a positive integer")
    max_entries = int(max_neighbour_entries)
    if max_entries < 1:
        raise ValueError("max_neighbour_entries must be a positive integer")
    n_rows = values.shape[0]
    if n_rows * order_k > max_entries:
        raise ValueError(
            "exact neighbour result exceeds its explicit bounded-memory limit; "
            "reduce k or configure a scalable neighbour backend"
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

    for query_index in range(n_rows):
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
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("exact kth-neighbour radii must be finite and positive")

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
    max_neighbour_entries: int = 1_000_000,
) -> ExactNeighbourResult:
    """Calculate exact kth other-location neighbours through the indexed CPU backend."""

    values = _validate_descriptors(descriptors)
    order_k = _validate_k(k)
    _validate_chunk_size(chunk_size)
    if isinstance(max_neighbour_entries, bool) or not isinstance(
        max_neighbour_entries, (int, np.integer)
    ):
        raise ValueError("max_neighbour_entries must be a positive integer")
    max_entries = int(max_neighbour_entries)
    if max_entries < 1:
        raise ValueError("max_neighbour_entries must be a positive integer")
    if values.shape[0] * order_k > max_entries:
        raise ValueError(
            "indexed exact neighbour result exceeds its explicit bounded-memory limit: "
            f"N={values.shape[0]}, required_entries={values.shape[0] * order_k}, "
            f"max_entries={max_entries}"
        )
    active_index = index if index is not None else IndexedCPUNeighbourIndex(values)
    if not np.array_equal(active_index.descriptors, values):
        raise ValueError("indexed neighbour index belongs to different descriptors")
    ordered_row_ids = _validate_row_ids(row_ids, values.shape[0])
    radii, neighbour_indices, neighbour_distances = active_index.query_neighbours(order_k)
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


def build_neighbour_index(
    descriptors: np.ndarray,
    *,
    backend: str = DEFAULT_NEIGHBOUR_BACKEND_ID,
    max_index_bytes: int | None = None,
) -> IndexedCPUNeighbourIndex | None:
    """Build the explicitly requested reusable index, with no fallback."""

    if backend == INDEXED_NEIGHBOUR_BACKEND_ID:
        return IndexedCPUNeighbourIndex(descriptors, max_index_bytes=max_index_bytes)
    if backend == NEIGHBOUR_BACKEND_ID:
        return None
    raise ValueError(f"configured neighbour backend is unavailable: {backend!r}")


def compute_radius_support(
    descriptors: np.ndarray,
    source_index: int,
    radius: float,
    *,
    backend: str = DEFAULT_NEIGHBOUR_BACKEND_ID,
    index: IndexedCPUNeighbourIndex | None = None,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strict compact-kernel support for one source row.

    The indexed backend expands unique-location hits to all original rows;
    the reference backend retains its bounded full-pool scan semantics.
    """

    values = _validate_descriptors(descriptors)
    block = _validate_chunk_size(chunk_size)
    if backend == INDEXED_NEIGHBOUR_BACKEND_ID:
        active_index = index if index is not None else IndexedCPUNeighbourIndex(values)
        if not np.array_equal(active_index.descriptors, values):
            raise ValueError("indexed radius-support index belongs to different descriptors")
        return active_index.radius_support(source_index, radius)
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
    targets: list[np.ndarray] = []
    distances_out: list[np.ndarray] = []
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
        if np.any(included):
            targets.append(np.arange(start, stop, dtype=np.int64)[included])
            distances_out.append(distances[included])
    if not targets:
        raise ValueError("positive kernel radius support omitted the source row")
    return np.concatenate(targets), np.concatenate(distances_out)


def compute_neighbours(
    descriptors: np.ndarray,
    k: int,
    *,
    backend: str = DEFAULT_NEIGHBOUR_BACKEND_ID,
    index: IndexedCPUNeighbourIndex | None = None,
    **kwargs: Any,
) -> ExactNeighbourResult:
    """Dispatch to the explicitly configured neighbour backend."""

    if backend == NEIGHBOUR_BACKEND_ID:
        if index is not None:
            raise ValueError("exact_cpu does not accept an indexed neighbour index")
        return compute_exact_neighbours(descriptors, k, **kwargs)
    if backend == INDEXED_NEIGHBOUR_BACKEND_ID:
        return compute_indexed_cpu_neighbours(descriptors, k, index=index, **kwargs)
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


__all__ = [
    "ExactNeighbourResult",
    "IndexedCPUNeighbourIndex",
    "NeighbourBackendCapabilities",
    "NeighbourResult",
    "calculate_kth_distinct_radii",
    "neighbour_backend_capabilities",
    "compute_exact_neighbours",
    "compute_indexed_cpu_neighbours",
    "compute_radius_support",
    "build_neighbour_index",
    "compute_kth_distinct_neighbour_radii",
    "compute_kth_distinct_neighbours",
    "compute_neighbours",
    "exact_neighbour_search",
    "exact_kth_distinct_neighbour_radii",
    "kth_distinct_neighbour_radii",
]
