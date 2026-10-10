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
    NEIGHBOUR_BACKEND_ID,
    NEIGHBOUR_BACKEND_VERSION,
    NEIGHBOUR_DISTINCT_POLICY,
    NEIGHBOUR_METRIC,
)

logger = logging.getLogger(__name__)


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
    if row_ids is None:
        ordered_row_ids = tuple(str(index) for index in range(n_rows))
    else:
        ordered_row_ids = tuple(str(value) for value in row_ids)
        if len(ordered_row_ids) != n_rows or any(not value.strip() for value in ordered_row_ids):
            raise ValueError("row_ids must be non-blank and align with descriptor rows")
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
    )


def compute_neighbours(
    descriptors: np.ndarray,
    k: int,
    *,
    backend: str = NEIGHBOUR_BACKEND_ID,
    **kwargs: Any,
) -> ExactNeighbourResult:
    """Dispatch to the explicitly configured neighbour backend."""

    if backend != NEIGHBOUR_BACKEND_ID:
        raise ValueError(f"configured neighbour backend is unavailable: {backend!r}")
    return compute_exact_neighbours(descriptors, k, **kwargs)


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
    "NeighbourResult",
    "calculate_kth_distinct_radii",
    "compute_exact_neighbours",
    "compute_kth_distinct_neighbour_radii",
    "compute_kth_distinct_neighbours",
    "compute_neighbours",
    "exact_neighbour_search",
    "exact_kth_distinct_neighbour_radii",
    "kth_distinct_neighbour_radii",
]
