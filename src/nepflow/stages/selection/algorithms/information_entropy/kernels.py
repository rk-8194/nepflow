"""Versioned Wendland kernels and bounded finite-pool calibration evaluation."""

from __future__ import annotations

import hashlib
import logging
import math
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json
from nepflow.resources.budget import (
    ResourceBudgetService,
    ResourceCapacityError,
    build_resource_budget,
)

from .models import (
    DEFAULT_RADIUS_QUERY_BYTES,
    INDEXED_NEIGHBOUR_BACKEND_ID,
    INDEXED_NEIGHBOUR_BACKEND_VERSION,
    KERNEL_OPERATOR_SCHEMA_VERSION,
    KERNEL_FAMILY,
    KERNEL_VERSION,
    NEIGHBOUR_BACKEND_ID,
    NEIGHBOUR_BACKEND_VERSION,
    NEIGHBOUR_METRIC,
    SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
    SPARSE_STREAMED_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
    SPARSE_KERNEL_GRAPH_SCHEMA_VERSION,
    SPARSE_NUMERICAL_TOLERANCE,
    BandwidthCalibrationResult,
    EntropyPool,
    FrozenBandwidths,
    KernelMetadata,
    SparseAtomicKernelGraph,
    SparseCandidateContributions,
    StreamedKernelExecutionSummary,
)
from .neighbours import (
    IndexedCPUNeighbourIndex,
    RadiusQueryCapacityError,
    _estimate_radius_workspace_bytes,
    _validate_chunk_size,
    _validate_descriptors,
    _validate_memory_limit,
    build_neighbour_index,
    compute_radius_support,
)

SourceProgressCallback = Callable[[int, int], None]
BatchSourceProgressCallback = Callable[[int, int], None]

logger = logging.getLogger(__name__)


_EDGE_SPOOL_DTYPE = np.dtype(([("target", "<i8"), ("value", "<f8")]))
_EDGE_SPOOL_HEADER = b"NEPFLOW-EXACT-EDGE-SPOOL\0v1\0target-i64-value-f64\0"
_EDGE_SPOOL_RECORD_BYTES = int(_EDGE_SPOOL_DTYPE.itemsize)


def _validate_spool_limit(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _validate_sparse_limit(value, name)


def _spool_write_header(handle: Any) -> Any:
    handle.write(_EDGE_SPOOL_HEADER)
    digest = hashlib.sha256()
    digest.update(_EDGE_SPOOL_HEADER)
    return digest


def _spool_write_records(handle: Any, digest: Any, records: np.ndarray) -> None:
    raw = memoryview(np.ascontiguousarray(records, dtype=_EDGE_SPOOL_DTYPE)).cast("B")
    digest.update(raw)
    handle.write(raw)


def _spool_read_records(
    handle: Any,
    digest: Any,
    count: int,
    *,
    chunk_records: int = 65_536,
) -> Iterator[np.ndarray]:
    remaining = count
    while remaining:
        raw = handle.read(min(remaining, chunk_records) * _EDGE_SPOOL_RECORD_BYTES)
        expected = min(remaining, chunk_records) * _EDGE_SPOOL_RECORD_BYTES
        if len(raw) != expected:
            raise ValueError("exact edge spool is truncated")
        digest.update(raw)
        yield np.frombuffer(raw, dtype=_EDGE_SPOOL_DTYPE)
        remaining -= min(remaining, chunk_records)


def _validate_bandwidths(bandwidths: Any, row_count: int) -> np.ndarray:
    values = np.asarray(bandwidths)
    if values.dtype != np.dtype(np.float64):
        raise ValueError("bandwidths must have dtype float64")
    if values.ndim != 1 or values.shape[0] != row_count:
        raise ValueError("bandwidths must contain one value per descriptor row")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("bandwidths must be finite and strictly positive")
    return np.ascontiguousarray(values, dtype=np.float64)


def wendland_kernel(u: Any) -> float | np.ndarray:
    """Evaluate the versioned compactly supported Wendland-type kernel."""

    values = np.asarray(u, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("kernel arguments must be finite and non-negative")
    result = np.zeros(values.shape, dtype=np.float64)
    supported = values < 1.0
    with np.errstate(over="raise", invalid="raise"):
        result[supported] = (1.0 - values[supported]) ** 4 * (4.0 * values[supported] + 1.0)
    if values.ndim == 0:
        return float(result)
    return result


wendland_c2 = wendland_kernel
wendland = wendland_kernel
evaluate_wendland_kernel = wendland_kernel


@dataclass(frozen=True, slots=True)
class NormalizedKernelColumn:
    """One sparse source column, retaining target row identity."""

    source_index: int
    target_indices: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        indices = np.array(self.target_indices, dtype=np.int64, copy=True)
        values = np.array(self.values, dtype=np.float64, copy=True)
        if indices.ndim != 1 or values.ndim != 1 or indices.shape != values.shape:
            raise ValueError("normalized kernel column arrays must be aligned vectors")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("normalized kernel column values must be finite and positive")
        indices.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "target_indices", indices)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class LeaveOneOutObjective:
    """Finite-pool leave-one-out probabilities and cross-entropy result."""

    valid: bool
    objective: float | None
    probabilities: np.ndarray
    reason: str | None = None

    def __post_init__(self) -> None:
        values = np.array(self.probabilities, dtype=np.float64, copy=True)
        if values.ndim != 1:
            raise ValueError("leave-one-out probabilities must be a vector")
        if self.valid and (not np.all(np.isfinite(values)) or np.any(values <= 0.0)):
            raise ValueError("valid leave-one-out probabilities must be finite and positive")
        values.setflags(write=False)
        object.__setattr__(self, "probabilities", values)


def _distance_block(
    descriptors: np.ndarray, source_index: int, start: int, stop: int
) -> np.ndarray:
    source = descriptors[source_index]
    with np.errstate(over="raise", invalid="raise"):
        try:
            distances = np.sqrt(np.sum((descriptors[start:stop] - source) ** 2, axis=1))
        except FloatingPointError as exc:
            raise ValueError("kernel distance overflowed or became invalid") from exc
    if not np.all(np.isfinite(distances)):
        raise ValueError("kernel distances must be finite")
    return distances


class _KernelSupportProvider:
    """Exact compact-support queries shared by normalisation and graph passes."""

    def __init__(
        self,
        descriptors: np.ndarray,
        *,
        backend: str,
        chunk_size: int,
        index: IndexedCPUNeighbourIndex | None = None,
        max_radius_query_bytes: int | None = None,
        resource_budget: ResourceBudgetService | None = None,
        capacity_context: str | None = None,
    ) -> None:
        self.descriptors = _validate_descriptors(descriptors)
        self.backend = backend
        self.chunk_size = _validate_chunk_size(chunk_size)
        self.resource_budget = (
            resource_budget
            or getattr(index, "resource_budget", None)
            or build_resource_budget()
        )
        if max_radius_query_bytes is None and index is not None:
            self.max_radius_query_bytes = index.max_radius_query_bytes
        else:
            service = self.resource_budget
            remaining = service.remaining_managed_budget
            if remaining is None:
                raise ResourceCapacityError(
                    service.unknown_memory_message("entropy kernel support queries"),
                    operation="entropy kernel support queries",
                )
            if remaining < 1:
                raise ResourceCapacityError(
                    "kernel support queries have no allocatable runtime memory headroom",
                    operation="entropy kernel support queries",
                    requested_bytes=1,
                    available_bytes=remaining,
                    reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
                )
            if max_radius_query_bytes is None:
                self.max_radius_query_bytes = max(1, remaining // 2)
            else:
                if isinstance(max_radius_query_bytes, bool) or not isinstance(
                    max_radius_query_bytes, (int, np.integer)
                ):
                    raise ValueError("max_radius_query_bytes must be a positive integer")
                self.max_radius_query_bytes = min(int(max_radius_query_bytes), remaining)
                if self.max_radius_query_bytes < 1:
                    raise ValueError("max_radius_query_bytes must be a positive integer")
        self.capacity_context = capacity_context
        if backend == INDEXED_NEIGHBOUR_BACKEND_ID:
            self.index = (
                index
                if index is not None
                else build_neighbour_index(
                    self.descriptors,
                    backend=backend,
                    max_radius_query_bytes=self.max_radius_query_bytes,
                    resource_budget=self.resource_budget,
                )
            )
            if not isinstance(self.index, IndexedCPUNeighbourIndex):
                raise ValueError("indexed exact kernel support requires its indexed CPU backend")
            if not np.array_equal(self.index.descriptors, self.descriptors):
                raise ValueError("indexed kernel support index belongs to different descriptors")
        elif backend == NEIGHBOUR_BACKEND_ID:
            if index is not None:
                raise ValueError("exact_cpu kernel support does not accept an indexed index")
            self.index = None
        else:
            raise ValueError(f"configured neighbour backend is unavailable: {backend!r}")
        self.distance_evaluations = 0
        self.index_queries = 0
        self.radius_queries = 0
        self.support_entries = 0
        self._normaliser_cache: dict[tuple[int, float], float] = {}
        self.current_support_count = 0
        self.min_support_count: int | None = None
        self.support_count_total = 0
        self.support_count_samples = 0
        self.max_support_count = 0
        self.query_workspace_peak_bytes = 0

    def support(self, source_index: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
        scale = float(radius)
        if self.index is not None:
            self.index_queries += 1
            target_indices, distances = self.index.radius_support(
                source_index,
                scale,
                context=self.capacity_context,
            )
            self.radius_queries = self.index.radius_queries
            self.query_workspace_peak_bytes = self.index.query_workspace_peak_bytes
            self.support_entries += int(target_indices.shape[0])
            self.current_support_count = int(target_indices.shape[0])
            if self.min_support_count is None:
                self.min_support_count = self.current_support_count
            else:
                self.min_support_count = min(self.min_support_count, self.current_support_count)
            self.support_count_total += self.current_support_count
            self.support_count_samples += 1
            self.max_support_count = max(self.max_support_count, self.current_support_count)
            return target_indices, distances
        self.radius_queries += 1
        targets, distances = compute_radius_support(
            self.descriptors,
            source_index,
            scale,
            backend=NEIGHBOUR_BACKEND_ID,
            chunk_size=self.chunk_size,
            max_radius_query_bytes=self.max_radius_query_bytes,
            resource_budget=self.resource_budget,
            context=self.capacity_context,
        )
        self.distance_evaluations += self.descriptors.shape[0]
        self.support_entries += int(targets.shape[0])
        self.current_support_count = int(targets.shape[0])
        if self.min_support_count is None:
            self.min_support_count = self.current_support_count
        else:
            self.min_support_count = min(self.min_support_count, self.current_support_count)
        self.support_count_total += self.current_support_count
        self.support_count_samples += 1
        self.max_support_count = max(self.max_support_count, self.current_support_count)
        return targets, distances

    def normaliser(
        self,
        source_index: int,
        radius: float,
        target_indices: np.ndarray | None = None,
        distances: np.ndarray | None = None,
    ) -> float:
        key = (int(source_index), float(radius))
        cached = self._normaliser_cache.get(key)
        if cached is not None:
            return cached
        if target_indices is None or distances is None:
            target_indices, distances = self.support(source_index, radius)
        del target_indices
        raw = np.asarray(wendland_kernel(distances / float(radius)), dtype=np.float64)
        total = float(np.sum(raw, dtype=np.float64))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(f"source kernel normaliser is invalid for source {source_index}")
        self._normaliser_cache[key] = total
        return total


def source_normalisers(
    descriptors: np.ndarray,
    bandwidths: np.ndarray,
    *,
    chunk_size: int = 1024,
    backend: str = NEIGHBOUR_BACKEND_ID,
    index: IndexedCPUNeighbourIndex | None = None,
    support_provider: _KernelSupportProvider | None = None,
    max_radius_query_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    capacity_context: str | None = None,
) -> np.ndarray:
    """Return ``Z_b`` over the complete exact compact-kernel support."""

    values = _validate_descriptors(descriptors)
    scales = _validate_bandwidths(bandwidths, values.shape[0])
    block = _validate_chunk_size(chunk_size)
    provider = support_provider or _KernelSupportProvider(
        values,
        backend=backend,
        chunk_size=block,
        index=index,
        max_radius_query_bytes=max_radius_query_bytes,
        resource_budget=resource_budget,
        capacity_context=capacity_context,
    )
    normalisers = np.empty(values.shape[0], dtype=np.float64)
    for source_index, scale in enumerate(scales):
        targets, distances = provider.support(source_index, float(scale))
        normalisers[source_index] = provider.normaliser(
            source_index,
            float(scale),
            targets,
            distances,
        )
    normalisers.setflags(write=False)
    return normalisers


def iter_normalized_kernel_columns(
    descriptors: np.ndarray,
    bandwidths: np.ndarray,
    *,
    chunk_size: int = 1024,
    progress_callback: SourceProgressCallback | None = None,
    backend: str = NEIGHBOUR_BACKEND_ID,
    index: IndexedCPUNeighbourIndex | None = None,
    support_provider: _KernelSupportProvider | None = None,
    max_radius_query_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    capacity_context: str | None = None,
    source_indices: Sequence[int] | np.ndarray | None = None,
) -> Iterator[NormalizedKernelColumn]:
    """Yield positive normalized source columns using bounded workspaces.

    A source's bandwidth is always ``bandwidths[source_index]``.  The
    normaliser includes every finite-pool target row, including self.
    """

    values = _validate_descriptors(descriptors)
    scales = _validate_bandwidths(bandwidths, values.shape[0])
    block = _validate_chunk_size(chunk_size)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    provider = support_provider or _KernelSupportProvider(
        values,
        backend=backend,
        chunk_size=block,
        index=index,
        max_radius_query_bytes=max_radius_query_bytes,
        resource_budget=resource_budget,
        capacity_context=capacity_context,
    )
    if source_indices is None:
        ordered_sources = range(values.shape[0])
    else:
        ordered_sources = tuple(int(source_index) for source_index in source_indices)
        if any(
            source_index < 0 or source_index >= values.shape[0]
            for source_index in ordered_sources
        ):
            raise ValueError("source_indices contains an out-of-range source row")
    for source_index in ordered_sources:
        scale = scales[source_index]
        target_indices, distances = provider.support(source_index, float(scale))
        normaliser = provider.normaliser(
            source_index,
            float(scale),
            target_indices,
            distances,
        )
        raw = np.asarray(wendland_kernel(distances / scale), dtype=np.float64)
        positive = raw > 0.0
        if not np.all(positive):
            target_indices = target_indices[positive]
            raw = raw[positive]
        normalized = raw / normaliser
        column_total = float(np.sum(normalized, dtype=np.float64))
        if target_indices.size:
            yield NormalizedKernelColumn(source_index, target_indices, normalized)
        if not math.isclose(column_total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"normalized source kernel column {source_index} does not sum to one")
        if progress_callback is not None:
            progress_callback(source_index + 1, values.shape[0])


def normalized_kernel_matrix(
    descriptors: np.ndarray,
    bandwidths: np.ndarray,
    *,
    chunk_size: int = 1024,
    max_dense_entries: int = 1_000_000,
    max_radius_query_bytes: int | None = None,
) -> np.ndarray:
    """Build a deliberately bounded dense oracle for small correctness tests.

    Production calibration uses :func:`iter_normalized_kernel_columns` and
    never calls this helper.  The guard prevents accidental whole-pool use.
    Rows are targets and columns are sources.
    """

    values = _validate_descriptors(descriptors)
    _validate_bandwidths(bandwidths, values.shape[0])
    if isinstance(max_dense_entries, bool) or not isinstance(max_dense_entries, int):
        raise ValueError("max_dense_entries must be a positive integer")
    if max_dense_entries < 1 or values.shape[0] * values.shape[0] > max_dense_entries:
        raise ValueError("dense normalized kernel oracle exceeds its explicit memory bound")
    matrix = np.zeros((values.shape[0], values.shape[0]), dtype=np.float64)
    for column in iter_normalized_kernel_columns(
        values,
        np.asarray(bandwidths, dtype=np.float64),
        chunk_size=chunk_size,
        max_radius_query_bytes=max_radius_query_bytes,
    ):
        matrix[column.target_indices, column.source_index] = column.values
    return matrix


normalised_kernel_matrix = normalized_kernel_matrix


def evaluate_leave_one_out_objective(
    descriptors: np.ndarray,
    probabilities: np.ndarray,
    bandwidths: np.ndarray,
    *,
    chunk_size: int = 1024,
    progress_callback: SourceProgressCallback | None = None,
    backend: str = NEIGHBOUR_BACKEND_ID,
    index: IndexedCPUNeighbourIndex | None = None,
    max_radius_query_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    capacity_context: str | None = None,
) -> LeaveOneOutObjective:
    """Evaluate the exact source-normalized finite-pool LOO objective."""

    values = _validate_descriptors(descriptors)
    masses = np.asarray(probabilities)
    if masses.dtype != np.dtype(np.float64):
        raise ValueError("probabilities must have dtype float64")
    if masses.ndim != 1 or masses.shape[0] != values.shape[0]:
        raise ValueError("probabilities must align with descriptor rows")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("probabilities must be finite and strictly positive")
    if not math.isclose(float(np.sum(masses, dtype=np.float64)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("probabilities must sum to one")
    if values.shape[0] < 2:
        raise ValueError("leave-one-out calibration requires at least two rows")
    denominators = 1.0 - masses
    if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0.0):
        raise ValueError("leave-one-out requires 1 - p_a to be strictly positive")

    numerator = np.zeros(values.shape[0], dtype=np.float64)
    for column in iter_normalized_kernel_columns(
        values,
        _validate_bandwidths(bandwidths, values.shape[0]),
        chunk_size=chunk_size,
        progress_callback=progress_callback,
        backend=backend,
        index=index,
        max_radius_query_bytes=max_radius_query_bytes,
        resource_budget=resource_budget,
        capacity_context=capacity_context,
    ):
        non_source = column.target_indices != column.source_index
        if np.any(non_source):
            numerator[column.target_indices[non_source]] += (
                masses[column.source_index] * column.values[non_source]
            )
    with np.errstate(divide="ignore", invalid="ignore", over="raise"):
        try:
            loo = numerator / denominators
        except FloatingPointError as exc:
            raise ValueError("leave-one-out probability reduction overflowed") from exc
    invalid = ~np.isfinite(loo) | (loo <= 0.0)
    if np.any(invalid):
        first = int(np.flatnonzero(invalid)[0])
        return LeaveOneOutObjective(
            valid=False,
            objective=None,
            probabilities=loo,
            reason=f"zero or non-finite leave-one-out support at target row {first}",
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        objective = float(-np.sum(masses * np.log(loo), dtype=np.float64))
    if not math.isfinite(objective):
        return LeaveOneOutObjective(
            valid=False,
            objective=None,
            probabilities=loo,
            reason="leave-one-out objective is non-finite",
        )
    return LeaveOneOutObjective(valid=True, objective=objective, probabilities=loo)


def evaluate_leave_one_out_objectives(
    descriptors: np.ndarray,
    probabilities: np.ndarray,
    bandwidths: np.ndarray,
    *,
    chunk_size: int = 1024,
    progress_callback: BatchSourceProgressCallback | None = None,
    backend: str = NEIGHBOUR_BACKEND_ID,
    index: IndexedCPUNeighbourIndex | None = None,
    max_radius_query_bytes: int | None = None,
    max_calibration_work_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    capacity_context: str | None = None,
) -> tuple[LeaveOneOutObjective, ...]:
    """Evaluate a bounded batch of exact LOO objectives source-by-source.

    ``bandwidths`` has shape ``(C, N)``.  The maximum source radius is queried
    once and filtered independently for each of the ``C`` bandwidth vectors.
    Each numerator row is updated in source-major order, matching the
    single-bandwidth reference evaluator.
    """

    values = _validate_descriptors(descriptors)
    masses = np.asarray(probabilities)
    if masses.dtype != np.dtype(np.float64):
        raise ValueError("probabilities must have dtype float64")
    if masses.ndim != 1 or masses.shape[0] != values.shape[0]:
        raise ValueError("probabilities must align with descriptor rows")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("probabilities must be finite and strictly positive")
    if not math.isclose(float(np.sum(masses, dtype=np.float64)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("probabilities must sum to one")
    if values.shape[0] < 2:
        raise ValueError("leave-one-out calibration requires at least two rows")
    denominators = 1.0 - masses
    if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0.0):
        raise ValueError("leave-one-out requires 1 - p_a to be strictly positive")
    scales = np.asarray(bandwidths)
    if scales.dtype != np.dtype(np.float64):
        raise ValueError("bandwidths must have dtype float64")
    if scales.ndim != 2 or scales.shape[1] != values.shape[0] or scales.shape[0] == 0:
        raise ValueError("batched bandwidths must have shape (C, N)")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("batched bandwidths must be finite and strictly positive")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    service = resource_budget or build_resource_budget()
    remaining = service.remaining_managed_budget
    if remaining is None:
        raise ResourceCapacityError(
            service.unknown_memory_message("entropy batched leave-one-out evaluation"),
            operation="entropy batched leave-one-out evaluation",
        )
    if remaining < 1:
        raise ResourceCapacityError(
            "batched leave-one-out evaluation has no allocatable runtime memory headroom",
            operation="entropy batched leave-one-out evaluation",
            requested_bytes=1,
            available_bytes=remaining,
            reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
        )
    if max_calibration_work_bytes is None or max_radius_query_bytes is None:
        work_limit = min(max_calibration_work_bytes or remaining, remaining)
        radius_limit = min(
            (
                index.max_radius_query_bytes
                if max_radius_query_bytes is None and index is not None
                else max_radius_query_bytes or max(1, remaining // 2)
            ),
            work_limit,
        )
    else:
        work_limit = _validate_memory_limit(
            max_calibration_work_bytes, "max_calibration_work_bytes"
        )
        radius_limit = min(
            _validate_memory_limit(max_radius_query_bytes, "max_radius_query_bytes"),
            work_limit,
        )
    block = _validate_chunk_size(chunk_size)
    batch_count = int(scales.shape[0])
    row_count = int(values.shape[0])
    # Keep a conservative allowance for the CxN numerator and bandwidth
    # batch, while the radius-query guard accounts for source-local geometry.
    batch_arrays_bytes = 16 * batch_count * row_count + 8192
    if batch_arrays_bytes > work_limit:
        details = [
            f"N={row_count}",
            f"d'={values.shape[1]}",
            f"batch_count={batch_count}",
            f"requested_bytes={batch_arrays_bytes}",
            f"max_calibration_work_bytes={work_limit}",
        ]
        if capacity_context:
            details.append(capacity_context)
        raise RadiusQueryCapacityError(
            "batched calibration workspace exceeds memory capacity: " + ", ".join(details)
        )
    if isinstance(index, IndexedCPUNeighbourIndex) and index.max_radius_query_bytes > radius_limit:
        raise RadiusQueryCapacityError(
            "indexed radius-query limit exceeds batched calibration workspace: "
            f"configured_max_radius_query_bytes={index.max_radius_query_bytes}, "
            f"max_calibration_work_bytes={work_limit}"
        )
    provider = _KernelSupportProvider(
        values,
        backend=backend,
        chunk_size=block,
        index=index,
        max_radius_query_bytes=radius_limit,
        resource_budget=service,
        capacity_context=capacity_context,
    )
    numerator = np.zeros((batch_count, row_count), dtype=np.float64)
    invalid_reasons: list[str | None] = [None] * batch_count
    peak_workspace_bytes = batch_arrays_bytes
    for source_index in range(row_count):
        source_scales = scales[:, source_index]
        maximum_scale = float(np.max(source_scales))
        target_indices, distances = provider.support(source_index, maximum_scale)
        support_count = int(target_indices.shape[0])
        query_bytes = int(getattr(provider.index, "query_workspace_peak_bytes", 0))
        requested_bytes = batch_arrays_bytes + query_bytes + 32 * support_count
        peak_workspace_bytes = max(peak_workspace_bytes, requested_bytes)
        if requested_bytes > work_limit:
            details = [
                f"N={row_count}",
                f"d'={values.shape[1]}",
                f"source_row={source_index}",
                f"radius={maximum_scale}",
                f"requested_bytes={requested_bytes}",
                f"max_calibration_work_bytes={work_limit}",
            ]
            if capacity_context:
                details.append(capacity_context)
            raise RadiusQueryCapacityError(
                "batched calibration workspace exceeds memory capacity: "
                + ", ".join(details)
            )
        for batch_index, scale in enumerate(source_scales):
            if invalid_reasons[batch_index] is not None:
                continue
            try:
                included = distances < float(scale)
                selected_targets = target_indices[included]
                selected_distances = distances[included]
                raw = np.asarray(
                    wendland_kernel(selected_distances / float(scale)),
                    dtype=np.float64,
                )
                positive = raw > 0.0
                if not np.all(positive):
                    selected_targets = selected_targets[positive]
                    raw = raw[positive]
                total = float(np.sum(raw, dtype=np.float64))
                if not math.isfinite(total) or total <= 0.0:
                    raise ValueError(
                        f"source kernel normaliser is invalid for source {source_index}"
                    )
                normalized = raw / total
                column_total = float(np.sum(normalized, dtype=np.float64))
                if not math.isclose(column_total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                    raise ValueError(
                        f"normalized source kernel column {source_index} does not sum to one"
                    )
                non_source = selected_targets != source_index
                if np.any(non_source):
                    numerator[batch_index, selected_targets[non_source]] += (
                        masses[source_index] * normalized[non_source]
                    )
            except (FloatingPointError, OverflowError, ValueError) as exc:
                invalid_reasons[batch_index] = str(exc)
        if progress_callback is not None:
            progress_callback(source_index + 1, row_count)

    results: list[LeaveOneOutObjective] = []
    for batch_index in range(batch_count):
        if invalid_reasons[batch_index] is not None:
            results.append(
                LeaveOneOutObjective(
                    valid=False,
                    objective=None,
                    probabilities=np.full(row_count, np.nan, dtype=np.float64),
                    reason=invalid_reasons[batch_index],
                )
            )
            continue
        with np.errstate(divide="ignore", invalid="ignore", over="raise"):
            try:
                loo = numerator[batch_index] / denominators
            except FloatingPointError as exc:
                raise ValueError("leave-one-out probability reduction overflowed") from exc
        invalid = ~np.isfinite(loo) | (loo <= 0.0)
        if np.any(invalid):
            first = int(np.flatnonzero(invalid)[0])
            results.append(
                LeaveOneOutObjective(
                    valid=False,
                    objective=None,
                    probabilities=loo,
                    reason=f"zero or non-finite leave-one-out support at target row {first}",
                )
            )
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            objective = float(-np.sum(masses * np.log(loo), dtype=np.float64))
        if not math.isfinite(objective):
            results.append(
                LeaveOneOutObjective(
                    valid=False,
                    objective=None,
                    probabilities=loo,
                    reason="leave-one-out objective is non-finite",
                )
            )
        else:
            results.append(LeaveOneOutObjective(valid=True, objective=objective, probabilities=loo))
    logger.info(
        "Batched leave-one-out calibration completed: batch_count=%d, "
        "peak_calibration_workspace_bytes=%d, support_count_min=%s, "
        "support_count_mean=%s, support_count_max=%d, radius_queries=%d, "
        "distance_evaluations=%d",
        batch_count,
        peak_workspace_bytes,
        str(provider.min_support_count),
        str(
            provider.support_count_total / provider.support_count_samples
            if provider.support_count_samples
            else None
        ),
        provider.max_support_count,
        provider.index.radius_queries if provider.index is not None else provider.radius_queries,
        (
            provider.index.distance_evaluations
            if provider.index is not None
            else provider.distance_evaluations
        ),
    )
    return tuple(results)


evaluate_batched_leave_one_out_objectives = evaluate_leave_one_out_objectives
evaluate_leave_one_out_batch = evaluate_leave_one_out_objectives


_DEFAULT_SPARSE_EDGE_LIMIT = 1_000_000


def _validate_sparse_limit(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _array_digest(values: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(values)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _resolve_frozen_bandwidths(
    bandwidths: FrozenBandwidths | BandwidthCalibrationResult,
) -> FrozenBandwidths:
    if isinstance(bandwidths, BandwidthCalibrationResult):
        return bandwidths.selected
    if isinstance(bandwidths, FrozenBandwidths):
        return bandwidths
    raise TypeError("sparse kernel construction requires frozen bandwidths")


def _validate_sparse_graph_inputs(
    pool: EntropyPool,
    bandwidths: FrozenBandwidths | BandwidthCalibrationResult,
) -> tuple[np.ndarray, FrozenBandwidths]:
    if not isinstance(pool, EntropyPool):
        raise TypeError("sparse kernel construction requires an EntropyPool")
    frozen = _resolve_frozen_bandwidths(bandwidths)
    values = _validate_descriptors(pool.descriptors)
    if pool.fingerprint != frozen.pool_fingerprint:
        raise ValueError("frozen bandwidths belong to a different entropy pool")
    if pool.transform_fingerprint != frozen.transform_fingerprint:
        raise ValueError("frozen bandwidths have a mismatched transform fingerprint")
    if values.shape[0] != pool.row_candidate_indices.shape[0]:
        raise ValueError("entropy pool row ownership does not align with descriptors")
    if frozen.bandwidths.dtype != np.dtype(np.float64):
        raise ValueError("frozen bandwidths must have dtype float64")
    if frozen.bandwidths.shape != (values.shape[0],):
        raise ValueError("frozen bandwidth count must equal the atomic row count")
    if not np.all(np.isfinite(frozen.bandwidths)) or np.any(frozen.bandwidths <= 0.0):
        raise ValueError("frozen bandwidths must be finite and strictly positive")
    if frozen.backend not in {NEIGHBOUR_BACKEND_ID, INDEXED_NEIGHBOUR_BACKEND_ID}:
        raise ValueError(
            "sparse kernel construction requires a supported exact neighbour backend; "
            f"received {frozen.backend!r}"
        )
    expected_version = (
        NEIGHBOUR_BACKEND_VERSION
        if frozen.backend == NEIGHBOUR_BACKEND_ID
        else INDEXED_NEIGHBOUR_BACKEND_VERSION
    )
    if frozen.backend_version != expected_version:
        raise ValueError(
            "sparse kernel construction requires the configured backend version; "
            f"received {frozen.backend_version!r}, expected {expected_version!r}"
        )
    if frozen.metric != NEIGHBOUR_METRIC:
        raise ValueError("sparse kernel construction requires the exact Euclidean metric")
    candidate_ids = tuple(pool.candidate_ids)
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("entropy pool candidate IDs must be non-empty and unique")
    owners = np.asarray(pool.row_candidate_indices)
    if owners.dtype != np.dtype(np.int64) or owners.ndim != 1:
        raise ValueError("entropy pool row ownership must be an int64 vector")
    if np.any(owners < 0) or np.any(owners >= len(candidate_ids)):
        raise ValueError("entropy pool row ownership contains an unknown candidate")
    return values, frozen


def _progress_timing(completed: int, total: int, started: float) -> tuple[float, float, float]:
    elapsed = max(0.0, time.perf_counter() - started)
    throughput = completed / elapsed if completed > 0 and elapsed > 0.0 else 0.0
    remaining = (total - completed) / throughput if throughput > 0.0 else 0.0
    return elapsed, throughput, max(0.0, remaining)


def _graph_fingerprint(
    pool: EntropyPool,
    bandwidths: FrozenBandwidths,
    source_indptr: np.ndarray,
    target_indices: np.ndarray,
    values: np.ndarray,
) -> str:
    payload = {
        "schema": SPARSE_KERNEL_GRAPH_SCHEMA_VERSION,
        "pool": pool.fingerprint,
        "transform": pool.transform_fingerprint,
        "bandwidth": bandwidths.fingerprint,
        "neighbour_backend": bandwidths.backend,
        "neighbour_backend_version": bandwidths.backend_version,
        "neighbour_backend_fingerprint": bandwidths.backend_fingerprint,
        "metric": bandwidths.metric,
        "kernel_family": KERNEL_FAMILY,
        "kernel_version": KERNEL_VERSION,
        "normalization": "finite-pool-source-column-v1",
        "source_bandwidth_orientation": "h_a",
        "self_membership": True,
        "source_order": "ordered-pool-rows",
        "target_order": "ascending-pool-row-index",
        "dtype": "float64",
        "numerical_tolerance": SPARSE_NUMERICAL_TOLERANCE,
        "candidate_ids": list(pool.candidate_ids),
        "row_candidate_indices": _array_digest(pool.row_candidate_indices),
        "source_indptr": _array_digest(source_indptr),
        "target_indices": _array_digest(target_indices),
        "values": _array_digest(values),
    }
    return sha256_canonical_json(payload)


def _kernel_operator_fingerprint(
    pool: EntropyPool,
    bandwidths: FrozenBandwidths,
) -> str:
    """Fingerprint the exact kernel operator without materialising its support."""

    return sha256_canonical_json(
        {
            "schema": KERNEL_OPERATOR_SCHEMA_VERSION,
            "pool": pool.fingerprint,
            "transform": pool.transform_fingerprint,
            "representation": pool.representation_fingerprint,
            "candidate_ids": list(pool.candidate_ids),
            "row_candidate_indices": _array_digest(pool.row_candidate_indices),
            "bandwidth": bandwidths.fingerprint,
            "bandwidth_values": _array_digest(bandwidths.bandwidths),
            "k": bandwidths.k,
            "c": bandwidths.c,
            "neighbour_backend": bandwidths.backend,
            "neighbour_backend_version": bandwidths.backend_version,
            "neighbour_backend_fingerprint": bandwidths.backend_fingerprint,
            "metric": bandwidths.metric,
            "distinct_location_policy": "exact-coordinate-location-v1",
            "kernel_family": KERNEL_FAMILY,
            "kernel_version": KERNEL_VERSION,
            "normalization": "finite-pool-source-column-v1",
            "source_bandwidth_orientation": "h_a",
            "duplicate_target_policy": "retain-all-original-pool-rows-v1",
            "source_reduction_order": "candidate-id-then-original-source-index-v1",
            "target_order": "ascending-original-pool-row-index",
            "dtype": "float64",
            "numerical_tolerance": SPARSE_NUMERICAL_TOLERANCE,
        }
    )


def _report_graph_progress(
    completed: int,
    total: int,
    edge_counts: np.ndarray,
    started: float,
    next_percent: list[int],
) -> None:
    if total < 100:
        should_report = True
    else:
        should_report = completed * 100 >= next_percent[0] * total
    if not should_report:
        return
    elapsed, throughput, remaining = _progress_timing(completed, total, started)
    edges = int(np.sum(edge_counts[:completed], dtype=np.int64))
    logger.info(
        "Sparse atomic kernel graph progress: sources=%d/%d (%.1f%%), edges=%d, "
        "throughput=%.3f sources/s, elapsed=%.3fs, estimated remaining=%.3fs",
        completed,
        total,
        100.0 * completed / total,
        edges,
        throughput,
        elapsed,
        remaining,
    )
    if total >= 100:
        while next_percent[0] <= 100 and completed * 100 >= next_percent[0] * total:
            next_percent[0] += 1


def build_sparse_atomic_kernel_graph(
    pool: EntropyPool,
    bandwidths: FrozenBandwidths | BandwidthCalibrationResult,
    *,
    chunk_size: int = 1024,
    max_edges: int = _DEFAULT_SPARSE_EDGE_LIMIT,
    max_graph_bytes: int | None = None,
    max_spool_bytes: int | None = None,
    progress_callback: SourceProgressCallback | None = None,
    max_radius_query_bytes: int = DEFAULT_RADIUS_QUERY_BYTES,
) -> SparseAtomicKernelGraph:
    """Build the exact source-major sparse finite-pool atomic kernel graph."""

    values, frozen = _validate_sparse_graph_inputs(pool, bandwidths)
    block = _validate_chunk_size(chunk_size)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    edge_limit = _validate_sparse_limit(max_edges, "max_edges")
    byte_limit = (
        None
        if max_graph_bytes is None
        else _validate_sparse_limit(max_graph_bytes, "max_graph_bytes")
    )
    spool_limit = _validate_spool_limit(max_spool_bytes, "max_spool_bytes")
    spool_limit = (
        spool_limit
        if spool_limit is not None
        else len(_EDGE_SPOOL_HEADER) + edge_limit * _EDGE_SPOOL_RECORD_BYTES
    )
    if spool_limit < len(_EDGE_SPOOL_HEADER):
        raise ValueError("max_spool_bytes is too small for the exact edge spool header")
    n_rows = values.shape[0]
    candidate_count = len(pool.candidate_ids)
    started = time.perf_counter()
    logger.info(
        "Sparse atomic kernel graph started: N=%d, M=%d, schema=%s, kernel=%s/%s, "
        "max_edges=%d, max_graph_bytes=%s, max_spool_bytes=%d, chunk_size=%d",
        n_rows,
        candidate_count,
        SPARSE_KERNEL_GRAPH_SCHEMA_VERSION,
        KERNEL_FAMILY,
        KERNEL_VERSION,
        edge_limit,
        byte_limit if byte_limit is not None else "unbounded",
        spool_limit,
        block,
    )
    edge_counts = np.zeros(n_rows, dtype=np.int64)
    support_provider = _KernelSupportProvider(
        values,
        backend=frozen.backend,
        chunk_size=block,
        max_radius_query_bytes=max_radius_query_bytes,
    )
    if (
        frozen.backend == INDEXED_NEIGHBOUR_BACKEND_ID
        and frozen.backend_fingerprint
        and support_provider.index is not None
        and frozen.backend_fingerprint != support_provider.index.fingerprint
    ):
        raise ValueError("frozen bandwidths have a mismatched indexed backend fingerprint")
    next_percent = [1]
    peak_estimated_bytes = 0
    materialized_spool_bytes = 0
    fixed_peak_bytes = 8 * (n_rows + 1 + n_rows)
    fixed_peak_bytes += (
        support_provider.index.index_bytes
        if support_provider.index is not None
        else 8 * (block + n_rows)
    )
    if byte_limit is not None and fixed_peak_bytes > byte_limit:
        raise ValueError(
            "sparse atomic kernel graph exceeds max_graph_bytes before source enumeration: "
            f"N={n_rows}, E=0, source=0, mean_support=0, max_support=0, "
            f"estimated_bytes={fixed_peak_bytes}, max_graph_bytes={byte_limit}"
        )
    peak_estimated_bytes = fixed_peak_bytes

    def report_source_progress(completed: int, total: int) -> None:
        _report_graph_progress(completed, total, edge_counts, started, next_percent)
        if progress_callback is not None:
            progress_callback(completed, total)

    try:
        with tempfile.TemporaryFile(mode="w+b") as spool:
            spool_digest = _spool_write_header(spool)
            spool_bytes = len(_EDGE_SPOOL_HEADER)
            edge_count = 0
            max_support = 0
            for column in iter_normalized_kernel_columns(
                values,
                frozen.bandwidths,
                chunk_size=block,
                progress_callback=report_source_progress,
                backend=frozen.backend,
                support_provider=support_provider,
                max_radius_query_bytes=max_radius_query_bytes,
            ):
                support = int(column.target_indices.shape[0])
                source_index = int(column.source_index)
                if support == 0 or np.any(np.diff(column.target_indices) <= 0):
                    raise ValueError(
                        f"sparse graph source {source_index} does not have strictly ascending support"
                    )
                prospective_edges = edge_count + support
                max_support = max(max_support, support)
                mean_support = prospective_edges / float(source_index + 1)
                if prospective_edges > edge_limit:
                    raise ValueError(
                        "sparse atomic kernel graph exceeds max_edges during source enumeration: "
                        f"N={n_rows}, E={prospective_edges}, required_edges={prospective_edges}, "
                        f"source={source_index}, "
                        f"mean_support={mean_support:.6g}, max_support={max_support}, "
                        f"estimated_bytes={8 * (n_rows + 1 + n_rows + 2 * prospective_edges)}, "
                        f"max_edges={edge_limit}"
                    )
                prospective_spool_bytes = spool_bytes + support * _EDGE_SPOOL_RECORD_BYTES
                if prospective_spool_bytes > spool_limit:
                    raise ValueError(
                        "sparse atomic kernel graph exact spool exceeds max_spool_bytes during "
                        f"source enumeration: N={n_rows}, E={prospective_edges}, source={source_index}, "
                        f"spool_bytes={prospective_spool_bytes}, max_spool_bytes={spool_limit}"
                    )
                required_bytes = 8 * (n_rows + 1 + n_rows + 2 * prospective_edges)
                estimated_peak_bytes = (
                    required_bytes + support_provider.index.index_bytes
                    if support_provider.index is not None
                    else required_bytes + 8 * (block + n_rows)
                )
                estimated_peak_bytes += support * _EDGE_SPOOL_RECORD_BYTES
                if byte_limit is not None and estimated_peak_bytes > byte_limit:
                    raise ValueError(
                        "sparse atomic kernel graph exceeds max_graph_bytes during source "
                        f"enumeration: N={n_rows}, E={prospective_edges}, source={source_index}, "
                        f"mean_support={mean_support:.6g}, max_support={max_support}, "
                        f"estimated_bytes={estimated_peak_bytes}, max_graph_bytes={byte_limit}"
                    )
                peak_estimated_bytes = max(peak_estimated_bytes, estimated_peak_bytes)
                records = np.empty(support, dtype=_EDGE_SPOOL_DTYPE)
                records["target"] = column.target_indices
                records["value"] = column.values
                _spool_write_records(spool, spool_digest, records)
                edge_counts[source_index] = support
                edge_count = prospective_edges
                spool_bytes = prospective_spool_bytes
                materialized_spool_bytes = spool_bytes
            if int(np.sum(edge_counts, dtype=np.int64)) != edge_count:
                raise ValueError("sparse graph source enumeration produced inconsistent edge counts")
            spool.flush()
            spool.seek(0)
            if spool.read(len(_EDGE_SPOOL_HEADER)) != _EDGE_SPOOL_HEADER:
                raise ValueError("exact edge spool has an invalid version header")
            source_indptr = np.empty(n_rows + 1, dtype=np.int64)
            source_indptr[0] = 0
            np.cumsum(edge_counts, dtype=np.int64, out=source_indptr[1:])
            target_indices = np.empty(edge_count, dtype=np.int64)
            edge_values = np.empty(edge_count, dtype=np.float64)
            positions = source_indptr[:-1].copy()
            read_digest = hashlib.sha256()
            read_digest.update(_EDGE_SPOOL_HEADER)
            source_cursor = 0
            edge_cursor = 0
            for records in _spool_read_records(spool, read_digest, edge_count):
                count = records.shape[0]
                # The spool is source-major and each source's position is known
                # from edge_counts; populate by consuming source-sized slices.
                offset = 0
                while offset < count:
                    while (
                        source_cursor < n_rows
                        and edge_cursor >= int(source_indptr[source_cursor + 1])
                    ):
                        source_cursor += 1
                    if source_cursor >= n_rows:
                        raise ValueError("exact edge spool contains too many records")
                    source_index = source_cursor
                    source_stop = int(source_indptr[source_index + 1])
                    take = min(count - offset, source_stop - int(positions[source_index]))
                    source_start = int(positions[source_index])
                    target_indices[source_start : source_start + take] = records["target"][
                        offset : offset + take
                    ]
                    edge_values[source_start : source_start + take] = records["value"][
                        offset : offset + take
                    ]
                    positions[source_index] += take
                    edge_cursor += take
                    offset += take
            if spool.read(1):
                raise ValueError("exact edge spool contains trailing records")
            if read_digest.digest() != spool_digest.digest():
                raise ValueError("exact edge spool checksum validation failed")
            if not np.array_equal(positions, source_indptr[1:]):
                raise ValueError("exact edge spool did not populate every source row")
        graph_fingerprint = _graph_fingerprint(
            pool,
            frozen,
            source_indptr,
            target_indices,
            edge_values,
        )
        graph = SparseAtomicKernelGraph(
            source_indptr=source_indptr,
            target_indices=target_indices,
            values=edge_values,
            row_candidate_indices=pool.row_candidate_indices,
            candidate_ids=tuple(pool.candidate_ids),
            pool_fingerprint=pool.fingerprint,
            bandwidth_fingerprint=frozen.fingerprint,
            transform_fingerprint=pool.transform_fingerprint,
            fingerprint=graph_fingerprint,
        )
    except Exception as exc:
        logger.error("Sparse atomic kernel graph failed: %s", exc)
        raise

    elapsed, _, _ = _progress_timing(n_rows, n_rows, started)
    support_sizes = graph.support_sizes
    logger.info(
        "Sparse atomic kernel graph completed: edges=%d, mean support=%.3f, "
        "max support=%d, density=%.6g, final_csr_bytes=%d, peak_estimated_bytes=%d, "
        "spool_bytes=%d, elapsed=%.3fs, budgets=satisfied",
        graph.edge_count,
        float(np.mean(support_sizes)),
        int(np.max(support_sizes)),
        graph.density,
        graph.array_bytes,
        peak_estimated_bytes,
        materialized_spool_bytes,
        elapsed,
    )
    return graph


def _measured_peak_memory_bytes() -> int | None:
    """Read the Linux high-water mark when available, without a dependency."""

    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def build_streamed_candidate_contributions(
    pool: EntropyPool,
    bandwidths: FrozenBandwidths | BandwidthCalibrationResult,
    *,
    chunk_size: int = 1024,
    max_entries: int | None = None,
    max_contribution_bytes: int | None = None,
    max_spool_bytes: int | None = None,
    progress_callback: SourceProgressCallback | None = None,
    max_radius_query_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
) -> tuple[SparseCandidateContributions, StreamedKernelExecutionSummary]:
    """Build exact candidate PMFs by streaming normalized source columns.

    Sources are queried once, in candidate-ID/source-row order.  A reusable
    dense accumulator is reduced to one candidate row before the next source
    group is processed; only the final candidate CSR is retained.
    """

    values, frozen = _validate_sparse_graph_inputs(pool, bandwidths)
    block = _validate_chunk_size(chunk_size)
    entry_limit = (
        None if max_entries is None else _validate_sparse_limit(max_entries, "max_entries")
    )
    byte_limit = (
        None
        if max_contribution_bytes is None
        else _validate_sparse_limit(max_contribution_bytes, "max_contribution_bytes")
    )
    spool_limit = _validate_spool_limit(max_spool_bytes, "max_contribution_spool_bytes")
    if spool_limit is not None and spool_limit < len(_EDGE_SPOOL_HEADER):
        raise ValueError(
            "max_contribution_spool_bytes is too small for the candidate contribution spool header"
        )
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")

    n_rows = int(values.shape[0])
    candidate_ids = tuple(pool.candidate_ids)
    candidate_count = len(candidate_ids)
    maximum_entry_count = candidate_count * n_rows
    if maximum_entry_count > np.iinfo(np.int64).max:
        raise ResourceCapacityError(
            "streamed candidate contributions exceed the int64 CSR capacity: "
            f"N={n_rows}, M={candidate_count}, Q_max={maximum_entry_count}",
            operation="entropy streamed candidate contributions",
            requested_bytes=maximum_entry_count * 16,
            available_bytes=None,
        )
    if entry_limit is None:
        entry_limit = maximum_entry_count
    runtime_budget = resource_budget or build_resource_budget()
    remaining_budget = runtime_budget.remaining_managed_budget
    if remaining_budget is None:
        raise ResourceCapacityError(
            runtime_budget.unknown_memory_message("entropy streamed candidate contributions"),
            operation="entropy streamed candidate contributions",
        )
    if max_radius_query_bytes is None:
        max_radius_query_bytes = max(1, remaining_budget // 2)
    else:
        max_radius_query_bytes = min(
            _validate_memory_limit(max_radius_query_bytes, "max_radius_query_bytes"),
            max(1, remaining_budget),
        )
    derived_spool_limit = spool_limit is None
    if spool_limit is None:
        scratch_limit = runtime_budget.budget.scratch_budget_bytes
        if scratch_limit is None:
            raise ResourceCapacityError(
                "streamed candidate contributions require known scratch capacity; "
                "provide [resources] scratch_budget_bytes or a usable temporary filesystem",
                operation="entropy streamed candidate contributions",
            )
        spool_limit = int(scratch_limit)
    if spool_limit < len(_EDGE_SPOOL_HEADER):
        if derived_spool_limit:
            raise ResourceCapacityError(
                "streamed candidate contributions have insufficient scratch capacity for "
                "their temporary spool header",
                operation="entropy streamed candidate contributions",
                requested_bytes=len(_EDGE_SPOOL_HEADER),
                available_bytes=runtime_budget.remaining_managed_budget,
                reserved_headroom_bytes=runtime_budget.budget.reserved_headroom_bytes,
                scratch_needed_bytes=len(_EDGE_SPOOL_HEADER),
                scratch_available_bytes=runtime_budget.budget.scratch_budget_bytes,
            )
        raise ValueError(
            "max_contribution_spool_bytes is too small for the candidate contribution spool header"
        )
    owners = np.asarray(pool.row_candidate_indices, dtype=np.int64)
    source_counts = np.bincount(owners, minlength=candidate_count).astype(np.int64)
    if np.any(source_counts <= 0):
        missing = candidate_ids[int(np.flatnonzero(source_counts <= 0)[0])]
        raise ValueError(f"candidate {missing!r} owns no source rows")
    source_order = np.argsort(owners, kind="stable")
    candidate_offsets = np.empty(candidate_count + 1, dtype=np.int64)
    candidate_offsets[0] = 0
    np.cumsum(source_counts, dtype=np.int64, out=candidate_offsets[1:])

    initial_runtime_bytes = int(
        16 * values.nbytes + owners.nbytes + source_order.nbytes + 256 * n_rows
    )
    operator_fingerprint = _kernel_operator_fingerprint(pool, frozen)
    try:
        lease = runtime_budget.acquire(
            "entropy streamed candidate contributions",
            max(1, initial_runtime_bytes),
            scratch_bytes=len(_EDGE_SPOOL_HEADER),
        )
    except ResourceCapacityError as exc:
        raise ResourceCapacityError(
            "streamed candidate contributions cannot reserve initial workspace: "
            f"N={n_rows}, M={candidate_count}, Q=0, E=0, "
            f"requested_bytes={initial_runtime_bytes}, "
            f"available_bytes={runtime_budget.remaining_managed_budget}, "
            f"scratch_available_bytes={runtime_budget.budget.scratch_budget_bytes}",
            operation="entropy streamed candidate contributions",
            requested_bytes=initial_runtime_bytes,
            available_bytes=runtime_budget.remaining_managed_budget,
            reserved_headroom_bytes=runtime_budget.budget.reserved_headroom_bytes,
            scratch_needed_bytes=len(_EDGE_SPOOL_HEADER),
            scratch_available_bytes=runtime_budget.budget.scratch_budget_bytes,
        ) from exc

    try:
        support_provider = _KernelSupportProvider(
            values,
            backend=frozen.backend,
            chunk_size=block,
            max_radius_query_bytes=max_radius_query_bytes,
            resource_budget=runtime_budget,
            capacity_context=(
                f"N={n_rows}, M={candidate_count}, operator={operator_fingerprint}"
            ),
        )
        if (
            frozen.backend == INDEXED_NEIGHBOUR_BACKEND_ID
            and frozen.backend_fingerprint
            and support_provider.index is not None
            and frozen.backend_fingerprint != support_provider.index.fingerprint
        ):
            raise ValueError("frozen bandwidths have a mismatched indexed backend fingerprint")
    except BaseException:
        lease.close()
        raise

    def estimated_peak_bytes(entry_count: int, support_count: int) -> int:
        final_arrays = 8 * (candidate_count + 1 + entry_count + entry_count + candidate_count)
        workspace = (
            values.nbytes
            + owners.nbytes
            + source_order.nbytes
            + source_counts.nbytes
            + candidate_offsets.nbytes
            + 8 * n_rows
            + n_rows
            + 32 * support_count
            + (support_provider.index.index_bytes if support_provider.index is not None else 0)
        )
        return int(final_arrays + workspace)

    logger.info(
        "Streamed candidate contributions started: N=%d, M=%d, operator=%s, "
        "max_entries=%d, max_contribution_bytes=%s, max_spool_bytes=%d",
        n_rows,
        candidate_count,
        operator_fingerprint,
        entry_limit,
        byte_limit if byte_limit is not None else "unbounded",
        spool_limit,
    )
    started = time.perf_counter()
    support_sizes = np.zeros(n_rows, dtype=np.int64)
    candidate_support_sizes = np.zeros(candidate_count, dtype=np.int64)
    accumulator = np.zeros(n_rows, dtype=np.float64)
    touched = np.zeros(n_rows, dtype=bool)
    implicit_edges = 0
    self_edges = 0
    source_mass_deviation = 0.0
    q_completed = 0
    q_active = 0
    q_observed = 0
    last_q_observed = 0

    def resize_workspace(
        requested_bytes: int,
        scratch_bytes: int,
        *,
        source_index: int | None = None,
    ) -> None:
        try:
            lease.resize(requested_bytes, scratch_bytes=scratch_bytes)
        except ResourceCapacityError as exc:
            raise ResourceCapacityError(
                "streamed candidate contributions exceed the runtime resource budget: "
                f"N={n_rows}, M={candidate_count}, Q={q_observed}, E={implicit_edges}, "
                f"source={source_index if source_index is not None else 'n/a'}, "
                f"requested_bytes={requested_bytes}, "
                f"available_bytes={runtime_budget.remaining_managed_budget}, "
                f"scratch_needed_bytes={scratch_bytes}, "
                f"scratch_available_bytes={runtime_budget.budget.scratch_budget_bytes}",
                operation="entropy streamed candidate contributions",
                requested_bytes=requested_bytes,
                available_bytes=runtime_budget.remaining_managed_budget,
                reserved_headroom_bytes=runtime_budget.budget.reserved_headroom_bytes,
                scratch_needed_bytes=scratch_bytes,
                scratch_available_bytes=runtime_budget.budget.scratch_budget_bytes,
            ) from exc

    def validate_q_invariants(candidate_index: int, source_index: int) -> None:
        nonlocal last_q_observed
        if not (
            0 <= q_active <= n_rows
            and 0 <= q_completed <= maximum_entry_count
            and 0 <= q_observed <= maximum_entry_count
            and q_observed == q_completed + q_active
            and q_observed >= last_q_observed
        ):
            candidate_label = (
                candidate_ids[candidate_index] if 0 <= candidate_index < candidate_count else None
            )
            raise RuntimeError(
                "invalid streamed candidate-entry counters: "
                f"candidate={candidate_label!r}, source={source_index}, "
                f"Q_completed={q_completed}, Q_active={q_active}, "
                f"Q_observed={q_observed}, previous_Q_observed={last_q_observed}, "
                f"N={n_rows}, M={candidate_count}"
            )
        last_q_observed = q_observed

    peak_estimated_bytes = estimated_peak_bytes(0, 0)
    if byte_limit is not None and peak_estimated_bytes > byte_limit:
        raise ValueError(
            "streamed candidate contributions exceed max_contribution_bytes before "
            "source enumeration: "
            f"N={n_rows}, M={candidate_count}, Q=0, E=0, requested_bytes={peak_estimated_bytes}, "
            f"limit={byte_limit}"
        )
    spool_bytes = len(_EDGE_SPOOL_HEADER)
    completed_sources = 0
    next_percent = [1]

    def report_source(source_index: int, support_count: int) -> None:
        nonlocal completed_sources
        completed_sources += 1
        if n_rows < 100 or completed_sources * 100 >= next_percent[0] * n_rows:
            logger.info(
                "Streamed candidate contributions progress: sources=%d/%d (%.1f%%), "
                "implicit_edges=%d, Q=%d (Q_observed=Q_completed+Q_active; "
                "Q_completed=%d, Q_active=%d)",
                completed_sources,
                n_rows,
                100.0 * completed_sources / n_rows,
                implicit_edges,
                q_observed,
                q_completed,
                q_active,
            )
            while next_percent[0] <= 100 and completed_sources * 100 >= next_percent[0] * n_rows:
                next_percent[0] += 1
        if progress_callback is not None:
            progress_callback(completed_sources, n_rows)

    try:
        scratch_directory = runtime_budget.budget.snapshot.scratch_path or None
        with tempfile.TemporaryFile(mode="w+b", dir=scratch_directory) as spool:
            spool_digest = _spool_write_header(spool)
            for candidate_index in range(candidate_count):
                accumulator.fill(0.0)
                touched.fill(False)
                q_active = 0
                q_observed = q_completed
                validate_q_invariants(candidate_index, -1)
                start_source = int(candidate_offsets[candidate_index])
                stop_source = int(candidate_offsets[candidate_index + 1])
                candidate_source_indices = source_order[start_source:stop_source]
                last_source_index = int(candidate_source_indices[-1])
                for column in iter_normalized_kernel_columns(
                    values,
                    frozen.bandwidths,
                    chunk_size=block,
                    backend=frozen.backend,
                    support_provider=support_provider,
                    max_radius_query_bytes=max_radius_query_bytes,
                    source_indices=candidate_source_indices,
                ):
                    source_index = int(column.source_index)
                    scale = float(frozen.bandwidths[source_index])
                    target_indices = np.asarray(column.target_indices, dtype=np.int64)
                    normalized = np.asarray(column.values, dtype=np.float64)
                    if (
                        target_indices.ndim != 1
                        or normalized.ndim != 1
                        or target_indices.shape != normalized.shape
                        or target_indices.size == 0
                        or not np.all(np.isfinite(normalized))
                        or np.any(normalized <= 0.0)
                    ):
                        raise ValueError(
                            f"invalid normalized support: N={n_rows}, M={candidate_count}, "
                            f"candidate={candidate_ids[candidate_index]!r}, source={source_index}, "
                            f"h_a={scale:.17g}, E={implicit_edges}"
                        )
                    if (
                        np.any(target_indices < 0)
                        or np.any(target_indices >= n_rows)
                        or np.any(target_indices[1:] <= target_indices[:-1])
                    ):
                        raise ValueError(
                            "normalized support target indices must be strictly ascending "
                            "and unique: "
                            f"N={n_rows}, M={candidate_count}, "
                            f"candidate={candidate_ids[candidate_index]!r}, source={source_index}"
                        )
                    column_mass = float(np.sum(normalized, dtype=np.float64))
                    if not math.isclose(
                        column_mass, 1.0, rel_tol=0.0, abs_tol=SPARSE_NUMERICAL_TOLERANCE
                    ):
                        raise ValueError(
                            f"source normalization failed: N={n_rows}, M={candidate_count}, "
                            f"candidate={candidate_ids[candidate_index]!r}, source={source_index}, "
                            f"h_a={scale:.17g}, mass={column_mass:.17g}, E={implicit_edges}"
                        )
                    support_count = int(target_indices.size)
                    support_sizes[source_index] = support_count
                    implicit_edges += support_count
                    self_edges += int(np.count_nonzero(target_indices == source_index))
                    source_mass_deviation = max(source_mass_deviation, abs(column_mass - 1.0))
                    previously_touched = touched[target_indices]
                    q_active += int(np.count_nonzero(~previously_touched))
                    touched[target_indices] = True
                    accumulator[target_indices] += normalized
                    q_observed = q_completed + q_active
                    validate_q_invariants(candidate_index, source_index)
                    if q_observed > entry_limit:
                        raise ValueError(
                            "streamed candidate contributions exceed max_entries during "
                            "source enumeration: "
                            f"N={n_rows}, M={candidate_count}, "
                            f"candidate={candidate_ids[candidate_index]!r}, "
                            f"source={source_index}, Q_completed={q_completed}, "
                            f"Q_active={q_active}, Q_observed={q_observed}, "
                            f"E={implicit_edges}, max_entries={entry_limit}, "
                            "failure=mid-candidate"
                        )
                    requested_bytes = estimated_peak_bytes(
                        q_observed,
                        support_count,
                    )
                    peak_estimated_bytes = max(peak_estimated_bytes, requested_bytes)
                    resize_workspace(
                        requested_bytes,
                        spool_bytes,
                        source_index=source_index,
                    )
                    if byte_limit is not None and requested_bytes > byte_limit:
                        raise ValueError(
                            "streamed candidate contributions exceed max_contribution_bytes: "
                            f"N={n_rows}, M={candidate_count}, "
                            f"candidate={candidate_ids[candidate_index]!r}, "
                            f"n_C={stop_source - start_source}, source={source_index}, "
                            f"h_a={scale:.17g}, Q_completed={q_completed}, "
                            f"Q_active={q_active}, Q_observed={q_observed}, "
                            f"E={implicit_edges}, requested_bytes={requested_bytes}, "
                            f"limit={byte_limit}"
                        )
                    report_source(source_index, support_count)

                targets = np.flatnonzero(touched).astype(np.int64, copy=False)
                values_for_candidate = np.asarray(
                    accumulator[targets] / float(source_counts[candidate_index]), dtype=np.float64
                )
                if not np.all(np.isfinite(values_for_candidate)) or np.any(
                    values_for_candidate <= 0.0
                ):
                    raise ValueError(
                        "streamed candidate contribution is non-finite or non-positive"
                    )
                mass = float(np.sum(values_for_candidate, dtype=np.float64))
                if not math.isclose(
                    mass, 1.0, rel_tol=0.0, abs_tol=SPARSE_NUMERICAL_TOLERANCE
                ):
                    raise ValueError(
                        f"candidate {candidate_ids[candidate_index]!r} contribution is not "
                        "normalized"
                    )
                support = int(targets.size)
                if support != q_active:
                    raise RuntimeError(
                        "streamed candidate support counter disagrees with finalized CSR row: "
                        f"candidate={candidate_ids[candidate_index]!r}, "
                        f"source={last_source_index}, "
                        f"Q_completed={q_completed}, Q_active={q_active}, support={support}"
                    )
                completed_entries = q_completed + support
                if completed_entries > entry_limit:
                    raise ValueError(
                        "streamed candidate contributions exceed max_entries at finalization: "
                        f"N={n_rows}, M={candidate_count}, "
                        f"candidate={candidate_ids[candidate_index]!r}, "
                        f"n_C={stop_source - start_source}, source={last_source_index}, "
                        f"Q_completed={q_completed}, Q_active={q_active}, "
                        f"Q_observed={completed_entries}, E={implicit_edges}, "
                        f"max_entries={entry_limit}, failure=finalization"
                    )
                prospective_spool_bytes = spool_bytes + support * _EDGE_SPOOL_RECORD_BYTES
                if prospective_spool_bytes > spool_limit:
                    if derived_spool_limit:
                        raise ResourceCapacityError(
                            "streamed candidate contributions exceed available scratch "
                            "capacity: "
                            f"N={n_rows}, M={candidate_count}, Q={completed_entries}, "
                            f"E={implicit_edges}, candidate={candidate_ids[candidate_index]!r}, "
                            f"requested_bytes={prospective_spool_bytes}, "
                            f"scratch_available_bytes={spool_limit}",
                            operation="entropy streamed candidate contributions",
                            requested_bytes=prospective_spool_bytes,
                            available_bytes=runtime_budget.remaining_managed_budget,
                            reserved_headroom_bytes=runtime_budget.budget.reserved_headroom_bytes,
                            scratch_needed_bytes=prospective_spool_bytes,
                            scratch_available_bytes=spool_limit,
                        )
                    raise ValueError(
                        "streamed candidate contribution spool exceeds its capacity: "
                        f"N={n_rows}, M={candidate_count}, "
                        f"candidate={candidate_ids[candidate_index]!r}, "
                        f"n_C={stop_source - start_source}, "
                        f"Q_completed={q_completed}, Q_active={q_active}, "
                        f"Q_observed={completed_entries}, "
                        f"E={implicit_edges}, "
                        f"requested_bytes={prospective_spool_bytes}, limit={spool_limit}"
                    )
                records = np.empty(support, dtype=_EDGE_SPOOL_DTYPE)
                records["target"] = targets
                records["value"] = values_for_candidate
                _spool_write_records(spool, spool_digest, records)
                spool_bytes = prospective_spool_bytes
                candidate_support_sizes[candidate_index] = support
                q_completed = completed_entries
                q_active = 0
                q_observed = q_completed
                validate_q_invariants(candidate_index, last_source_index)
                resize_workspace(
                    estimated_peak_bytes(q_completed, 0),
                    spool_bytes,
                    source_index=last_source_index,
                )

            if completed_sources != n_rows:
                raise ValueError("streamed candidate contribution did not query every source row")
            entry_count = q_completed
            if entry_count > np.iinfo(np.int64).max:
                raise ValueError(
                    "streamed candidate contribution entry count exceeds int64 CSR capacity: "
                    f"N={n_rows}, M={candidate_count}, Q={entry_count}"
                )
            candidate_indptr = np.empty(candidate_count + 1, dtype=np.int64)
            candidate_indptr[0] = 0
            np.cumsum(candidate_support_sizes, dtype=np.int64, out=candidate_indptr[1:])
            if int(candidate_indptr[-1]) != entry_count:
                raise RuntimeError(
                    "streamed candidate CSR offsets disagree with finalized entry count: "
                    f"N={n_rows}, M={candidate_count}, Q_completed={entry_count}, "
                    f"csr_entries={int(candidate_indptr[-1])}"
                )
            final_bytes = estimated_peak_bytes(entry_count, 0)
            peak_estimated_bytes = max(peak_estimated_bytes, final_bytes)
            resize_workspace(final_bytes, spool_bytes)
            if byte_limit is not None and final_bytes > byte_limit:
                raise ValueError(
                    "streamed candidate contributions exceed max_contribution_bytes "
                    "before final CSR: "
                    f"N={n_rows}, M={candidate_count}, Q={entry_count}, E={implicit_edges}, "
                    f"requested_bytes={final_bytes}, limit={byte_limit}"
                )
            target_indices = np.empty(entry_count, dtype=np.int64)
            contribution_values = np.empty(entry_count, dtype=np.float64)
            spool.seek(0)
            if spool.read(len(_EDGE_SPOOL_HEADER)) != _EDGE_SPOOL_HEADER:
                raise ValueError("candidate contribution spool has an invalid version header")
            read_digest = hashlib.sha256()
            read_digest.update(_EDGE_SPOOL_HEADER)
            cursor = 0
            for records in _spool_read_records(spool, read_digest, entry_count):
                count = records.shape[0]
                target_indices[cursor : cursor + count] = records["target"]
                contribution_values[cursor : cursor + count] = records["value"]
                cursor += count
            if spool.read(1) or cursor != entry_count:
                raise ValueError("candidate contribution spool contains an invalid record count")
            if read_digest.digest() != spool_digest.digest():
                raise ValueError("candidate contribution spool checksum validation failed")

        candidate_fingerprint = sha256_canonical_json(
            {
                "schema": SPARSE_STREAMED_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
                "kernel_operator": operator_fingerprint,
                "candidate_ids": list(candidate_ids),
                "candidate_indptr": _array_digest(candidate_indptr),
                "target_indices": _array_digest(target_indices),
                "values": _array_digest(contribution_values),
                "candidate_source_counts": _array_digest(source_counts),
            }
        )
        contributions = SparseCandidateContributions(
            candidate_indptr=candidate_indptr,
            target_indices=target_indices,
            values=contribution_values,
            candidate_ids=candidate_ids,
            graph_fingerprint="",
            kernel_operator_fingerprint=operator_fingerprint,
            fingerprint=candidate_fingerprint,
            row_count=n_rows,
            candidate_source_counts=source_counts,
            pool_fingerprint=pool.fingerprint,
            sparse_schema_version=SPARSE_STREAMED_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
        )
    except BaseException as exc:
        logger.error("Streamed candidate contributions failed: %s", exc)
        lease.close()
        if isinstance(exc, OSError):
            raise ResourceCapacityError(
                "streamed candidate contribution spool could not be created or written: "
                f"N={n_rows}, M={candidate_count}, Q={q_completed}, E={implicit_edges}, "
                f"scratch_needed_bytes={spool_bytes}, "
                f"scratch_available_bytes={runtime_budget.budget.scratch_budget_bytes}",
                operation="entropy streamed candidate contributions",
                requested_bytes=peak_estimated_bytes,
                available_bytes=runtime_budget.remaining_managed_budget,
                reserved_headroom_bytes=runtime_budget.budget.reserved_headroom_bytes,
                scratch_needed_bytes=spool_bytes,
                scratch_available_bytes=runtime_budget.budget.scratch_budget_bytes,
            ) from exc
        raise

    lease.close()
    summary = StreamedKernelExecutionSummary(
        atomic_row_count=n_rows,
        implicit_edge_count=implicit_edges,
        source_support_min=int(np.min(support_sizes)),
        source_support_mean=float(np.mean(support_sizes)),
        source_support_max=int(np.max(support_sizes)),
        self_edge_count=self_edges,
        source_mass_max_deviation=source_mass_deviation,
        candidate_entry_count=contributions.entry_count,
        candidate_support_min=int(np.min(candidate_support_sizes)),
        candidate_support_mean=float(np.mean(candidate_support_sizes)),
        candidate_support_max=int(np.max(candidate_support_sizes)),
        candidate_pmf_max_deviation=float(
            np.max(
                np.abs(
                    np.add.reduceat(
                        contributions.values, contributions.candidate_indptr[:-1]
                    )
                    - 1.0
                )
            )
        ),
        candidate_csr_array_bytes=contributions.array_bytes,
        contribution_spool_bytes=spool_bytes,
        estimated_peak_memory_bytes=peak_estimated_bytes,
        measured_peak_memory_bytes=_measured_peak_memory_bytes(),
        pool_fingerprint=pool.fingerprint,
        bandwidth_fingerprint=frozen.fingerprint,
        transform_fingerprint=pool.transform_fingerprint,
        backend=frozen.backend,
        metric=frozen.metric,
        backend_version=frozen.backend_version,
        kernel_operator_fingerprint=operator_fingerprint,
        source_query_count=completed_sources,
    )
    logger.info(
        "Streamed candidate contributions completed: E=%d (implicit), Q=%d, "
        "atomic_graph_materialized=false, final_csr_bytes=%d, spool_bytes=%d, "
        "peak_estimated_bytes=%d, elapsed=%.3fs",
        summary.E,
        summary.Q,
        summary.candidate_csr_array_bytes,
        summary.contribution_spool_bytes,
        summary.estimated_peak_memory_bytes,
        max(0.0, time.perf_counter() - started),
    )
    return contributions, summary


def _coerce_graph_and_pool(
    first: EntropyPool | SparseAtomicKernelGraph,
    second: EntropyPool | SparseAtomicKernelGraph | None,
) -> tuple[SparseAtomicKernelGraph, EntropyPool | None]:
    if isinstance(first, SparseAtomicKernelGraph):
        graph = first
        pool = second if isinstance(second, EntropyPool) else None
    elif isinstance(first, EntropyPool) and isinstance(second, SparseAtomicKernelGraph):
        graph = second
        pool = first
    else:
        raise TypeError("candidate aggregation requires a sparse graph and optional EntropyPool")
    if second is not None and pool is None and second is not graph:
        raise TypeError("candidate aggregation received incompatible graph/pool arguments")
    if pool is not None:
        if pool.fingerprint != graph.pool_fingerprint:
            raise ValueError("sparse graph belongs to a different entropy pool")
        if pool.transform_fingerprint != graph.transform_fingerprint:
            raise ValueError("sparse graph has a mismatched transform fingerprint")
        if tuple(pool.candidate_ids) != graph.candidate_ids:
            raise ValueError("sparse graph candidate IDs do not match the entropy pool")
        if not np.array_equal(pool.row_candidate_indices, graph.row_candidate_indices):
            raise ValueError("sparse graph row ownership does not match the entropy pool")
    return graph, pool


def _candidate_fingerprint(
    graph: SparseAtomicKernelGraph,
    candidate_indptr: np.ndarray,
    target_indices: np.ndarray,
    values: np.ndarray,
    source_counts: np.ndarray,
) -> str:
    payload = {
        "schema": SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
        "graph": graph.fingerprint,
        "pool": graph.pool_fingerprint,
        "candidate_ids": list(graph.candidate_ids),
        "row_candidate_indices": _array_digest(graph.row_candidate_indices),
        "aggregation": "source-column-average-by-candidate-row-count",
        "source_order": "ordered-pool-rows",
        "target_order": "ascending-pool-row-index",
        "dtype": "float64",
        "numerical_tolerance": SPARSE_NUMERICAL_TOLERANCE,
        "candidate_indptr": _array_digest(candidate_indptr),
        "target_indices": _array_digest(target_indices),
        "values": _array_digest(values),
        "candidate_source_counts": _array_digest(source_counts),
    }
    return sha256_canonical_json(payload)


def _report_candidate_progress(
    completed: int,
    total: int,
    started: float,
    next_percent: list[int],
    candidate_id: str,
) -> None:
    if total < 100:
        should_report = True
    else:
        should_report = completed * 100 >= next_percent[0] * total
    if not should_report:
        return
    elapsed, throughput, remaining = _progress_timing(completed, total, started)
    logger.info(
        "Sparse candidate contribution progress: candidates=%d/%d (%.1f%%), "
        "completed_candidate_id=%s, throughput=%.3f candidates/s, elapsed=%.3fs, "
        "estimated remaining=%.3fs",
        completed,
        total,
        100.0 * completed / total,
        candidate_id,
        throughput,
        elapsed,
        remaining,
    )
    if total >= 100:
        while next_percent[0] <= 100 and completed * 100 >= next_percent[0] * total:
            next_percent[0] += 1


def aggregate_candidate_contributions(
    first: EntropyPool | SparseAtomicKernelGraph,
    second: EntropyPool | SparseAtomicKernelGraph | None = None,
    *,
    max_entries: int = _DEFAULT_SPARSE_EDGE_LIMIT,
    max_graph_bytes: int | None = None,
    max_spool_bytes: int | None = None,
    progress_callback: SourceProgressCallback | None = None,
) -> SparseCandidateContributions:
    """Aggregate source-major graph columns into normalized candidate rows."""

    graph, pool = _coerce_graph_and_pool(first, second)
    entry_limit = _validate_sparse_limit(max_entries, "max_entries")
    byte_limit = (
        None
        if max_graph_bytes is None
        else _validate_sparse_limit(max_graph_bytes, "max_graph_bytes")
    )
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    candidate_count = len(graph.candidate_ids)
    started = time.perf_counter()
    aggregate_peak_bytes = 0
    aggregate_spool_bytes = 0
    logger.info(
        "Sparse candidate contribution aggregation started: M=%d, schema=%s, "
        "max_entries=%d, max_graph_bytes=%s, max_spool_bytes=%s",
        candidate_count,
        SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
        entry_limit,
        byte_limit if byte_limit is not None else "unbounded",
        max_spool_bytes if max_spool_bytes is not None else "derived-from-max_entries",
    )
    try:
        source_counts = np.bincount(
            graph.row_candidate_indices,
            minlength=candidate_count,
        ).astype(np.int64)
        if np.any(source_counts < 1):
            missing = graph.candidate_ids[int(np.flatnonzero(source_counts < 1)[0])]
            raise ValueError(f"candidate {missing!r} owns no source rows")
        source_order = np.argsort(graph.row_candidate_indices, kind="stable")
        candidate_offsets = np.empty(candidate_count + 1, dtype=np.int64)
        candidate_offsets[0] = 0
        np.cumsum(source_counts, dtype=np.int64, out=candidate_offsets[1:])

        def estimated_peak_bytes(entry_count: int) -> int:
            final_arrays = 8 * (candidate_count + 1 + entry_count + entry_count + candidate_count)
            # CPython dict entries retain boxed integer keys and float values;
            # 72 bytes per live entry is deliberately conservative for the
            # temporary single-candidate accumulator and table slack.
            accumulator = 128 + 72 * entry_count + 16 * entry_count
            return int(graph.array_bytes + source_order.nbytes + final_arrays + accumulator)

        support_counts = np.zeros(candidate_count, dtype=np.int64)
        completed_candidates = 0
        next_percent = [1]
        spool_limit = _validate_spool_limit(max_spool_bytes, "max_spool_bytes")
        spool_limit = (
            spool_limit
            if spool_limit is not None
            else len(_EDGE_SPOOL_HEADER) + entry_limit * _EDGE_SPOOL_RECORD_BYTES
        )
        if spool_limit < len(_EDGE_SPOOL_HEADER):
            raise ValueError("max_spool_bytes is too small for the candidate contribution spool header")

        # One graph walk per candidate is enough.  The normalized, sorted
        # candidate rows are streamed to a bounded scratch spool while the
        # exact final CSR size is learned.  The second pass below reads bytes,
        # never re-enters the graph or repeats Python hash/reduction work.
        with tempfile.TemporaryFile(mode="w+b") as spool:
            spool_digest = _spool_write_header(spool)
            spool_bytes = len(_EDGE_SPOOL_HEADER)
            for candidate_index in range(candidate_count):
                accumulator: dict[int, float] = {}
                start_source = int(candidate_offsets[candidate_index])
                stop_source = int(candidate_offsets[candidate_index + 1])
                for position in range(start_source, stop_source):
                    source_index = int(source_order[position])
                    start_edge = int(graph.source_indptr[source_index])
                    stop_edge = int(graph.source_indptr[source_index + 1])
                    for edge_index in range(start_edge, stop_edge):
                        target = int(graph.target_indices[edge_index])
                        weight = float(graph.values[edge_index])
                        if target in accumulator:
                            accumulator[target] += weight
                            continue
                        prospective_count = len(accumulator) + 1
                        if prospective_count > entry_limit:
                            raise ValueError(
                                "sparse candidate contributions exceed max_entries: "
                                f"M={candidate_count}, required_entries>{entry_limit}, "
                                f"max_entries={entry_limit}"
                            )
                        estimated_bytes = estimated_peak_bytes(prospective_count)
                        aggregate_peak_bytes = max(aggregate_peak_bytes, estimated_bytes)
                        if byte_limit is not None:
                            if estimated_bytes > byte_limit:
                                raise ValueError(
                                    "sparse candidate contributions exceed max_graph_bytes: "
                                    f"M={candidate_count}, required_entries>={prospective_count}, "
                                    f"estimated_peak_bytes={estimated_bytes}, "
                                    f"max_graph_bytes={byte_limit}"
                                )
                        accumulator[target] = weight
                support = len(accumulator)
                support_counts[candidate_index] = support
                completed_entries = int(
                    np.sum(support_counts[: candidate_index + 1], dtype=np.int64)
                )
                if completed_entries > entry_limit:
                    raise ValueError(
                        "sparse candidate contributions exceed max_entries: "
                        f"M={candidate_count}, required_entries={completed_entries}, "
                        f"max_entries={entry_limit}"
                    )
                targets = sorted(accumulator)
                records = np.empty(support, dtype=_EDGE_SPOOL_DTYPE)
                records["target"] = targets
                records["value"] = [
                    accumulator[target] / float(source_counts[candidate_index])
                    for target in targets
                ]
                if not np.all(np.isfinite(records["value"])) or np.any(records["value"] <= 0.0):
                    raise ValueError("candidate contribution is non-finite or non-positive")
                if not math.isclose(
                    float(np.sum(records["value"], dtype=np.float64)),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=SPARSE_NUMERICAL_TOLERANCE,
                ):
                    raise ValueError(
                        f"candidate {graph.candidate_ids[candidate_index]!r} contribution is not normalized"
                    )
                prospective_spool_bytes = spool_bytes + support * _EDGE_SPOOL_RECORD_BYTES
                if prospective_spool_bytes > spool_limit:
                    raise ValueError(
                        "sparse candidate contributions exact spool exceeds its bounded capacity: "
                        f"M={candidate_count}, Q={completed_entries}, "
                        f"candidate={graph.candidate_ids[candidate_index]!r}, "
                        f"spool_bytes={prospective_spool_bytes}, max_spool_bytes={spool_limit}"
                    )
                _spool_write_records(spool, spool_digest, records)
                spool_bytes = prospective_spool_bytes
                aggregate_spool_bytes = spool_bytes
                completed_candidates += 1
                _report_candidate_progress(
                    completed_candidates,
                    candidate_count,
                    started,
                    next_percent,
                    graph.candidate_ids[candidate_index],
                )
                if progress_callback is not None:
                    progress_callback(completed_candidates, candidate_count)

            candidate_indptr = np.empty(candidate_count + 1, dtype=np.int64)
            candidate_indptr[0] = 0
            np.cumsum(support_counts, dtype=np.int64, out=candidate_indptr[1:])
            entry_count = int(candidate_indptr[-1])
            if entry_count > entry_limit:
                raise ValueError(
                    "sparse candidate contributions exceed max_entries: "
                    f"M={candidate_count}, required_entries={entry_count}, max_entries={entry_limit}"
                )
            required_bytes = 8 * (candidate_count + 1 + entry_count + entry_count + candidate_count)
            estimated_peak = estimated_peak_bytes(entry_count)
            aggregate_peak_bytes = max(aggregate_peak_bytes, estimated_peak)
            if byte_limit is not None and estimated_peak > byte_limit:
                raise ValueError(
                    "sparse candidate contributions exceed max_graph_bytes: "
                    f"M={candidate_count}, required_bytes={required_bytes}, "
                    f"estimated_peak_bytes={estimated_peak}, max_graph_bytes={byte_limit}"
                )
            target_indices = np.empty(entry_count, dtype=np.int64)
            contribution_values = np.empty(entry_count, dtype=np.float64)
            spool.seek(0)
            if spool.read(len(_EDGE_SPOOL_HEADER)) != _EDGE_SPOOL_HEADER:
                raise ValueError("candidate contribution spool has an invalid version header")
            read_digest = hashlib.sha256()
            read_digest.update(_EDGE_SPOOL_HEADER)
            entry_cursor = 0
            for records in _spool_read_records(spool, read_digest, entry_count):
                count = records.shape[0]
                target_indices[entry_cursor : entry_cursor + count] = records["target"]
                contribution_values[entry_cursor : entry_cursor + count] = records["value"]
                entry_cursor += count
            if spool.read(1):
                raise ValueError("candidate contribution spool contains trailing records")
            if read_digest.digest() != spool_digest.digest() or entry_cursor != entry_count:
                raise ValueError("candidate contribution spool checksum validation failed")
        contribution_fingerprint = _candidate_fingerprint(
            graph,
            candidate_indptr,
            target_indices,
            contribution_values,
            source_counts,
        )
        contributions = SparseCandidateContributions(
            candidate_indptr=candidate_indptr,
            target_indices=target_indices,
            values=contribution_values,
            candidate_ids=graph.candidate_ids,
            graph_fingerprint=graph.fingerprint,
            fingerprint=contribution_fingerprint,
            row_count=graph.n_targets,
            candidate_source_counts=source_counts,
            pool_fingerprint=graph.pool_fingerprint,
        )
    except Exception as exc:
        logger.error("Sparse candidate contribution aggregation failed: %s", exc)
        raise

    elapsed, _, _ = _progress_timing(candidate_count, candidate_count, started)
    support_sizes = contributions.support_sizes
    logger.info(
        "Sparse candidate contribution aggregation completed: entries=%d, mean support=%.3f, "
        "max support=%d, final_csr_bytes=%d, peak_estimated_bytes=%d, spool_bytes=%d, "
        "elapsed=%.3fs",
        contributions.entry_count,
        float(np.mean(support_sizes)),
        int(np.max(support_sizes)),
        contributions.array_bytes,
        aggregate_peak_bytes,
        aggregate_spool_bytes,
        elapsed,
    )
    return contributions


build_direct_candidate_contributions = build_streamed_candidate_contributions
stream_candidate_contributions = build_streamed_candidate_contributions
build_sparse_candidate_contributions = aggregate_candidate_contributions
build_candidate_contributions = aggregate_candidate_contributions
compute_candidate_contributions = aggregate_candidate_contributions
construct_sparse_atomic_kernel_graph = build_sparse_atomic_kernel_graph
build_atomic_kernel_graph = build_sparse_atomic_kernel_graph
construct_atomic_kernel_graph = build_sparse_atomic_kernel_graph
build_sparse_kernel_graph = build_sparse_atomic_kernel_graph
create_sparse_atomic_kernel_graph = build_sparse_atomic_kernel_graph
aggregate_sparse_candidate_contributions = aggregate_candidate_contributions


evaluate_loo_objective = evaluate_leave_one_out_objective


__all__ = [
    "KERNEL_FAMILY",
    "KERNEL_VERSION",
    "KERNEL_OPERATOR_SCHEMA_VERSION",
    "KernelMetadata",
    "LeaveOneOutObjective",
    "NormalizedKernelColumn",
    "SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION",
    "SPARSE_KERNEL_GRAPH_SCHEMA_VERSION",
    "SPARSE_NUMERICAL_TOLERANCE",
    "SourceProgressCallback",
    "BatchSourceProgressCallback",
    "SparseAtomicKernelGraph",
    "SparseCandidateContributions",
    "StreamedKernelExecutionSummary",
    "aggregate_candidate_contributions",
    "aggregate_sparse_candidate_contributions",
    "build_atomic_kernel_graph",
    "build_candidate_contributions",
    "build_sparse_atomic_kernel_graph",
    "build_sparse_candidate_contributions",
    "build_streamed_candidate_contributions",
    "build_direct_candidate_contributions",
    "stream_candidate_contributions",
    "build_sparse_kernel_graph",
    "compute_candidate_contributions",
    "construct_atomic_kernel_graph",
    "construct_sparse_atomic_kernel_graph",
    "create_sparse_atomic_kernel_graph",
    "evaluate_leave_one_out_objective",
    "evaluate_leave_one_out_objectives",
    "evaluate_batched_leave_one_out_objectives",
    "evaluate_leave_one_out_batch",
    "evaluate_loo_objective",
    "evaluate_wendland_kernel",
    "iter_normalized_kernel_columns",
    "normalised_kernel_matrix",
    "normalized_kernel_matrix",
    "source_normalisers",
    "wendland_c2",
    "wendland",
    "wendland_kernel",
]
