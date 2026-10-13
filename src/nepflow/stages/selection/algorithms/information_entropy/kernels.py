"""Versioned Wendland kernels and bounded finite-pool calibration evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json
from nepflow.resources.budget import (
    ResourceBudgetService,
    ResourceCapacityError,
    build_resource_budget,
)

from .checkpointing import (
    CandidateAggregationCheckpoint,
    CheckpointValidationError,
    VersionedCheckpointStore,
)
from .models import (
    DEFAULT_LOO_CHECKPOINT_EVERY_SOURCES,
    DEFAULT_RADIUS_QUERY_BYTES,
    GPU_NEIGHBOUR_BACKEND_ID,
    GPU_NEIGHBOUR_BACKEND_VERSION,
    INDEXED_NEIGHBOUR_BACKEND_ID,
    INDEXED_NEIGHBOUR_BACKEND_VERSION,
    KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE,
    KERNEL_BANDWIDTH_RULE_SYMMETRIC_MAX,
    KERNEL_FAMILY,
    KERNEL_OPERATOR_SCHEMA_VERSION,
    KERNEL_SUPPORT_EXECUTION_CPU,
    KERNEL_SUPPORT_EXECUTION_GPU,
    KERNEL_SUPPORT_EXECUTION_GPU_VERSION,
    KERNEL_SUPPORT_EXECUTIONS,
    KERNEL_VERSION,
    NEIGHBOUR_BACKEND_ID,
    NEIGHBOUR_BACKEND_VERSION,
    NEIGHBOUR_METRIC,
    SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
    SPARSE_KERNEL_GRAPH_SCHEMA_VERSION,
    SPARSE_NUMERICAL_TOLERANCE,
    SPARSE_STREAMED_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
    TILED_NEIGHBOUR_BACKEND_ID,
    TILED_NEIGHBOUR_BACKEND_VERSION,
    BandwidthCalibrationResult,
    EntropyPool,
    FrozenBandwidths,
    KernelMetadata,
    OutOfCoreSparseCandidateContributions,
    SparseAtomicKernelGraph,
    SparseCandidateContributions,
    StreamedKernelExecutionSummary,
    normalize_kernel_bandwidth_rule,
)
from .neighbours import (
    ExactKNNPrefixSupport,
    ExactNeighbourResult,
    IndexedCPUNeighbourIndex,
    RadiusQueryCapacityError,
    TiledCPUNeighbourIndex,
    _process_memory_metrics,
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


def effective_pair_bandwidths(
    source_bandwidth: float,
    target_bandwidths: np.ndarray,
    *,
    kernel_bandwidth_rule: str = KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE,
) -> np.ndarray:
    """Evaluate the canonical effective bandwidth ``H(a, b)`` for one source.

    ``target_bandwidths`` is ordered by target row ``b`` and ``source_bandwidth``
    is the scalar ``h_a``.  This helper is intentionally CPU/reference-facing;
    production support queries still use the legacy source-radius path until
    the symmetric integration issue is complete.
    """

    rule = normalize_kernel_bandwidth_rule(kernel_bandwidth_rule)
    source = float(source_bandwidth)
    if not math.isfinite(source) or source <= 0.0:
        raise ValueError("source_bandwidth must be finite and strictly positive")
    targets = np.asarray(target_bandwidths)
    if targets.dtype != np.dtype(np.float64):
        raise ValueError("target_bandwidths must have dtype float64")
    if targets.ndim != 1 or not np.all(np.isfinite(targets)) or np.any(targets <= 0.0):
        raise ValueError("target_bandwidths must be a finite, strictly positive vector")
    if rule == KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE:
        return np.full(targets.shape, source, dtype=np.float64)
    return np.maximum(source, targets)


def evaluate_raw_kernel_pairs(
    distances: np.ndarray,
    source_bandwidth: float,
    target_bandwidths: np.ndarray,
    *,
    kernel_bandwidth_rule: str = KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(H, strict_support, W)`` for one source column.

    Support uses the strict FP64 predicate ``d < H``.  Distances and
    bandwidths are required to be finite, so extreme finite ratios that are
    outside support are screened without evaluating an infinite ratio.  A
    positive in-support Wendland value that underflows in FP64 remains zero;
    no floor or other noncanonical substitute is introduced.  Coincident rows
    therefore retain their multiplicity and receive the exact value
    ``wendland_kernel(0) == 1``.
    """

    distance_values = np.asarray(distances)
    targets = np.asarray(target_bandwidths)
    if distance_values.dtype != np.dtype(np.float64):
        raise ValueError("distances must have dtype float64")
    if targets.dtype != np.dtype(np.float64):
        raise ValueError("target_bandwidths must have dtype float64")
    if distance_values.ndim != 1:
        raise ValueError("distances must be a one-dimensional vector")
    if targets.ndim != 1 or targets.shape != distance_values.shape:
        raise ValueError("target_bandwidths must align with distances")
    if not np.all(np.isfinite(distance_values)) or np.any(distance_values < 0.0):
        raise ValueError("distances must be finite and non-negative")
    effective = effective_pair_bandwidths(
        source_bandwidth,
        targets,
        kernel_bandwidth_rule=kernel_bandwidth_rule,
    )
    supported = distance_values < effective
    raw = np.zeros(distance_values.shape, dtype=np.float64)
    if np.any(supported):
        with np.errstate(divide="raise", over="raise", invalid="raise"):
            ratios = distance_values[supported] / effective[supported]
        raw[supported] = np.asarray(wendland_kernel(ratios), dtype=np.float64)
    return effective, supported, raw


# Descriptive aliases make the reference contract discoverable without
# proliferating independent implementations.
evaluate_raw_kernel = evaluate_raw_kernel_pairs


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
        index: IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None = None,
        max_radius_query_bytes: int | None = None,
        resource_budget: ResourceBudgetService | None = None,
        capacity_context: str | None = None,
        kernel_support_execution: str = KERNEL_SUPPORT_EXECUTION_CPU,
        gpu_device: int = 0,
        gpu_devices: Sequence[int] | str | int | None = None,
        gpu_tile_size: int = 128,
        max_gpu_support_bytes: int | None = None,
        max_gpu_neighbour_bytes: int | None = None,
        knn_support: ExactNeighbourResult | None = None,
        knn_support_row_ids: Sequence[str] | None = None,
        max_calibration_work_bytes: int | None = None,
    ) -> None:
        self.descriptors = _validate_descriptors(descriptors)
        self.backend = backend
        self.chunk_size = _validate_chunk_size(chunk_size)
        self.resource_budget = (
            resource_budget or getattr(index, "resource_budget", None) or build_resource_budget()
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
        self.support_execution = str(kernel_support_execution).strip().lower()
        if self.support_execution == "exact_cpu":
            self.support_execution = KERNEL_SUPPORT_EXECUTION_CPU
        if self.support_execution not in KERNEL_SUPPORT_EXECUTIONS:
            raise ValueError("kernel_support_execution must be 'cpu' or 'exact_tiled_gpu'")
        self.support_execution_version = (
            KERNEL_SUPPORT_EXECUTION_GPU_VERSION
            if self.support_execution == KERNEL_SUPPORT_EXECUTION_GPU
            else "cpu-canonical-v1"
        )
        self.gpu_support: Any = None
        from .gpu_devices import normalize_gpu_devices

        self.gpu_devices = normalize_gpu_devices(gpu_devices, gpu_device=gpu_device)
        if max_gpu_neighbour_bytes is None and backend == GPU_NEIGHBOUR_BACKEND_ID:
            # Older public kernel entry points expose only the support limit.
            # Reusing it as a conservative device ceiling preserves their
            # bounded-memory contract; the dedicated bandwidth configuration
            # passes max_gpu_neighbour_bytes explicitly at index construction.
            max_gpu_neighbour_bytes = max_gpu_support_bytes
        if backend in {
            INDEXED_NEIGHBOUR_BACKEND_ID,
            TILED_NEIGHBOUR_BACKEND_ID,
            GPU_NEIGHBOUR_BACKEND_ID,
        }:
            self.index = (
                index
                if index is not None
                else build_neighbour_index(
                    self.descriptors,
                    backend=backend,
                    tile_size=chunk_size,
                    max_radius_query_bytes=self.max_radius_query_bytes,
                    gpu_device=gpu_device,
                    gpu_devices=self.gpu_devices,
                    gpu_tile_size=gpu_tile_size,
                    max_gpu_neighbour_bytes=max_gpu_neighbour_bytes,
                    resource_budget=self.resource_budget,
                )
            )
            if backend == GPU_NEIGHBOUR_BACKEND_ID:
                from .gpu_neighbours import ExactTiledGPUNeighbourIndex

                expected_type = ExactTiledGPUNeighbourIndex
            else:
                expected_type = (
                    IndexedCPUNeighbourIndex
                    if backend == INDEXED_NEIGHBOUR_BACKEND_ID
                    else TiledCPUNeighbourIndex
                )
            if not isinstance(self.index, expected_type):
                raise ValueError("exact kernel support requires its configured CPU backend")
            if not np.array_equal(self.index.descriptors, self.descriptors):
                raise ValueError("indexed kernel support index belongs to different descriptors")
        elif backend == NEIGHBOUR_BACKEND_ID:
            if index is not None:
                raise ValueError("exact_cpu kernel support does not accept an indexed index")
            self.index = None
        else:
            raise ValueError(f"configured neighbour backend is unavailable: {backend!r}")
        if self.support_execution == KERNEL_SUPPORT_EXECUTION_GPU:
            from .gpu_support import ExactTiledGPUSupport

            self.gpu_support = ExactTiledGPUSupport(
                self.descriptors,
                device=gpu_device,
                gpu_devices=self.gpu_devices,
                tile_size=gpu_tile_size,
                max_device_bytes=max_gpu_support_bytes,
                max_host_bytes=self.max_radius_query_bytes,
                resource_budget=self.resource_budget,
                capacity_context=capacity_context,
            )
            self.support_batch_size = self.gpu_support.source_tile_size
        else:
            self.support_batch_size = self.chunk_size
        self.knn_prefix_support: ExactKNNPrefixSupport | None = None
        if knn_support is not None:
            self.knn_prefix_support = ExactKNNPrefixSupport(
                self.descriptors,
                knn_support,
                grouping_source=self.index,
                expected_row_ids=knn_support_row_ids,
                max_radius_query_bytes=self.max_radius_query_bytes,
                max_calibration_work_bytes=max_calibration_work_bytes,
                resource_budget=self.resource_budget,
                capacity_context=capacity_context,
            )
        self.knn_prefix_support_used = False
        self.generic_support_used = False
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
        self.support_query_count = 0
        self.support_query_seconds = 0.0
        self.support_query_peak_scratch_bytes = 0

    def support(self, source_index: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
        scale = float(radius)
        started = time.perf_counter()
        if self.gpu_support is not None:
            offsets, targets, distances = self.gpu_support.support_many(
                np.asarray([source_index], dtype=np.int64),
                np.asarray([scale], dtype=np.float64),
            )
            result_targets = targets[: int(offsets[1])]
            result_distances = distances[: int(offsets[1])]
            self.support_entries += int(result_targets.size)
            self.current_support_count = int(result_targets.size)
            if self.min_support_count is None:
                self.min_support_count = self.current_support_count
            else:
                self.min_support_count = min(self.min_support_count, self.current_support_count)
            self.support_count_total += self.current_support_count
            self.support_count_samples += 1
            self.max_support_count = max(self.max_support_count, self.current_support_count)
            self.support_query_count += 1
            self.support_query_seconds += max(0.0, time.perf_counter() - started)
            return result_targets, result_distances
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
            self.support_query_count += 1
            self.support_query_seconds += max(0.0, time.perf_counter() - started)
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
        self.support_query_count += 1
        self.support_query_seconds += max(0.0, time.perf_counter() - started)
        return targets, distances

    def support_many(
        self,
        source_indices: Sequence[int] | np.ndarray,
        radii: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return source supports as bounded CSR arrays.

        ``radii`` may be a one-dimensional source-radius vector or a
        ``(candidate_count, source_count)`` matrix.  The GPU provider uses the
        latter to screen one conservative per-source superset and the caller
        applies each candidate's strict radius to the canonical distances.
        CPU providers use the same maximum-radius superset contract, keeping
        the accumulation order identical across execution modes.

        Indexed CPU support queries share their tree preflight and canonical
        distance pass.  Backends without a batched radius primitive retain the
        exact source-by-source implementation and are assembled into the same
        CSR representation for callers.
        """

        sources = np.asarray(source_indices, dtype=np.int64)
        requested_radii = np.asarray(radii, dtype=np.float64)
        if sources.ndim != 1 or requested_radii.ndim not in (1, 2):
            raise ValueError(
                "source_indices must be one-dimensional and radii must have shape "
                "(source_count,) or (candidate_count, source_count)"
            )
        if requested_radii.shape[-1] != sources.size:
            raise ValueError("source_indices and radii must have aligned source dimensions")
        if requested_radii.ndim == 2 and requested_radii.shape[0] == 0:
            raise ValueError("support_many requires at least one radius candidate")
        if sources.size == 0:
            raise ValueError("support_many requires at least one source")
        if not np.all(np.isfinite(requested_radii)) or np.any(requested_radii <= 0.0):
            raise ValueError("support radii must be finite and strictly positive")
        if (
            self.knn_prefix_support is not None
            and self.knn_prefix_support.eligible(sources, requested_radii)
        ):
            started = time.perf_counter()
            offsets, targets, distances = self.knn_prefix_support.support_many(
                sources,
                requested_radii,
            )
            for count in np.diff(offsets):
                support_count = int(count)
                self.current_support_count = support_count
                if self.min_support_count is None:
                    self.min_support_count = support_count
                else:
                    self.min_support_count = min(self.min_support_count, support_count)
                self.support_count_total += support_count
                self.support_count_samples += 1
                self.max_support_count = max(self.max_support_count, support_count)
            self.support_entries += int(targets.size)
            self.support_query_count += int(sources.size)
            self.support_query_seconds += max(0.0, time.perf_counter() - started)
            self.knn_prefix_support_used = True
            return offsets, targets, distances
        if self.knn_prefix_support is not None:
            self.generic_support_used = True
        scales = (
            requested_radii
            if requested_radii.ndim == 1
            else np.max(requested_radii, axis=0)
        )
        started = time.perf_counter()
        if self.gpu_support is not None:
            offsets, targets, distances = self.gpu_support.support_many(
                sources, requested_radii
            )
            counts = np.diff(offsets)
            for count in counts:
                support_count = int(count)
                self.current_support_count = support_count
                if self.min_support_count is None:
                    self.min_support_count = support_count
                else:
                    self.min_support_count = min(self.min_support_count, support_count)
                self.support_count_total += support_count
                self.support_count_samples += 1
                self.max_support_count = max(self.max_support_count, support_count)
            self.support_entries += int(targets.size)
            self.support_query_count += int(sources.size)
            self.support_query_seconds += max(0.0, time.perf_counter() - started)
            return offsets, targets, distances
        indexed_batch = isinstance(self.index, IndexedCPUNeighbourIndex)
        if indexed_batch:
            offsets, targets, distances = self.index.radius_support_many(
                sources,
                scales,
                chunk_size=self.chunk_size,
                context=self.capacity_context,
            )
            self.index_queries += int(sources.size)
            self.radius_queries = self.index.radius_queries
            self.query_workspace_peak_bytes = self.index.query_workspace_peak_bytes
        else:
            target_parts: list[np.ndarray] = []
            distance_parts: list[np.ndarray] = []
            offsets = np.zeros(sources.size + 1, dtype=np.int64)
            for offset, (source, scale) in enumerate(zip(sources, scales, strict=True)):
                target, distance = self.support(int(source), float(scale))
                target_parts.append(target)
                distance_parts.append(distance)
                offsets[offset + 1] = offsets[offset] + int(target.size)
            targets = np.concatenate(target_parts).astype(np.int64, copy=False)
            distances = np.asarray(np.concatenate(distance_parts), dtype=np.float64)

        if indexed_batch:
            counts = np.diff(offsets)
            for count in counts:
                support_count = int(count)
                self.current_support_count = support_count
                if self.min_support_count is None:
                    self.min_support_count = support_count
                else:
                    self.min_support_count = min(self.min_support_count, support_count)
                self.support_count_total += support_count
                self.support_count_samples += 1
                self.max_support_count = max(self.max_support_count, support_count)
            self.support_entries += int(targets.size)
            self.support_query_count += int(sources.size)
            self.support_query_seconds += max(0.0, time.perf_counter() - started)
        return offsets, targets, distances

    def operational_metrics(self) -> dict[str, Any]:
        """Return support/query evidence without exposing it to science IDs."""

        index_metrics = self.index.memory_metrics() if self.index is not None else {}
        count = int(self.support_count_samples)
        total = int(self.support_count_total)
        metrics: dict[str, Any] = {
            "support_query_count": int(self.support_query_count),
            "support_query_seconds": float(self.support_query_seconds),
            "support_query_throughput_per_second": (
                self.support_query_count / self.support_query_seconds
                if self.support_query_seconds > 0.0
                else 0.0
            ),
            "support_entries": int(self.support_entries),
            "support_density_mean": total / (count * self.descriptors.shape[0]) if count else 0.0,
            "support_count_min": self.min_support_count,
            "support_count_mean": total / count if count else None,
            "support_count_max": int(self.max_support_count),
            "current_support_count": int(self.current_support_count),
            "query_workspace_peak_bytes": int(
                max(
                    self.query_workspace_peak_bytes,
                    int(index_metrics.get("query_workspace_peak_bytes", 0) or 0),
                )
            ),
            "distance_evaluations": int(
                index_metrics.get("distance_evaluations", self.distance_evaluations) or 0
            ),
            "unique_distance_evaluations": int(
                index_metrics.get("unique_distance_evaluations", 0) or 0
            ),
            "atomic_distance_evaluations": int(
                index_metrics.get("atomic_distance_evaluations", 0) or 0
            ),
            "radius_queries": int(index_metrics.get("radius_queries", self.radius_queries) or 0),
            "index_queries": int(index_metrics.get("index_queries", self.index_queries) or 0),
            "process_peak_rss_bytes": _process_memory_metrics().get("process_peak_rss_bytes"),
            "cgroup_memory_current_bytes": _process_memory_metrics().get(
                "cgroup_memory_current_bytes"
            ),
            "support_execution": self.support_execution,
            "support_execution_version": self.support_execution_version,
        }
        if self.gpu_support is not None:
            metrics.update(self.gpu_support.operational_metrics())
            metrics["support_provider"] = "active_gpu_support"
            metrics["support_provider_stats_status"] = "active"
            metrics["support_provider_radius_queries"] = None
            metrics["support_provider_distance_evaluations"] = int(
                metrics.get("gpu_fp64_evaluations", 0) or 0
            )
            metrics["support_provider_unique_distance_evaluations"] = None
            metrics["support_provider_atomic_distance_evaluations"] = None
            if "gpu_per_device" in index_metrics:
                # The support provider owns the generic per-device key.  Keep
                # the neighbour scheduler's independent device timings as a
                # separate field so neither execution stage is hidden.
                metrics["knn_gpu_per_device"] = index_metrics["gpu_per_device"]
        else:
            metrics["support_provider"] = "active_cpu_support"
            metrics["support_provider_stats_status"] = "active"
            metrics["support_provider_radius_queries"] = int(self.radius_queries)
            metrics["support_provider_distance_evaluations"] = int(
                index_metrics.get("distance_evaluations", self.distance_evaluations) or 0
            )
            metrics["support_provider_unique_distance_evaluations"] = int(
                index_metrics.get("unique_distance_evaluations", 0) or 0
            )
            metrics["support_provider_atomic_distance_evaluations"] = int(
                index_metrics.get("atomic_distance_evaluations", 0) or 0
            )
        if self.knn_prefix_support is not None:
            metrics.update(self.knn_prefix_support.operational_metrics())
            if self.knn_prefix_support_used and self.generic_support_used:
                metrics["support_provider"] = "mixed_exact_knn_prefix_and_generic"
            elif self.knn_prefix_support_used:
                metrics.update(
                    {
                        "support_provider": "exact_knn_prefix",
                        "support_provider_stats_status": "active",
                        "support_provider_radius_queries": 0,
                        "support_provider_distance_evaluations": 0,
                        "support_provider_unique_distance_evaluations": 0,
                        "support_provider_atomic_distance_evaluations": 0,
                        "distance_evaluations": 0,
                        "unique_distance_evaluations": 0,
                        "atomic_distance_evaluations": 0,
                        "radius_queries": 0,
                        "index_queries": 0,
                        "gpu_fp64_evaluations": 0,
                        "gpu_source_tiles": 0,
                        "gpu_target_tiles": 0,
                        "gpu_tile_scans": 0,
                        "gpu_screening_seconds": 0.0,
                        "query_workspace_peak_bytes": int(
                            self.knn_prefix_support.query_workspace_peak_bytes
                        ),
                    }
                )
        for key, value in index_metrics.items():
            metrics.setdefault(key, value)
        return metrics

    def close(self) -> None:
        """Release optional GPU support pools without owning the neighbour index."""

        if self.gpu_support is not None:
            self.gpu_support.close()

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
    index: IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None = None,
    support_provider: _KernelSupportProvider | None = None,
    max_radius_query_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    capacity_context: str | None = None,
    kernel_support_execution: str = KERNEL_SUPPORT_EXECUTION_CPU,
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_gpu_support_bytes: int | None = None,
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
        kernel_support_execution=kernel_support_execution,
        gpu_device=gpu_device,
        gpu_devices=gpu_devices,
        gpu_tile_size=gpu_tile_size,
        max_gpu_support_bytes=max_gpu_support_bytes,
    )
    normalisers = np.empty(values.shape[0], dtype=np.float64)
    query_block = min(block, provider.support_batch_size)
    for start in range(0, values.shape[0], query_block):
        stop = min(start + query_block, values.shape[0])
        offsets, targets, distances = provider.support_many(
            np.arange(start, stop, dtype=np.int64),
            scales[start:stop],
        )
        for offset, source_index in enumerate(range(start, stop)):
            begin = int(offsets[offset])
            end = int(offsets[offset + 1])
            normalisers[source_index] = provider.normaliser(
                source_index,
                float(scales[source_index]),
                targets[begin:end],
                distances[begin:end],
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
    index: IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None = None,
    support_provider: _KernelSupportProvider | None = None,
    max_radius_query_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    capacity_context: str | None = None,
    source_indices: Sequence[int] | np.ndarray | None = None,
    kernel_support_execution: str = KERNEL_SUPPORT_EXECUTION_CPU,
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_gpu_support_bytes: int | None = None,
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
        kernel_support_execution=kernel_support_execution,
        gpu_device=gpu_device,
        gpu_devices=gpu_devices,
        gpu_tile_size=gpu_tile_size,
        max_gpu_support_bytes=max_gpu_support_bytes,
    )
    if source_indices is None:
        ordered_sources = range(values.shape[0])
    else:
        ordered_sources = tuple(int(source_index) for source_index in source_indices)
        if any(
            source_index < 0 or source_index >= values.shape[0] for source_index in ordered_sources
        ):
            raise ValueError("source_indices contains an out-of-range source row")
    query_block = min(block, provider.support_batch_size)
    for start in range(0, len(ordered_sources), query_block):
        source_batch = ordered_sources[start : start + query_block]
        source_array = np.asarray(source_batch, dtype=np.int64)
        offsets, targets, distances = provider.support_many(
            source_array,
            scales[source_array],
        )
        for offset, source_index in enumerate(source_batch):
            begin = int(offsets[offset])
            end = int(offsets[offset + 1])
            scale = scales[source_index]
            target_indices = targets[begin:end]
            source_distances = distances[begin:end]
            normaliser = provider.normaliser(
                source_index,
                float(scale),
                target_indices,
                source_distances,
            )
            raw = np.asarray(wendland_kernel(source_distances / scale), dtype=np.float64)
            positive = raw > 0.0
            if not np.all(positive):
                target_indices = target_indices[positive]
                raw = raw[positive]
            normalized = raw / normaliser
            column_total = float(np.sum(normalized, dtype=np.float64))
            if target_indices.size:
                yield NormalizedKernelColumn(source_index, target_indices, normalized)
            if not math.isclose(column_total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(
                    f"normalized source kernel column {source_index} does not sum to one"
                )
            if progress_callback is not None:
                progress_callback(source_index + 1, values.shape[0])


def normalized_kernel_matrix(
    descriptors: np.ndarray,
    bandwidths: np.ndarray,
    *,
    chunk_size: int = 1024,
    max_dense_entries: int = 1_000_000,
    max_radius_query_bytes: int | None = None,
    kernel_bandwidth_rule: str = KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE,
) -> np.ndarray:
    """Build a deliberately bounded dense oracle for small correctness tests.

    Production calibration uses :func:`iter_normalized_kernel_columns` and
    never calls this helper.  The guard prevents accidental whole-pool use.
    Rows are targets and columns are sources.
    """

    values = _validate_descriptors(descriptors)
    scales = _validate_bandwidths(bandwidths, values.shape[0])
    rule = normalize_kernel_bandwidth_rule(kernel_bandwidth_rule)
    if isinstance(max_dense_entries, bool) or not isinstance(max_dense_entries, int):
        raise ValueError("max_dense_entries must be a positive integer")
    if max_dense_entries < 1 or values.shape[0] * values.shape[0] > max_dense_entries:
        raise ValueError("dense normalized kernel oracle exceeds its explicit memory bound")
    matrix = np.zeros((values.shape[0], values.shape[0]), dtype=np.float64)
    if rule == KERNEL_BANDWIDTH_RULE_SYMMETRIC_MAX:
        # Deliberately dense and bounded: this is the small CPU oracle only.
        # Rows are targets and columns are sources, matching the production
        # source-column orientation.
        for source_index in range(values.shape[0]):
            with np.errstate(over="raise", invalid="raise"):
                try:
                    distances = np.sqrt(
                        np.sum((values - values[source_index]) ** 2, axis=1)
                    )
                except FloatingPointError as exc:
                    raise ValueError("kernel distance overflowed or became invalid") from exc
            _effective, _supported, raw = evaluate_raw_kernel_pairs(
                distances,
                float(scales[source_index]),
                scales,
                kernel_bandwidth_rule=rule,
            )
            normaliser = float(np.sum(raw, dtype=np.float64))
            if not math.isfinite(normaliser) or normaliser <= 0.0:
                raise ValueError(f"source kernel normaliser is invalid for source {source_index}")
            matrix[:, source_index] = raw / normaliser
        totals = np.sum(matrix, axis=0, dtype=np.float64)
        if not np.allclose(totals, 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("reference normalized source kernel columns do not sum to one")
        return matrix
    for column in iter_normalized_kernel_columns(
        values,
        scales,
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
    index: IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None = None,
    max_radius_query_bytes: int | None = None,
    max_calibration_work_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    capacity_context: str | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_name: str = "loo",
    checkpoint_every_sources: int = DEFAULT_LOO_CHECKPOINT_EVERY_SOURCES,
    checkpoint_every_seconds: float | None = None,
    resume: bool = True,
    preserve_checkpoint: bool = False,
    diagnostics: dict[str, Any] | None = None,
    kernel_support_execution: str = KERNEL_SUPPORT_EXECUTION_CPU,
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_gpu_support_bytes: int | None = None,
    kernel_bandwidth_rule: str = KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE,
    knn_support: ExactNeighbourResult | None = None,
    knn_support_row_ids: Sequence[str] | None = None,
) -> LeaveOneOutObjective:
    """Evaluate the exact source-normalized finite-pool LOO objective."""

    bandwidth_rule = normalize_kernel_bandwidth_rule(kernel_bandwidth_rule)
    if bandwidth_rule != KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE:
        raise ValueError(
            "symmetric_max leave-one-out evaluation is not integrated in this release; "
            "use normalized_kernel_matrix as the bounded CPU reference oracle"
        )

    if checkpoint_path is not None:
        batched = evaluate_leave_one_out_objectives(
            descriptors,
            probabilities,
            np.asarray(bandwidths, dtype=np.float64).reshape(1, -1),
            chunk_size=chunk_size,
            progress_callback=progress_callback,
            backend=backend,
            index=index,
            max_radius_query_bytes=max_radius_query_bytes,
            max_calibration_work_bytes=max_calibration_work_bytes,
            resource_budget=resource_budget,
            capacity_context=capacity_context,
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            checkpoint_every_sources=checkpoint_every_sources,
            checkpoint_every_seconds=checkpoint_every_seconds,
            resume=resume,
            preserve_checkpoint=preserve_checkpoint,
            diagnostics=diagnostics,
            kernel_support_execution=kernel_support_execution,
            gpu_device=gpu_device,
            gpu_devices=gpu_devices,
            gpu_tile_size=gpu_tile_size,
            max_gpu_support_bytes=max_gpu_support_bytes,
            knn_support=knn_support,
            knn_support_row_ids=knn_support_row_ids,
        )
        return batched[0]

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

    provider = _KernelSupportProvider(
        values,
        backend=backend,
        chunk_size=chunk_size,
        index=index,
        max_radius_query_bytes=max_radius_query_bytes,
        resource_budget=resource_budget,
        capacity_context=capacity_context,
        kernel_support_execution=kernel_support_execution,
        gpu_device=gpu_device,
        gpu_devices=gpu_devices,
        gpu_tile_size=gpu_tile_size,
        max_gpu_support_bytes=max_gpu_support_bytes,
        knn_support=knn_support,
        knn_support_row_ids=knn_support_row_ids,
        max_calibration_work_bytes=max_calibration_work_bytes,
    )
    numerator = np.zeros(values.shape[0], dtype=np.float64)
    loo_started = time.perf_counter()
    for column in iter_normalized_kernel_columns(
        values,
        _validate_bandwidths(bandwidths, values.shape[0]),
        chunk_size=chunk_size,
        progress_callback=progress_callback,
        backend=backend,
        index=index,
        support_provider=provider,
        max_radius_query_bytes=max_radius_query_bytes,
        resource_budget=resource_budget,
        capacity_context=capacity_context,
        kernel_support_execution=kernel_support_execution,
        gpu_device=gpu_device,
        gpu_devices=gpu_devices,
        gpu_tile_size=gpu_tile_size,
        max_gpu_support_bytes=max_gpu_support_bytes,
    ):
        non_source = column.target_indices != column.source_index
        if np.any(non_source):
            numerator[column.target_indices[non_source]] += (
                masses[column.source_index] * column.values[non_source]
            )
    provider_metrics = provider.operational_metrics()
    if diagnostics is not None:
        diagnostics["loo_wall_seconds"] = max(0.0, time.perf_counter() - loo_started)
        diagnostics["loo_support_query_seconds"] = float(
            provider_metrics.get("support_query_seconds", 0.0) or 0.0
        )
        diagnostics["loo_distance_evaluations"] = int(
            provider_metrics.get("distance_evaluations", 0) or 0
        )
        for metric_name, metric_value in provider_metrics.items():
            if metric_name in {
                "support_execution",
                "support_execution_version",
                "support_provider",
                "support_provider_stats_status",
                "gpu_device",
                "gpu_devices",
                "gpu_device_count",
                "gpu_per_device",
                "knn_gpu_per_device",
                "gpu_rescan_reasons",
            }:
                diagnostics[metric_name] = metric_value
            elif isinstance(metric_value, bool):
                diagnostics[metric_name] = metric_value
            elif isinstance(metric_value, (int, float, np.integer, np.floating)):
                diagnostics[metric_name] = metric_value
        diagnostics["support_size_per_candidate"] = [
            provider_metrics.get("support_count_mean")
        ]
        diagnostics["support_max_per_candidate"] = [
            provider_metrics.get("support_count_max")
        ]
        diagnostics["conservative_candidate_count_per_candidate"] = [
            provider_metrics.get("support_count_mean")
        ]
        diagnostics["conservative_candidate_max_per_candidate"] = [
            provider_metrics.get("support_count_max")
        ]
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
    index: IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None = None,
    max_radius_query_bytes: int | None = None,
    max_calibration_work_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    capacity_context: str | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_name: str = "loo",
    checkpoint_every_sources: int = DEFAULT_LOO_CHECKPOINT_EVERY_SOURCES,
    checkpoint_every_seconds: float | None = None,
    resume: bool = True,
    preserve_checkpoint: bool = False,
    diagnostics: dict[str, Any] | None = None,
    kernel_support_execution: str = KERNEL_SUPPORT_EXECUTION_CPU,
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_gpu_support_bytes: int | None = None,
    kernel_bandwidth_rule: str = KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE,
    knn_support: ExactNeighbourResult | None = None,
    knn_support_row_ids: Sequence[str] | None = None,
) -> tuple[LeaveOneOutObjective, ...]:
    """Evaluate a bounded batch of exact LOO objectives source-by-source.

    ``bandwidths`` has shape ``(C, N)``.  The maximum source radius is queried
    once and filtered independently for each of the ``C`` bandwidth vectors.
    Each numerator row is updated in source-major order, matching the
    single-bandwidth reference evaluator.  Checkpoint publication is an
    operational row/elapsed-time trigger (4096 completed sources by default),
    with a final durable cursor and a bounded rolling retention policy; it
    never changes the accumulation order.
    """

    bandwidth_rule = normalize_kernel_bandwidth_rule(kernel_bandwidth_rule)
    if bandwidth_rule != KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE:
        raise ValueError(
            "symmetric_max leave-one-out evaluation is not integrated in this release; "
            "use normalized_kernel_matrix as the bounded CPU reference oracle"
        )

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
    if (
        isinstance(index, (IndexedCPUNeighbourIndex, TiledCPUNeighbourIndex))
        and index.max_radius_query_bytes > radius_limit
    ):
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
        kernel_support_execution=kernel_support_execution,
        gpu_device=gpu_device,
        gpu_devices=gpu_devices,
        gpu_tile_size=gpu_tile_size,
        max_gpu_support_bytes=max_gpu_support_bytes,
        knn_support=knn_support,
        knn_support_row_ids=knn_support_row_ids,
        max_calibration_work_bytes=work_limit,
    )
    loo_started = time.perf_counter()
    checkpoint_seconds = 0.0
    checkpoint_bytes = 0
    checkpoint_writes = 0
    peak_scratch_bytes = int(service.reserved_scratch_bytes)
    peak_lease_bytes = int(service.reserved_memory_bytes)
    normalization_seconds = 0.0
    numerator = np.zeros((batch_count, row_count), dtype=np.float64)
    invalid_reasons: list[str | None] = [None] * batch_count
    support_totals = np.zeros(batch_count, dtype=np.int64)
    support_samples = np.zeros(batch_count, dtype=np.int64)
    support_maxima = np.zeros(batch_count, dtype=np.int64)
    conservative_candidate_totals = np.zeros(batch_count, dtype=np.int64)
    conservative_candidate_maxima = np.zeros(batch_count, dtype=np.int64)
    peak_workspace_bytes = batch_arrays_bytes
    checkpoint_store = (
        None
        if checkpoint_path is None
        else VersionedCheckpointStore(
            checkpoint_path,
            retention_generations=2,
            resource_budget=service,
            retention_name=checkpoint_name,
        )
    )
    if (
        isinstance(checkpoint_every_sources, bool)
        or not isinstance(checkpoint_every_sources, (int, np.integer))
        or int(checkpoint_every_sources) < 1
    ):
        raise ValueError("checkpoint_every_sources must be a positive integer")
    if checkpoint_every_seconds is not None:
        if isinstance(checkpoint_every_seconds, bool):
            raise ValueError("checkpoint_every_seconds must be finite and positive")
        checkpoint_seconds_limit = float(checkpoint_every_seconds)
        if not math.isfinite(checkpoint_seconds_limit) or checkpoint_seconds_limit <= 0.0:
            raise ValueError("checkpoint_every_seconds must be finite and positive")
    else:
        checkpoint_seconds_limit = None
    checkpoint_identity: dict[str, Any] | None = None
    source_start = 0
    if checkpoint_store is not None:
        normalized_support_execution = str(kernel_support_execution).strip().lower()
        if normalized_support_execution == "exact_cpu":
            normalized_support_execution = KERNEL_SUPPORT_EXECUTION_CPU
        checkpoint_identity = {
            "phase": "leave-one-out",
            "schema": "loo-checkpoint-v1",
            "descriptors": _array_digest(values),
            "probabilities": _array_digest(masses),
            "bandwidths": _array_digest(scales),
            "backend": backend,
            "backend_version": getattr(index, "backend_version", NEIGHBOUR_BACKEND_VERSION),
            "metric": getattr(index, "metric", NEIGHBOUR_METRIC),
            "backend_fingerprint": getattr(index, "fingerprint", ""),
            "kernel_support_execution": kernel_support_execution,
            "source_order": "ascending-original-pool-row-index",
            "k_c_batch_shape": [batch_count, row_count],
            "numerical_tolerance": SPARSE_NUMERICAL_TOLERANCE,
        }
        if normalized_support_execution == KERNEL_SUPPORT_EXECUTION_GPU:
            checkpoint_identity["kernel_support_execution_version"] = (
                KERNEL_SUPPORT_EXECUTION_GPU_VERSION
            )
        if not resume:
            checkpoint_store.clear(checkpoint_name)
        if resume and checkpoint_store.exists(checkpoint_name):
            _identity, checkpoint_state, checkpoint_arrays = checkpoint_store.load(
                checkpoint_name,
                expected_identity=checkpoint_identity,
            )
            source_start = int(checkpoint_state.get("cursor", -1))
            if source_start < 0 or source_start > row_count:
                raise CheckpointValidationError("LOO checkpoint cursor is outside the source order")
            saved_numerator = checkpoint_arrays.get("numerator")
            if saved_numerator is None or saved_numerator.shape != numerator.shape:
                raise CheckpointValidationError("LOO checkpoint numerator shape is incompatible")
            numerator[...] = saved_numerator
            saved_invalid = checkpoint_state.get("invalid_reasons", invalid_reasons)
            saved_totals = checkpoint_state.get("support_totals", support_totals.tolist())
            saved_samples = checkpoint_state.get("support_samples", support_samples.tolist())
            saved_maxima = checkpoint_state.get("support_maxima", support_maxima.tolist())
            saved_conservative_totals = checkpoint_state.get(
                "conservative_candidate_totals", conservative_candidate_totals.tolist()
            )
            saved_conservative_maxima = checkpoint_state.get(
                "conservative_candidate_maxima", conservative_candidate_maxima.tolist()
            )
            if not (
                isinstance(saved_invalid, list)
                and len(saved_invalid) == batch_count
                and len(saved_totals) == batch_count
                and len(saved_samples) == batch_count
                and len(saved_maxima) == batch_count
                and isinstance(saved_conservative_totals, list)
                and len(saved_conservative_totals) == batch_count
                and isinstance(saved_conservative_maxima, list)
                and len(saved_conservative_maxima) == batch_count
            ):
                raise CheckpointValidationError("LOO checkpoint cumulative state is incompatible")
            invalid_reasons = [
                value if value is None or isinstance(value, str) else str(value)
                for value in saved_invalid
            ]
            support_totals[...] = np.asarray(saved_totals, dtype=np.int64)
            support_samples[...] = np.asarray(saved_samples, dtype=np.int64)
            support_maxima[...] = np.asarray(saved_maxima, dtype=np.int64)
            conservative_candidate_totals[...] = np.asarray(
                saved_conservative_totals, dtype=np.int64
            )
            conservative_candidate_maxima[...] = np.asarray(
                saved_conservative_maxima, dtype=np.int64
            )
            logger.info(
                "Resuming exact batched leave-one-out from source %d/%d",
                source_start,
                row_count,
            )
    source_phase_started = time.perf_counter()
    query_block = min(block, provider.support_batch_size)
    if diagnostics is not None:
        diagnostics["loo_active_support_metrics"] = {}
    last_completed_source = source_start
    last_checkpoint_cursor = source_start if source_start > 0 else -1
    last_checkpoint_started = time.perf_counter()

    def commit_checkpoint(cursor: int) -> None:
        """Publish one complete rolling snapshot and account for its I/O."""

        nonlocal checkpoint_seconds, checkpoint_bytes, checkpoint_writes
        nonlocal last_checkpoint_cursor, last_checkpoint_started
        if checkpoint_store is None or checkpoint_identity is None:
            return
        checkpoint_started = time.perf_counter()
        before = checkpoint_store.metrics()
        checkpoint_store.write(
            checkpoint_name,
            identity=checkpoint_identity,
            state={
                "cursor": int(cursor),
                "invalid_reasons": invalid_reasons,
                "support_totals": support_totals.tolist(),
                "support_samples": support_samples.tolist(),
                "support_maxima": support_maxima.tolist(),
                "conservative_candidate_totals": conservative_candidate_totals.tolist(),
                "conservative_candidate_maxima": conservative_candidate_maxima.tolist(),
            },
            arrays={"numerator": numerator},
        )
        after = checkpoint_store.metrics()
        checkpoint_seconds += max(0.0, time.perf_counter() - checkpoint_started)
        checkpoint_bytes += max(
            0,
            int(after.get("array_bytes_written", 0))
            - int(before.get("array_bytes_written", 0)),
        )
        checkpoint_writes += 1
        last_checkpoint_cursor = int(cursor)
        last_checkpoint_started = time.perf_counter()

    for batch_start in range(source_start, row_count, query_block):
        batch_stop = min(batch_start + query_block, row_count)
        source_indices = np.arange(batch_start, batch_stop, dtype=np.int64)
        source_scales = scales[:, batch_start:batch_stop]
        support_offsets, support_targets, support_distances = provider.support_many(
            source_indices,
            source_scales,
        )
        peak_scratch_bytes = max(peak_scratch_bytes, int(service.reserved_scratch_bytes))
        peak_lease_bytes = max(peak_lease_bytes, int(service.reserved_memory_bytes))
        query_bytes = int(getattr(provider.index, "query_workspace_peak_bytes", 0))
        requested_bytes = batch_arrays_bytes + query_bytes + 32 * int(support_targets.size)
        peak_workspace_bytes = max(peak_workspace_bytes, requested_bytes)
        if requested_bytes > work_limit:
            details = [
                f"N={row_count}",
                f"d'={values.shape[1]}",
                f"source_row={batch_start}",
                f"radius={float(np.max(source_scales))}",
                f"requested_bytes={requested_bytes}",
                f"max_calibration_work_bytes={work_limit}",
            ]
            if capacity_context:
                details.append(capacity_context)
            raise RadiusQueryCapacityError(
                "batched calibration workspace exceeds memory capacity: " + ", ".join(details)
            )
        for local_index, source_index_value in enumerate(source_indices):
            source_index = int(source_index_value)
            source_scales = scales[:, source_index]
            begin = int(support_offsets[local_index])
            end = int(support_offsets[local_index + 1])
            target_indices = support_targets[begin:end]
            distances = support_distances[begin:end]
            conservative_count = int(target_indices.shape[0])
            conservative_candidate_totals += conservative_count
            conservative_candidate_maxima = np.maximum(
                conservative_candidate_maxima,
                conservative_count,
            )
            normalization_started = time.perf_counter()
            for batch_index, scale in enumerate(source_scales):
                if invalid_reasons[batch_index] is not None:
                    continue
                try:
                    included = distances < float(scale)
                    selected_targets = target_indices[included]
                    selected_distances = distances[included]
                    support_size = int(selected_targets.shape[0])
                    support_totals[batch_index] += support_size
                    support_samples[batch_index] += 1
                    support_maxima[batch_index] = max(
                        int(support_maxima[batch_index]),
                        support_size,
                    )
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
            normalization_seconds += max(0.0, time.perf_counter() - normalization_started)
            last_completed_source = source_index + 1
            if diagnostics is not None and (
                last_completed_source == source_start + 1
                or last_completed_source == row_count
                or last_completed_source % max(64, query_block) == 0
            ):
                # This is the active LOO provider, not the neighbour index
                # used to construct the frozen bandwidths.
                diagnostics["loo_active_support_metrics"] = provider.operational_metrics()
            if progress_callback is not None:
                try:
                    progress_callback(source_index + 1, row_count)
                except (KeyboardInterrupt, SystemExit):
                    if last_completed_source > last_checkpoint_cursor:
                        commit_checkpoint(last_completed_source)
                    raise
            elapsed_since_checkpoint = time.perf_counter() - last_checkpoint_started
            if (
                checkpoint_store is not None
                and checkpoint_identity is not None
                and last_completed_source > last_checkpoint_cursor
                and (
                    last_completed_source % int(checkpoint_every_sources) == 0
                    or (
                        checkpoint_seconds_limit is not None
                        and elapsed_since_checkpoint >= checkpoint_seconds_limit
                    )
                )
            ):
                commit_checkpoint(last_completed_source)
                peak_scratch_bytes = max(peak_scratch_bytes, int(service.reserved_scratch_bytes))
                peak_lease_bytes = max(peak_lease_bytes, int(service.reserved_memory_bytes))

    # The final source cursor is durable before objective reduction.  This is
    # useful for preemption during reduction and is cleared only after the
    # complete calibration pair has returned successfully.
    if (
        checkpoint_store is not None
        and checkpoint_identity is not None
        and last_completed_source > last_checkpoint_cursor
    ):
        commit_checkpoint(last_completed_source)

    source_phase_seconds = max(0.0, time.perf_counter() - source_phase_started)

    reduction_started = time.perf_counter()
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
    reduction_seconds = max(0.0, time.perf_counter() - reduction_started)
    if checkpoint_store is not None and not preserve_checkpoint:
        checkpoint_store.clear(checkpoint_name)
        checkpoint_store.clean_stale(checkpoint_names={checkpoint_name})
    provider_metrics = provider.operational_metrics()
    provider.close()
    total_seconds = max(0.0, time.perf_counter() - loo_started)
    operational_metrics = {
        "wall_seconds": total_seconds,
        "source_phase_seconds": source_phase_seconds,
        "objective_reduction_seconds": reduction_seconds,
        "normalization_seconds": normalization_seconds,
        "source_throughput_per_second": (
            row_count / source_phase_seconds if source_phase_seconds > 0.0 else 0.0
        ),
        "batch_count": batch_count,
        "source_count": row_count,
        "support_size_per_candidate": support_totals.tolist(),
        "support_max_per_candidate": support_maxima.tolist(),
        "conservative_candidate_count_per_candidate": conservative_candidate_totals.tolist(),
        "conservative_candidate_max_per_candidate": conservative_candidate_maxima.tolist(),
        "checkpoint_writes": checkpoint_writes,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_seconds": checkpoint_seconds,
        "peak_scratch_bytes": peak_scratch_bytes,
        "resource_lease_peak_bytes": peak_lease_bytes,
        **provider_metrics,
    }
    if checkpoint_store is not None:
        checkpoint_diagnostics = checkpoint_store.metrics()
        for metric_name, metric_value in checkpoint_diagnostics.items():
            key = (
                metric_name
                if metric_name.startswith("checkpoint_")
                else f"checkpoint_{metric_name}"
            )
            operational_metrics[key] = metric_value
    logger.info(
        "Batched leave-one-out calibration completed: batch_count=%d, "
        "peak_calibration_workspace_bytes=%d, support_count_min=%s, "
        "support_count_mean=%s, support_count_max=%d, radius_queries=%d, "
        "distance_evaluations=%d, wall_seconds=%.6f, source_throughput=%.3f, "
        "support_density=%.6g, support_query_seconds=%.6f, "
        "normalization_seconds=%.6f, gpu_strict_radius_screening_seconds=%.6f, "
        "gpu_h2d_seconds=%.6f, gpu_d2h_seconds=%.6f, "
        "gpu_cpu_verification_seconds=%.6f, checkpoint_bytes=%d, "
        "checkpoint_seconds=%.6f, checkpoint_wall_seconds=%.6f, "
        "checkpoint_fsync_seconds=%.6f, checkpoint_digest_seconds=%.6f, "
        "checkpoint_cleanup_seconds=%.6f, checkpoint_stale_generations=%d, "
        "checkpoint_stale_generation_bytes=%d, process_peak_rss_bytes=%s",
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
        total_seconds,
        operational_metrics["source_throughput_per_second"],
        operational_metrics["support_density_mean"],
        operational_metrics["support_query_seconds"],
        operational_metrics["normalization_seconds"],
        float(provider_metrics.get("gpu_strict_radius_screening_seconds", 0.0) or 0.0),
        float(provider_metrics.get("gpu_h2d_seconds", 0.0) or 0.0),
        float(provider_metrics.get("gpu_d2h_seconds", 0.0) or 0.0),
        float(provider_metrics.get("gpu_cpu_verification_seconds", 0.0) or 0.0),
        checkpoint_bytes,
        checkpoint_seconds,
        float(operational_metrics.get("checkpoint_wall_seconds", 0.0) or 0.0),
        float(operational_metrics.get("checkpoint_fsync_seconds", 0.0) or 0.0),
        float(operational_metrics.get("checkpoint_digest_seconds", 0.0) or 0.0),
        float(operational_metrics.get("checkpoint_cleanup_seconds", 0.0) or 0.0),
        int(operational_metrics.get("checkpoint_stale_generation_count", 0) or 0),
        int(operational_metrics.get("checkpoint_stale_generation_bytes", 0) or 0),
        operational_metrics.get("process_peak_rss_bytes"),
    )
    logger.info(
        "Batched leave-one-out support work by candidate: samples=%s, mean=%s, max=%s",
        support_samples.tolist(),
        [
            float(total) / int(samples) if int(samples) else None
            for total, samples in zip(support_totals, support_samples, strict=True)
        ],
        support_maxima.tolist(),
    )
    if diagnostics is not None:
        diagnostics["loo_wall_seconds"] = diagnostics.get("loo_wall_seconds", 0.0) + total_seconds
        diagnostics["loo_source_phase_seconds"] = (
            diagnostics.get("loo_source_phase_seconds", 0.0) + source_phase_seconds
        )
        diagnostics["loo_normalization_seconds"] = (
            diagnostics.get("loo_normalization_seconds", 0.0) + normalization_seconds
        )
        diagnostics["loo_objective_reduction_seconds"] = (
            diagnostics.get("loo_objective_reduction_seconds", 0.0) + reduction_seconds
        )
        diagnostics["loo_checkpoint_bytes"] = (
            diagnostics.get("loo_checkpoint_bytes", 0) + checkpoint_bytes
        )
        diagnostics["loo_checkpoint_seconds"] = (
            diagnostics.get("loo_checkpoint_seconds", 0.0) + checkpoint_seconds
        )
        if checkpoint_store is not None:
            for metric_name, metric_value in checkpoint_store.metrics().items():
                key = (
                    metric_name
                    if metric_name.startswith("checkpoint_")
                    else f"checkpoint_{metric_name}"
                )
                diagnostics[f"loo_{key}"] = metric_value
        diagnostics["loo_peak_scratch_bytes"] = max(
            int(diagnostics.get("loo_peak_scratch_bytes", 0)), peak_scratch_bytes
        )
        diagnostics["loo_resource_lease_peak_bytes"] = max(
            int(diagnostics.get("loo_resource_lease_peak_bytes", 0)), peak_lease_bytes
        )
        diagnostics["loo_support_query_seconds"] = diagnostics.get(
            "loo_support_query_seconds", 0.0
        ) + float(provider_metrics["support_query_seconds"])
        diagnostics["loo_distance_evaluations"] = diagnostics.get(
            "loo_distance_evaluations", 0
        ) + int(provider_metrics["distance_evaluations"])
        cumulative_gpu_metrics = {
            "gpu_source_tiles",
            "gpu_target_tiles",
            "gpu_fp64_evaluations",
            "gpu_launch_count",
            "gpu_distance_block_count",
            "gpu_kernel_launch_equivalent_count",
            "gpu_d2h_transfer_count",
            "gpu_ambiguous_boundary_count",
            "gpu_cpu_rescanned_count",
            "gpu_completed_locations",
            "gpu_distance_evaluations",
            "gpu_unique_distance_evaluations",
            "gpu_atomic_distance_evaluations",
            "gpu_cpu_tile_refinements",
            "gpu_h2d_bytes",
            "gpu_d2h_bytes",
            "gpu_full_tile_d2h_bytes",
            "gpu_compact_candidate_d2h_bytes",
            "gpu_full_tile_d2h_transfer_count",
            "gpu_compact_candidate_d2h_transfer_count",
            "gpu_seconds",
            "gpu_screening_seconds",
            "gpu_kernel_wall_seconds",
            "gpu_distance_reduction_seconds",
            "gpu_tile_wall_seconds",
            "gpu_candidate_compaction_seconds",
            "gpu_synchronization_seconds",
            "gpu_active_kernel_seconds_sum",
            "gpu_active_operation_seconds_sum",
            "gpu_operation_seconds_inclusive",
            "gpu_h2d_seconds",
            "gpu_d2h_seconds",
            "gpu_full_tile_d2h_seconds",
            "gpu_compact_candidate_d2h_seconds",
            "gpu_launch_allocation_seconds",
            "gpu_memory_pool_peak_bytes",
            "gpu_memory_pool_used_peak_bytes",
            "gpu_allocation_bytes",
            "gpu_allocation_count",
            "gpu_allocation_seconds",
            "gpu_certification_seconds",
            "gpu_cpu_verification_seconds",
            "gpu_cpu_tile_refinement_seconds",
            "gpu_conservative_candidate_count",
            "gpu_grouped_radius_query_count",
            "gpu_reused_radius_scans",
            "support_query_count",
            "support_entries",
            "support_query_seconds",
        }
        for metric_name, metric_value in provider_metrics.items():
            if metric_name in {
                "support_execution",
                "support_execution_version",
                "support_provider",
                "support_provider_stats_status",
                "gpu_device",
                "gpu_devices",
                "gpu_device_count",
                "gpu_per_device",
                "knn_gpu_per_device",
                "gpu_rescan_reasons",
            }:
                diagnostics[metric_name] = metric_value
            elif isinstance(metric_value, bool):
                diagnostics[metric_name] = metric_value
            elif isinstance(metric_value, (int, float, np.integer, np.floating)):
                if metric_name in cumulative_gpu_metrics:
                    diagnostics[metric_name] = diagnostics.get(metric_name, 0) + metric_value
                elif metric_name.endswith("_peak_bytes"):
                    diagnostics[metric_name] = max(diagnostics.get(metric_name, 0), metric_value)
                else:
                    diagnostics[metric_name] = metric_value
        gpu_pairs = int(diagnostics.get("gpu_fp64_evaluations", 0) or 0)
        if gpu_pairs:
            diagnostics["gpu_conservative_candidate_density"] = (
                int(diagnostics.get("gpu_conservative_candidate_count", 0) or 0)
                / gpu_pairs
            )
        diagnostics["support_size_per_candidate"] = provider_metrics.get(
            "support_size_per_candidate", operational_metrics["support_size_per_candidate"]
        )
        diagnostics["support_max_per_candidate"] = provider_metrics.get(
            "support_max_per_candidate", operational_metrics["support_max_per_candidate"]
        )
        diagnostics["conservative_candidate_count_per_candidate"] = provider_metrics.get(
            "conservative_candidate_count_per_candidate",
            operational_metrics["conservative_candidate_count_per_candidate"],
        )
        diagnostics["conservative_candidate_max_per_candidate"] = provider_metrics.get(
            "conservative_candidate_max_per_candidate",
            operational_metrics["conservative_candidate_max_per_candidate"],
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
    flat = contiguous.reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.size, 131_072):
        digest.update(flat[start : start + 131_072].tobytes())
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": digest.hexdigest(),
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
    if frozen.backend not in {
        NEIGHBOUR_BACKEND_ID,
        INDEXED_NEIGHBOUR_BACKEND_ID,
        TILED_NEIGHBOUR_BACKEND_ID,
        GPU_NEIGHBOUR_BACKEND_ID,
    }:
        raise ValueError(
            "sparse kernel construction requires a supported exact neighbour backend; "
            f"received {frozen.backend!r}"
        )
    expected_version = {
        NEIGHBOUR_BACKEND_ID: NEIGHBOUR_BACKEND_VERSION,
        INDEXED_NEIGHBOUR_BACKEND_ID: INDEXED_NEIGHBOUR_BACKEND_VERSION,
        TILED_NEIGHBOUR_BACKEND_ID: TILED_NEIGHBOUR_BACKEND_VERSION,
        GPU_NEIGHBOUR_BACKEND_ID: GPU_NEIGHBOUR_BACKEND_VERSION,
    }[frozen.backend]
    if frozen.backend_version != expected_version:
        raise ValueError(
            "sparse kernel construction requires the configured backend version; "
            f"received {frozen.backend_version!r}, expected {expected_version!r}"
        )
    if frozen.metric != NEIGHBOUR_METRIC:
        raise ValueError("sparse kernel construction requires the exact Euclidean metric")
    if frozen.kernel_bandwidth_rule != KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE:
        raise ValueError(
            "symmetric_max bandwidths are reference-only in this release; "
            "production sparse kernel construction is not yet integrated"
        )
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
    payload: dict[str, Any] = {
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
    if bandwidths.kernel_bandwidth_rule != KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE:
        payload["kernel_bandwidth_rule"] = bandwidths.kernel_bandwidth_rule
        payload["kernel_bandwidth_rule_version"] = bandwidths.kernel_bandwidth_rule_version
    return sha256_canonical_json(payload)


def _kernel_operator_fingerprint(
    pool: EntropyPool,
    bandwidths: FrozenBandwidths,
) -> str:
    """Fingerprint the exact kernel operator without materialising its support."""

    payload: dict[str, Any] = {
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
    if bandwidths.kernel_bandwidth_rule != KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE:
        payload["kernel_bandwidth_rule"] = bandwidths.kernel_bandwidth_rule
        payload["kernel_bandwidth_rule_version"] = bandwidths.kernel_bandwidth_rule_version
    return sha256_canonical_json(payload)


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
        frozen.backend
        in {INDEXED_NEIGHBOUR_BACKEND_ID, TILED_NEIGHBOUR_BACKEND_ID, GPU_NEIGHBOUR_BACKEND_ID}
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
                raise ValueError(
                    "sparse graph source enumeration produced inconsistent edge counts"
                )
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
                    while source_cursor < n_rows and edge_cursor >= int(
                        source_indptr[source_cursor + 1]
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
        support_provider.close()
        raise

    support_provider.close()
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_out_of_core_candidate_store(
    *,
    directory: str | Path | None,
    scratch_directory: str | Path | None,
    spool: Any,
    spool_digest: Any,
    entry_count: int,
    candidate_indptr: np.ndarray,
    candidate_ids: tuple[str, ...],
    source_counts: np.ndarray,
    pool: EntropyPool,
    operator_fingerprint: str,
    spool_header: bytes,
    implicit_edge_count: int,
    source_support_min: int,
    source_support_mean: float,
    source_support_max: int,
    self_edge_count: int,
    source_mass_max_deviation: float,
    candidate_support_min: int,
    candidate_support_mean: float,
    candidate_support_max: int,
    contribution_spool_bytes: int,
    estimated_peak_memory_bytes: int,
) -> OutOfCoreSparseCandidateContributions:
    """Promote a verified candidate spool into memory-mapped CSR arrays."""

    if directory is None:
        seed = Path(tempfile.mkdtemp(prefix="nepflow-candidate-store-", dir=scratch_directory))
        seed.rmdir()
        root = seed
        root_created = True
    else:
        root = Path(directory)
        if root.exists():
            raise ValueError(f"out-of-core candidate store path already exists: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        root_created = False
    temporary = root.parent / f".{root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False)
    try:
        indptr_path = temporary / "candidate_indptr.npy"
        counts_path = temporary / "candidate_source_counts.npy"
        target_path = temporary / "target_indices.npy"
        values_path = temporary / "values.npy"
        with indptr_path.open("wb") as handle:
            np.save(handle, np.asarray(candidate_indptr, dtype=np.int64), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        with counts_path.open("wb") as handle:
            np.save(handle, np.asarray(source_counts, dtype=np.int64), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        targets = np.lib.format.open_memmap(
            target_path, mode="w+", dtype=np.int64, shape=(entry_count,)
        )
        values = np.lib.format.open_memmap(
            values_path, mode="w+", dtype=np.float64, shape=(entry_count,)
        )
        spool.seek(0)
        if spool.read(len(spool_header)) != spool_header:
            raise ValueError("candidate contribution spool has an invalid version header")
        read_digest = hashlib.sha256()
        read_digest.update(spool_header)
        cursor = 0
        for records in _spool_read_records(spool, read_digest, entry_count):
            count = records.shape[0]
            targets[cursor : cursor + count] = records["target"]
            values[cursor : cursor + count] = records["value"]
            cursor += count
        if spool.read(1) or cursor != entry_count:
            raise ValueError("candidate contribution spool contains an invalid record count")
        if read_digest.digest() != spool_digest.digest():
            raise ValueError("candidate contribution spool checksum validation failed")
        targets.flush()
        values.flush()
        del targets, values
        persisted_indptr = np.load(indptr_path, allow_pickle=False)
        persisted_values = np.load(values_path, mmap_mode="r", allow_pickle=False)
        candidate_pmf_max_deviation = float(
            np.max(np.abs(np.add.reduceat(persisted_values, persisted_indptr[:-1]) - 1.0))
        )
        file_entries: dict[str, dict[str, Any]] = {}
        for name, path in (
            ("candidate_indptr", indptr_path),
            ("candidate_source_counts", counts_path),
            ("target_indices", target_path),
            ("values", values_path),
        ):
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            file_entries[name] = {
                "name": path.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": _file_sha256(path),
            }
        candidate_fingerprint = sha256_canonical_json(
            {
                "schema": SPARSE_STREAMED_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
                "kernel_operator": operator_fingerprint,
                "candidate_ids": list(candidate_ids),
                "candidate_indptr": _array_digest(np.load(indptr_path, allow_pickle=False)),
                "target_indices": _array_digest(
                    np.load(target_path, mmap_mode="r", allow_pickle=False)
                ),
                "values": _array_digest(np.load(values_path, mmap_mode="r", allow_pickle=False)),
                "candidate_source_counts": _array_digest(np.load(counts_path, allow_pickle=False)),
            }
        )
        manifest = {
            "schema": OutOfCoreSparseCandidateContributions.SCHEMA_VERSION,
            "candidate_ids": list(candidate_ids),
            "row_count": int(pool.descriptors.shape[0]),
            "graph_fingerprint": "",
            "fingerprint": candidate_fingerprint,
            "pool_fingerprint": pool.fingerprint,
            "sparse_schema_version": SPARSE_STREAMED_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
            "numerical_tolerance": SPARSE_NUMERICAL_TOLERANCE,
            "kernel_operator_fingerprint": operator_fingerprint,
            "implicit_edge_count": int(implicit_edge_count),
            "source_support_min": int(source_support_min),
            "source_support_mean": float(source_support_mean),
            "source_support_max": int(source_support_max),
            "self_edge_count": int(self_edge_count),
            "source_mass_max_deviation": float(source_mass_max_deviation),
            "candidate_support_min": int(candidate_support_min),
            "candidate_support_mean": float(candidate_support_mean),
            "candidate_support_max": int(candidate_support_max),
            "candidate_pmf_max_deviation": candidate_pmf_max_deviation,
            "contribution_spool_bytes": int(contribution_spool_bytes),
            "estimated_peak_memory_bytes": int(estimated_peak_memory_bytes),
            "files": file_entries,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root)
        return OutOfCoreSparseCandidateContributions(root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if root_created:
            shutil.rmtree(root, ignore_errors=True)
        raise


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
    index: IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None = None,
    storage: str = "auto",
    storage_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_name: str = "candidate-aggregation",
    resume: bool = True,
    kernel_support_execution: str = KERNEL_SUPPORT_EXECUTION_CPU,
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_gpu_support_bytes: int | None = None,
) -> tuple[
    SparseCandidateContributions | OutOfCoreSparseCandidateContributions,
    StreamedKernelExecutionSummary,
]:
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
    storage_mode = str(storage).strip().lower()
    if storage_mode not in {"auto", "memory", "disk"}:
        raise ValueError("storage must be 'auto', 'memory', or 'disk'")
    if storage_path is not None and storage_mode == "memory":
        raise ValueError("storage_path is only valid for disk-backed contributions")

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
    runtime_budget = (
        resource_budget or getattr(index, "resource_budget", None) or build_resource_budget()
    )
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
    prospective_final_bytes = int(8 * (candidate_count + 1 + candidate_count))
    prospective_final_bytes += int(16 * maximum_entry_count)
    if storage_mode == "auto":
        storage_mode = (
            "disk"
            if storage_path is not None
            or (byte_limit is not None and prospective_final_bytes > byte_limit)
            or prospective_final_bytes > max(1, remaining_budget // 2)
            else "memory"
        )
    if storage_mode == "memory" and prospective_final_bytes > remaining_budget:
        raise ResourceCapacityError(
            "in-memory candidate contribution plan exceeds the runtime memory allowance; "
            "select storage='disk' for the exact out-of-core plan: "
            f"N={n_rows}, M={candidate_count}, Q_max={maximum_entry_count}, "
            f"required_bytes={prospective_final_bytes}, available_bytes={remaining_budget}",
            operation="entropy streamed candidate contributions planning",
            requested_bytes=prospective_final_bytes,
            available_bytes=remaining_budget,
        )
    if derived_spool_limit:
        prospective_scratch_bytes = int(len(_EDGE_SPOOL_HEADER) + 16 * int(entry_limit))
        if storage_mode == "disk":
            prospective_scratch_bytes += 8 * (candidate_count + 1 + candidate_count)
        if prospective_scratch_bytes > spool_limit:
            raise ResourceCapacityError(
                "candidate contribution plan exceeds available scratch capacity: "
                f"N={n_rows}, M={candidate_count}, Q_max={int(entry_limit)}, "
                f"required_scratch_bytes={prospective_scratch_bytes}, "
                f"scratch_available_bytes={spool_limit}",
                operation="entropy streamed candidate contributions planning",
                requested_bytes=prospective_scratch_bytes,
                available_bytes=remaining_budget,
                scratch_needed_bytes=prospective_scratch_bytes,
                scratch_available_bytes=spool_limit,
            )
    logger.info(
        "Candidate contribution storage plan: storage=%s, N=%d, M=%d, Q_max=%d, "
        "prospective_final_bytes=%d, memory_limit=%s, scratch_limit=%d",
        storage_mode,
        n_rows,
        candidate_count,
        maximum_entry_count,
        prospective_final_bytes,
        byte_limit if byte_limit is not None else "unbounded",
        spool_limit,
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
    if storage_mode == "disk" and storage_path is not None:
        existing_path = Path(storage_path)
        if existing_path.is_dir():
            try:
                existing = OutOfCoreSparseCandidateContributions(existing_path)
            except ValueError as exc:
                raise ValueError(
                    "existing out-of-core candidate contribution store cannot be reused; "
                    f"validation failed for {existing_path}"
                ) from exc
            if (
                existing.pool_fingerprint != pool.fingerprint
                or existing.kernel_operator_fingerprint != operator_fingerprint
                or tuple(existing.candidate_ids) != candidate_ids
                or existing.row_count != n_rows
            ):
                raise ValueError(
                    "existing out-of-core candidate contribution store belongs to a "
                    "different ordered entropy computation"
                )
            if existing.entry_count > entry_limit:
                raise ValueError(
                    "existing out-of-core candidate contribution store exceeds max_entries: "
                    f"Q={existing.entry_count}, max_entries={entry_limit}"
                )
            metadata = existing.manifest
            required_metadata = (
                "implicit_edge_count",
                "source_support_min",
                "source_support_mean",
                "source_support_max",
                "self_edge_count",
                "source_mass_max_deviation",
                "candidate_support_min",
                "candidate_support_mean",
                "candidate_support_max",
                "candidate_pmf_max_deviation",
                "contribution_spool_bytes",
                "estimated_peak_memory_bytes",
            )
            if any(name not in metadata for name in required_metadata):
                raise ValueError(
                    "existing out-of-core candidate contribution store has no complete "
                    "execution summary and cannot be resumed safely"
                )
            if checkpoint_path is not None:
                CandidateAggregationCheckpoint(Path(checkpoint_path) / checkpoint_name).clear()
            summary = StreamedKernelExecutionSummary(
                atomic_row_count=n_rows,
                implicit_edge_count=int(metadata["implicit_edge_count"]),
                source_support_min=int(metadata["source_support_min"]),
                source_support_mean=float(metadata["source_support_mean"]),
                source_support_max=int(metadata["source_support_max"]),
                self_edge_count=int(metadata["self_edge_count"]),
                source_mass_max_deviation=float(metadata["source_mass_max_deviation"]),
                candidate_entry_count=existing.entry_count,
                candidate_support_min=int(metadata["candidate_support_min"]),
                candidate_support_mean=float(metadata["candidate_support_mean"]),
                candidate_support_max=int(metadata["candidate_support_max"]),
                candidate_pmf_max_deviation=float(metadata["candidate_pmf_max_deviation"]),
                candidate_csr_array_bytes=existing.array_bytes,
                contribution_spool_bytes=int(metadata["contribution_spool_bytes"]),
                estimated_peak_memory_bytes=int(metadata["estimated_peak_memory_bytes"]),
                measured_peak_memory_bytes=_measured_peak_memory_bytes(),
                pool_fingerprint=pool.fingerprint,
                bandwidth_fingerprint=frozen.fingerprint,
                transform_fingerprint=pool.transform_fingerprint,
                backend=frozen.backend,
                metric=frozen.metric,
                backend_version=frozen.backend_version,
                kernel_operator_fingerprint=operator_fingerprint,
                source_query_count=n_rows,
                candidate_storage="disk",
                candidate_storage_path=str(existing.directory),
                candidate_resident_bytes=existing.resident_array_bytes,
            )
            logger.info(
                "Reusing verified out-of-core candidate contributions: path=%s, Q=%d",
                existing.directory,
                existing.entry_count,
            )
            return existing, summary
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
            index=index,
            max_radius_query_bytes=max_radius_query_bytes,
            resource_budget=runtime_budget,
            capacity_context=(f"N={n_rows}, M={candidate_count}, operator={operator_fingerprint}"),
            kernel_support_execution=kernel_support_execution,
            gpu_device=gpu_device,
            gpu_devices=gpu_devices,
            gpu_tile_size=gpu_tile_size,
            max_gpu_support_bytes=max_gpu_support_bytes,
        )
        if (
            frozen.backend
            in {
                INDEXED_NEIGHBOUR_BACKEND_ID,
                TILED_NEIGHBOUR_BACKEND_ID,
                GPU_NEIGHBOUR_BACKEND_ID,
            }
            and frozen.backend_fingerprint
            and support_provider.index is not None
            and frozen.backend_fingerprint != support_provider.index.fingerprint
        ):
            raise ValueError("frozen bandwidths have a mismatched indexed backend fingerprint")
    except BaseException:
        lease.close()
        raise

    def estimated_peak_bytes(entry_count: int, support_count: int) -> int:
        resident_entry_count = entry_count if storage_mode == "memory" else 0
        final_arrays = 8 * (
            candidate_count + 1 + resident_entry_count + resident_entry_count + candidate_count
        )
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
    phase_seconds: dict[str, float] = {}
    phase_counts: dict[str, int] = {}
    phase_bytes: dict[str, int] = {}
    checkpoint_seconds = 0.0
    checkpoint_bytes = 0
    checkpoint_writes = 0
    peak_lease_bytes = int(runtime_budget.reserved_memory_bytes)
    peak_scratch_bytes = max(
        int(runtime_budget.reserved_scratch_bytes),
        len(_EDGE_SPOOL_HEADER),
    )
    spool_bytes = len(_EDGE_SPOOL_HEADER)

    def record_phase(name: str, elapsed: float, *, count: int = 1, bytes_count: int = 0) -> None:
        phase_seconds[name] = phase_seconds.get(name, 0.0) + max(0.0, float(elapsed))
        phase_counts[name] = phase_counts.get(name, 0) + int(count)
        if bytes_count:
            phase_bytes[name] = phase_bytes.get(name, 0) + int(bytes_count)

    def sample_lease() -> None:
        nonlocal peak_lease_bytes, peak_scratch_bytes
        peak_lease_bytes = max(peak_lease_bytes, int(runtime_budget.reserved_memory_bytes))
        peak_scratch_bytes = max(
            peak_scratch_bytes,
            int(runtime_budget.reserved_scratch_bytes),
            int(spool_bytes),
        )

    source_loop_seconds = 0.0
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
    aggregation_checkpoint = (
        None
        if checkpoint_path is None
        else CandidateAggregationCheckpoint(Path(checkpoint_path) / checkpoint_name)
    )
    completed_rows: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    resume_candidate_count = 0
    resumed_sources = 0
    if aggregation_checkpoint is not None:
        if not resume and aggregation_checkpoint.manifest_path.is_file():
            aggregation_checkpoint.clear()
        checkpoint_identity = {
            "phase": "candidate-aggregation",
            "schema": "candidate-aggregation-checkpoint-v2",
            "pool": pool.fingerprint,
            "operator": operator_fingerprint,
            "candidate_ids": list(candidate_ids),
            "source_order": "candidate-id-then-original-source-index",
            "target_order": "ascending-original-pool-row-index",
            "backend": frozen.backend,
            "backend_version": frozen.backend_version,
            "kernel_support_execution": kernel_support_execution,
            "gpu_device": int(gpu_device),
            "gpu_tile_size": int(gpu_tile_size),
            "max_gpu_support_bytes": max_gpu_support_bytes,
            "bandwidth": frozen.fingerprint,
            "entry_limit": int(entry_limit),
            "numerical_tolerance": SPARSE_NUMERICAL_TOLERANCE,
        }
        normalized_support_execution = str(kernel_support_execution).strip().lower()
        if normalized_support_execution == "exact_cpu":
            normalized_support_execution = KERNEL_SUPPORT_EXECUTION_CPU
        if normalized_support_execution == KERNEL_SUPPORT_EXECUTION_GPU:
            checkpoint_identity["kernel_support_execution_version"] = (
                KERNEL_SUPPORT_EXECUTION_GPU_VERSION
            )
        aggregation_checkpoint.initialize(checkpoint_identity, candidate_ids)
        aggregation_checkpoint.clean_stale()
        if resume and aggregation_checkpoint.manifest_path.is_file():
            completed_rows = aggregation_checkpoint.load_rows()
            while resume_candidate_count in completed_rows:
                targets, values_for_candidate, row_manifest = completed_rows[resume_candidate_count]
                if row_manifest.get("candidate_id") != candidate_ids[resume_candidate_count]:
                    raise CheckpointValidationError(
                        "candidate aggregation checkpoint row ID does not match candidate order"
                    )
                expected_source_cursor = int(candidate_offsets[resume_candidate_count + 1])
                if int(row_manifest.get("source_cursor", -1)) != expected_source_cursor:
                    raise CheckpointValidationError(
                        "candidate aggregation checkpoint source cursor does not match the pool"
                    )
                if targets.dtype != np.dtype(np.int64) or values_for_candidate.dtype != np.dtype(
                    np.float64
                ):
                    raise CheckpointValidationError(
                        "candidate aggregation checkpoint row dtype is invalid"
                    )
                if (
                    targets.ndim != 1
                    or values_for_candidate.shape != targets.shape
                    or targets.size == 0
                    or np.any(targets < 0)
                    or np.any(targets >= n_rows)
                    or np.any(np.diff(targets) <= 0)
                    or not np.all(np.isfinite(values_for_candidate))
                    or np.any(values_for_candidate <= 0.0)
                    or not math.isclose(
                        float(np.sum(values_for_candidate, dtype=np.float64)),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=SPARSE_NUMERICAL_TOLERANCE,
                    )
                ):
                    raise CheckpointValidationError(
                        "candidate aggregation checkpoint row is not canonical CSR"
                    )
                support = int(targets.size)
                candidate_support_sizes[resume_candidate_count] = support
                q_completed += support
                if int(row_manifest.get("q_completed", -1)) != q_completed:
                    raise CheckpointValidationError(
                        "candidate aggregation checkpoint Q cursor is not cumulative"
                    )
                candidate_source_indices = source_order[
                    int(candidate_offsets[resume_candidate_count]) : int(
                        candidate_offsets[resume_candidate_count + 1]
                    )
                ]
                saved_source_support = row_manifest.get("_source_support")
                if not isinstance(saved_source_support, np.ndarray) or (
                    saved_source_support.dtype != np.dtype(np.int64)
                    or saved_source_support.shape != candidate_source_indices.shape
                    or np.any(saved_source_support < 1)
                ):
                    raise CheckpointValidationError(
                        "candidate aggregation checkpoint source support is incomplete"
                    )
                support_sizes[candidate_source_indices] = saved_source_support
                self_edges += int(row_manifest.get("self_edge_count", 0))
                source_mass_deviation = max(
                    source_mass_deviation,
                    float(row_manifest.get("source_mass_max_deviation", 0.0)),
                )
                restored_implicit_edges = int(row_manifest.get("implicit_edges", -1))
                if restored_implicit_edges < implicit_edges:
                    raise CheckpointValidationError(
                        "candidate aggregation checkpoint E cursor is not cumulative"
                    )
                implicit_edges = restored_implicit_edges
                resume_candidate_count += 1
            if len(completed_rows) != resume_candidate_count:
                raise CheckpointValidationError(
                    "candidate aggregation checkpoint has a non-contiguous completed prefix"
                )
            q_observed = q_completed
            last_q_observed = q_completed
            resumed_sources = int(np.sum(source_counts[:resume_candidate_count], dtype=np.int64))
            logger.info(
                "Resuming exact candidate aggregation from candidate %d/%d",
                resume_candidate_count,
                candidate_count,
            )

    def resize_workspace(
        requested_bytes: int,
        scratch_bytes: int,
        *,
        source_index: int | None = None,
    ) -> None:
        try:
            lease.resize(requested_bytes, scratch_bytes=scratch_bytes)
            sample_lease()
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
    completed_sources = resumed_sources
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

    source_phase_started = time.perf_counter()
    try:
        scratch_directory = runtime_budget.budget.snapshot.scratch_path or None
        with tempfile.TemporaryFile(mode="w+b", dir=scratch_directory) as spool:
            spool_digest = _spool_write_header(spool)
            for completed_index in range(resume_candidate_count):
                records = np.empty(
                    int(completed_rows[completed_index][0].size), dtype=_EDGE_SPOOL_DTYPE
                )
                records["target"] = completed_rows[completed_index][0]
                records["value"] = completed_rows[completed_index][1]
                spool_started = time.perf_counter()
                _spool_write_records(spool, spool_digest, records)
                record_phase(
                    "spool_serialization",
                    time.perf_counter() - spool_started,
                    bytes_count=int(records.nbytes),
                )
                spool_bytes += int(records.nbytes)
            for candidate_index in range(resume_candidate_count, candidate_count):
                accumulator.fill(0.0)
                touched.fill(False)
                q_active = 0
                q_observed = q_completed
                validate_q_invariants(candidate_index, -1)
                start_source = int(candidate_offsets[candidate_index])
                stop_source = int(candidate_offsets[candidate_index + 1])
                candidate_source_indices = source_order[start_source:stop_source]
                last_source_index = int(candidate_source_indices[-1])
                candidate_self_edges = 0
                candidate_source_mass_deviation = 0.0
                for column in iter_normalized_kernel_columns(
                    values,
                    frozen.bandwidths,
                    chunk_size=block,
                    backend=frozen.backend,
                    support_provider=support_provider,
                    max_radius_query_bytes=max_radius_query_bytes,
                    source_indices=candidate_source_indices,
                    kernel_support_execution=kernel_support_execution,
                    gpu_device=gpu_device,
                    gpu_devices=gpu_devices,
                    gpu_tile_size=gpu_tile_size,
                    max_gpu_support_bytes=max_gpu_support_bytes,
                ):
                    normalization_started = time.perf_counter()
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
                    source_self_edges = int(np.count_nonzero(target_indices == source_index))
                    self_edges += source_self_edges
                    candidate_self_edges += source_self_edges
                    source_deviation = abs(column_mass - 1.0)
                    source_mass_deviation = max(source_mass_deviation, source_deviation)
                    candidate_source_mass_deviation = max(
                        candidate_source_mass_deviation, source_deviation
                    )
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
                    record_phase(
                        "source_normalization_and_accumulation",
                        time.perf_counter() - normalization_started,
                        count=1,
                    )

                targets = np.flatnonzero(touched).astype(np.int64, copy=False)
                reduction_started = time.perf_counter()
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
                if not math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=SPARSE_NUMERICAL_TOLERANCE):
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
                record_phase(
                    "candidate_reduction",
                    time.perf_counter() - reduction_started,
                    count=1,
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
                spool_started = time.perf_counter()
                _spool_write_records(spool, spool_digest, records)
                record_phase(
                    "spool_serialization",
                    time.perf_counter() - spool_started,
                    bytes_count=int(records.nbytes),
                )
                spool_bytes = prospective_spool_bytes
                sample_lease()
                candidate_support_sizes[candidate_index] = support
                q_completed = completed_entries
                q_active = 0
                q_observed = q_completed
                validate_q_invariants(candidate_index, last_source_index)
                if aggregation_checkpoint is not None:
                    checkpoint_started = time.perf_counter()
                    aggregation_checkpoint.write_row(
                        candidate_index,
                        candidate_ids[candidate_index],
                        targets,
                        values_for_candidate,
                        source_cursor=stop_source,
                        q_completed=q_completed,
                        implicit_edges=implicit_edges,
                        source_support=support_sizes[candidate_source_indices].copy(),
                        self_edge_count=candidate_self_edges,
                        source_mass_max_deviation=candidate_source_mass_deviation,
                    )
                    checkpoint_elapsed = max(0.0, time.perf_counter() - checkpoint_started)
                    checkpoint_seconds += checkpoint_elapsed
                    checkpoint_bytes += int(
                        targets.nbytes
                        + values_for_candidate.nbytes
                        + support_sizes[candidate_source_indices].nbytes
                    )
                    checkpoint_writes += 1
                    record_phase(
                        "checkpoint_serialization",
                        checkpoint_elapsed,
                        bytes_count=int(
                            targets.nbytes
                            + values_for_candidate.nbytes
                            + support_sizes[candidate_source_indices].nbytes
                        ),
                    )
                    sample_lease()
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
            final_scratch_bytes = spool_bytes
            if storage_mode == "disk":
                final_scratch_bytes += int(
                    8 * (candidate_count + 1 + candidate_count) + 16 * entry_count
                )
            resize_workspace(final_bytes, final_scratch_bytes)
            if byte_limit is not None and final_bytes > byte_limit and storage_mode == "memory":
                raise ValueError(
                    "streamed candidate contributions exceed max_contribution_bytes "
                    "before final CSR: "
                    f"N={n_rows}, M={candidate_count}, Q={entry_count}, E={implicit_edges}, "
                    f"requested_bytes={final_bytes}, limit={byte_limit}"
                )
            finalization_started = time.perf_counter()
            if storage_mode == "memory":
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
                    raise ValueError(
                        "candidate contribution spool contains an invalid record count"
                    )
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
            else:
                contributions = _publish_out_of_core_candidate_store(
                    directory=storage_path,
                    scratch_directory=scratch_directory,
                    spool=spool,
                    spool_digest=spool_digest,
                    entry_count=entry_count,
                    candidate_indptr=candidate_indptr,
                    candidate_ids=candidate_ids,
                    source_counts=source_counts,
                    pool=pool,
                    operator_fingerprint=operator_fingerprint,
                    spool_header=_EDGE_SPOOL_HEADER,
                    implicit_edge_count=implicit_edges,
                    source_support_min=int(np.min(support_sizes)),
                    source_support_mean=float(np.mean(support_sizes)),
                    source_support_max=int(np.max(support_sizes)),
                    self_edge_count=self_edges,
                    source_mass_max_deviation=source_mass_deviation,
                    candidate_support_min=int(np.min(candidate_support_sizes)),
                    candidate_support_mean=float(np.mean(candidate_support_sizes)),
                    candidate_support_max=int(np.max(candidate_support_sizes)),
                    contribution_spool_bytes=spool_bytes,
                    estimated_peak_memory_bytes=peak_estimated_bytes,
                )
            record_phase(
                "candidate_csr_finalization",
                time.perf_counter() - finalization_started,
                count=1,
                bytes_count=int(contributions.array_bytes),
            )
        if aggregation_checkpoint is not None:
            aggregation_checkpoint.clear()
        source_loop_seconds = max(0.0, time.perf_counter() - source_phase_started)
    except BaseException as exc:
        logger.error("Streamed candidate contributions failed: %s", exc)
        support_provider.close()
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

    sample_lease()
    provider_metrics = support_provider.operational_metrics()
    support_provider.close()
    operational_metrics = {
        "wall_seconds": max(0.0, time.perf_counter() - started),
        "source_loop_seconds": source_loop_seconds,
        "source_throughput_per_second": (
            completed_sources / source_loop_seconds if source_loop_seconds > 0.0 else 0.0
        ),
        "candidate_count": candidate_count,
        "source_count": n_rows,
        "entry_count": int(q_completed),
        "source_support_density_mean": float(np.mean(support_sizes)) / n_rows,
        "candidate_support_density_mean": float(np.mean(candidate_support_sizes)) / n_rows,
        "peak_estimated_workspace_bytes": int(peak_estimated_bytes),
        "peak_scratch_bytes": int(max(peak_scratch_bytes, spool_bytes)),
        "resource_lease_peak_bytes": int(peak_lease_bytes),
        "checkpoint_writes": checkpoint_writes,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_seconds": checkpoint_seconds,
        "process_peak_rss_bytes": _measured_peak_memory_bytes(),
        "phase_seconds": dict(phase_seconds),
        "phase_counts": dict(phase_counts),
        "phase_bytes": dict(phase_bytes),
        **provider_metrics,
    }
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
                    np.add.reduceat(contributions.values, contributions.candidate_indptr[:-1]) - 1.0
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
        candidate_storage=storage_mode,
        candidate_storage_path=(
            str(contributions.directory)
            if isinstance(contributions, OutOfCoreSparseCandidateContributions)
            else None
        ),
        candidate_resident_bytes=(
            contributions.resident_array_bytes
            if isinstance(contributions, OutOfCoreSparseCandidateContributions)
            else contributions.array_bytes
        ),
        support_execution=support_provider.support_execution,
        support_execution_version=support_provider.support_execution_version,
        gpu_device=(
            int(gpu_device)
            if support_provider.support_execution == KERNEL_SUPPORT_EXECUTION_GPU
            else None
        ),
        gpu_devices=(
            tuple(int(device) for device in support_provider.gpu_devices)
            if support_provider.support_execution == KERNEL_SUPPORT_EXECUTION_GPU
            else None
        ),
        operational_metrics=operational_metrics,
    )
    logger.info(
        "Streamed candidate contributions completed: E=%d (implicit), Q=%d, "
        "atomic_graph_materialized=false, final_csr_bytes=%d, spool_bytes=%d, "
        "peak_estimated_bytes=%d, elapsed=%.3fs, source_throughput=%.3f, "
        "source_support_density=%.6g, candidate_support_density=%.6g, "
        "support_query_seconds=%.6f, distance_evaluations=%s, "
        "checkpoint_bytes=%d, checkpoint_seconds=%.6f, "
        "peak_scratch_bytes=%d, process_peak_rss_bytes=%s",
        summary.E,
        summary.Q,
        summary.candidate_csr_array_bytes,
        summary.contribution_spool_bytes,
        summary.estimated_peak_memory_bytes,
        operational_metrics["source_throughput_per_second"],
        operational_metrics["source_support_density_mean"],
        operational_metrics["candidate_support_density_mean"],
        operational_metrics["support_query_seconds"],
        operational_metrics.get("distance_evaluations"),
        checkpoint_bytes,
        checkpoint_seconds,
        operational_metrics["peak_scratch_bytes"],
        operational_metrics["process_peak_rss_bytes"],
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
            raise ValueError(
                "max_spool_bytes is too small for the candidate contribution spool header"
            )

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
    "KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE",
    "KERNEL_BANDWIDTH_RULE_SYMMETRIC_MAX",
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
    "effective_pair_bandwidths",
    "evaluate_raw_kernel",
    "evaluate_raw_kernel_pairs",
    "iter_normalized_kernel_columns",
    "normalised_kernel_matrix",
    "normalized_kernel_matrix",
    "source_normalisers",
    "wendland_c2",
    "wendland",
    "wendland_kernel",
]
