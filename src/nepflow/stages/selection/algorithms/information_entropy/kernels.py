"""Versioned Wendland kernels and bounded finite-pool calibration evaluation."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json

from .models import (
    KERNEL_FAMILY,
    KERNEL_VERSION,
    SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
    SPARSE_KERNEL_GRAPH_SCHEMA_VERSION,
    SPARSE_NUMERICAL_TOLERANCE,
    BandwidthCalibrationResult,
    EntropyPool,
    FrozenBandwidths,
    KernelMetadata,
    SparseAtomicKernelGraph,
    SparseCandidateContributions,
)
from .neighbours import _validate_chunk_size, _validate_descriptors

SourceProgressCallback = Callable[[int, int], None]

logger = logging.getLogger(__name__)


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


def source_normalisers(
    descriptors: np.ndarray,
    bandwidths: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Return ``Z_b`` using every target row and no target probability weights."""

    values = _validate_descriptors(descriptors)
    scales = _validate_bandwidths(bandwidths, values.shape[0])
    block = _validate_chunk_size(chunk_size)
    normalisers = np.empty(values.shape[0], dtype=np.float64)
    for source_index, scale in enumerate(scales):
        raw_values = np.empty(values.shape[0], dtype=np.float64)
        for start in range(0, values.shape[0], block):
            stop = min(start + block, values.shape[0])
            distances = _distance_block(values, source_index, start, stop)
            raw = np.asarray(wendland_kernel(distances / scale), dtype=np.float64)
            raw_values[start:stop] = raw
        total = float(np.sum(raw_values, dtype=np.float64))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(f"source kernel normaliser is invalid for source {source_index}")
        normalisers[source_index] = total
    normalisers.setflags(write=False)
    return normalisers


def iter_normalized_kernel_columns(
    descriptors: np.ndarray,
    bandwidths: np.ndarray,
    *,
    chunk_size: int = 1024,
    progress_callback: SourceProgressCallback | None = None,
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
    normalisers = source_normalisers(values, scales, chunk_size=block)
    for source_index, scale in enumerate(scales):
        normaliser = normalisers[source_index]
        column_total = 0.0
        for start in range(0, values.shape[0], block):
            stop = min(start + block, values.shape[0])
            distances = _distance_block(values, source_index, start, stop)
            raw = np.asarray(wendland_kernel(distances / scale), dtype=np.float64)
            positive = raw > 0.0
            if not np.any(positive):
                continue
            target_indices = np.arange(start, stop, dtype=np.int64)[positive]
            normalized = raw[positive] / normaliser
            column_total += float(np.sum(normalized, dtype=np.float64))
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
    if frozen.backend != "exact_cpu" or frozen.metric != "euclidean":
        raise ValueError("sparse kernel construction requires the exact Euclidean backend")
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
    progress_callback: SourceProgressCallback | None = None,
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
    n_rows = values.shape[0]
    candidate_count = len(pool.candidate_ids)
    started = time.perf_counter()
    logger.info(
        "Sparse atomic kernel graph started: N=%d, M=%d, schema=%s, kernel=%s/%s, "
        "max_edges=%d, max_graph_bytes=%s, chunk_size=%d",
        n_rows,
        candidate_count,
        SPARSE_KERNEL_GRAPH_SCHEMA_VERSION,
        KERNEL_FAMILY,
        KERNEL_VERSION,
        edge_limit,
        byte_limit if byte_limit is not None else "unbounded",
        block,
    )
    edge_counts = np.zeros(n_rows, dtype=np.int64)
    next_percent = [1]

    def report_source_progress(completed: int, total: int) -> None:
        _report_graph_progress(completed, total, edge_counts, started, next_percent)
        if progress_callback is not None:
            progress_callback(completed, total)

    try:
        for column in iter_normalized_kernel_columns(
            values,
            frozen.bandwidths,
            chunk_size=block,
            progress_callback=report_source_progress,
        ):
            edge_counts[column.source_index] += np.int64(column.target_indices.shape[0])
        edge_count = int(np.sum(edge_counts, dtype=np.int64))
        if edge_count > edge_limit:
            raise ValueError(
                "sparse atomic kernel graph exceeds max_edges: "
                f"N={n_rows}, required_edges={edge_count}, max_edges={edge_limit}"
            )
        required_bytes = 8 * (n_rows + 1 + edge_count + edge_count + n_rows)
        if byte_limit is not None and required_bytes > byte_limit:
            raise ValueError(
                "sparse atomic kernel graph exceeds max_graph_bytes: "
                f"N={n_rows}, required_bytes={required_bytes}, max_graph_bytes={byte_limit}"
            )

        source_indptr = np.empty(n_rows + 1, dtype=np.int64)
        source_indptr[0] = 0
        np.cumsum(edge_counts, dtype=np.int64, out=source_indptr[1:])
        target_indices = np.empty(edge_count, dtype=np.int64)
        edge_values = np.empty(edge_count, dtype=np.float64)
        positions = source_indptr[:-1].copy()
        for column in iter_normalized_kernel_columns(
            values,
            frozen.bandwidths,
            chunk_size=block,
        ):
            start = int(positions[column.source_index])
            stop = start + column.target_indices.shape[0]
            target_indices[start:stop] = column.target_indices
            edge_values[start:stop] = column.values
            positions[column.source_index] = stop
        if not np.array_equal(positions, source_indptr[:-1] + edge_counts):
            raise ValueError("sparse graph edge preflight disagreed with graph construction")
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
        "max support=%d, density=%.6g, array_bytes=%d, elapsed=%.3fs, budgets=satisfied",
        graph.edge_count,
        float(np.mean(support_sizes)),
        int(np.max(support_sizes)),
        graph.density,
        graph.array_bytes,
        elapsed,
    )
    return graph


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
    logger.info(
        "Sparse candidate contribution aggregation started: M=%d, schema=%s, "
        "max_entries=%d, max_graph_bytes=%s",
        candidate_count,
        SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
        entry_limit,
        byte_limit if byte_limit is not None else "unbounded",
    )
    try:
        last_source = np.full(candidate_count, -1, dtype=np.int64)
        for source_index, candidate_index in enumerate(graph.row_candidate_indices):
            last_source[int(candidate_index)] = source_index
        if np.any(last_source < 0):
            missing = graph.candidate_ids[int(np.flatnonzero(last_source < 0)[0])]
            raise ValueError(f"candidate {missing!r} owns no source rows")
        source_counts = np.bincount(
            graph.row_candidate_indices,
            minlength=candidate_count,
        ).astype(np.int64)
        accumulators: list[dict[int, float]] = [dict() for _ in range(candidate_count)]
        entry_count = 0
        completed_candidates = 0
        next_percent = [1]
        for source_index, candidate_index_value in enumerate(graph.row_candidate_indices):
            candidate_index = int(candidate_index_value)
            start = int(graph.source_indptr[source_index])
            stop = int(graph.source_indptr[source_index + 1])
            accumulator = accumulators[candidate_index]
            for edge_index in range(start, stop):
                target = int(graph.target_indices[edge_index])
                weight = float(graph.values[edge_index])
                if target in accumulator:
                    accumulator[target] += weight
                else:
                    entry_count += 1
                    if entry_count > entry_limit:
                        raise ValueError(
                            "sparse candidate contributions exceed max_entries: "
                            f"M={candidate_count}, required_entries>{entry_limit}, "
                            f"max_entries={entry_limit}"
                        )
                    if byte_limit is not None:
                        estimated_bytes = 8 * (
                            candidate_count + 1 + entry_count + entry_count + candidate_count
                        )
                        if estimated_bytes > byte_limit:
                            raise ValueError(
                                "sparse candidate contributions exceed max_graph_bytes: "
                                f"M={candidate_count}, required_bytes>={estimated_bytes}, "
                                f"max_graph_bytes={byte_limit}"
                            )
                    accumulator[target] = weight
            if source_index == int(last_source[candidate_index]):
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
        support_counts = np.asarray(
            [len(accumulator) for accumulator in accumulators],
            dtype=np.int64,
        )
        np.cumsum(support_counts, dtype=np.int64, out=candidate_indptr[1:])
        entry_count = int(candidate_indptr[-1])
        if entry_count > entry_limit:
            raise ValueError(
                "sparse candidate contributions exceed max_entries: "
                f"M={candidate_count}, required_entries={entry_count}, max_entries={entry_limit}"
            )
        required_bytes = 8 * (candidate_count + 1 + entry_count + entry_count + candidate_count)
        if byte_limit is not None and required_bytes > byte_limit:
            raise ValueError(
                "sparse candidate contributions exceed max_graph_bytes: "
                f"M={candidate_count}, required_bytes={required_bytes}, "
                f"max_graph_bytes={byte_limit}"
            )
        target_indices = np.empty(entry_count, dtype=np.int64)
        contribution_values = np.empty(entry_count, dtype=np.float64)
        for candidate_index, accumulator in enumerate(accumulators):
            start = int(candidate_indptr[candidate_index])
            for offset, target in enumerate(sorted(accumulator)):
                value = accumulator[target] / float(source_counts[candidate_index])
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError("candidate contribution is non-finite or non-positive")
                target_indices[start + offset] = target
                contribution_values[start + offset] = value
            if not math.isclose(
                float(
                    np.sum(
                        contribution_values[start : int(candidate_indptr[candidate_index + 1])],
                        dtype=np.float64,
                    )
                ),
                1.0,
                rel_tol=0.0,
                abs_tol=SPARSE_NUMERICAL_TOLERANCE,
            ):
                raise ValueError(
                    f"candidate {graph.candidate_ids[candidate_index]!r} contribution is not normalized"
                )
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
        "max support=%d, array_bytes=%d, elapsed=%.3fs",
        contributions.entry_count,
        float(np.mean(support_sizes)),
        int(np.max(support_sizes)),
        contributions.array_bytes,
        elapsed,
    )
    return contributions


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
    "KernelMetadata",
    "LeaveOneOutObjective",
    "NormalizedKernelColumn",
    "SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION",
    "SPARSE_KERNEL_GRAPH_SCHEMA_VERSION",
    "SPARSE_NUMERICAL_TOLERANCE",
    "SourceProgressCallback",
    "SparseAtomicKernelGraph",
    "SparseCandidateContributions",
    "aggregate_candidate_contributions",
    "aggregate_sparse_candidate_contributions",
    "build_atomic_kernel_graph",
    "build_candidate_contributions",
    "build_sparse_atomic_kernel_graph",
    "build_sparse_candidate_contributions",
    "build_sparse_kernel_graph",
    "compute_candidate_contributions",
    "construct_atomic_kernel_graph",
    "construct_sparse_atomic_kernel_graph",
    "create_sparse_atomic_kernel_graph",
    "evaluate_leave_one_out_objective",
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
