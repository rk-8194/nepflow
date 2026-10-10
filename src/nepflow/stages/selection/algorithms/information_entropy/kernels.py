"""Versioned Wendland kernels and bounded finite-pool calibration evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import KERNEL_FAMILY, KERNEL_VERSION, KernelMetadata
from .neighbours import _validate_chunk_size, _validate_descriptors


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
        total = 0.0
        for start in range(0, values.shape[0], block):
            stop = min(start + block, values.shape[0])
            distances = _distance_block(values, source_index, start, stop)
            raw = np.asarray(wendland_kernel(distances / scale), dtype=np.float64)
            total += float(np.sum(raw, dtype=np.float64))
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
) -> Iterator[NormalizedKernelColumn]:
    """Yield positive normalized source columns using bounded workspaces.

    A source's bandwidth is always ``bandwidths[source_index]``.  The
    normaliser includes every finite-pool target row, including self.
    """

    values = _validate_descriptors(descriptors)
    scales = _validate_bandwidths(bandwidths, values.shape[0])
    block = _validate_chunk_size(chunk_size)
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


evaluate_loo_objective = evaluate_leave_one_out_objective


__all__ = [
    "KERNEL_FAMILY",
    "KERNEL_VERSION",
    "KernelMetadata",
    "LeaveOneOutObjective",
    "NormalizedKernelColumn",
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
