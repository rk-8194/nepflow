"""Shared, bounded estimation of the entropy reference measure.

This module deliberately has no dependency on representation or entropy pool
classes.  Both the preliminary local-representation pass and the final pool
therefore use the same versioned estimator without introducing an import
cycle.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json

REFERENCE_MEASURE_VERSION = "descriptor-local-occupancy-v1"
REFERENCE_MEASURE_ESTIMATOR = "fixed-radius-euclidean-occupancy"
REFERENCE_MEASURE_WEIGHTING_VERSION = "entropy-reference-weights-v1"
DEFAULT_REFERENCE_MEASURE_TOLERANCE = 1.0e-6
DEFAULT_REFERENCE_MEASURE_CHUNK_SIZE = 1024


@dataclass(frozen=True, slots=True)
class ReferenceMeasure:
    """Reference probabilities and complete estimator evidence."""

    probabilities: np.ndarray
    version: str
    config: dict[str, Any]
    weights_fingerprint: str
    diagnostics: dict[str, Any]


def _array_fingerprint(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    payload = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }
    return sha256_canonical_json(payload)


def reference_weights_fingerprint(values: np.ndarray) -> str:
    """Return the canonical identity used for persisted target probabilities."""

    return _array_fingerprint(np.asarray(values, dtype=np.float64))


def _validate_inputs(
    descriptors: np.ndarray,
    tolerance: float,
    chunk_size: int,
) -> tuple[np.ndarray, float, int]:
    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("reference-measure descriptors must be a non-empty 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("reference-measure descriptors must be finite")
    radius = float(tolerance)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("reference_measure_tolerance must be finite and positive")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise ValueError("reference_measure_chunk_size must be a positive integer")
    block = int(chunk_size)
    if block < 1:
        raise ValueError("reference_measure_chunk_size must be a positive integer")
    return values, radius, block


def _fixed_radius_occupancy(
    descriptors: np.ndarray,
    *,
    tolerance: float,
    chunk_size: int,
) -> np.ndarray:
    """Count inclusive descriptor balls without materialising an N x N array."""

    values, radius, configured_block = _validate_inputs(descriptors, tolerance, chunk_size)
    n_rows = int(values.shape[0])
    counts = np.zeros(n_rows, dtype=np.int64)
    radius_squared = radius * radius
    block = min(configured_block, max(1, math.isqrt(n_rows)))
    for source_start in range(0, n_rows, block):
        source_stop = min(source_start + block, n_rows)
        source = values[source_start:source_stop]
        for target_start in range(0, n_rows, block):
            target_stop = min(target_start + block, n_rows)
            target = values[target_start:target_stop]
            with np.errstate(over="ignore", invalid="ignore"):
                difference = source[:, None, :] - target[None, :, :]
                squared = np.sum(difference * difference, axis=2, dtype=np.float64)
            counts[source_start:source_stop] += np.count_nonzero(
                squared <= radius_squared, axis=1
            )
    if np.any(counts < 1):
        raise ValueError("reference-measure occupancy must include every row itself")
    return counts


def estimate_reference_measure(
    descriptors: np.ndarray,
    *,
    tolerance: float = DEFAULT_REFERENCE_MEASURE_TOLERANCE,
    chunk_size: int = DEFAULT_REFERENCE_MEASURE_CHUNK_SIZE,
    preprocessing_transform_fingerprint: str = "",
) -> ReferenceMeasure:
    """Estimate the canonical density-aware target probabilities.

    The entropy representation uses a deterministic two-pass procedure:
    unweighted preliminary whitening, fixed-radius occupancy estimation, then
    final whitening with these probabilities.  The preliminary transform
    identity is recorded so the measure is reproducible and auditable.
    """

    values, radius, configured_block = _validate_inputs(descriptors, tolerance, chunk_size)
    occupancy = _fixed_radius_occupancy(
        values,
        tolerance=radius,
        chunk_size=configured_block,
    )
    raw_weights = 1.0 / occupancy.astype(np.float64)
    total = float(math.fsum(float(value) for value in raw_weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("reference-measure weights must have a finite positive sum")
    probabilities = np.asarray(raw_weights / total, dtype=np.float64)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("reference-measure weights must be finite and positive")
    if not math.isclose(float(np.sum(probabilities)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("reference-measure weights must sum to one")
    probabilities.setflags(write=False)
    hhi = float(np.sum(probabilities * probabilities, dtype=np.float64))
    config = {
        "estimator": REFERENCE_MEASURE_ESTIMATOR,
        "metric": "euclidean",
        "tolerance": radius,
        "boundary": "inclusive",
        "self_included": True,
        "rare_environment_policy": "retain_positive_mass",
        "preprocessing_order": "preliminary_unweighted_whitening_then_weighted_whitening",
        "weighting_version": REFERENCE_MEASURE_WEIGHTING_VERSION,
        "preprocessing_transform_fingerprint": str(preprocessing_transform_fingerprint),
        # Chunk width is retained for operational diagnostics but omitted from
        # scientific identities by callers.
        "chunk_size": configured_block,
    }
    diagnostics = {
        "occupancy_fingerprint": _array_fingerprint(np.asarray(occupancy, dtype=np.int64)),
        "occupancy_count": {
            "minimum": int(np.min(occupancy)),
            "median": float(np.median(occupancy.astype(np.float64))),
            "mean": float(np.mean(occupancy.astype(np.float64))),
            "maximum": int(np.max(occupancy)),
        },
        "effective_support": 1.0 / hhi,
        "effective_weighted_sample_mass": 1.0 / hhi,
        "maximum_row_mass": float(np.max(probabilities)),
        "mass_concentration_hhi": hhi,
        "positive_row_count": int(probabilities.size),
        "weight_squared_sum": hhi,
    }
    return ReferenceMeasure(
        probabilities=probabilities,
        version=REFERENCE_MEASURE_VERSION,
        config=config,
        weights_fingerprint=reference_weights_fingerprint(probabilities),
        diagnostics=diagnostics,
    )


__all__ = [
    "DEFAULT_REFERENCE_MEASURE_CHUNK_SIZE",
    "DEFAULT_REFERENCE_MEASURE_TOLERANCE",
    "REFERENCE_MEASURE_ESTIMATOR",
    "REFERENCE_MEASURE_VERSION",
    "REFERENCE_MEASURE_WEIGHTING_VERSION",
    "ReferenceMeasure",
    "estimate_reference_measure",
    "reference_weights_fingerprint",
]
