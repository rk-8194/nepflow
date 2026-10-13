"""Shared, bounded estimation of the entropy reference measure.

This module deliberately has no dependency on representation or entropy pool
classes.  Both the preliminary local-representation pass and the final pool
therefore use the same versioned estimator without introducing an import
cycle.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json
from nepflow.resources.budget import ResourceBudgetService

from .indexed_radius import IndexedRadiusCountService

logger = logging.getLogger(__name__)

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
    resource_budget: ResourceBudgetService | None = None,
) -> np.ndarray:
    """Count inclusive descriptor balls with an indexed exact radius service."""

    counts, _diagnostics = _fixed_radius_occupancy_with_diagnostics(
        descriptors,
        tolerance=tolerance,
        chunk_size=chunk_size,
        resource_budget=resource_budget,
    )
    return counts


def _fixed_radius_occupancy_with_diagnostics(
    descriptors: np.ndarray,
    *,
    tolerance: float,
    chunk_size: int,
    resource_budget: ResourceBudgetService | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return exact occupancies and the operational indexed-query evidence."""

    values, radius, configured_block = _validate_inputs(descriptors, tolerance, chunk_size)
    service = IndexedRadiusCountService(
        values,
        chunk_size=configured_block,
        resource_budget=resource_budget,
    )
    return service.count(radius)


def estimate_reference_measure(
    descriptors: np.ndarray,
    *,
    tolerance: float = DEFAULT_REFERENCE_MEASURE_TOLERANCE,
    chunk_size: int = DEFAULT_REFERENCE_MEASURE_CHUNK_SIZE,
    preprocessing_transform_fingerprint: str = "",
    resource_budget: ResourceBudgetService | None = None,
) -> ReferenceMeasure:
    """Estimate the canonical density-aware target probabilities.

    The entropy representation uses a deterministic two-pass procedure:
    unweighted preliminary whitening, fixed-radius occupancy estimation, then
    final whitening with these probabilities.  The preliminary transform
    identity is recorded so the measure is reproducible and auditable.
    """

    values, radius, configured_block = _validate_inputs(descriptors, tolerance, chunk_size)
    occupancy, occupancy_diagnostics = _fixed_radius_occupancy_with_diagnostics(
        values,
        tolerance=radius,
        chunk_size=configured_block,
        resource_budget=resource_budget,
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
        **occupancy_diagnostics,
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
    singleton_fraction = float(occupancy_diagnostics["singleton_row_fraction"])
    if singleton_fraction >= 0.99:
        logger.warning(
            "Reference-measure occupancy is almost entirely singleton at tolerance=%g: "
            "singleton_fraction=%.6f; density correction therefore approximates uniform "
            "row weighting; the configured tolerance was not changed",
            radius,
            singleton_fraction,
        )
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
