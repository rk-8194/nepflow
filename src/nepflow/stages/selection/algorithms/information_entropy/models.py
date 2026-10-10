"""Typed settings for the information-entropy selector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

NEIGHBOUR_BACKEND_ID = "exact_cpu"
NEIGHBOUR_BACKEND_VERSION = "exact-cpu-v1"
NEIGHBOUR_METRIC = "euclidean"
NEIGHBOUR_DISTINCT_POLICY = "exact-coordinate-location-v1"
BANDWIDTH_SCHEMA_VERSION = "entropy-bandwidth-v1"
CALIBRATION_OPTIMIZER_ID = "bounded-grid"
CALIBRATION_OPTIMIZER_VERSION = "bounded-grid-v1"
KERNEL_FAMILY = "wendland_c2"
KERNEL_VERSION = "wendland-c2-v1"


def _readonly_float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional float64 array")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class InformationEntropyConfig:
    """Numerical controls for the deterministic finite-pool objective."""

    background_mass: float = 1.0e-12
    kernel_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.background_mass)) or self.background_mass <= 0.0:
            raise ValueError("background_mass must be finite and positive")
        if not math.isfinite(float(self.kernel_scale)) or self.kernel_scale <= 0.0:
            raise ValueError("kernel_scale must be finite and positive")


@dataclass(frozen=True, slots=True)
class EntropyBandwidthSettings:
    """Validated manual or deterministic automatic bandwidth settings."""

    mode: str = "automatic"
    k: int | None = None
    c: float | None = None
    k_candidates: tuple[int, ...] = (1, 2, 4, 8)
    c_candidates: tuple[float, ...] = (1.5, 2.0, 4.0, 8.0)
    backend: str = NEIGHBOUR_BACKEND_ID
    metric: str = NEIGHBOUR_METRIC
    chunk_size: int = 1024

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"manual", "automatic"}:
            raise ValueError("bandwidth mode must be 'manual' or 'automatic'")
        if self.backend != NEIGHBOUR_BACKEND_ID:
            raise ValueError(f"unsupported neighbour backend: {self.backend!r}")
        if self.metric != NEIGHBOUR_METRIC:
            raise ValueError(f"unsupported neighbour metric: {self.metric!r}")
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, (int, np.integer)):
            raise ValueError("chunk_size must be a positive integer")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        k_candidates = _validate_ordered_integer_domain(self.k_candidates, "k_candidates")
        c_candidates = _validate_ordered_float_domain(self.c_candidates, "c_candidates")
        if mode == "manual":
            if self.k is None or self.c is None:
                raise ValueError("manual bandwidth mode requires explicit k and c")
            _validate_positive_integer(self.k, "k")
            _validate_positive_float(self.c, "c")
        elif self.k is not None or self.c is not None:
            raise ValueError("automatic bandwidth mode does not accept manual k or c")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "k_candidates", k_candidates)
        object.__setattr__(self, "c_candidates", c_candidates)
        object.__setattr__(self, "chunk_size", int(self.chunk_size))


def _validate_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and strictly positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _validate_ordered_integer_domain(value: Any, name: str) -> tuple[int, ...]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty ordered domain") from exc
    if not values:
        raise ValueError(f"{name} must be non-empty")
    validated = tuple(_validate_positive_integer(item, name) for item in values)
    if tuple(sorted(set(validated))) != validated:
        raise ValueError(f"{name} must be strictly increasing and unique")
    return validated


def _validate_ordered_float_domain(value: Any, name: str) -> tuple[float, ...]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty ordered domain") from exc
    if not values:
        raise ValueError(f"{name} must be non-empty")
    validated = tuple(_validate_positive_float(item, name) for item in values)
    if tuple(sorted(set(validated))) != validated:
        raise ValueError(f"{name} must be strictly increasing and unique")
    return validated


@dataclass(frozen=True, slots=True)
class EntropyPool:
    """Validated, ordered local-row pool and canonical target probability mass."""

    descriptors: np.ndarray
    rows: tuple[Any, ...]
    candidate_ids: tuple[str, ...]
    row_candidate_indices: np.ndarray
    probabilities: np.ndarray
    transform_fingerprint: str
    representation_fingerprint: str
    fingerprint: str

    @property
    def target_weights(self) -> np.ndarray:
        """Alias for the canonical row-level target probability masses."""

        return self.probabilities

    @property
    def row_owners(self) -> np.ndarray:
        """Alias for the ordered row-to-candidate index mapping."""

        return self.row_candidate_indices

    def __post_init__(self) -> None:
        descriptors = np.array(self.descriptors, dtype=np.float64, copy=True)
        if descriptors.ndim != 2 or descriptors.shape[0] == 0 or descriptors.shape[1] == 0:
            raise ValueError("entropy descriptors must be a non-empty 2D array")
        if not np.all(np.isfinite(descriptors)):
            raise ValueError("entropy descriptors must be finite")
        row_candidates = np.array(self.row_candidate_indices, dtype=np.int64, copy=True)
        probabilities = _readonly_float_array(self.probabilities, name="probabilities")
        if len(self.rows) != descriptors.shape[0] or len(row_candidates) != descriptors.shape[0]:
            raise ValueError("entropy rows, owner mapping, and descriptors must align")
        if len(probabilities) != descriptors.shape[0]:
            raise ValueError("probabilities must align with entropy descriptor rows")
        if np.any(row_candidates < 0) or np.any(row_candidates >= len(self.candidate_ids)):
            raise ValueError("entropy row owner mapping contains an unknown candidate")
        if not self.transform_fingerprint or not self.representation_fingerprint:
            raise ValueError("entropy pool transform and representation identities are required")
        descriptors.setflags(write=False)
        row_candidates.setflags(write=False)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "row_candidate_indices", row_candidates)
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True, slots=True)
class FrozenBandwidths:
    """Frozen full-pool radii and source bandwidths for one ``(k, c)`` pair."""

    radii: np.ndarray
    bandwidths: np.ndarray
    k: int
    c: float
    neighbour_fingerprint: str
    pool_fingerprint: str
    transform_fingerprint: str
    fingerprint: str
    backend: str = NEIGHBOUR_BACKEND_ID
    metric: str = NEIGHBOUR_METRIC
    schema_version: str = BANDWIDTH_SCHEMA_VERSION

    @property
    def h(self) -> np.ndarray:
        """Mathematical alias for the frozen source bandwidth vector."""

        return self.bandwidths

    def __post_init__(self) -> None:
        radii = _readonly_float_array(self.radii, name="radii")
        bandwidths = _readonly_float_array(self.bandwidths, name="bandwidths")
        if radii.shape != bandwidths.shape:
            raise ValueError("radii and bandwidths must have the same shape")
        _validate_positive_integer(self.k, "k")
        _validate_positive_float(self.c, "c")
        if not self.neighbour_fingerprint or not self.pool_fingerprint:
            raise ValueError("bandwidth identities must not be blank")
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "c", float(self.c))


@dataclass(frozen=True, slots=True)
class CalibrationAttempt:
    """One deterministic automatic/manual calibration evaluation."""

    k: int
    c: float
    status: str
    objective: float | None = None
    reason: str | None = None
    bandwidth_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class BandwidthCalibrationResult:
    """Complete, auditable result of bandwidth calibration."""

    mode: str
    selected: FrozenBandwidths
    objective: float | None
    attempts: tuple[CalibrationAttempt, ...]
    k_domain: tuple[int, ...]
    c_domain: tuple[float, ...]
    optimizer_id: str
    optimizer_version: str
    evaluation_count: int
    termination_reason: str
    pool_fingerprint: str
    calibration_fingerprint: str
    loo_probabilities: np.ndarray | None = None

    @property
    def k(self) -> int:
        """Selected neighbourhood order."""

        return self.selected.k

    @property
    def c(self) -> float:
        """Selected global bandwidth scale."""

        return self.selected.c

    @property
    def radii(self) -> np.ndarray:
        """Frozen kth-neighbour radii."""

        return self.selected.radii

    @property
    def bandwidths(self) -> np.ndarray:
        """Frozen source bandwidths."""

        return self.selected.bandwidths

    @property
    def search_domain(self) -> tuple[tuple[int, float], ...]:
        """Ordered Cartesian search domain recorded by the calibration."""

        return tuple(
            (candidate_k, candidate_c)
            for candidate_k in self.k_domain
            for candidate_c in self.c_domain
        )

    def __post_init__(self) -> None:
        if self.mode not in {"manual", "automatic"}:
            raise ValueError("calibration mode must be manual or automatic")
        if self.evaluation_count != len(self.attempts):
            raise ValueError("evaluation_count must equal the number of attempts")
        if self.loo_probabilities is not None:
            values = np.array(self.loo_probabilities, dtype=np.float64, copy=True)
            if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError("loo_probabilities must be finite and positive")
            values.setflags(write=False)
            object.__setattr__(self, "loo_probabilities", values)


@dataclass(frozen=True, slots=True)
class KernelMetadata:
    """Versioned metadata for the finite-pool normalized kernel primitive."""

    family: str = KERNEL_FAMILY
    version: str = KERNEL_VERSION
    normalization: str = "finite-pool-source-column-v1"
    source_bandwidth_orientation: str = "h_b"
    self_membership: bool = True


BandwidthSettings = EntropyBandwidthSettings
CalibrationSettings = EntropyBandwidthSettings
BandwidthResult = FrozenBandwidths
CalibrationResult = BandwidthCalibrationResult


__all__ = [
    "BANDWIDTH_SCHEMA_VERSION",
    "CALIBRATION_OPTIMIZER_ID",
    "CALIBRATION_OPTIMIZER_VERSION",
    "BandwidthCalibrationResult",
    "BandwidthResult",
    "BandwidthSettings",
    "CalibrationAttempt",
    "CalibrationResult",
    "CalibrationSettings",
    "EntropyBandwidthSettings",
    "EntropyPool",
    "FrozenBandwidths",
    "InformationEntropyConfig",
    "KernelMetadata",
    "KERNEL_FAMILY",
    "KERNEL_VERSION",
    "NEIGHBOUR_BACKEND_ID",
    "NEIGHBOUR_BACKEND_VERSION",
    "NEIGHBOUR_DISTINCT_POLICY",
    "NEIGHBOUR_METRIC",
]
