"""Typed settings for the information-entropy selector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

NEIGHBOUR_BACKEND_ID = "exact_cpu"
NEIGHBOUR_BACKEND_VERSION = "exact-cpu-v1"
INDEXED_NEIGHBOUR_BACKEND_ID = "exact_indexed_cpu"
INDEXED_NEIGHBOUR_BACKEND_VERSION = "exact-indexed-cpu-v1"
DEFAULT_NEIGHBOUR_BACKEND_ID = INDEXED_NEIGHBOUR_BACKEND_ID
NEIGHBOUR_METRIC = "euclidean"
NEIGHBOUR_DISTINCT_POLICY = "exact-coordinate-location-v1"
BANDWIDTH_SCHEMA_VERSION = "entropy-bandwidth-v1"
CALIBRATION_OPTIMIZER_ID = "bounded-grid"
CALIBRATION_OPTIMIZER_VERSION = "bounded-grid-v1"
KERNEL_FAMILY = "wendland_c2"
KERNEL_VERSION = "wendland-c2-v1"
SPARSE_KERNEL_GRAPH_SCHEMA_VERSION = "sparse-atomic-kernel-graph-v1"
SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION = "sparse-candidate-contributions-v1"
SPARSE_STREAMED_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION = (
    "sparse-candidate-contributions-streamed-v1"
)
KERNEL_OPERATOR_SCHEMA_VERSION = "kernel-operator-v1"
SPARSE_NUMERICAL_TOLERANCE = 1.0e-12
ENTROPY_OBJECTIVE_SCHEMA_VERSION = "entropy-objective-v1"
DEFAULT_RADIUS_QUERY_BYTES = 256 * 1024 * 1024
DEFAULT_CALIBRATION_WORK_BYTES = 512 * 1024 * 1024


def _readonly_float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional float64 array")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    array.setflags(write=False)
    return array


def _readonly_exact_array(value: Any, *, dtype: np.dtype[Any], name: str) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    array.setflags(write=False)
    return array


def _exact_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


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
    backend: str = DEFAULT_NEIGHBOUR_BACKEND_ID
    metric: str = NEIGHBOUR_METRIC
    chunk_size: int = 1024
    max_neighbour_entries: int = 1_000_000
    max_index_bytes: int | None = None
    max_radius_query_bytes: int = DEFAULT_RADIUS_QUERY_BYTES
    max_calibration_work_bytes: int = DEFAULT_CALIBRATION_WORK_BYTES
    calibration_batch_size: int | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"manual", "automatic"}:
            raise ValueError("bandwidth mode must be 'manual' or 'automatic'")
        if self.backend not in {NEIGHBOUR_BACKEND_ID, INDEXED_NEIGHBOUR_BACKEND_ID}:
            raise ValueError(f"unsupported neighbour backend: {self.backend!r}")
        if self.metric != NEIGHBOUR_METRIC:
            raise ValueError(f"unsupported neighbour metric: {self.metric!r}")
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, (int, np.integer)):
            raise ValueError("chunk_size must be a positive integer")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if (
            isinstance(self.max_neighbour_entries, bool)
            or not isinstance(self.max_neighbour_entries, (int, np.integer))
            or int(self.max_neighbour_entries) < 1
        ):
            raise ValueError("max_neighbour_entries must be a positive integer")
        if self.max_index_bytes is not None and (
            isinstance(self.max_index_bytes, bool)
            or not isinstance(self.max_index_bytes, (int, np.integer))
            or int(self.max_index_bytes) < 1
        ):
            raise ValueError("max_index_bytes must be a positive integer when provided")
        if (
            isinstance(self.max_radius_query_bytes, bool)
            or not isinstance(self.max_radius_query_bytes, (int, np.integer))
            or int(self.max_radius_query_bytes) < 1
        ):
            raise ValueError("max_radius_query_bytes must be a positive integer")
        if (
            isinstance(self.max_calibration_work_bytes, bool)
            or not isinstance(self.max_calibration_work_bytes, (int, np.integer))
            or int(self.max_calibration_work_bytes) < 1
        ):
            raise ValueError("max_calibration_work_bytes must be a positive integer")
        if self.calibration_batch_size is not None and (
            isinstance(self.calibration_batch_size, bool)
            or not isinstance(self.calibration_batch_size, (int, np.integer))
            or int(self.calibration_batch_size) < 1
        ):
            raise ValueError("calibration_batch_size must be a positive integer when provided")
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
        object.__setattr__(self, "max_neighbour_entries", int(self.max_neighbour_entries))
        object.__setattr__(self, "max_radius_query_bytes", int(self.max_radius_query_bytes))
        if self.max_index_bytes is not None:
            object.__setattr__(self, "max_index_bytes", int(self.max_index_bytes))
        object.__setattr__(self, "max_calibration_work_bytes", int(self.max_calibration_work_bytes))
        if self.calibration_batch_size is not None:
            object.__setattr__(self, "calibration_batch_size", int(self.calibration_batch_size))

    @property
    def operational_work_bytes(self) -> int:
        """Return the effective per-source calibration/query workspace limit."""

        return int(self.max_calibration_work_bytes)

    @property
    def radius_query_bytes(self) -> int:
        """Return the effective single-source exact-query limit."""

        return min(int(self.max_radius_query_bytes), self.operational_work_bytes)


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
    backend_version: str = NEIGHBOUR_BACKEND_VERSION
    backend_fingerprint: str = ""

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
        expected_backend_version = (
            INDEXED_NEIGHBOUR_BACKEND_VERSION
            if self.backend == INDEXED_NEIGHBOUR_BACKEND_ID
            else NEIGHBOUR_BACKEND_VERSION
        )
        if self.backend not in {NEIGHBOUR_BACKEND_ID, INDEXED_NEIGHBOUR_BACKEND_ID}:
            raise ValueError(f"unsupported neighbour backend: {self.backend!r}")
        if self.backend_version != expected_backend_version:
            raise ValueError(
                "bandwidth backend version does not match its configured backend: "
                f"{self.backend!r} requires {expected_backend_version!r}"
            )
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
    backend: str = NEIGHBOUR_BACKEND_ID
    backend_version: str = NEIGHBOUR_BACKEND_VERSION
    backend_fingerprint: str = ""

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
        if self.pool_fingerprint != self.selected.pool_fingerprint:
            raise ValueError("calibration pool identity must match selected bandwidths")
        if (
            self.backend != self.selected.backend
            or self.backend_version != self.selected.backend_version
            or self.backend_fingerprint != self.selected.backend_fingerprint
        ):
            raise ValueError("calibration backend identity must match selected bandwidths")
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


@dataclass(frozen=True, slots=True)
class SparseAtomicKernelRow:
    """Read-only source-major row of the sparse atomic kernel graph."""

    source_index: int
    candidate_index: int
    target_indices: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        targets = _readonly_exact_array(
            self.target_indices,
            dtype=np.dtype(np.int64),
            name="source-row target_indices",
        )
        values = _readonly_exact_array(
            self.values,
            dtype=np.dtype(np.float64),
            name="source-row values",
        )
        if targets.ndim != 1 or values.ndim != 1 or targets.shape != values.shape:
            raise ValueError("source-row targets and values must be aligned vectors")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("source-row values must be finite and strictly positive")
        object.__setattr__(self, "target_indices", targets)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self, "source_index", _exact_integer(self.source_index, name="source_index")
        )
        object.__setattr__(
            self,
            "candidate_index",
            _exact_integer(self.candidate_index, name="candidate_index"),
        )

    @property
    def targets(self) -> np.ndarray:
        return self.target_indices

    @property
    def weights(self) -> np.ndarray:
        return self.values


@dataclass(frozen=True, slots=True)
class SparseAtomicKernelGraph:
    """Immutable source-major CSR representation of the finite-pool kernel."""

    source_indptr: np.ndarray
    target_indices: np.ndarray
    values: np.ndarray
    row_candidate_indices: np.ndarray
    candidate_ids: tuple[str, ...]
    pool_fingerprint: str
    bandwidth_fingerprint: str
    transform_fingerprint: str
    fingerprint: str
    kernel_family: str = KERNEL_FAMILY
    kernel_version: str = KERNEL_VERSION
    normalization: str = "finite-pool-source-column-v1"
    source_bandwidth_orientation: str = "h_a"
    self_membership: bool = True
    sparse_schema_version: str = SPARSE_KERNEL_GRAPH_SCHEMA_VERSION
    numerical_tolerance: float = SPARSE_NUMERICAL_TOLERANCE

    def __post_init__(self) -> None:
        source_indptr = _readonly_exact_array(
            self.source_indptr,
            dtype=np.dtype(np.int64),
            name="source_indptr",
        )
        target_indices = _readonly_exact_array(
            self.target_indices,
            dtype=np.dtype(np.int64),
            name="target_indices",
        )
        values = _readonly_exact_array(self.values, dtype=np.dtype(np.float64), name="values")
        row_candidates = _readonly_exact_array(
            self.row_candidate_indices,
            dtype=np.dtype(np.int64),
            name="row_candidate_indices",
        )
        candidate_ids = tuple(self.candidate_ids)
        if not candidate_ids or any(
            not isinstance(value, str) or not value.strip() for value in candidate_ids
        ):
            raise ValueError("candidate_ids must be non-empty and non-blank")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if source_indptr.ndim != 1 or row_candidates.ndim != 1:
            raise ValueError("source graph row arrays must be one-dimensional")
        source_count = row_candidates.shape[0]
        if source_indptr.shape[0] != source_count + 1:
            raise ValueError("source_indptr must have one more entry than source rows")
        if target_indices.ndim != 1 or values.ndim != 1:
            raise ValueError("source graph edge arrays must be one-dimensional")
        edge_count = target_indices.shape[0]
        if values.shape[0] != edge_count:
            raise ValueError("target_indices and values must have equal lengths")
        if source_indptr[0] != 0 or source_indptr[-1] != edge_count:
            raise ValueError("source_indptr endpoints must cover all graph edges")
        if np.any(np.diff(source_indptr) < 0):
            raise ValueError("source_indptr must be non-decreasing")
        if np.any(row_candidates < 0) or np.any(row_candidates >= len(candidate_ids)):
            raise ValueError("source row ownership contains an unknown candidate")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("graph values must be finite and strictly positive")
        tolerance = float(self.numerical_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("numerical_tolerance must be finite and strictly positive")
        for source_index in range(source_count):
            start = int(source_indptr[source_index])
            stop = int(source_indptr[source_index + 1])
            targets = target_indices[start:stop]
            if targets.size == 0 or not np.any(targets == source_index):
                raise ValueError(f"source row {source_index} is missing its self-edge")
            if np.any(targets < 0) or np.any(targets >= source_count):
                raise ValueError("graph target index is outside the source-row range")
            if targets.size > 1 and np.any(np.diff(targets) <= 0):
                raise ValueError("graph target indices must be strictly increasing per source")
            if not math.isclose(
                float(np.sum(values[start:stop], dtype=np.float64)),
                1.0,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError(f"source row {source_index} is not normalized")
        for name, value in (
            ("pool_fingerprint", self.pool_fingerprint),
            ("bandwidth_fingerprint", self.bandwidth_fingerprint),
            ("transform_fingerprint", self.transform_fingerprint),
            ("fingerprint", self.fingerprint),
            ("kernel_family", self.kernel_family),
            ("kernel_version", self.kernel_version),
            ("normalization", self.normalization),
            ("source_bandwidth_orientation", self.source_bandwidth_orientation),
            ("sparse_schema_version", self.sparse_schema_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        if self.kernel_family != KERNEL_FAMILY or self.kernel_version != KERNEL_VERSION:
            raise ValueError("unsupported sparse graph kernel identity")
        if self.normalization != "finite-pool-source-column-v1":
            raise ValueError("unsupported sparse graph normalization convention")
        if self.source_bandwidth_orientation != "h_a":
            raise ValueError("sparse graph source bandwidth orientation must be h_a")
        if self.sparse_schema_version != SPARSE_KERNEL_GRAPH_SCHEMA_VERSION:
            raise ValueError("unsupported sparse graph schema version")
        if not math.isclose(
            tolerance,
            SPARSE_NUMERICAL_TOLERANCE,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("unsupported sparse graph numerical tolerance")
        if not self.self_membership:
            raise ValueError("source-major graph must retain self membership")
        object.__setattr__(self, "source_indptr", source_indptr)
        object.__setattr__(self, "target_indices", target_indices)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "row_candidate_indices", row_candidates)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "numerical_tolerance", tolerance)

    @property
    def n_sources(self) -> int:
        return int(self.row_candidate_indices.shape[0])

    @property
    def n_targets(self) -> int:
        return self.n_sources

    @property
    def edge_count(self) -> int:
        return int(self.target_indices.shape[0])

    @property
    def nnz(self) -> int:
        return self.edge_count

    @property
    def array_bytes(self) -> int:
        return int(
            self.source_indptr.nbytes
            + self.target_indices.nbytes
            + self.values.nbytes
            + self.row_candidate_indices.nbytes
        )

    @property
    def support_sizes(self) -> np.ndarray:
        values = np.diff(self.source_indptr).astype(np.int64, copy=True)
        values.setflags(write=False)
        return values

    @property
    def density(self) -> float:
        denominator = self.n_sources * self.n_targets
        return self.edge_count / denominator if denominator else 0.0

    @property
    def source_offsets(self) -> np.ndarray:
        return self.source_indptr

    @property
    def indptr(self) -> np.ndarray:
        return self.source_indptr

    @property
    def indices(self) -> np.ndarray:
        return self.target_indices

    @property
    def data(self) -> np.ndarray:
        return self.values

    @property
    def source_candidate_indices(self) -> np.ndarray:
        return self.row_candidate_indices

    def source_row(self, source_index: int) -> SparseAtomicKernelRow:
        index = int(source_index)
        if index < 0 or index >= self.n_sources:
            raise IndexError("source index is outside the graph")
        start = int(self.source_indptr[index])
        stop = int(self.source_indptr[index + 1])
        return SparseAtomicKernelRow(
            source_index=index,
            candidate_index=int(self.row_candidate_indices[index]),
            target_indices=self.target_indices[start:stop],
            values=self.values[start:stop],
        )

    get_source_row = source_row

    def iter_source_rows(self):
        for source_index in range(self.n_sources):
            yield self.source_row(source_index)


@dataclass(frozen=True, slots=True)
class StreamedKernelExecutionSummary:
    """Immutable execution evidence for direct source-column aggregation.

    This record deliberately describes an implicit exact atomic operator.  It
    is not an empty or synthetic ``SparseAtomicKernelGraph``.
    """

    atomic_row_count: int
    implicit_edge_count: int
    source_support_min: int
    source_support_mean: float
    source_support_max: int
    self_edge_count: int
    source_mass_max_deviation: float
    candidate_entry_count: int
    candidate_support_min: int
    candidate_support_mean: float
    candidate_support_max: int
    candidate_pmf_max_deviation: float
    candidate_csr_array_bytes: int
    contribution_spool_bytes: int
    estimated_peak_memory_bytes: int
    measured_peak_memory_bytes: int | None
    pool_fingerprint: str
    bandwidth_fingerprint: str
    transform_fingerprint: str
    backend: str
    metric: str
    backend_version: str
    kernel_operator_fingerprint: str
    atomic_graph_materialized: bool = False
    atomic_graph_csr_bytes: int = 0
    source_query_count: int = 0
    numerical_tolerance: float = SPARSE_NUMERICAL_TOLERANCE

    def __post_init__(self) -> None:
        for name in (
            "atomic_row_count",
            "implicit_edge_count",
            "source_support_min",
            "source_support_max",
            "self_edge_count",
            "candidate_entry_count",
            "candidate_support_min",
            "candidate_support_max",
            "candidate_csr_array_bytes",
            "contribution_spool_bytes",
            "estimated_peak_memory_bytes",
            "atomic_graph_csr_bytes",
            "source_query_count",
        ):
            value = _exact_integer(getattr(self, name), name=name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.atomic_row_count < 1 or self.source_query_count != self.atomic_row_count:
            raise ValueError("streamed execution must query every atomic source exactly once")
        if self.implicit_edge_count < self.atomic_row_count:
            raise ValueError("streamed execution must retain at least one self support per source")
        if self.source_support_min < 1 or self.source_support_max < self.source_support_min:
            raise ValueError("streamed source support distribution is invalid")
        if (
            self.candidate_support_min < 1
            or self.candidate_support_max < self.candidate_support_min
        ):
            raise ValueError("streamed candidate support distribution is invalid")
        for name in (
            "source_support_mean",
            "source_mass_max_deviation",
            "candidate_support_mean",
            "candidate_pmf_max_deviation",
            "numerical_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.numerical_tolerance <= 0.0:
            raise ValueError("numerical_tolerance must be strictly positive")
        if self.atomic_graph_materialized:
            raise ValueError("streamed execution summary cannot claim a materialized graph")
        if self.atomic_graph_csr_bytes != 0:
            raise ValueError("streamed execution must report zero atomic graph CSR bytes")
        for name, value in (
            ("backend", self.backend),
            ("metric", self.metric),
            ("backend_version", self.backend_version),
            ("pool_fingerprint", self.pool_fingerprint),
            ("bandwidth_fingerprint", self.bandwidth_fingerprint),
            ("transform_fingerprint", self.transform_fingerprint),
            ("kernel_operator_fingerprint", self.kernel_operator_fingerprint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-blank")
        if self.measured_peak_memory_bytes is not None:
            value = _exact_integer(
                self.measured_peak_memory_bytes, name="measured_peak_memory_bytes"
            )
            if value < 0:
                raise ValueError("measured_peak_memory_bytes must be non-negative")
            object.__setattr__(self, "measured_peak_memory_bytes", value)

    @property
    def E(self) -> int:
        return self.implicit_edge_count

    @property
    def Q(self) -> int:
        return self.candidate_entry_count

    @property
    def edge_count(self) -> int:
        return self.implicit_edge_count

    @property
    def csr_array_bytes(self) -> int:
        return self.atomic_graph_csr_bytes

    @property
    def array_bytes(self) -> int:
        """Atomic graph bytes; streamed execution intentionally reports zero."""

        return self.atomic_graph_csr_bytes

    @property
    def atomic_graph_CSR_bytes(self) -> int:
        """Compatibility spelling matching the diagnostics terminology."""

        return self.atomic_graph_csr_bytes

    @property
    def peak_resident_memory_bytes(self) -> int | None:
        return self.measured_peak_memory_bytes

    @property
    def source_normalization_max_deviation(self) -> float:
        return self.source_mass_max_deviation

    @property
    def kernel_source_fingerprint(self) -> str:
        return self.kernel_operator_fingerprint

    def to_manifest(self) -> dict[str, Any]:
        return {
            "atomic_graph_materialized": False,
            "atomic_row_count": self.atomic_row_count,
            "edge_count": self.implicit_edge_count,
            "edge_per_row": self.implicit_edge_count / self.atomic_row_count,
            "self_edge_count": self.self_edge_count,
            "source_mass_deviation": self.source_mass_max_deviation,
            "source_support": {
                "minimum": self.source_support_min,
                "mean": self.source_support_mean,
                "maximum": self.source_support_max,
            },
            "candidate_entry_count": self.candidate_entry_count,
            "candidate_support": {
                "minimum": self.candidate_support_min,
                "mean": self.candidate_support_mean,
                "maximum": self.candidate_support_max,
            },
            "candidate_pmf_max_deviation": self.candidate_pmf_max_deviation,
            "memory": {
                "csr_array_bytes": 0,
                "atomic_graph_csr_bytes": 0,
                "candidate_csr_array_bytes": self.candidate_csr_array_bytes,
                "measured_peak_memory_bytes": self.measured_peak_memory_bytes,
                "estimated_peak_memory_bytes": self.estimated_peak_memory_bytes,
                "contribution_spool_bytes": self.contribution_spool_bytes,
            },
            "backend": self.backend,
            "metric": self.metric,
            "backend_version": self.backend_version,
            "pool_fingerprint": self.pool_fingerprint,
            "bandwidth_fingerprint": self.bandwidth_fingerprint,
            "transform_fingerprint": self.transform_fingerprint,
            "kernel_operator_fingerprint": self.kernel_operator_fingerprint,
            "source_query_count": self.source_query_count,
            "numerical_tolerance": self.numerical_tolerance,
        }


@dataclass(frozen=True, slots=True)
class SparseCandidateContributionRow:
    """Read-only candidate-major sparse contribution row."""

    candidate_index: int
    candidate_id: str
    target_indices: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        targets = _readonly_exact_array(
            self.target_indices,
            dtype=np.dtype(np.int64),
            name="candidate target_indices",
        )
        values = _readonly_exact_array(
            self.values,
            dtype=np.dtype(np.float64),
            name="candidate values",
        )
        if targets.ndim != 1 or values.ndim != 1 or targets.shape != values.shape:
            raise ValueError("candidate targets and values must be aligned vectors")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("candidate values must be finite and strictly positive")
        object.__setattr__(self, "target_indices", targets)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "candidate_index",
            _exact_integer(self.candidate_index, name="candidate_index"),
        )
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-blank string")

    @property
    def targets(self) -> np.ndarray:
        return self.target_indices

    @property
    def weights(self) -> np.ndarray:
        return self.values


@dataclass(frozen=True, slots=True)
class SparseCandidateContributions:
    """Immutable candidate-major CSR representation of ``q_C``."""

    candidate_indptr: np.ndarray
    target_indices: np.ndarray
    values: np.ndarray
    candidate_ids: tuple[str, ...]
    graph_fingerprint: str
    fingerprint: str
    row_count: int = -1
    candidate_source_counts: np.ndarray | None = None
    pool_fingerprint: str = ""
    sparse_schema_version: str = SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION
    numerical_tolerance: float = SPARSE_NUMERICAL_TOLERANCE
    kernel_operator_fingerprint: str = ""

    def __post_init__(self) -> None:
        indptr = _readonly_exact_array(
            self.candidate_indptr,
            dtype=np.dtype(np.int64),
            name="candidate_indptr",
        )
        targets = _readonly_exact_array(
            self.target_indices,
            dtype=np.dtype(np.int64),
            name="candidate target_indices",
        )
        values = _readonly_exact_array(
            self.values, dtype=np.dtype(np.float64), name="candidate values"
        )
        candidate_ids = tuple(self.candidate_ids)
        if not candidate_ids or any(
            not isinstance(value, str) or not value.strip() for value in candidate_ids
        ):
            raise ValueError("candidate_ids must be non-empty and non-blank")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if indptr.ndim != 1 or indptr.shape[0] != len(candidate_ids) + 1:
            raise ValueError("candidate_indptr must have one more entry than candidates")
        if targets.ndim != 1 or values.ndim != 1 or targets.shape != values.shape:
            raise ValueError("candidate target_indices and values must be aligned")
        if indptr[0] != 0 or indptr[-1] != targets.shape[0] or np.any(np.diff(indptr) < 0):
            raise ValueError("candidate_indptr must be a non-decreasing CSR offset array")
        row_count = _exact_integer(self.row_count, name="row_count")
        if row_count < 0:
            row_count = int(np.max(targets)) + 1 if targets.size else 0
        if row_count < 1:
            raise ValueError("candidate contributions require at least one target row")
        if np.any(targets < 0) or np.any(targets >= row_count):
            raise ValueError("candidate target index is outside the pool row range")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("candidate contributions must be finite and strictly positive")
        tolerance = float(self.numerical_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("numerical_tolerance must be finite and strictly positive")
        for candidate_index in range(len(candidate_ids)):
            start = int(indptr[candidate_index])
            stop = int(indptr[candidate_index + 1])
            candidate_targets = targets[start:stop]
            if candidate_targets.size == 0:
                raise ValueError(f"candidate {candidate_ids[candidate_index]!r} has empty support")
            if candidate_targets.size > 1 and np.any(np.diff(candidate_targets) <= 0):
                raise ValueError("candidate target indices must be strictly increasing")
            if not math.isclose(
                float(np.sum(values[start:stop], dtype=np.float64)),
                1.0,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError(f"candidate {candidate_ids[candidate_index]!r} is not normalized")
        source_counts = self.candidate_source_counts
        if source_counts is not None:
            source_counts = _readonly_exact_array(
                source_counts,
                dtype=np.dtype(np.int64),
                name="candidate_source_counts",
            )
            if source_counts.ndim != 1 or source_counts.shape[0] != len(candidate_ids):
                raise ValueError("candidate_source_counts must align with candidate_ids")
            if np.any(source_counts <= 0):
                raise ValueError("candidate source counts must be strictly positive")
        for name, value in (
            ("fingerprint", self.fingerprint),
            ("sparse_schema_version", self.sparse_schema_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        if self.sparse_schema_version not in {
            SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
            SPARSE_STREAMED_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported sparse candidate contribution schema version")
        if not math.isclose(
            tolerance,
            SPARSE_NUMERICAL_TOLERANCE,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("unsupported sparse candidate numerical tolerance")
        object.__setattr__(self, "candidate_indptr", indptr)
        object.__setattr__(self, "target_indices", targets)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "candidate_source_counts", source_counts)
        object.__setattr__(self, "numerical_tolerance", tolerance)
        operator_fingerprint = self.kernel_operator_fingerprint or self.graph_fingerprint
        if not isinstance(operator_fingerprint, str) or not operator_fingerprint.strip():
            raise ValueError("kernel_operator_fingerprint must be non-blank")
        if (
            self.sparse_schema_version == SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION
            and (not isinstance(self.graph_fingerprint, str) or not self.graph_fingerprint.strip())
        ):
            raise ValueError("graph_fingerprint is required for graph-backed contributions")
        object.__setattr__(self, "kernel_operator_fingerprint", operator_fingerprint)

    @property
    def n_candidates(self) -> int:
        return len(self.candidate_ids)

    @property
    def n_targets(self) -> int:
        return self.row_count

    @property
    def entry_count(self) -> int:
        return int(self.target_indices.shape[0])

    @property
    def nnz(self) -> int:
        return self.entry_count

    @property
    def kernel_source_fingerprint(self) -> str:
        """Canonical identity of the exact kernel operator that produced q_C."""

        return self.kernel_operator_fingerprint

    @property
    def array_bytes(self) -> int:
        total = self.candidate_indptr.nbytes + self.target_indices.nbytes + self.values.nbytes
        if self.candidate_source_counts is not None:
            total += self.candidate_source_counts.nbytes
        return int(total)

    @property
    def support_sizes(self) -> np.ndarray:
        values = np.diff(self.candidate_indptr).astype(np.int64, copy=True)
        values.setflags(write=False)
        return values

    @property
    def indptr(self) -> np.ndarray:
        return self.candidate_indptr

    @property
    def indices(self) -> np.ndarray:
        return self.target_indices

    @property
    def data(self) -> np.ndarray:
        return self.values

    def candidate_index(self, candidate: int | str) -> int:
        if isinstance(candidate, bool):
            raise TypeError("candidate selector must be an index or candidate ID")
        if isinstance(candidate, str):
            try:
                return self.candidate_ids.index(candidate)
            except ValueError as exc:
                raise KeyError(candidate) from exc
        index = _exact_integer(candidate, name="candidate selector")
        if index < 0 or index >= self.n_candidates:
            raise IndexError("candidate index is outside the contribution record")
        return index

    def candidate_row(self, candidate: int | str) -> SparseCandidateContributionRow:
        index = self.candidate_index(candidate)
        start = int(self.candidate_indptr[index])
        stop = int(self.candidate_indptr[index + 1])
        return SparseCandidateContributionRow(
            candidate_index=index,
            candidate_id=self.candidate_ids[index],
            target_indices=self.target_indices[start:stop],
            values=self.values[start:stop],
        )

    get_candidate = candidate_row

    def iter_candidates(self):
        for candidate_index in range(self.n_candidates):
            yield self.candidate_row(candidate_index)


def _readonly_objective_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.dtype != np.dtype(np.float64) or array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional float64 array")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    array.setflags(write=False)
    return array


@dataclass(slots=True)
class EntropyObjectiveState:
    """Mutable selected-set state over immutable finite-pool objective inputs."""

    probabilities: np.ndarray
    baseline: np.ndarray
    s: np.ndarray
    candidate_ids: tuple[str, ...]
    selected_indices: tuple[int, ...]
    selected_candidate_ids: tuple[str, ...]
    budget: int
    beta: float
    objective: float
    anchor_objective: float
    pool_fingerprint: str
    graph_fingerprint: str
    contributions_fingerprint: str
    fingerprint: str
    numerical_tolerance: float = SPARSE_NUMERICAL_TOLERANCE
    schema_version: str = ENTROPY_OBJECTIVE_SCHEMA_VERSION
    kernel_operator_fingerprint: str = ""

    def __post_init__(self) -> None:
        probabilities = _readonly_objective_vector(self.probabilities, name="probabilities")
        baseline = _readonly_objective_vector(self.baseline, name="baseline")
        support = _readonly_objective_vector(self.s, name="objective support")
        if not (probabilities.shape == baseline.shape == support.shape):
            raise ValueError("objective vectors must have identical shapes")
        candidate_ids = tuple(self.candidate_ids)
        if not candidate_ids or any(
            not isinstance(value, str) or not value.strip() for value in candidate_ids
        ):
            raise ValueError("objective candidate_ids must be non-empty and non-blank")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("objective candidate_ids must be unique")
        selected_indices = tuple(
            _exact_integer(value, name="selected candidate index")
            for value in self.selected_indices
        )
        selected_ids = tuple(self.selected_candidate_ids)
        if len(selected_indices) != len(selected_ids):
            raise ValueError("selected candidate indices and IDs must align")
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("selected candidates must be unique")
        if any(index < 0 or index >= len(candidate_ids) for index in selected_indices):
            raise ValueError("selected candidate index is outside candidate_ids")
        if selected_ids != tuple(candidate_ids[index] for index in selected_indices):
            raise ValueError("selected candidate IDs do not match selected indices")
        budget = _exact_integer(self.budget, name="objective budget")
        if budget < 0 or budget > len(candidate_ids):
            raise ValueError("objective budget must satisfy 0 <= budget <= candidate count")
        if len(selected_indices) > budget:
            raise ValueError("selected candidate count exceeds objective budget")
        if isinstance(self.beta, bool):
            raise ValueError("beta must be finite and strictly positive")
        beta = float(self.beta)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be finite and strictly positive")
        tolerance = float(self.numerical_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("objective numerical_tolerance must be finite and positive")
        if not math.isclose(
            tolerance,
            SPARSE_NUMERICAL_TOLERANCE,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("unsupported objective numerical tolerance")
        if self.schema_version != ENTROPY_OBJECTIVE_SCHEMA_VERSION:
            raise ValueError("unsupported entropy objective schema version")
        for name, value in (
            ("pool_fingerprint", self.pool_fingerprint),
            ("contributions_fingerprint", self.contributions_fingerprint),
            ("fingerprint", self.fingerprint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        if not math.isfinite(float(self.objective)):
            raise ValueError("objective value must be finite")
        if not math.isfinite(float(self.anchor_objective)):
            raise ValueError("anchor objective must be finite")
        with np.errstate(over="raise", invalid="raise"):
            try:
                expected_baseline = beta * probabilities
            except FloatingPointError as exc:
                raise ValueError("objective baseline overflowed") from exc
        if not np.array_equal(expected_baseline, baseline):
            raise ValueError("objective baseline does not equal beta times probabilities")
        total_mass = float(np.sum(support, dtype=np.float64))
        expected_mass = beta + len(selected_indices)
        if not math.isclose(total_mass, expected_mass, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("objective support mass does not match beta plus selected count")
        probabilities.setflags(write=False)
        baseline.setflags(write=False)
        support.setflags(write=True)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "s", support)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "selected_indices", selected_indices)
        object.__setattr__(self, "selected_candidate_ids", selected_ids)
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "objective", float(self.objective))
        object.__setattr__(self, "anchor_objective", float(self.anchor_objective))
        object.__setattr__(self, "numerical_tolerance", tolerance)
        operator_fingerprint = self.kernel_operator_fingerprint or self.graph_fingerprint
        if not isinstance(operator_fingerprint, str) or not operator_fingerprint.strip():
            raise ValueError("kernel_operator_fingerprint must be non-blank")
        object.__setattr__(self, "kernel_operator_fingerprint", operator_fingerprint)

    @property
    def support(self) -> np.ndarray:
        """The owned mutable ``s_i(A)`` vector used by the update primitive."""

        return self.s

    @property
    def target_probabilities(self) -> np.ndarray:
        return self.probabilities

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return self.selected_candidate_ids

    @property
    def selected_count(self) -> int:
        return len(self.selected_indices)

    @property
    def n_targets(self) -> int:
        return int(self.s.shape[0])

    @property
    def is_final(self) -> bool:
        return self.selected_count == self.budget

    @property
    def anchor_normalized_objective(self) -> float:
        return self.objective - self.anchor_objective


@dataclass(frozen=True, slots=True)
class EntropyObjectiveDiagnostics:
    """Cross-entropy and forward-KL diagnostics for the current prefix."""

    selected_count: int
    budget: int
    denominator: float
    total_mass: float
    objective: float
    anchor_normalized_objective: float
    cross_entropy: float
    shannon_entropy: float
    kl_divergence: float
    beta: float
    is_final: bool
    state_fingerprint: str

    def __post_init__(self) -> None:
        selected_count = _exact_integer(self.selected_count, name="diagnostic selected count")
        budget = _exact_integer(self.budget, name="diagnostic budget")
        if selected_count < 0 or budget < selected_count:
            raise ValueError("diagnostic selected count is incompatible with budget")
        if isinstance(self.beta, bool):
            raise ValueError("diagnostic beta must be finite and strictly positive")
        beta = float(self.beta)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("diagnostic beta must be finite and strictly positive")
        tolerance = SPARSE_NUMERICAL_TOLERANCE
        for name, value in (
            ("denominator", self.denominator),
            ("total_mass", self.total_mass),
            ("objective", self.objective),
            ("anchor_normalized_objective", self.anchor_normalized_objective),
            ("cross_entropy", self.cross_entropy),
            ("shannon_entropy", self.shannon_entropy),
            ("kl_divergence", self.kl_divergence),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"diagnostic {name} must be finite")
        if self.denominator <= 0.0 or self.total_mass <= 0.0:
            raise ValueError("diagnostic mass and denominator must be positive")
        if not math.isclose(
            self.total_mass,
            self.denominator,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("diagnostic total mass must equal its denominator")
        if self.kl_divergence < -tolerance:
            raise ValueError("diagnostic KL divergence is materially negative")
        if not isinstance(self.state_fingerprint, str) or not self.state_fingerprint.strip():
            raise ValueError("diagnostic state fingerprint must be non-blank")
        object.__setattr__(self, "selected_count", selected_count)
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "beta", beta)

    @property
    def normalized_prefix_mass(self) -> float:
        return self.total_mass / self.denominator

    @property
    def is_final_budget(self) -> bool:
        return self.is_final


@dataclass(frozen=True, slots=True)
class EntropySelectionStep:
    """One immutable anchor or greedy acquisition history entry."""

    candidate_id: str
    structure_id: str | None
    acquisition_iteration: int
    reason: str
    marginal_gain: float | None
    objective: float
    anchor_normalized_objective: float
    selected_count: int
    cross_entropy: float | None = None
    forward_kl: float | None = None
    state_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("selection step candidate_id must be non-blank")
        if self.structure_id is not None and (
            not isinstance(self.structure_id, str) or not self.structure_id.strip()
        ):
            raise ValueError("selection step structure_id must be non-blank when supplied")
        if self.reason not in {"anchor", "greedy"}:
            raise ValueError("selection step reason must be 'anchor' or 'greedy'")
        iteration = _exact_integer(
            self.acquisition_iteration,
            name="selection step acquisition_iteration",
        )
        selected_count = _exact_integer(self.selected_count, name="selection step selected_count")
        if iteration < 0 or selected_count < 1:
            raise ValueError("selection step iteration and selected_count must be positive")
        if self.reason == "anchor":
            if self.marginal_gain is not None:
                raise ValueError("anchor selection steps cannot claim a marginal gain")
            if iteration != 0:
                raise ValueError("anchor selection steps must use acquisition_iteration zero")
        elif self.marginal_gain is None:
            raise ValueError("greedy selection steps require a marginal gain")
        if self.marginal_gain is not None and (
            not math.isfinite(float(self.marginal_gain)) or self.marginal_gain <= 0.0
        ):
            raise ValueError("selection step marginal gain must be finite and positive")
        for name, value in (
            ("objective", self.objective),
            ("anchor_normalized_objective", self.anchor_normalized_objective),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"selection step {name} must be finite")
        for name, value in (
            ("cross_entropy", self.cross_entropy),
            ("forward_kl", self.forward_kl),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"selection step {name} must be finite when supplied")
        if not isinstance(self.state_fingerprint, str) or not self.state_fingerprint.strip():
            raise ValueError("selection step state fingerprint must be non-blank")
        object.__setattr__(self, "acquisition_iteration", iteration)
        object.__setattr__(self, "selected_count", selected_count)
        if self.marginal_gain is not None:
            object.__setattr__(self, "marginal_gain", float(self.marginal_gain))


@dataclass(frozen=True, slots=True)
class EntropySelectionHistory:
    """Immutable acquisition order and per-step records."""

    acquisition_order: tuple[str, ...]
    steps: tuple[EntropySelectionStep, ...]

    def __post_init__(self) -> None:
        order = tuple(self.acquisition_order)
        steps = tuple(self.steps)
        if len(set(order)) != len(order):
            raise ValueError("selection acquisition order must contain unique candidate IDs")
        if tuple(step.candidate_id for step in steps) != order:
            raise ValueError("selection history steps must follow acquisition order")
        anchor_count = sum(step.reason == "anchor" for step in steps)
        if any(step.reason == "anchor" for step in steps[anchor_count:]):
            raise ValueError("anchor steps must precede greedy steps")
        object.__setattr__(self, "acquisition_order", order)
        object.__setattr__(self, "steps", steps)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return self.acquisition_order

    @property
    def records(self) -> tuple[EntropySelectionStep, ...]:
        return self.steps


@dataclass(frozen=True, slots=True)
class EntropySelectionPerformance:
    """Operational counters kept separate from the scientific result."""

    gain_evaluations: int
    initial_gain_evaluations: int
    refreshed_gain_evaluations: int
    update_gain_evaluations: int
    heap_pops: int = 0
    heap_reinsertions: int = 0
    certifications: int = 0
    support_work: int = 0
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        names = (
            "gain_evaluations",
            "initial_gain_evaluations",
            "refreshed_gain_evaluations",
            "update_gain_evaluations",
            "heap_pops",
            "heap_reinsertions",
            "certifications",
            "support_work",
        )
        for name in names:
            value = _exact_integer(getattr(self, name), name=name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.gain_evaluations != (
            self.initial_gain_evaluations
            + self.refreshed_gain_evaluations
            + self.update_gain_evaluations
        ):
            raise ValueError("gain_evaluations must equal its component counters")
        if not math.isfinite(float(self.elapsed_seconds)) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")

    @property
    def gain_calls(self) -> int:
        return self.gain_evaluations


@dataclass(frozen=True, slots=True)
class EntropySelectionResult:
    """Immutable direct-optimizer result ready for later stage adaptation."""

    method: str
    method_version: str
    budget: int
    beta: float
    anchor_ids: tuple[str, ...]
    acquisition_order: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    selected_indices: tuple[int, ...]
    history: EntropySelectionHistory
    anchor_objective: float
    final_objective: float
    final_anchor_normalized_objective: float
    final_cross_entropy: float
    final_shannon_entropy: float
    final_forward_kl: float
    pool_fingerprint: str
    graph_fingerprint: str
    contributions_fingerprint: str
    state_fingerprint: str
    performance: EntropySelectionPerformance
    final_normalized_support: tuple[float, ...]
    structure_provenance: tuple[tuple[str, str], ...] = ()
    kernel_operator_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.method not in {"full_greedy", "lazy_greedy"}:
            raise ValueError("selection method must be 'full_greedy' or 'lazy_greedy'")
        if not isinstance(self.method_version, str) or not self.method_version.strip():
            raise ValueError("selection method_version must be non-blank")
        budget = _exact_integer(self.budget, name="selection budget")
        if budget < 0:
            raise ValueError("selection budget must be non-negative")
        beta = float(self.beta)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("selection beta must be finite and strictly positive")
        anchor_ids = tuple(self.anchor_ids)
        acquisition_order = tuple(self.acquisition_order)
        selected_ids = tuple(self.selected_candidate_ids)
        selected_indices = tuple(
            _exact_integer(value, name="selection index") for value in self.selected_indices
        )
        if len(acquisition_order) != budget or len(set(acquisition_order)) != budget:
            raise ValueError("acquisition order must contain exactly budget unique IDs")
        if len(anchor_ids) > budget or acquisition_order[: len(anchor_ids)] != anchor_ids:
            raise ValueError("anchors must be the first acquisition IDs")
        if len(selected_ids) != budget or len(selected_indices) != budget:
            raise ValueError("selection membership must contain exactly budget candidates")
        if tuple(sorted(selected_indices)) != selected_indices:
            raise ValueError("selection indices must be in canonical index order")
        if set(selected_ids) != set(acquisition_order):
            raise ValueError("selection membership must match acquisition order")
        if self.history.acquisition_order != acquisition_order:
            raise ValueError("selection history does not match acquisition order")
        for name, value in (
            ("anchor_objective", self.anchor_objective),
            ("final_objective", self.final_objective),
            ("final_anchor_normalized_objective", self.final_anchor_normalized_objective),
            ("final_cross_entropy", self.final_cross_entropy),
            ("final_shannon_entropy", self.final_shannon_entropy),
            ("final_forward_kl", self.final_forward_kl),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name, value in (
            ("pool_fingerprint", self.pool_fingerprint),
            ("contributions_fingerprint", self.contributions_fingerprint),
            ("state_fingerprint", self.state_fingerprint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-blank")
        operator_fingerprint = self.kernel_operator_fingerprint or self.graph_fingerprint
        if not isinstance(operator_fingerprint, str) or not operator_fingerprint.strip():
            raise ValueError("kernel_operator_fingerprint must be non-blank")
        object.__setattr__(self, "kernel_operator_fingerprint", operator_fingerprint)
        normalized_support = tuple(float(value) for value in self.final_normalized_support)
        if not normalized_support or any(
            not math.isfinite(value) or value <= 0.0 for value in normalized_support
        ):
            raise ValueError("final_normalized_support must be finite and strictly positive")
        if not math.isclose(
            sum(normalized_support),
            1.0,
            rel_tol=0.0,
            abs_tol=SPARSE_NUMERICAL_TOLERANCE,
        ):
            raise ValueError("final_normalized_support must sum to one")
        provenance = tuple(self.structure_provenance)
        if len({candidate_id for candidate_id, _ in provenance}) != len(provenance):
            raise ValueError("structure provenance must contain unique candidate IDs")
        if any(
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or not isinstance(structure_id, str)
            or not structure_id.strip()
            for candidate_id, structure_id in provenance
        ):
            raise ValueError("structure provenance IDs must be non-blank strings")
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "anchor_ids", anchor_ids)
        object.__setattr__(self, "acquisition_order", acquisition_order)
        object.__setattr__(self, "selected_candidate_ids", selected_ids)
        object.__setattr__(self, "selected_indices", selected_indices)
        object.__setattr__(self, "final_normalized_support", normalized_support)
        object.__setattr__(self, "structure_provenance", provenance)

    @property
    def algorithm_id(self) -> str:
        return "information_entropy"

    @property
    def algorithm_version(self) -> str:
        return self.method_version

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return self.selected_candidate_ids

    @property
    def acquisition_ids(self) -> tuple[str, ...]:
        return self.acquisition_order

    @property
    def steps(self) -> tuple[EntropySelectionStep, ...]:
        return self.history.steps

    @property
    def membership(self) -> frozenset[str]:
        return frozenset(self.selected_candidate_ids)

    @property
    def K(self) -> int:
        return self.budget

    @property
    def final_F(self) -> float:
        return self.final_objective

    @property
    def objective_history(self) -> tuple[float, ...]:
        return (self.anchor_objective,) + tuple(
            step.objective for step in self.history.steps if step.reason == "greedy"
        )

    @property
    def marginal_gains(self) -> tuple[float, ...]:
        return tuple(
            step.marginal_gain
            for step in self.history.steps
            if step.reason == "greedy" and step.marginal_gain is not None
        )

    @property
    def anchor_count(self) -> int:
        return len(self.anchor_ids)

    @property
    def initial_objective(self) -> float:
        return self.anchor_objective

    @property
    def final_diagnostics(self) -> tuple[float, float, float]:
        return (self.final_cross_entropy, self.final_shannon_entropy, self.final_forward_kl)

    @property
    def final_q(self) -> tuple[float, ...]:
        return self.final_normalized_support

    @property
    def final_normalized_q(self) -> tuple[float, ...]:
        return self.final_normalized_support


# Short aliases keep the optimizer API discoverable without duplicating model classes.
GreedySelectionStep = EntropySelectionStep
GreedySelectionHistory = EntropySelectionHistory
GreedySelectionPerformance = EntropySelectionPerformance
GreedySelectionResult = EntropySelectionResult


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
    "EntropyObjectiveDiagnostics",
    "EntropyObjectiveState",
    "EntropySelectionHistory",
    "EntropySelectionPerformance",
    "EntropySelectionResult",
    "EntropySelectionStep",
    "FrozenBandwidths",
    "InformationEntropyConfig",
    "KernelMetadata",
    "KERNEL_FAMILY",
    "KERNEL_VERSION",
    "NEIGHBOUR_BACKEND_ID",
    "NEIGHBOUR_BACKEND_VERSION",
    "INDEXED_NEIGHBOUR_BACKEND_ID",
    "INDEXED_NEIGHBOUR_BACKEND_VERSION",
    "DEFAULT_NEIGHBOUR_BACKEND_ID",
    "DEFAULT_RADIUS_QUERY_BYTES",
    "DEFAULT_CALIBRATION_WORK_BYTES",
    "NEIGHBOUR_DISTINCT_POLICY",
    "NEIGHBOUR_METRIC",
    "SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION",
    "SPARSE_STREAMED_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION",
    "KERNEL_OPERATOR_SCHEMA_VERSION",
    "ENTROPY_OBJECTIVE_SCHEMA_VERSION",
    "SPARSE_KERNEL_GRAPH_SCHEMA_VERSION",
    "SPARSE_NUMERICAL_TOLERANCE",
    "SparseAtomicKernelGraph",
    "SparseAtomicKernelRow",
    "SparseCandidateContributionRow",
    "SparseCandidateContributions",
    "StreamedKernelExecutionSummary",
    "GreedySelectionHistory",
    "GreedySelectionPerformance",
    "GreedySelectionResult",
    "GreedySelectionStep",
]
