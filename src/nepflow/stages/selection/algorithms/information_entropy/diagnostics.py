"""Immutable scientific diagnostics for finite-pool entropy selection.

This module deliberately sits after the scientific selector.  It consumes the
validated pool, calibration, graph, contribution and selection records; it
does not refit, retune, reorder, or acquire anything.  Scientific values and
identities are fingerprinted separately from the retained operational evidence
(timings, memory, and provider counters), so runtime variation cannot change a
run fingerprint.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from nepflow.io.atomic import atomic_write_stream
from nepflow.io.hashing import sha256_bytes, sha256_canonical_json, sha256_file
from nepflow.resources.budget import (
    ResourceBudgetService,
    ResourceCapacityError,
    build_resource_budget,
)

from .gpu_devices import (
    contiguous_gpu_ranges,
    normalize_gpu_devices,
    validate_visible_gpu_devices,
)
from .gpu_neighbours import _canonical_distances as _canonical_gpu_distances
from .gpu_support import (
    GPUCapacityError,
    GPUCertificationError,
    GPUDeviceValidationError,
    _distance_intervals,
)
from .models import (
    KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE,
    KERNEL_BANDWIDTH_RULE_SYMMETRIC_MAX,
    SPARSE_NUMERICAL_TOLERANCE,
    BandwidthCalibrationResult,
    CalibrationAttempt,
    EntropyPool,
    EntropySelectionHistory,
    EntropySelectionResult,
    FrozenBandwidths,
    OutOfCoreSparseCandidateContributions,
    SparseAtomicKernelGraph,
    SparseCandidateContributions,
    StreamedKernelExecutionSummary,
    kernel_bandwidth_rule_version,
    normalize_kernel_bandwidth_rule,
)

logger = logging.getLogger(__name__)

DIAGNOSTICS_SCHEMA_VERSION = "entropy-scientific-diagnostics-v4"
DIAGNOSTICS_ARRAY_ARTIFACT_SCHEMA = "entropy-diagnostic-arrays-v1"
PREFIX_COVERAGE_SCHEMA_VERSION = "entropy-prefix-coverage-v1"
DIAGNOSTICS_NUMERICAL_TOLERANCE = SPARSE_NUMERICAL_TOLERANCE
QUANTILE_CONVENTION = "weighted-inverse-empirical-cdf-stable-value-row-index-v1"


def _whitening_manifest_with_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Add conditioning labels while retaining the exact transform manifest."""

    result = dict(value)
    eigenvalues = np.asarray(value.get("eigenvalues", ()), dtype=np.float64)
    tolerance = float(value.get("tolerance", DIAGNOSTICS_NUMERICAL_TOLERANCE))
    retained = tuple(int(item) for item in value.get("retained_indices", ()))
    singular = tuple(int(index) for index, item in enumerate(eigenvalues) if item <= tolerance)
    effective = eigenvalues[list(retained)] if retained else np.empty(0, dtype=np.float64)
    if str(value.get("singular_policy")) == "regularize" and effective.size:
        effective = np.maximum(effective, tolerance) + float(value.get("regularization", 0.0))
    condition = (
        None
        if effective.size == 0 or float(np.min(effective)) <= 0.0
        else float(np.max(effective) / np.min(effective))
    )
    result.update(
        {
            "singular_indices": list(singular),
            "retained_indices": list(retained),
            "condition_number": condition,
            "spectrum_unit": "descriptor covariance eigenvalue",
            "identity_verified_against": "WhiteningTransform.to_manifest",
        }
    )
    return result


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _tuple_floats(values: Sequence[Any], *, name: str) -> np.ndarray:
    result = np.asarray([_finite(value, name=name) for value in values], dtype=np.float64)
    result.setflags(write=False)
    return result


def _array_digest(values: Sequence[Any], *, dtype: str = "<f8") -> str:
    array = np.asarray(values, dtype=np.dtype(dtype))
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def _diagnostic_array_metadata(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_digest(array),
    }


def _inline_diagnostic_arrays(
    probabilities: np.ndarray,
    final_q: np.ndarray,
    row_ids: Sequence[str],
) -> dict[str, Any]:
    buffer = BytesIO()
    np.savez(buffer, p=probabilities, q=final_q)
    payload = buffer.getvalue()
    return {
        "schema": DIAGNOSTICS_ARRAY_ARTIFACT_SCHEMA,
        "storage": "inline",
        "encoding": "base64",
        "payload": base64.b64encode(payload).decode("ascii"),
        "sha256": sha256_bytes(payload),
        "row_count": int(probabilities.shape[0]),
        "row_ids_sha256": sha256_canonical_json(list(row_ids)),
        "arrays": {
            "p": _diagnostic_array_metadata(probabilities),
            "q": _diagnostic_array_metadata(final_q),
        },
    }


def _load_diagnostic_arrays(
    artifact: Mapping[str, Any],
    row_ids: Sequence[str],
    *,
    artifact_root: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if artifact.get("schema") != DIAGNOSTICS_ARRAY_ARTIFACT_SCHEMA:
        raise ValueError("unsupported entropy diagnostics numeric artifact schema")
    storage = artifact.get("storage")
    if storage == "inline":
        if artifact.get("encoding") != "base64" or not isinstance(artifact.get("payload"), str):
            raise ValueError("entropy diagnostics inline numeric artifact is malformed")
        try:
            payload = base64.b64decode(artifact["payload"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "entropy diagnostics inline numeric artifact is not valid base64"
            ) from exc
    elif storage == "file":
        path_value = artifact.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError("entropy diagnostics numeric artifact path is missing")
        path = Path(path_value)
        if not path.is_absolute():
            if artifact_root is None:
                raise ValueError(
                    "entropy diagnostics numeric artifact requires its artifact root for restore"
                )
            path = Path(artifact_root) / path
        if not path.is_file() or sha256_file(path, required=True) != artifact.get("sha256"):
            raise ValueError("entropy diagnostics numeric artifact is missing or corrupt")
        payload = path.read_bytes()
    else:
        raise ValueError("entropy diagnostics numeric artifact storage is unsupported")
    if sha256_bytes(payload) != artifact.get("sha256"):
        raise ValueError("entropy diagnostics numeric artifact checksum is invalid")
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            probabilities = np.asarray(archive["p"], dtype=np.float64)
            final_q = np.asarray(archive["q"], dtype=np.float64)
    except (KeyError, OSError, ValueError, EOFError) as exc:
        raise ValueError("entropy diagnostics numeric artifact cannot be read") from exc
    if (
        probabilities.dtype.str != "<f8"
        or final_q.dtype.str != "<f8"
        or probabilities.ndim != 1
        or final_q.shape != probabilities.shape
        or probabilities.shape[0] != len(row_ids)
        or artifact.get("row_count") != len(row_ids)
        or artifact.get("row_ids_sha256") != sha256_canonical_json(list(row_ids))
    ):
        raise ValueError("entropy diagnostics numeric artifact row identity or shape is invalid")
    array_records = artifact.get("arrays")
    if not isinstance(array_records, Mapping):
        raise ValueError("entropy diagnostics numeric array metadata is missing")
    for name, array in (("p", probabilities), ("q", final_q)):
        record = array_records.get(name)
        if (
            not isinstance(record, Mapping)
            or record.get("dtype") != array.dtype.str
            or record.get("shape") != list(array.shape)
            or record.get("sha256") != _array_digest(array)
        ):
            raise ValueError(f"entropy diagnostics numeric array {name} is corrupt")
    probabilities.setflags(write=False)
    final_q.setflags(write=False)
    return probabilities, final_q


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    quantile: float,
    weights: Sequence[float] | np.ndarray | None = None,
    *,
    original_indices: Sequence[int] | np.ndarray | None = None,
) -> float:
    """Return a deterministic weighted empirical quantile.

    Ties are ordered by the original row index.  The returned value is the
    smallest observed value whose cumulative normalized weight reaches the
    requested probability, including the exact ``u=0`` minimum convention.
    """

    u = _finite(quantile, name="quantile")
    if not 0.0 <= u <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("quantile values must be a non-empty finite vector")
    if weights is None:
        mass = np.ones(data.shape[0], dtype=np.float64)
    else:
        mass = np.asarray(weights, dtype=np.float64)
        if mass.shape != data.shape or not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError("quantile weights must be finite and strictly positive")
    if original_indices is None:
        indices = np.arange(data.shape[0], dtype=np.int64)
    else:
        indices = np.asarray(original_indices, dtype=np.int64)
        if indices.shape != data.shape:
            raise ValueError("quantile original_indices must align with values")
    order = np.lexsort((indices, data))
    ordered_values = data[order]
    ordered_weights = mass[order]
    total = float(np.sum(ordered_weights, dtype=np.float64))
    target = u * total
    position = int(np.searchsorted(np.cumsum(ordered_weights), target, side="left"))
    return float(ordered_values[min(position, len(ordered_values) - 1)])


@dataclass(frozen=True, slots=True)
class DiagnosticDistribution:
    """A defined numerical distribution with population and weighting labels."""

    count: int
    minimum: float | None
    median: float | None
    mean: float | None
    q95: float | None
    q99: float | None
    maximum: float | None
    unit: str
    population: str
    weighting: str
    denominator: str
    tolerance: float = DIAGNOSTICS_NUMERICAL_TOLERANCE
    quantile_convention: str = QUANTILE_CONVENTION

    @property
    def q50(self) -> float | None:
        return self.median

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or int(self.count) != self.count or self.count < 0:
            raise ValueError("diagnostic distribution count must be a non-negative integer")
        if not self.unit.strip() or not self.population.strip() or not self.weighting.strip():
            raise ValueError("diagnostic distribution labels must be non-blank")
        if not self.denominator.strip() or not self.quantile_convention.strip():
            raise ValueError(
                "diagnostic distribution denominator and quantile convention are required"
            )
        tolerance = _finite(self.tolerance, name="diagnostic distribution tolerance")
        if tolerance <= 0.0:
            raise ValueError("diagnostic distribution tolerance must be positive")
        for value in (
            self.minimum,
            self.median,
            self.mean,
            self.q95,
            self.q99,
            self.maximum,
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("diagnostic distribution values must be finite")
        if self.count == 0 and any(
            value is not None
            for value in (self.minimum, self.median, self.mean, self.q95, self.q99, self.maximum)
        ):
            raise ValueError("empty diagnostic distributions must not contain summary values")
        if self.count > 0 and any(
            value is None
            for value in (self.minimum, self.median, self.mean, self.q95, self.q99, self.maximum)
        ):
            raise ValueError("non-empty diagnostic distributions require all summary values")
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "tolerance", tolerance)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "minimum": self.minimum,
            "median": self.median,
            "mean": self.mean,
            "q50": self.median,
            "q95": self.q95,
            "q99": self.q99,
            "maximum": self.maximum,
            "unit": self.unit,
            "population": self.population,
            "weighting": self.weighting,
            "denominator": self.denominator,
            "tolerance": self.tolerance,
            "quantile_convention": self.quantile_convention,
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "DiagnosticDistribution":
        if not isinstance(value, Mapping):
            raise ValueError("diagnostic distribution manifest must be an object")
        return cls(
            count=int(value["count"]),
            minimum=None if value.get("minimum") is None else float(value["minimum"]),
            median=None if value.get("median") is None else float(value["median"]),
            mean=None if value.get("mean") is None else float(value["mean"]),
            q95=None if value.get("q95") is None else float(value["q95"]),
            q99=None if value.get("q99") is None else float(value["q99"]),
            maximum=None if value.get("maximum") is None else float(value["maximum"]),
            unit=str(value["unit"]),
            population=str(value["population"]),
            weighting=str(value["weighting"]),
            denominator=str(value["denominator"]),
            tolerance=float(value.get("tolerance", DIAGNOSTICS_NUMERICAL_TOLERANCE)),
            quantile_convention=str(value.get("quantile_convention", QUANTILE_CONVENTION)),
        )


def _distribution(
    values: Sequence[float] | np.ndarray,
    *,
    population: str,
    weighting: str,
    denominator: str,
    unit: str = "dimensionless",
    weights: Sequence[float] | np.ndarray | None = None,
    original_indices: Sequence[int] | np.ndarray | None = None,
) -> DiagnosticDistribution:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not np.all(np.isfinite(data)):
        raise ValueError(f"{population} values must be a finite vector")
    if data.size == 0:
        return DiagnosticDistribution(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            unit,
            population,
            weighting,
            denominator,
        )
    if weights is None:
        mass = np.ones(data.shape[0], dtype=np.float64)
    else:
        mass = np.asarray(weights, dtype=np.float64)
        if mass.shape != data.shape or not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError(f"{population} weights must be finite and positive")
    total = float(np.sum(mass, dtype=np.float64))
    mean = float(np.sum(data * mass, dtype=np.float64) / total)
    return DiagnosticDistribution(
        int(data.size),
        float(np.min(data)),
        weighted_quantile(data, 0.5, mass, original_indices=original_indices),
        mean,
        weighted_quantile(data, 0.95, mass, original_indices=original_indices),
        weighted_quantile(data, 0.99, mass, original_indices=original_indices),
        float(np.max(data)),
        unit,
        population,
        weighting,
        denominator,
    )


@dataclass(frozen=True, slots=True)
class AtomicCoverageDiagnostics:
    """Nearest-selected atomic-row coverage for one explicitly named population."""

    population: str
    candidate_ids: tuple[str, ...]
    target_row_indices: tuple[int, ...]
    distances: tuple[float, ...]
    conditional_mass: float
    summary: DiagnosticDistribution
    source_candidate_support: DiagnosticDistribution

    def __post_init__(self) -> None:
        if not self.population.strip():
            raise ValueError("coverage population must be non-blank")
        if len(self.target_row_indices) != len(self.distances):
            raise ValueError("coverage target rows and distances must align")
        if any(int(value) < 0 for value in self.target_row_indices):
            raise ValueError("coverage target row indices must be non-negative")
        distances = _tuple_floats(self.distances, name="coverage distance")
        if any(value < 0.0 for value in distances):
            raise ValueError("coverage distances must be non-negative")
        mass = _finite(self.conditional_mass, name="coverage conditional mass")
        if mass < 0.0:
            raise ValueError("coverage conditional mass must be non-negative")
        object.__setattr__(
            self, "target_row_indices", tuple(int(v) for v in self.target_row_indices)
        )
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "conditional_mass", mass)
        object.__setattr__(self, "candidate_ids", tuple(str(v) for v in self.candidate_ids))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "population": self.population,
            "candidate_ids": list(self.candidate_ids),
            "target_row_indices": list(self.target_row_indices),
            "distances": list(self.distances),
            "distance_unit": "whitened_euclidean",
            "conditional_mass": self.conditional_mass,
            "summary": self.summary.to_manifest(),
            "source_candidate_support": self.source_candidate_support.to_manifest(),
        }


@dataclass(frozen=True, slots=True)
class GraphDiagnostics:
    """Read-only graph and memory accounting, with distinct byte categories."""

    atomic_row_count: int
    edge_count: int
    edge_per_row: float
    edge_density: float
    self_edge_count: int
    source_mass_deviation: float
    source_support: DiagnosticDistribution
    candidate_entry_count: int
    candidate_support: DiagnosticDistribution
    candidate_pmf_max_deviation: float
    csr_array_bytes: int
    candidate_csr_array_bytes: int
    indexed_workspace_bytes: int | None
    measured_peak_memory_bytes: int | None
    estimated_peak_memory_bytes: int | None
    backend: str
    metric: str
    backend_version: str
    kernel_bandwidth_rule: str = KERNEL_BANDWIDTH_RULE_SYMMETRIC_MAX
    kernel_bandwidth_rule_version: str = ""
    numerical_tolerance: float = DIAGNOSTICS_NUMERICAL_TOLERANCE
    atomic_graph_materialized: bool = True
    atomic_graph_csr_bytes: int = -1
    kernel_operator_fingerprint: str | None = None
    contribution_spool_bytes: int | None = None
    candidate_storage: str = "memory"
    candidate_storage_path: str | None = None
    candidate_resident_bytes: int = 0
    operational_metrics: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        def nonnegative_integer(value: Any, *, name: str) -> int:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"graph {name} must be an integer")
            normalized = int(value)
            if normalized < 0:
                raise ValueError(f"graph {name} must be non-negative")
            return normalized

        for name in (
            "atomic_row_count",
            "edge_count",
            "self_edge_count",
            "candidate_entry_count",
            "csr_array_bytes",
            "candidate_csr_array_bytes",
        ):
            object.__setattr__(
                self,
                name,
                nonnegative_integer(getattr(self, name), name=name),
            )
        if self.atomic_row_count < 1 or self.edge_count < 1:
            raise ValueError("graph diagnostics require a non-empty graph")
        for name in (
            "edge_per_row",
            "edge_density",
            "source_mass_deviation",
            "candidate_pmf_max_deviation",
        ):
            if _finite(getattr(self, name), name=f"graph {name}") < 0.0:
                raise ValueError(f"graph {name} must be non-negative")
        for name in (
            "indexed_workspace_bytes",
            "measured_peak_memory_bytes",
            "estimated_peak_memory_bytes",
            "contribution_spool_bytes",
            "candidate_resident_bytes",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    nonnegative_integer(value, name=name),
                )
        if not self.backend.strip() or not self.metric.strip() or not self.backend_version.strip():
            raise ValueError("graph backend identity is required")
        rule = normalize_kernel_bandwidth_rule(self.kernel_bandwidth_rule)
        expected_rule_version = kernel_bandwidth_rule_version(rule)
        if self.kernel_bandwidth_rule_version not in {"", expected_rule_version}:
            raise ValueError("graph kernel bandwidth rule version is incompatible")
        object.__setattr__(self, "kernel_bandwidth_rule", rule)
        object.__setattr__(self, "kernel_bandwidth_rule_version", expected_rule_version)
        if not isinstance(self.atomic_graph_materialized, bool):
            raise ValueError("atomic_graph_materialized must be boolean")
        atomic_bytes = int(self.atomic_graph_csr_bytes)
        if atomic_bytes == -1:
            atomic_bytes = self.csr_array_bytes if self.atomic_graph_materialized else 0
        if atomic_bytes < 0:
            raise ValueError("atomic_graph_csr_bytes must be non-negative")
        if self.atomic_graph_materialized and atomic_bytes != self.csr_array_bytes:
            raise ValueError("materialized graph bytes must match csr_array_bytes")
        if not self.atomic_graph_materialized and (atomic_bytes != 0 or self.csr_array_bytes != 0):
            raise ValueError("streamed diagnostics must report zero atomic graph CSR bytes")
        object.__setattr__(self, "atomic_graph_csr_bytes", atomic_bytes)
        if (
            self.kernel_operator_fingerprint is not None
            and not self.kernel_operator_fingerprint.strip()
        ):
            raise ValueError("kernel_operator_fingerprint must be non-blank when supplied")
        if not isinstance(self.candidate_storage, str) or self.candidate_storage not in {
            "memory",
            "disk",
        }:
            raise ValueError("candidate_storage must be 'memory' or 'disk'")
        if self.candidate_storage_path is not None and not isinstance(
            self.candidate_storage_path, str
        ):
            raise ValueError("candidate_storage_path must be a string when supplied")
        if self.candidate_storage == "memory" and self.candidate_storage_path is not None:
            raise ValueError("memory candidate storage must not have a storage path")
        if self.candidate_storage == "disk" and (
            self.candidate_storage_path is None or not self.candidate_storage_path.strip()
        ):
            raise ValueError("disk candidate storage requires a non-empty storage path")
        if not isinstance(self.operational_metrics, dict):
            raise ValueError("graph operational_metrics must be a dictionary")
        object.__setattr__(self, "operational_metrics", dict(self.operational_metrics))

    def to_manifest(self) -> dict[str, Any]:
        manifest = {
            "atomic_row_count": self.atomic_row_count,
            "edge_count": self.edge_count,
            "edge_per_row": self.edge_per_row,
            "edge_density": self.edge_density,
            "self_edge_count": self.self_edge_count,
            "source_mass_deviation": self.source_mass_deviation,
            "source_support": self.source_support.to_manifest(),
            "candidate_entry_count": self.candidate_entry_count,
            "candidate_support": self.candidate_support.to_manifest(),
            "candidate_pmf_max_deviation": self.candidate_pmf_max_deviation,
            "memory": {
                "csr_array_bytes": self.csr_array_bytes,
                "candidate_csr_array_bytes": self.candidate_csr_array_bytes,
                "indexed_workspace_bytes": self.indexed_workspace_bytes,
                "measured_peak_memory_bytes": self.measured_peak_memory_bytes,
                "estimated_peak_memory_bytes": self.estimated_peak_memory_bytes,
                "contribution_spool_bytes": self.contribution_spool_bytes,
                "candidate_storage": self.candidate_storage,
                "candidate_storage_path": self.candidate_storage_path,
                "candidate_resident_bytes": self.candidate_resident_bytes,
            },
            "backend": self.backend,
            "metric": self.metric,
            "backend_version": self.backend_version,
            "atomic_graph_materialized": self.atomic_graph_materialized,
            "atomic_graph_csr_bytes": self.atomic_graph_csr_bytes,
            "kernel_operator_fingerprint": self.kernel_operator_fingerprint,
            "numerical_tolerance": self.numerical_tolerance,
            "operations": dict(self.operational_metrics),
            "kernel_bandwidth_rule": self.kernel_bandwidth_rule,
            "kernel_bandwidth_rule_version": self.kernel_bandwidth_rule_version,
        }
        return manifest

    @property
    def E(self) -> int:
        return self.edge_count

    @property
    def Q(self) -> int:
        return self.candidate_entry_count


@dataclass(frozen=True, slots=True)
class EntropyScientificDiagnostics:
    """The versioned, scientific closure record for one entropy selection."""

    schema_version: str
    candidate_ids: tuple[str, ...]
    structure_ids: tuple[str, ...]
    row_ids: tuple[str, ...]
    row_candidate_ids: tuple[str, ...]
    candidate_count: int
    atomic_row_count: int
    descriptor_dimension_before: int
    descriptor_dimension_after: int
    descriptor_backend: str
    magnetic_mode: str
    representation_fingerprint: str
    pool_fingerprint: str
    transform_fingerprint: str
    whitening_manifest_json: str
    radius_summary: DiagnosticDistribution
    bandwidth_summary: DiagnosticDistribution
    calibration: BandwidthCalibrationResult | None
    graph: GraphDiagnostics
    selected_candidate_ids: tuple[str, ...]
    acquisition_order: tuple[str, ...]
    test_candidate_ids: tuple[str, ...]
    target_probabilities: np.ndarray
    final_q: np.ndarray
    beta: float
    final_objective: float
    final_cross_entropy: float
    final_shannon_entropy: float
    final_forward_kl: float
    final_support_mass: float
    target_mass: float
    q_mass: float
    mass_discrepancy: float
    coverage: tuple[AtomicCoverageDiagnostics, ...]
    provenance_json: str
    reference_measure_version: str
    reference_measure_config_json: str
    reference_weights_fingerprint: str
    reference_measure_diagnostics_json: str
    scientific_fingerprint: str = ""
    kernel_operator_fingerprint: str | None = None
    tolerance: float = DIAGNOSTICS_NUMERICAL_TOLERANCE
    # These are validated, immutable upstream records.  They are intentionally
    # excluded from ``to_manifest`` and from the scientific fingerprint.
    pool: EntropyPool | None = field(default=None, repr=False, compare=False)
    bandwidths: FrozenBandwidths | None = field(default=None, repr=False, compare=False)
    graph_record: SparseAtomicKernelGraph | None = field(default=None, repr=False, compare=False)
    contributions: SparseCandidateContributions | OutOfCoreSparseCandidateContributions | None = (
        field(default=None, repr=False, compare=False)
    )
    entropy_result: EntropySelectionResult | None = field(default=None, repr=False, compare=False)
    calibration_manifest_json: str = field(default="", repr=False, compare=False)
    graph_source_fingerprint: str | None = field(default=None, repr=False, compare=False)
    contributions_source_fingerprint: str | None = field(default=None, repr=False, compare=False)
    selection_history: EntropySelectionHistory | None = field(
        default=None, repr=False, compare=False
    )
    selection_history_json: str = field(default="", repr=False, compare=False)
    # Prefix metrics are additive presentation evidence.  They are deliberately
    # excluded from the v3 scientific fingerprint so old scientific identities
    # remain stable while reports become reproducible from persisted inputs.
    prefix_coverage_json: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != DIAGNOSTICS_SCHEMA_VERSION:
            raise ValueError("unsupported entropy diagnostics schema version")
        if self.calibration is None and not self.calibration_manifest_json:
            raise ValueError("entropy diagnostics require the complete calibration trace")
        ids = tuple(str(value) for value in self.candidate_ids)
        structures = tuple(str(value) for value in self.structure_ids)
        rows = tuple(str(value) for value in self.row_ids)
        owners = tuple(str(value) for value in self.row_candidate_ids)
        if not ids or len(set(ids)) != len(ids) or len(structures) != len(ids):
            raise ValueError("diagnostics candidate and structure identities are invalid")
        if len(rows) != len(owners):
            raise ValueError("diagnostics row identities and owners must align")
        if any(value not in ids for value in owners):
            raise ValueError("diagnostics row owner is absent from candidate identities")
        if int(self.candidate_count) != len(ids) or int(self.atomic_row_count) != len(rows):
            raise ValueError("diagnostics dimensions do not match ordered identities")
        if int(self.descriptor_dimension_before) < 1 or int(self.descriptor_dimension_after) < 1:
            raise ValueError("diagnostics descriptor dimensions must be positive")
        for name, value in (
            ("descriptor_backend", self.descriptor_backend),
            ("magnetic_mode", self.magnetic_mode),
            ("representation_fingerprint", self.representation_fingerprint),
            ("pool_fingerprint", self.pool_fingerprint),
            ("transform_fingerprint", self.transform_fingerprint),
        ):
            if not str(value).strip():
                raise ValueError(f"diagnostics {name} is required")
        if not str(self.reference_measure_version).strip():
            raise ValueError("diagnostics reference-measure version is required")
        for name, raw_value in (
            ("reference_measure_config_json", self.reference_measure_config_json),
            ("reference_measure_diagnostics_json", self.reference_measure_diagnostics_json),
        ):
            try:
                parsed = json.loads(raw_value)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"diagnostics {name} must contain JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"diagnostics {name} must contain an object")
        if not str(self.reference_weights_fingerprint).strip():
            raise ValueError("diagnostics reference weights fingerprint is required")
        target = _tuple_floats(self.target_probabilities, name="target probability")
        final_q = _tuple_floats(self.final_q, name="final q")
        if len(target) != len(rows) or len(final_q) != len(rows):
            raise ValueError("diagnostics p and q arrays must match atomic row count")
        if np.any(target <= 0.0) or np.any(final_q <= 0.0):
            raise ValueError("diagnostics p and q arrays must be strictly positive")
        beta = _finite(self.beta, name="diagnostics beta")
        if beta <= 0.0:
            raise ValueError("diagnostics beta must be positive")
        tolerance = _finite(self.tolerance, name="diagnostics tolerance")
        if tolerance <= 0.0:
            raise ValueError("diagnostics tolerance must be positive")
        metrics = recompute_final_entropy_metrics(
            target,
            final_q,
            beta=beta,
            selected_count=len(self.selected_candidate_ids),
            tolerance=tolerance,
        )
        for name, expected in (
            ("final_support_mass", metrics["support_mass"]),
            ("target_mass", metrics["target_mass"]),
            ("q_mass", metrics["q_mass"]),
            (
                "mass_discrepancy",
                abs(metrics["support_mass"] - (beta + len(self.selected_candidate_ids))),
            ),
        ):
            if not math.isclose(
                float(getattr(self, name)), expected, rel_tol=0.0, abs_tol=tolerance
            ):
                raise ValueError(f"diagnostics {name} cannot be independently recomputed")
        for name in (
            "final_objective",
            "final_cross_entropy",
            "final_shannon_entropy",
            "final_forward_kl",
        ):
            metric_name = {
                "final_objective": "objective",
                "final_cross_entropy": "cross_entropy",
                "final_shannon_entropy": "shannon_entropy",
                "final_forward_kl": "forward_kl",
            }[name]
            if not math.isclose(
                float(getattr(self, name)),
                metrics[metric_name],
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError(f"diagnostics {name} cannot be independently recomputed")
        if self.final_forward_kl < -tolerance:
            raise ValueError("diagnostics forward KL is materially negative")
        operator_fingerprint = self.kernel_operator_fingerprint
        if operator_fingerprint is None:
            operator_fingerprint = self.graph_source_fingerprint
        if operator_fingerprint is None and self.graph_record is not None:
            operator_fingerprint = self.graph_record.fingerprint
        if not isinstance(operator_fingerprint, str) or not operator_fingerprint.strip():
            raise ValueError("diagnostics kernel operator identity is required")
        selected = tuple(str(value) for value in self.selected_candidate_ids)
        acquisition = tuple(str(value) for value in self.acquisition_order)
        if len(set(selected)) != len(selected) or set(selected) != set(acquisition):
            raise ValueError("diagnostics selected membership and acquisition order disagree")
        if any(value not in ids for value in selected + tuple(self.test_candidate_ids)):
            raise ValueError("diagnostics selected/test candidate identity is unknown")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "structure_ids", structures)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "row_candidate_ids", owners)
        object.__setattr__(self, "candidate_count", len(ids))
        object.__setattr__(self, "atomic_row_count", len(rows))
        object.__setattr__(
            self, "descriptor_dimension_before", int(self.descriptor_dimension_before)
        )
        object.__setattr__(self, "descriptor_dimension_after", int(self.descriptor_dimension_after))
        object.__setattr__(self, "selected_candidate_ids", selected)
        object.__setattr__(self, "acquisition_order", acquisition)
        object.__setattr__(
            self, "test_candidate_ids", tuple(str(v) for v in self.test_candidate_ids)
        )
        object.__setattr__(self, "target_probabilities", target)
        object.__setattr__(self, "final_q", final_q)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "kernel_operator_fingerprint", operator_fingerprint)
        object.__setattr__(self, "target_mass", float(np.sum(target, dtype=np.float64)))
        object.__setattr__(self, "q_mass", float(np.sum(final_q, dtype=np.float64)))
        object.__setattr__(self, "final_support_mass", metrics["support_mass"])
        object.__setattr__(
            self, "mass_discrepancy", abs(metrics["support_mass"] - (beta + len(selected)))
        )
        fingerprint = self._fingerprint()
        if self.scientific_fingerprint and self.scientific_fingerprint != fingerprint:
            raise ValueError("entropy diagnostics fingerprint does not match its scientific values")
        object.__setattr__(self, "scientific_fingerprint", fingerprint)

    def _fingerprint_payload(self) -> dict[str, Any]:
        calibration_record = self._calibration_manifest()
        if calibration_record is not None:
            calibration_record = dict(calibration_record)
            if self.kernel_bandwidth_rule == KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE:
                # Preserve the pre-rule scientific fingerprint while still
                # exposing explicit legacy provenance in the user manifest.
                calibration_record.pop("kernel_bandwidth_rule", None)
                calibration_record.pop("kernel_bandwidth_rule_version", None)
        graph_record = self.graph.to_manifest()
        graph_memory = graph_record.get("memory")
        if isinstance(graph_memory, dict):
            # Storage and spool values are publication telemetry, not scientific
            # identity. Keep fingerprints compatible with earlier v3 records.
            graph_memory = dict(graph_memory)
            graph_memory.pop("contribution_spool_bytes", None)
            graph_memory.pop("candidate_storage", None)
            graph_memory.pop("candidate_storage_path", None)
            graph_memory.pop("candidate_resident_bytes", None)
            graph_record["memory"] = graph_memory
        # Runtime timings, memory observations, and provider counters are
        # evidence, not scientific identity.
        graph_record.pop("operations", None)
        if self.kernel_bandwidth_rule == KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE:
            graph_record.pop("kernel_bandwidth_rule", None)
            graph_record.pop("kernel_bandwidth_rule_version", None)
        return {
            "schema_version": self.schema_version,
            "candidate_ids": list(self.candidate_ids),
            "structure_ids": list(self.structure_ids),
            "row_ids": list(self.row_ids),
            "row_candidate_ids": list(self.row_candidate_ids),
            "dimensions": {
                "M": self.candidate_count,
                "N": self.atomic_row_count,
                "d": self.descriptor_dimension_before,
                "d_prime": self.descriptor_dimension_after,
            },
            "descriptor_backend": self.descriptor_backend,
            "magnetic_mode": self.magnetic_mode,
            "representation_fingerprint": self.representation_fingerprint,
            "pool_fingerprint": self.pool_fingerprint,
            "transform_fingerprint": self.transform_fingerprint,
            "source_fingerprints": {
                "reference_weights": self.reference_weights_fingerprint,
                "calibration": (
                    None
                    if calibration_record is None
                    else calibration_record.get("calibration_fingerprint")
                ),
                "graph": (None if self.graph_record is None else self.graph_record.fingerprint),
                "kernel_operator": self.kernel_operator_fingerprint,
                "contributions": self.contributions_source_fingerprint
                if self.contributions_source_fingerprint is not None
                else (None if self.contributions is None else self.contributions.fingerprint),
            },
            "whitening_manifest": json.loads(self.whitening_manifest_json),
            "radius_summary": self.radius_summary.to_manifest(),
            "bandwidth_summary": self.bandwidth_summary.to_manifest(),
            "calibration": self._calibration_manifest(),
            "graph": graph_record,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "acquisition_order": list(self.acquisition_order),
            "test_candidate_ids": list(self.test_candidate_ids),
            "selection_history": self._selection_history_manifest(),
            "p_sha256": _array_digest(self.target_probabilities),
            "q_sha256": _array_digest(self.final_q),
            "numeric_array_shape": [self.atomic_row_count],
            "beta": self.beta,
            "final_objective": self.final_objective,
            "final_cross_entropy": self.final_cross_entropy,
            "final_shannon_entropy": self.final_shannon_entropy,
            "final_forward_kl": self.final_forward_kl,
            "coverage": [item.to_manifest() for item in self.coverage],
            "provenance": json.loads(self.provenance_json),
            "reference_measure": {
                "version": self.reference_measure_version,
                "config": json.loads(self.reference_measure_config_json),
                "weights_fingerprint": self.reference_weights_fingerprint,
                "diagnostics": json.loads(self.reference_measure_diagnostics_json),
            },
            "tolerance": self.tolerance,
        }

    def _fingerprint(self) -> str:
        return sha256_canonical_json(self._fingerprint_payload())

    def _calibration_manifest(self) -> dict[str, Any] | None:
        if self.calibration is not None:
            return calibration_manifest(self.calibration)
        if self.calibration_manifest_json:
            value = json.loads(self.calibration_manifest_json)
            if not isinstance(value, dict):
                raise ValueError("restored calibration manifest is not an object")
            return value
        return None

    def _selection_history_manifest(self) -> list[dict[str, Any]]:
        if self.selection_history is not None:
            return [dict(asdict(step)) for step in self.selection_history.steps]
        if self.selection_history_json:
            value = json.loads(self.selection_history_json)
            if not isinstance(value, list):
                raise ValueError("restored selection history is not an array")
            return value
        return []

    def _prefix_coverage_manifest(self) -> dict[str, Any] | None:
        if not self.prefix_coverage_json:
            return None
        value = json.loads(self.prefix_coverage_json)
        if not isinstance(value, dict):
            raise ValueError("restored prefix coverage manifest is not an object")
        return value

    @property
    def fingerprint(self) -> str:
        return self.scientific_fingerprint

    @property
    def kernel_bandwidth_rule(self) -> str:
        """Return the persisted scientific pair-bandwidth rule."""

        if self.bandwidths is not None:
            return self.bandwidths.kernel_bandwidth_rule
        if self.calibration is not None:
            return self.calibration.kernel_bandwidth_rule
        manifest = self._calibration_manifest()
        if manifest is not None:
            return normalize_kernel_bandwidth_rule(
                manifest.get("kernel_bandwidth_rule", KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE)
            )
        return KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE

    @property
    def whitening_manifest(self) -> dict[str, Any]:
        return json.loads(self.whitening_manifest_json)

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self.provenance_json)

    @property
    def composition(self) -> dict[str, Any]:
        return dict(self.provenance.get("composition", {}))

    @property
    def provenance_diagnostics(self) -> dict[str, Any]:
        return self.provenance

    @property
    def magnetic(self) -> dict[str, Any]:
        return dict(self.provenance.get("magnetic_state", {}))

    @property
    def M(self) -> int:
        return self.candidate_count

    @property
    def N(self) -> int:
        return self.atomic_row_count

    @property
    def d(self) -> int:
        return self.descriptor_dimension_before

    @property
    def d_prime(self) -> int:
        return self.descriptor_dimension_after

    @property
    def radii(self) -> np.ndarray:
        if self.bandwidths is None:
            raise ValueError(
                "raw calibrated radii are unavailable in this restored diagnostics record"
            )
        values = np.asarray(self.bandwidths.radii, dtype=np.float64).copy()
        values.setflags(write=False)
        return values

    @property
    def source_bandwidths(self) -> np.ndarray:
        if self.bandwidths is None:
            raise ValueError(
                "raw calibrated source bandwidths are unavailable in this restored diagnostics record"
            )
        values = np.asarray(self.bandwidths.bandwidths, dtype=np.float64).copy()
        values.setflags(write=False)
        return values

    @property
    def selected_bandwidth_fingerprint(self) -> str:
        """Return the selected h_a identity, including after persisted restore."""

        if self.bandwidths is not None:
            return self.bandwidths.fingerprint
        calibration = self._calibration_manifest()
        if not isinstance(calibration, Mapping):
            raise ValueError("selected bandwidth identity is unavailable")
        selected = calibration.get("selected")
        if not isinstance(selected, Mapping):
            raise ValueError("selected bandwidth identity is unavailable")
        fingerprint = selected.get("bandwidth_fingerprint", selected.get("bandwidth"))
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("selected bandwidth identity is unavailable")
        return fingerprint

    @property
    def selection_history_records(self) -> list[dict[str, Any]]:
        return self._selection_history_manifest()

    @property
    def prefix_coverage_records(self) -> dict[str, Any] | None:
        """Return persisted exact prefix coverage metrics, if available."""

        return self._prefix_coverage_manifest()

    @property
    def calibration_trace(self) -> list[dict[str, Any]]:
        calibration = self._calibration_manifest()
        return [] if calibration is None else list(calibration["attempts"])

    def recompute_final(self) -> dict[str, float]:
        """Recompute final science from retained numeric ``p`` and ``q`` arrays."""

        return recompute_final_entropy_metrics(
            self.target_probabilities,
            self.final_q,
            beta=self.beta,
            selected_count=len(self.selected_candidate_ids),
            tolerance=self.tolerance,
        )

    def as_mapping(self) -> dict[str, Any]:
        return self.to_manifest()

    def __getitem__(self, key: str) -> Any:
        return self.to_manifest()[key]

    def to_manifest(self, artifact_dir: Path | None = None) -> dict[str, Any]:
        """Return the complete structured diagnostic payload."""

        calibration_record = self._calibration_manifest()
        if calibration_record is None:
            raise ValueError("entropy diagnostics require the complete calibration trace")
        if artifact_dir is None:
            numeric_artifact = _inline_diagnostic_arrays(
                self.target_probabilities,
                self.final_q,
                self.row_ids,
            )
        else:
            artifact_dir = Path(artifact_dir)
            artifact_path = artifact_dir / (
                f"entropy_diagnostic_arrays-{self.scientific_fingerprint}.npz"
            )
            artifact_sha256 = atomic_write_stream(
                artifact_path,
                lambda handle: np.savez(
                    handle,
                    p=self.target_probabilities,
                    q=self.final_q,
                ),
            )
            numeric_artifact = {
                "schema": DIAGNOSTICS_ARRAY_ARTIFACT_SCHEMA,
                "storage": "file",
                "path": artifact_path.name,
                "sha256": artifact_sha256,
                "row_count": self.atomic_row_count,
                "row_ids_sha256": sha256_canonical_json(list(self.row_ids)),
                "arrays": {
                    "p": _diagnostic_array_metadata(self.target_probabilities),
                    "q": _diagnostic_array_metadata(self.final_q),
                },
            }
        manifest = {
            "schema_version": self.schema_version,
            "scientific_fingerprint": self.scientific_fingerprint,
            "source_fingerprints": {
                "representation": self.representation_fingerprint,
                "pool": self.pool_fingerprint,
                "transform": self.transform_fingerprint,
                "reference_weights": self.reference_weights_fingerprint,
                "calibration": (
                    None
                    if calibration_record is None
                    else calibration_record.get("calibration_fingerprint")
                ),
                "graph": (None if self.graph_record is None else self.graph_record.fingerprint),
                "kernel_operator": self.kernel_operator_fingerprint,
                "contributions": self.contributions_source_fingerprint
                if self.contributions_source_fingerprint is not None
                else (None if self.contributions is None else self.contributions.fingerprint),
            },
            "identities": {
                "candidate_ids": list(self.candidate_ids),
                "structure_ids": list(self.structure_ids),
                "row_ids": list(self.row_ids),
                "row_candidate_ids": list(self.row_candidate_ids),
                "selected_candidate_ids": list(self.selected_candidate_ids),
                "acquisition_order": list(self.acquisition_order),
                "test_candidate_ids": list(self.test_candidate_ids),
            },
            "selection_history": self._selection_history_manifest(),
            "dimensions": {
                "candidate_count_M": self.candidate_count,
                "atomic_row_count_N": self.atomic_row_count,
                "descriptor_dimension_before_d": self.descriptor_dimension_before,
                "descriptor_dimension_after_d_prime": self.descriptor_dimension_after,
            },
            "representation": {
                "backend": self.descriptor_backend,
                "magnetic_mode": self.magnetic_mode,
                "fingerprint": self.representation_fingerprint,
                "whitening": self.whitening_manifest,
            },
            "reference_measure": {
                "version": self.reference_measure_version,
                "config": json.loads(self.reference_measure_config_json),
                "weights_fingerprint": self.reference_weights_fingerprint,
                "diagnostics": json.loads(self.reference_measure_diagnostics_json),
            },
            "bandwidth": {
                "radii": self.radius_summary.to_manifest(),
                "source_bandwidths": self.bandwidth_summary.to_manifest(),
                "relation": "h_a=c*r_{k,a}",
                "kernel_bandwidth_rule": self.kernel_bandwidth_rule,
                "kernel_bandwidth_rule_version": (
                    calibration_record.get("kernel_bandwidth_rule_version", "")
                ),
                "effective_pair_bandwidth": (
                    "H_ab=max(h_a,h_b)"
                    if self.kernel_bandwidth_rule == KERNEL_BANDWIDTH_RULE_SYMMETRIC_MAX
                    else "H_ab=h_a"
                ),
                "selected_k": calibration_record["selected"]["k"],
                "selected_c": calibration_record["selected"]["c"],
                "calibration": calibration_record,
            },
            "graph": self.graph.to_manifest(),
            "objective": {
                "beta": self.beta,
                "numeric_arrays": numeric_artifact,
                "objective_F": self.final_objective,
                "cross_entropy_nats": self.final_cross_entropy,
                "shannon_entropy_nats": self.final_shannon_entropy,
                "forward_kl_nats": self.final_forward_kl,
                "target_mass": self.target_mass,
                "q_mass": self.q_mass,
                "final_support_mass": self.final_support_mass,
                "mass_discrepancy": self.mass_discrepancy,
                "tolerance": self.tolerance,
            },
            "coverage": {item.population: item.to_manifest() for item in self.coverage},
            "provenance": self.provenance,
            "integrity": {
                "schema_version": self.schema_version,
                "scientific_fingerprint": self.scientific_fingerprint,
                "recomputation": "float64-p-q-v1",
            },
        }
        prefix_coverage = self._prefix_coverage_manifest()
        if prefix_coverage is not None:
            manifest["prefix_coverage"] = prefix_coverage
        return manifest

    @classmethod
    def from_manifest(
        cls,
        value: Mapping[str, Any],
        *,
        artifact_root: Path | None = None,
    ) -> "EntropyScientificDiagnostics":
        """Restore a diagnostics record and verify its independent ``p/q`` science."""

        try:
            source = value["source_fingerprints"]
            identity = value["identities"]
            dimensions = value["dimensions"]
            representation = value["representation"]
            objective = value["objective"]
            bandwidth = value["bandwidth"]
            graph_manifest = value["graph"]
        except (KeyError, TypeError) as exc:
            raise ValueError("entropy diagnostics manifest is missing required fields") from exc
        if not isinstance(source, Mapping) or not isinstance(identity, Mapping):
            raise ValueError("entropy diagnostics identities are malformed")
        if not isinstance(objective, Mapping) or not isinstance(representation, Mapping):
            raise ValueError("entropy diagnostics science payload is malformed")
        if not isinstance(graph_manifest, Mapping) or not isinstance(bandwidth, Mapping):
            raise ValueError("entropy diagnostics graph/bandwidth payload is malformed")
        reference_measure = value.get("reference_measure")
        if not isinstance(reference_measure, Mapping):
            raise ValueError("entropy diagnostics reference-measure payload is missing")
        for name in ("version", "config", "weights_fingerprint", "diagnostics"):
            if name not in reference_measure:
                raise ValueError("entropy diagnostics reference-measure payload is incomplete")
        if source.get("reference_weights") != reference_measure["weights_fingerprint"]:
            raise ValueError("entropy diagnostics reference-weight identity does not match")
        operator_source = source.get("kernel_operator", source.get("graph"))
        if (
            any(
                not isinstance(source.get(name), str) or not str(source.get(name)).strip()
                for name in ("representation", "pool", "transform", "calibration", "contributions")
            )
            or not isinstance(operator_source, str)
            or not operator_source.strip()
        ):
            raise ValueError("entropy diagnostics source fingerprints are incomplete")
        atomic_graph_materialized = bool(graph_manifest.get("atomic_graph_materialized", True))
        if not atomic_graph_materialized and "kernel_operator" not in source:
            raise ValueError("streamed diagnostics require an explicit kernel operator fingerprint")
        if atomic_graph_materialized and (
            not isinstance(source.get("graph"), str) or not str(source.get("graph")).strip()
        ):
            raise ValueError("materialized graph diagnostics require a graph fingerprint")
        if not atomic_graph_materialized and source.get("graph") is not None:
            raise ValueError("streamed diagnostics must not claim a materialized graph fingerprint")
        graph_operator = graph_manifest.get("kernel_operator_fingerprint")
        if not isinstance(graph_operator, str) or not graph_operator.strip():
            if not atomic_graph_materialized:
                raise ValueError("streamed diagnostics graph record lacks operator identity")
        elif graph_operator != operator_source:
            raise ValueError("entropy diagnostics operator identity does not match graph record")
        whitening = representation.get("whitening")
        if (
            not isinstance(whitening, Mapping)
            or whitening.get("fingerprint") != source["transform"]
        ):
            raise ValueError("entropy diagnostics whitening identity does not match its source")
        numeric_artifact = objective.get("numeric_arrays")
        if not isinstance(numeric_artifact, Mapping):
            raise ValueError(
                "legacy JSON-array diagnostics are unsupported; regenerate the completed result "
                "with the versioned numeric artifact schema"
            )
        probabilities, final_q = _load_diagnostic_arrays(
            numeric_artifact,
            tuple(identity.get("row_ids", ())),
            artifact_root=artifact_root,
        )
        graph = _graph_from_manifest(graph_manifest)
        coverage_values = value.get("coverage", {})
        if not isinstance(coverage_values, Mapping):
            raise ValueError("entropy diagnostics coverage payload is malformed")
        coverage = tuple(_coverage_from_manifest(item) for item in coverage_values.values())
        calibration_value = bandwidth.get("calibration")
        calibration = None
        calibration_json = ""
        if calibration_value is not None:
            calibration_from_manifest(calibration_value)
            calibration_json = json.dumps(calibration_value, sort_keys=True, separators=(",", ":"))
        history_value = value.get("selection_history", [])
        if not isinstance(history_value, list):
            raise ValueError("entropy diagnostics selection history is malformed")
        prefix_coverage_value = value.get("prefix_coverage")
        if prefix_coverage_value is not None and not isinstance(prefix_coverage_value, Mapping):
            raise ValueError("entropy diagnostics prefix coverage payload is malformed")
        diagnostics = cls(
            schema_version=str(value["schema_version"]),
            candidate_ids=tuple(identity["candidate_ids"]),
            structure_ids=tuple(identity["structure_ids"]),
            row_ids=tuple(identity["row_ids"]),
            row_candidate_ids=tuple(identity["row_candidate_ids"]),
            candidate_count=int(dimensions["candidate_count_M"]),
            atomic_row_count=int(dimensions["atomic_row_count_N"]),
            descriptor_dimension_before=int(dimensions["descriptor_dimension_before_d"]),
            descriptor_dimension_after=int(dimensions["descriptor_dimension_after_d_prime"]),
            descriptor_backend=str(representation["backend"]),
            magnetic_mode=str(representation["magnetic_mode"]),
            representation_fingerprint=str(source["representation"]),
            pool_fingerprint=str(source["pool"]),
            transform_fingerprint=str(source["transform"]),
            whitening_manifest_json=json.dumps(
                representation["whitening"], sort_keys=True, separators=(",", ":")
            ),
            radius_summary=DiagnosticDistribution.from_manifest(bandwidth["radii"]),
            bandwidth_summary=DiagnosticDistribution.from_manifest(bandwidth["source_bandwidths"]),
            calibration=calibration,
            graph=graph,
            selected_candidate_ids=tuple(identity["selected_candidate_ids"]),
            acquisition_order=tuple(identity["acquisition_order"]),
            test_candidate_ids=tuple(identity["test_candidate_ids"]),
            target_probabilities=probabilities,
            final_q=final_q,
            beta=float(objective["beta"]),
            final_objective=float(objective["objective_F"]),
            final_cross_entropy=float(objective["cross_entropy_nats"]),
            final_shannon_entropy=float(objective["shannon_entropy_nats"]),
            final_forward_kl=float(objective["forward_kl_nats"]),
            final_support_mass=float(objective["final_support_mass"]),
            target_mass=float(objective["target_mass"]),
            q_mass=float(objective["q_mass"]),
            mass_discrepancy=float(objective["mass_discrepancy"]),
            coverage=coverage,
            provenance_json=json.dumps(
                value.get("provenance", {}), sort_keys=True, separators=(",", ":")
            ),
            reference_measure_version=str(reference_measure["version"]),
            reference_measure_config_json=json.dumps(
                reference_measure["config"], sort_keys=True, separators=(",", ":")
            ),
            reference_weights_fingerprint=str(reference_measure["weights_fingerprint"]),
            reference_measure_diagnostics_json=json.dumps(
                reference_measure["diagnostics"], sort_keys=True, separators=(",", ":")
            ),
            scientific_fingerprint=str(value.get("scientific_fingerprint", "")),
            kernel_operator_fingerprint=str(operator_source),
            tolerance=float(objective.get("tolerance", DIAGNOSTICS_NUMERICAL_TOLERANCE)),
            calibration_manifest_json=calibration_json,
            graph_source_fingerprint=(
                None if source.get("graph") is None else str(source.get("graph"))
            ),
            contributions_source_fingerprint=(
                None if source.get("contributions") is None else str(source.get("contributions"))
            ),
            selection_history_json=json.dumps(history_value, sort_keys=True, separators=(",", ":")),
            prefix_coverage_json=(
                ""
                if prefix_coverage_value is None
                else json.dumps(prefix_coverage_value, sort_keys=True, separators=(",", ":"))
            ),
        )
        if value.get("scientific_fingerprint") != diagnostics.scientific_fingerprint:
            raise ValueError("entropy diagnostics scientific fingerprint does not match payload")
        if prefix_coverage_value is not None:
            restored_prefix_coverage = diagnostics.prefix_coverage_records
            if restored_prefix_coverage is None:
                raise ValueError("entropy diagnostics prefix coverage payload is unavailable")
            validate_prefix_coverage_manifest(restored_prefix_coverage, diagnostics)
        return diagnostics


def _build_prefix_coverage_manifest(
    probabilities: Sequence[float] | np.ndarray,
    *,
    beta: float,
    history: EntropySelectionHistory,
    contributions: SparseCandidateContributions | OutOfCoreSparseCandidateContributions,
    final_q: Sequence[float] | np.ndarray,
    final_forward_kl: float,
    tolerance: float = DIAGNOSTICS_NUMERICAL_TOLERANCE,
    pool_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Build exact support-union metrics in one acquisition-order pass.

    The candidate store is accessed only through ``candidate_row``.  This
    keeps the calculation bounded by the union of selected sparse supports and
    an O(N) covered bitset, rather than materialising or rescanning an MxN
    candidate matrix.
    """

    p = np.asarray(probabilities, dtype=np.float64)
    q_final = np.asarray(final_q, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("prefix coverage requires a finite positive target probability vector")
    if q_final.shape != p.shape or not np.all(np.isfinite(q_final)) or np.any(q_final <= 0.0):
        raise ValueError("prefix coverage final q does not align with target probabilities")
    beta_value = _finite(beta, name="prefix coverage beta")
    tolerance_value = _finite(tolerance, name="prefix coverage tolerance")
    if beta_value <= 0.0 or tolerance_value <= 0.0:
        raise ValueError("prefix coverage beta and tolerance must be positive")
    target_mass = float(np.sum(p, dtype=np.float64))
    if not math.isclose(target_mass, 1.0, rel_tol=0.0, abs_tol=tolerance_value):
        raise ValueError("prefix coverage target probabilities must sum to one")
    if contributions.n_targets != p.size:
        raise ValueError("prefix coverage contribution target count does not match p")
    if tuple(history.acquisition_order) != tuple(step.candidate_id for step in history.steps):
        raise ValueError("prefix coverage history order is malformed")

    covered = np.zeros(p.size, dtype=bool)
    selected_sum = np.zeros(p.size, dtype=np.float64)
    covered_mass = 0.0
    covered_rows = 0
    records: list[dict[str, Any]] = []

    def append_record(
        *,
        kind: str,
        candidate_ids: Sequence[str],
        acquisition_iteration: int,
        selected_count: int,
        cross_entropy: float | None,
        kl_total: float,
    ) -> None:
        nonlocal covered_mass, covered_rows
        uncovered_mass = 1.0 - covered_mass
        if covered_mass < -tolerance_value or covered_mass > 1.0 + tolerance_value:
            raise ValueError("prefix coverage target mass is outside [0, 1]")
        if uncovered_mass < -tolerance_value or uncovered_mass > 1.0 + tolerance_value:
            raise ValueError("prefix coverage uncovered target mass is outside [0, 1]")
        covered_mass_value = min(1.0, max(0.0, covered_mass))
        uncovered_mass_value = 1.0 - covered_mass_value
        if not math.isfinite(float(kl_total)):
            raise ValueError("prefix coverage history forward KL must be finite")
        if float(kl_total) < -tolerance_value:
            raise ValueError("prefix coverage total forward KL is materially negative")
        if selected_count == 0:
            expected_uncovered_kl = 0.0
        else:
            expected_uncovered_kl = uncovered_mass_value * math.log1p(
                selected_count / beta_value
            )
        covered_kl = float(kl_total) - expected_uncovered_kl
        if not math.isfinite(covered_kl):
            raise ValueError("prefix coverage signed covered KL is not finite")
        if not math.isclose(
            float(kl_total),
            covered_kl + expected_uncovered_kl,
            rel_tol=0.0,
            abs_tol=tolerance_value,
        ):
            raise ValueError("prefix coverage KL decomposition is inconsistent")
        records.append(
            {
                "kind": kind,
                "candidate_ids": [str(value) for value in candidate_ids],
                "acquisition_iteration": int(acquisition_iteration),
                "selected_count": int(selected_count),
                "covered_row_count": int(covered_rows),
                "uncovered_row_count": int(p.size - covered_rows),
                "row_coverage_fraction": float(covered_rows / p.size),
                "target_mass_covered": covered_mass_value,
                "target_mass_uncovered": uncovered_mass_value,
                "cross_entropy": None if cross_entropy is None else float(cross_entropy),
                "kl_total": float(kl_total),
                "kl_covered": covered_kl,
                "kl_uncovered": expected_uncovered_kl,
            }
        )

    anchor_steps = [step for step in history.steps if step.reason == "anchor"]
    greedy_steps = [step for step in history.steps if step.reason == "greedy"]
    if len(anchor_steps) + len(greedy_steps) != len(history.steps):
        raise ValueError("prefix coverage history contains an unsupported step reason")

    for step in anchor_steps:
        try:
            row = contributions.candidate_row(step.candidate_id)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"prefix coverage candidate support is missing for {step.candidate_id!r}"
            ) from exc
        targets = np.asarray(row.target_indices, dtype=np.int64)
        values = np.asarray(row.values, dtype=np.float64)
        if targets.size == 0 or np.any(targets < 0) or np.any(targets >= p.size):
            raise ValueError(f"prefix coverage support is invalid for {step.candidate_id!r}")
        new_targets = ~covered[targets]
        covered[targets[new_targets]] = True
        covered_mass += float(np.sum(p[targets[new_targets]], dtype=np.float64))
        covered_rows += int(np.count_nonzero(new_targets))
        np.add.at(selected_sum, targets, values)

    if anchor_steps:
        anchor = anchor_steps[0]
        if any(
            step.selected_count != anchor.selected_count
            or step.cross_entropy is None
            or anchor.cross_entropy is None
            or not math.isclose(
                float(step.cross_entropy),
                float(anchor.cross_entropy),
                rel_tol=0.0,
                abs_tol=tolerance_value,
            )
            or step.forward_kl is None
            or anchor.forward_kl is None
            or not math.isclose(
                float(step.forward_kl),
                float(anchor.forward_kl),
                rel_tol=0.0,
                abs_tol=tolerance_value,
            )
            for step in anchor_steps
        ):
            raise ValueError("prefix coverage anchor history is not one aggregate state")
        if anchor.selected_count != len(anchor_steps):
            raise ValueError("prefix coverage anchor selected_count is inconsistent")
        append_record(
            kind="anchor",
            candidate_ids=[step.candidate_id for step in anchor_steps],
            acquisition_iteration=0,
            selected_count=anchor.selected_count,
            cross_entropy=anchor.cross_entropy,
            kl_total=float(anchor.forward_kl),
        )
    else:
        # There is no saved greedy zero-state metric.  This point is explicitly
        # q_0=p, so it is a labelled baseline and does not claim support coverage.
        shannon_entropy = float(-np.sum(p * np.log(p), dtype=np.float64))
        append_record(
            kind="baseline",
            candidate_ids=(),
            acquisition_iteration=0,
            selected_count=0,
            cross_entropy=shannon_entropy,
            kl_total=0.0,
        )

    expected_iteration = 1
    expected_count = len(anchor_steps)
    for step in greedy_steps:
        if step.acquisition_iteration != expected_iteration:
            raise ValueError("prefix coverage greedy iterations are not contiguous")
        expected_count += 1
        if step.selected_count != expected_count:
            raise ValueError("prefix coverage greedy selected_count is inconsistent")
        if step.cross_entropy is None or step.forward_kl is None:
            raise ValueError("prefix coverage requires saved cross-entropy and forward KL history")
        try:
            row = contributions.candidate_row(step.candidate_id)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"prefix coverage candidate support is missing for {step.candidate_id!r}"
            ) from exc
        targets = np.asarray(row.target_indices, dtype=np.int64)
        values = np.asarray(row.values, dtype=np.float64)
        if targets.size == 0 or np.any(targets < 0) or np.any(targets >= p.size):
            raise ValueError(f"prefix coverage support is invalid for {step.candidate_id!r}")
        new_targets = ~covered[targets]
        covered[targets[new_targets]] = True
        covered_mass += float(np.sum(p[targets[new_targets]], dtype=np.float64))
        covered_rows += int(np.count_nonzero(new_targets))
        np.add.at(selected_sum, targets, values)
        append_record(
            kind="greedy",
            candidate_ids=(step.candidate_id,),
            acquisition_iteration=step.acquisition_iteration,
            selected_count=step.selected_count,
            cross_entropy=step.cross_entropy,
            kl_total=float(step.forward_kl),
        )
        expected_iteration += 1

    selected_ids = tuple(history.acquisition_order)
    if expected_count != len(selected_ids):
        raise ValueError("prefix coverage history selected count does not match acquisition order")
    denominator = beta_value + len(selected_ids)
    direct_q = (beta_value * p + selected_sum) / denominator
    if not np.allclose(direct_q, q_final, rtol=0.0, atol=tolerance_value):
        raise ValueError("prefix coverage terminal q does not match selected sparse supports")
    direct_kl = float(np.sum(p * np.log(p / direct_q), dtype=np.float64))
    if not math.isclose(direct_kl, float(final_forward_kl), rel_tol=0.0, abs_tol=tolerance_value):
        raise ValueError("prefix coverage terminal forward KL does not match final q")
    if records and not math.isclose(
        records[-1]["kl_total"], float(final_forward_kl), rel_tol=0.0, abs_tol=tolerance_value
    ):
        raise ValueError("prefix coverage terminal history does not match final forward KL")
    return {
        "schema_version": PREFIX_COVERAGE_SCHEMA_VERSION,
        "target_row_count": int(p.size),
        "target_mass": target_mass,
        "beta": beta_value,
        "pool_fingerprint": pool_fingerprint,
        "candidate_contributions_fingerprint": str(contributions.fingerprint),
        "acquisition_order": list(selected_ids),
        "records": records,
    }


def validate_prefix_coverage_manifest(
    value: Mapping[str, Any],
    diagnostics: EntropyScientificDiagnostics,
) -> None:
    """Validate persisted prefix metrics against restored v3 diagnostics."""

    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != PREFIX_COVERAGE_SCHEMA_VERSION
    ):
        raise ValueError("prefix coverage schema version is unsupported")
    tolerance = float(diagnostics.tolerance)
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("prefix coverage records are missing")
    if value.get("target_row_count") != diagnostics.atomic_row_count:
        raise ValueError("prefix coverage row count disagrees with diagnostics")
    if not math.isclose(
        float(value.get("target_mass")),
        diagnostics.target_mass,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("prefix coverage target mass disagrees with diagnostics")
    if not math.isclose(float(value.get("beta")), diagnostics.beta, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("prefix coverage beta disagrees with diagnostics")
    if value.get("pool_fingerprint") != diagnostics.pool_fingerprint:
        raise ValueError("prefix coverage pool identity disagrees with diagnostics")
    contribution_fingerprint = diagnostics.contributions_source_fingerprint
    if (
        not isinstance(value.get("candidate_contributions_fingerprint"), str)
        or not str(value.get("candidate_contributions_fingerprint")).strip()
        or contribution_fingerprint is None
        or value.get("candidate_contributions_fingerprint") != contribution_fingerprint
    ):
        raise ValueError("prefix coverage contribution identity disagrees with diagnostics")
    if tuple(value.get("acquisition_order", ())) != tuple(diagnostics.acquisition_order):
        raise ValueError("prefix coverage acquisition order disagrees with diagnostics")
    history = diagnostics.selection_history_records
    history_ids: list[str] = []
    greedy_started = False
    for index, item in enumerate(history):
        if not isinstance(item, Mapping):
            raise ValueError(f"prefix coverage history record {index} is malformed")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"prefix coverage history candidate {index} is invalid")
        if candidate_id in history_ids:
            raise ValueError("prefix coverage history candidate IDs are duplicated")
        history_ids.append(candidate_id)
        reason = item.get("reason")
        if reason == "anchor":
            if greedy_started or item.get("acquisition_iteration") != 0:
                raise ValueError("prefix coverage anchor history is out of order")
        elif reason == "greedy":
            greedy_started = True
        else:
            raise ValueError("prefix coverage history reason is invalid")
    if tuple(history_ids) != tuple(diagnostics.acquisition_order):
        raise ValueError("prefix coverage history order disagrees with diagnostics")
    anchors = [item for item in history if item.get("reason") == "anchor"]
    greedy = [item for item in history if item.get("reason") == "greedy"]
    for iteration, item in enumerate(greedy, start=1):
        if item.get("acquisition_iteration") != iteration:
            raise ValueError("prefix coverage greedy history iterations are not contiguous")
    expected_selected_count = len(anchors)
    for item in greedy:
        expected_selected_count += 1
        if item.get("selected_count") != expected_selected_count:
            raise ValueError("prefix coverage greedy selected_count is inconsistent")
    if anchors:
        anchor = anchors[0]
        for item in anchors:
            if item.get("selected_count") != anchor.get("selected_count"):
                raise ValueError("prefix coverage anchor selected_count is inconsistent")
            for name in ("cross_entropy", "forward_kl"):
                if item.get(name) is None or anchor.get(name) is None or not math.isclose(
                    float(item[name]), float(anchor[name]), rel_tol=0.0, abs_tol=tolerance
                ):
                    raise ValueError("prefix coverage anchor metrics are inconsistent")
        if anchor.get("selected_count") != len(anchors):
            raise ValueError("prefix coverage anchor count is inconsistent")
    expected_records = 1 + len(greedy)
    if len(records) != expected_records:
        raise ValueError("prefix coverage record count disagrees with selection history")
    previous_mass = -tolerance
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"prefix coverage record {index} is malformed")
        kind = record.get("kind")
        if kind not in {"baseline", "anchor", "greedy"}:
            raise ValueError(f"prefix coverage record {index} kind is invalid")
        for name in (
            "acquisition_iteration",
            "selected_count",
            "covered_row_count",
            "uncovered_row_count",
        ):
            item = record.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"prefix coverage record {index}.{name} is invalid")
        for name in (
            "row_coverage_fraction",
            "target_mass_covered",
            "target_mass_uncovered",
            "kl_total",
            "kl_covered",
            "kl_uncovered",
        ):
            number = float(record.get(name))
            if not math.isfinite(number):
                raise ValueError(f"prefix coverage record {index}.{name} is not finite")
        if record["kl_total"] < -tolerance:
            raise ValueError(f"prefix coverage record {index}.kl_total is materially negative")
        cross_entropy = record.get("cross_entropy")
        if cross_entropy is None or not math.isfinite(float(cross_entropy)):
            raise ValueError(f"prefix coverage record {index}.cross_entropy is invalid")
        if (
            record["covered_row_count"] + record["uncovered_row_count"]
            != diagnostics.atomic_row_count
        ):
            raise ValueError(f"prefix coverage record {index} row counts do not add to N")
        if not 0.0 - tolerance <= record["row_coverage_fraction"] <= 1.0 + tolerance:
            raise ValueError(f"prefix coverage record {index} row fraction is invalid")
        if not 0.0 - tolerance <= record["target_mass_covered"] <= 1.0 + tolerance:
            raise ValueError(f"prefix coverage record {index} covered mass is invalid")
        if not math.isclose(
            record["target_mass_uncovered"],
            1.0 - record["target_mass_covered"],
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(f"prefix coverage record {index} masses do not complement")
        if record["target_mass_covered"] + tolerance < previous_mass:
            raise ValueError("prefix coverage target mass is not monotone")
        previous_mass = record["target_mass_covered"]
        if not math.isclose(
            record["covered_row_count"] / diagnostics.atomic_row_count,
            record["row_coverage_fraction"],
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(f"prefix coverage record {index} row fraction is inconsistent")
        selected_count = record["selected_count"]
        expected_uncovered = record["target_mass_uncovered"] * math.log1p(
            selected_count / diagnostics.beta
        )
        if not math.isclose(
            record["kl_uncovered"],
            expected_uncovered,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(f"prefix coverage record {index} uncovered KL is inconsistent")
        if not math.isclose(
            record["kl_total"],
            record["kl_covered"] + record["kl_uncovered"],
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(f"prefix coverage record {index} KL components do not add")

    first = records[0]
    if anchors:
        if first["kind"] != "anchor" or first["candidate_ids"] != [
            item["candidate_id"] for item in anchors
        ]:
            raise ValueError("prefix coverage anchor aggregate is inconsistent")
        if first["acquisition_iteration"] != 0:
            raise ValueError("prefix coverage anchor iteration is inconsistent")
        if first["selected_count"] != len(anchors):
            raise ValueError("prefix coverage anchor count is inconsistent")
        history_points = [anchors[0]] + greedy
    else:
        if (
            first["kind"] != "baseline"
            or first["selected_count"] != 0
            or first["candidate_ids"] != []
            or first["acquisition_iteration"] != 0
        ):
            raise ValueError("prefix coverage baseline is inconsistent")
        baseline_shannon = float(
            -np.sum(
                diagnostics.target_probabilities * np.log(diagnostics.target_probabilities),
                dtype=np.float64,
            )
        )
        if not math.isclose(
            first["cross_entropy"], baseline_shannon, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError("prefix coverage baseline is not the explicit q0=p state")
        history_points = [None] + greedy
    for index, history_item in enumerate(history_points):
        record = records[index]
        if history_item is None:
            if record["kind"] != "baseline":
                raise ValueError("prefix coverage baseline kind is inconsistent")
            if not math.isclose(record["kl_total"], 0.0, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError("prefix coverage baseline KL must be zero")
            continue
        expected_kind = "anchor" if anchors and index == 0 else "greedy"
        if record["kind"] != expected_kind:
            raise ValueError("prefix coverage record kind is inconsistent with history")
        if record["kind"] == "greedy":
            if record["candidate_ids"] != [history_item["candidate_id"]]:
                raise ValueError("prefix coverage greedy identity is inconsistent")
            if record["acquisition_iteration"] != history_item["acquisition_iteration"]:
                raise ValueError("prefix coverage greedy iteration is inconsistent")
        if record["selected_count"] != history_item["selected_count"]:
            raise ValueError("prefix coverage history selected_count is inconsistent")
        if history_item.get("cross_entropy") is None or history_item.get("forward_kl") is None:
            raise ValueError("prefix coverage history metrics are incomplete")
        if not math.isclose(
            record["cross_entropy"],
            history_item["cross_entropy"],
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("prefix coverage cross-entropy disagrees with history")
        if not math.isclose(
            record["kl_total"],
            history_item["forward_kl"],
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("prefix coverage total KL disagrees with history")
    if not math.isclose(
        records[-1]["kl_total"],
        diagnostics.final_forward_kl,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("prefix coverage final KL disagrees with diagnostics")


def recompute_final_entropy_metrics(
    probabilities: Sequence[float] | np.ndarray,
    final_q: Sequence[float] | np.ndarray,
    *,
    beta: float,
    selected_count: int | None = None,
    tolerance: float = DIAGNOSTICS_NUMERICAL_TOLERANCE,
) -> dict[str, float]:
    """Independently derive ``F``, entropy and forward KL from ``p`` and ``q``."""

    p = np.asarray(probabilities, dtype=np.float64)
    q = np.asarray(final_q, dtype=np.float64)
    if p.ndim != 1 or q.shape != p.shape or p.size == 0:
        raise ValueError("independent entropy recomputation requires aligned non-empty p and q")
    if (
        not np.all(np.isfinite(p))
        or not np.all(np.isfinite(q))
        or np.any(p <= 0.0)
        or np.any(q <= 0.0)
    ):
        raise ValueError("independent entropy recomputation requires finite positive p and q")
    beta_value = _finite(beta, name="beta")
    tolerance_value = _finite(tolerance, name="tolerance")
    if beta_value <= 0.0 or tolerance_value <= 0.0:
        raise ValueError("beta and tolerance must be positive")
    target_mass = float(np.sum(p, dtype=np.float64))
    q_mass = float(np.sum(q, dtype=np.float64))
    if not math.isclose(target_mass, 1.0, rel_tol=0.0, abs_tol=tolerance_value):
        raise ValueError("independent entropy target probability mass is not one")
    if not math.isclose(q_mass, 1.0, rel_tol=0.0, abs_tol=tolerance_value):
        raise ValueError("independent entropy final q mass is not one")
    # q_A = s_A/(beta+K), so K is recovered from the retained final mass.
    if selected_count is None:
        selected_count_value = 1
    else:
        if (
            isinstance(selected_count, bool)
            or int(selected_count) != selected_count
            or selected_count < 0
        ):
            raise ValueError("selected_count must be a non-negative integer")
        selected_count_value = int(selected_count)
    denominator = beta_value + selected_count_value
    support = q * denominator
    # The actual denominator is not encoded in p/q alone.  Diagnostics callers
    # pass q from a fixed budget; the optional convention below is overridden by
    # the exact support mass in the caller when available.
    objective = float(np.sum(p * np.log(support), dtype=np.float64))
    cross_entropy = float(-np.sum(p * np.log(q), dtype=np.float64))
    shannon = float(-np.sum(p * np.log(p), dtype=np.float64))
    kl = cross_entropy - shannon
    if not all(math.isfinite(v) for v in (objective, cross_entropy, shannon, kl)):
        raise ValueError("independent entropy recomputation produced a non-finite value")
    if kl < -tolerance_value:
        raise ValueError("independent entropy recomputation produced materially negative KL")
    return {
        "objective": objective,
        "cross_entropy": cross_entropy,
        "shannon_entropy": shannon,
        "forward_kl": max(0.0, kl),
        "target_mass": target_mass,
        "q_mass": q_mass,
        "support_mass": denominator,
    }


def calibration_manifest(calibration: BandwidthCalibrationResult) -> dict[str, Any]:
    """Serialize the complete ordered calibration trace without telemetry."""

    manifest = {
        "mode": calibration.mode,
        "k_domain": list(calibration.k_domain),
        "c_domain": list(calibration.c_domain),
        "attempts": [calibration_attempt_manifest(attempt) for attempt in calibration.attempts],
        "selected": {
            "k": calibration.k,
            "c": calibration.c,
            "objective": calibration.objective,
            "bandwidth_fingerprint": calibration.selected.fingerprint,
        },
        "optimizer_id": calibration.optimizer_id,
        "optimizer_version": calibration.optimizer_version,
        "evaluation_count": calibration.evaluation_count,
        "termination_reason": calibration.termination_reason,
        "pool_fingerprint": calibration.pool_fingerprint,
        "calibration_fingerprint": calibration.calibration_fingerprint,
        "backend": calibration.backend,
        "backend_version": calibration.backend_version,
        "backend_fingerprint": calibration.backend_fingerprint,
        "metric": calibration.selected.metric,
        "kernel_bandwidth_rule": calibration.kernel_bandwidth_rule,
        "kernel_bandwidth_rule_version": (
            calibration.selected.kernel_bandwidth_rule_version
        ),
    }
    return manifest


def calibration_attempt_manifest(attempt: CalibrationAttempt) -> dict[str, Any]:
    return {
        "k": int(attempt.k),
        "c": float(attempt.c),
        "status": attempt.status,
        "objective": None if attempt.objective is None else float(attempt.objective),
        "reason": attempt.reason,
        "bandwidth_fingerprint": attempt.bandwidth_fingerprint,
    }


def calibration_from_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate persisted calibration completeness.

    Reconstructing the full frozen numerical object requires the upstream pool
    artifact, so restore validation intentionally returns no fake calibration
    object.  The trace is checked by :func:`validate_calibration_manifest`.
    """

    validate_calibration_manifest(value)
    return dict(value)


def validate_calibration_manifest(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("calibration manifest must be an object")
    rule = normalize_kernel_bandwidth_rule(
        value.get("kernel_bandwidth_rule", KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE)
    )
    expected_rule_version = kernel_bandwidth_rule_version(rule)
    supplied_rule_version = value.get("kernel_bandwidth_rule_version", "")
    if rule != KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE and supplied_rule_version != (
        expected_rule_version
    ):
        raise ValueError("symmetric_max calibration manifests require their rule version")
    if rule == KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE and supplied_rule_version not in {
        "",
        expected_rule_version,
    }:
        raise ValueError("calibration manifest has an incompatible kernel bandwidth rule version")
    attempts = value.get("attempts")
    k_domain = value.get("k_domain")
    c_domain = value.get("c_domain")
    if (
        not isinstance(attempts, list)
        or not isinstance(k_domain, list)
        or not isinstance(c_domain, list)
    ):
        raise ValueError("calibration manifest is missing its ordered trace")
    for item in attempts:
        if not isinstance(item, Mapping):
            raise ValueError("calibration trace contains a malformed attempt")
        status = item.get("status")
        if status not in {"valid", "invalid"}:
            raise ValueError("calibration attempt status must be valid or invalid")
        if status == "valid":
            objective = item.get("objective")
            if objective is None or not math.isfinite(float(objective)):
                raise ValueError("valid calibration attempts require a finite objective")
        elif item.get("objective") is not None or not str(item.get("reason", "")).strip():
            raise ValueError("invalid calibration attempts require a reason and no objective")
    expected = [(int(k), float(c)) for k in k_domain for c in c_domain]
    actual = [(int(item["k"]), float(item["c"])) for item in attempts if isinstance(item, Mapping)]
    if actual != expected:
        raise ValueError("calibration trace is incomplete or out of order")
    if int(value.get("evaluation_count", -1)) != len(attempts):
        raise ValueError("calibration evaluation_count does not match its trace")
    selected = value.get("selected")
    if not isinstance(selected, Mapping) or selected.get("k") is None or selected.get("c") is None:
        raise ValueError("calibration selected pair is missing")
    selected_k: Any = selected.get("k")
    selected_c: Any = selected.get("c")
    selected_pair_present = False
    for item in attempts:
        if not isinstance(item, Mapping):
            continue
        item_k: Any = item.get("k")
        item_c: Any = item.get("c")
        if (
            item.get("status") == "valid"
            and item_k is not None
            and item_c is not None
            and int(item_k) == int(selected_k)
            and float(item_c) == float(selected_c)
        ):
            selected_pair_present = True
            break
    if not selected_pair_present:
        raise ValueError("calibration selected pair is absent from valid trace")


def _graph_from_manifest(value: Mapping[str, Any]) -> GraphDiagnostics:
    source = DiagnosticDistribution.from_manifest(value["source_support"])
    candidate = DiagnosticDistribution.from_manifest(value["candidate_support"])
    memory = value.get("memory", {})
    if not isinstance(memory, Mapping):
        raise ValueError("graph memory payload is malformed")

    def nonnegative_integer(name: str, *, default: Any = None) -> int:
        raw = memory.get(name, default)
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise ValueError(f"graph memory {name} must be an integer")
        normalized = int(raw)
        if normalized < 0:
            raise ValueError(f"graph memory {name} must be non-negative")
        return normalized

    def nullable_nonnegative_integer(name: str) -> int | None:
        raw = memory.get(name)
        if raw is None:
            return None
        return nonnegative_integer(name)

    candidate_storage = memory.get("candidate_storage", "memory")
    if not isinstance(candidate_storage, str) or candidate_storage not in {"memory", "disk"}:
        raise ValueError("graph memory candidate_storage must be 'memory' or 'disk'")
    candidate_storage_path = memory.get("candidate_storage_path")
    if candidate_storage_path is not None and not isinstance(candidate_storage_path, str):
        raise ValueError("graph memory candidate_storage_path must be a string or null")
    if candidate_storage == "memory" and candidate_storage_path is not None:
        raise ValueError("memory candidate storage must not have a storage path")
    if candidate_storage == "disk" and (
        candidate_storage_path is None or not candidate_storage_path.strip()
    ):
        raise ValueError("disk candidate storage requires a non-empty storage path")
    return GraphDiagnostics(
        atomic_row_count=int(value["atomic_row_count"]),
        edge_count=int(value["edge_count"]),
        edge_per_row=float(value["edge_per_row"]),
        edge_density=float(value["edge_density"]),
        self_edge_count=int(value["self_edge_count"]),
        source_mass_deviation=float(value["source_mass_deviation"]),
        source_support=source,
        candidate_entry_count=int(value["candidate_entry_count"]),
        candidate_support=candidate,
        candidate_pmf_max_deviation=float(value["candidate_pmf_max_deviation"]),
        csr_array_bytes=nonnegative_integer("csr_array_bytes"),
        candidate_csr_array_bytes=nonnegative_integer("candidate_csr_array_bytes", default=0),
        indexed_workspace_bytes=nullable_nonnegative_integer("indexed_workspace_bytes"),
        measured_peak_memory_bytes=nullable_nonnegative_integer("measured_peak_memory_bytes"),
        estimated_peak_memory_bytes=nullable_nonnegative_integer("estimated_peak_memory_bytes"),
        backend=str(value["backend"]),
        metric=str(value["metric"]),
        backend_version=str(value["backend_version"]),
        kernel_bandwidth_rule=str(
            value.get("kernel_bandwidth_rule", KERNEL_BANDWIDTH_RULE_ASYMMETRIC_SOURCE)
        ),
        kernel_bandwidth_rule_version=str(value.get("kernel_bandwidth_rule_version", "")),
        atomic_graph_materialized=bool(value.get("atomic_graph_materialized", True)),
        atomic_graph_csr_bytes=int(
            value.get("atomic_graph_csr_bytes", memory.get("csr_array_bytes", 0))
        ),
        kernel_operator_fingerprint=(
            None
            if value.get("kernel_operator_fingerprint") is None
            else str(value["kernel_operator_fingerprint"])
        ),
        contribution_spool_bytes=(
            nullable_nonnegative_integer("contribution_spool_bytes")
        ),
        candidate_storage=candidate_storage,
        candidate_storage_path=candidate_storage_path,
        candidate_resident_bytes=nonnegative_integer("candidate_resident_bytes", default=0),
        operational_metrics=dict(value.get("operations", {})),
        numerical_tolerance=float(
            value.get("numerical_tolerance", DIAGNOSTICS_NUMERICAL_TOLERANCE)
        ),
    )


def _coverage_from_manifest(value: Mapping[str, Any]) -> AtomicCoverageDiagnostics:
    return AtomicCoverageDiagnostics(
        population=str(value["population"]),
        candidate_ids=tuple(value.get("candidate_ids", ())),
        target_row_indices=tuple(value.get("target_row_indices", ())),
        distances=tuple(value.get("distances", ())),
        conditional_mass=float(value["conditional_mass"]),
        summary=DiagnosticDistribution.from_manifest(value["summary"]),
        source_candidate_support=DiagnosticDistribution.from_manifest(
            value["source_candidate_support"]
        ),
    )


def _coverage_summary(
    pool: EntropyPool,
    distances: np.ndarray,
    row_indices: np.ndarray,
    *,
    population: str,
    owners: np.ndarray,
) -> AtomicCoverageDiagnostics:
    if distances.shape != row_indices.shape:
        raise ValueError("coverage distances and row indices must align")
    probabilities = np.asarray(pool.probabilities, dtype=np.float64)[row_indices]
    conditional_mass = float(np.sum(probabilities, dtype=np.float64))
    if distances.size:
        weights = probabilities / conditional_mass
        summary = _distribution(
            distances,
            population=population,
            weighting="redundancy-aware reference p_i, renormalized within this conditional population",
            denominator="sum of authoritative p_i over target rows in this population",
            unit="whitened Euclidean distance",
            weights=weights,
            original_indices=row_indices,
        )
    else:
        summary = _distribution(
            distances,
            population=population,
            weighting="redundancy-aware reference p_i, conditional population empty",
            denominator="sum of authoritative p_i over target rows in this population",
            unit="whitened Euclidean distance",
        )
    support = (
        np.bincount(owners, minlength=len(pool.candidate_ids))
        if owners.size
        else np.zeros(len(pool.candidate_ids))
    )
    return AtomicCoverageDiagnostics(
        population=population,
        candidate_ids=tuple(pool.candidate_ids),
        target_row_indices=tuple(int(value) for value in row_indices),
        distances=tuple(float(value) for value in distances),
        conditional_mass=conditional_mass,
        summary=summary,
        source_candidate_support=_distribution(
            support,
            population=f"{population} source candidate support",
            weighting="unweighted candidate count",
            denominator="number of candidates in the full ordered pool",
            unit="target rows",
        ),
    )


_COVERAGE_CPU_ORACLE_PAIR_LIMIT = 250_000
_COVERAGE_CPU_PAIR_LIMIT = 100_000_000
_COVERAGE_PROGRESS_INTERVAL_SECONDS = 10.0


def _coverage_source_indices(
    pool: EntropyPool, selected_candidate_ids: Sequence[str]
) -> np.ndarray:
    selected = {str(value) for value in selected_candidate_ids}
    if not selected:
        raise ValueError("coverage requires at least one selected candidate")
    candidate_positions = {value: index for index, value in enumerate(pool.candidate_ids)}
    missing = selected.difference(candidate_positions)
    if missing:
        raise ValueError("coverage selected candidate IDs are absent from the pool")
    source_indices = np.flatnonzero(
        np.isin(
            np.asarray(pool.row_candidate_indices, dtype=np.int64),
            [candidate_positions[value] for value in selected],
        )
    )
    if source_indices.size == 0:
        raise ValueError("coverage selected candidates own no atomic rows")
    return source_indices.astype(np.int64, copy=False)


def _coverage_log_progress(
    *,
    backend: str,
    completed: int,
    total: int,
    started: float,
    device: int | None = None,
) -> None:
    elapsed = max(0.0, time.perf_counter() - started)
    throughput = completed / elapsed if elapsed > 0.0 else 0.0
    device_label = "" if device is None else f", device={device}"
    logger.info(
        "Exact coverage progress: backend=%s%s, query_rows=%d/%d, elapsed=%.1fs, "
        "throughput=%.1f rows/s",
        backend,
        device_label,
        completed,
        total,
        elapsed,
        throughput,
    )


def _coverage_cpu_oracle(
    descriptors: np.ndarray,
    source_indices: np.ndarray,
    *,
    chunk_size: int,
    started: float,
    progress_callback: Any | None,
) -> np.ndarray:
    """Small, explicitly bounded reference implementation for tests/oracles."""

    from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

    tree = cKDTree(descriptors[source_indices], compact_nodes=True)
    distances = np.empty(descriptors.shape[0], dtype=np.float64)
    last_log = started
    for start in range(0, descriptors.shape[0], chunk_size):
        stop = min(start + chunk_size, descriptors.shape[0])
        distances[start:stop] = tree.query(descriptors[start:stop], k=1, workers=1)[0]
        if (
            stop == descriptors.shape[0]
            or stop - start == chunk_size
            and time.perf_counter() - last_log >= _COVERAGE_PROGRESS_INTERVAL_SECONDS
        ):
            _coverage_log_progress(
                backend="cpu_oracle",
                completed=stop,
                total=descriptors.shape[0],
                started=started,
            )
            last_log = time.perf_counter()
        if progress_callback is not None:
            progress_callback(stop, descriptors.shape[0])
    return distances


def _coverage_cpu_tiled(
    descriptors: np.ndarray,
    source_indices: np.ndarray,
    *,
    chunk_size: int,
    resource_budget: ResourceBudgetService,
    max_pair_work: int,
    started: float,
    progress_callback: Any | None,
) -> np.ndarray:
    total_rows, dimension = descriptors.shape
    pair_work = int(total_rows) * int(source_indices.size)
    if pair_work > int(max_pair_work):
        raise ResourceCapacityError(
            "exact CPU coverage exceeds its bounded pair-work admission: "
            f"N={total_rows}, S={source_indices.size}, pairs={pair_work}, "
            f"max_pairs={max_pair_work}; select the explicit GPU coverage backend "
            "or raise max_cpu_pair_work",
            operation="entropy exact CPU coverage",
            requested_bytes=None,
            available_bytes=resource_budget.remaining_managed_budget,
        )
    query_tile = max(1, min(int(chunk_size), total_rows))
    reference_tile = max(1, min(256, int(source_indices.size)))
    references = np.ascontiguousarray(descriptors[source_indices], dtype=np.float64)
    distances = np.empty(total_rows, dtype=np.float64)
    last_log = started
    for query_start in range(0, total_rows, query_tile):
        query_stop = min(query_start + query_tile, total_rows)
        query = descriptors[query_start:query_stop]
        best = np.full(query_stop - query_start, np.inf, dtype=np.float64)
        for reference_start in range(0, references.shape[0], reference_tile):
            reference_stop = min(reference_start + reference_tile, references.shape[0])
            reference = references[reference_start:reference_stop]
            requested = int(
                8
                * (
                    4 * query.shape[0] * dimension
                    + 4 * reference.shape[0] * dimension
                    + 2 * query.shape[0] * reference.shape[0]
                )
                + 65_536
            )
            with resource_budget.acquire("entropy exact CPU coverage tile", max(1, requested)):
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        delta = query[:, None, :] - reference[None, :, :]
                        tile_distances = np.sqrt(
                            np.sum(delta * delta, axis=2, dtype=np.float64)
                        )
                except FloatingPointError as exc:
                    raise ValueError(
                        "exact CPU coverage distance overflowed or became invalid"
                    ) from exc
            if not np.all(np.isfinite(tile_distances)) or np.any(tile_distances < 0.0):
                raise ValueError("coverage nearest distances are not finite and non-negative")
            best = np.minimum(best, np.min(tile_distances, axis=1))
        distances[query_start:query_stop] = best
        now = time.perf_counter()
        if (
            query_stop == total_rows
            or now - last_log >= _COVERAGE_PROGRESS_INTERVAL_SECONDS
        ):
            _coverage_log_progress(
                backend="cpu_tiled",
                completed=query_stop,
                total=total_rows,
                started=started,
            )
            last_log = now
        if progress_callback is not None:
            progress_callback(query_stop, total_rows)
    return distances


def _coverage_gpu_tiled(
    descriptors: np.ndarray,
    source_indices: np.ndarray,
    *,
    chunk_size: int,
    gpu_device: int,
    gpu_devices: Sequence[int] | str | int | None,
    gpu_tile_size: int,
    max_gpu_support_bytes: int | None,
    resource_budget: ResourceBudgetService,
    started: float,
    progress_callback: Any | None,
    operational_metrics: dict[str, Any] | None,
) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GPUDeviceValidationError(
            "exact_tiled_gpu coverage requires optional CuPy; no CPU fallback is performed"
        ) from exc
    try:
        devices = normalize_gpu_devices(gpu_devices, gpu_device=gpu_device)
        validate_visible_gpu_devices(cp, devices)
    except Exception as exc:
        if isinstance(exc, GPUDeviceValidationError):
            raise
        raise GPUDeviceValidationError(
            f"requested coverage GPU devices are unavailable: {exc}"
        ) from exc
    if isinstance(gpu_tile_size, bool) or int(gpu_tile_size) < 1:
        raise ValueError("coverage gpu_tile_size must be positive")
    tile_size = int(gpu_tile_size)
    total_rows, dimension = descriptors.shape
    query_tile = max(1, min(int(chunk_size), tile_size, total_rows))
    reference_tile = max(1, min(tile_size, int(source_indices.size)))

    device_limits: dict[int, int] = {}
    for device in devices:
        try:
            with cp.cuda.Device(device):
                free_bytes, _ = cp.cuda.runtime.memGetInfo()
        except Exception as exc:
            raise GPUDeviceValidationError(
                f"unable to inspect coverage GPU device {device}: {exc}"
            ) from exc
        limit = int(free_bytes)
        if max_gpu_support_bytes is not None:
            if isinstance(max_gpu_support_bytes, bool) or int(max_gpu_support_bytes) < 1:
                raise ValueError("max_gpu_support_bytes must be positive")
            limit = min(limit, int(max_gpu_support_bytes))
        device_limits[int(device)] = limit
    minimum_tile_bytes = int(
        4
        * 8
        * (query_tile * dimension + reference_tile * dimension + query_tile * reference_tile)
        + 65_536
    )
    for device, limit in device_limits.items():
        if limit < minimum_tile_bytes:
            raise GPUCapacityError(
                "coverage GPU tile exceeds the declared device workspace: "
                f"device={device}, requested_bytes={minimum_tile_bytes}, available_bytes={limit}",
                operation="entropy exact tiled GPU coverage",
                requested_bytes=minimum_tile_bytes,
                available_bytes=limit,
            )
    logger.info(
        "Exact coverage start: N=%d, S=%d, d'=%d, backend=exact_tiled_gpu, devices=%s, "
        "query_tile=%d, reference_tile=%d, estimated_pairs=%d",
        total_rows,
        source_indices.size,
        dimension,
        devices,
        query_tile,
        reference_tile,
        int(total_rows) * int(source_indices.size),
    )
    def run_device(
        device: int, reference_start: int, reference_stop: int
    ) -> tuple[np.ndarray, int, int, int, int]:
        device_references = np.ascontiguousarray(
            descriptors[source_indices[reference_start:reference_stop]], dtype=np.float64
        )
        device_distances = np.full(total_rows, np.inf, dtype=np.float64)
        device_pair_count = 0
        device_candidate_transfer_count = 0
        device_cpu_refined_count = 0
        device_cpu_seed_count = 0
        for query_start in range(0, total_rows, query_tile):
            query_stop = min(query_start + query_tile, total_rows)
            query = np.ascontiguousarray(descriptors[query_start:query_stop], dtype=np.float64)
            query_count = query_stop - query_start
            gpu_best_reported = None
            gpu_best_positions = None
            for tile_start in range(0, device_references.shape[0], reference_tile):
                tile_stop = min(tile_start + reference_tile, device_references.shape[0])
                references = device_references[tile_start:tile_stop]
                requested_host = int(
                    8
                    * (
                        4 * query.shape[0] * dimension
                        + 4 * references.shape[0] * dimension
                        + 2 * query.shape[0] * references.shape[0]
                    )
                    + 65_536
                )
                with resource_budget.acquire(
                    "entropy exact GPU coverage certification tile", max(1, requested_host)
                ):
                    try:
                        with cp.cuda.Device(device):
                            gpu_query = cp.asarray(
                                query, dtype=getattr(cp, "float64", np.float64)
                            )
                            gpu_reference = cp.asarray(
                                references, dtype=getattr(cp, "float64", np.float64)
                            )
                            reported_device, _lower_device, finite_device = _distance_intervals(
                                cp,
                                gpu_query,
                                gpu_reference,
                            )
                            finite_reported = cp.where(
                                finite_device,
                                reported_device,
                                np.inf,
                            )
                            tile_positions = cp.argmin(finite_reported, axis=1)
                            tile_reported = cp.min(finite_reported, axis=1)
                            tile_positions = tile_positions + tile_start
                            if gpu_best_reported is None:
                                gpu_best_reported = tile_reported
                                gpu_best_positions = tile_positions
                            else:
                                improved = tile_reported < gpu_best_reported
                                gpu_best_reported = cp.where(
                                    improved, tile_reported, gpu_best_reported
                                )
                                gpu_best_positions = cp.where(
                                    improved, tile_positions, gpu_best_positions
                                )
                            del (
                                gpu_query,
                                gpu_reference,
                                reported_device,
                                _lower_device,
                                finite_device,
                                finite_reported,
                                tile_positions,
                                tile_reported,
                            )
                    except Exception as exc:
                        raise GPUCertificationError(
                            f"coverage GPU distance evaluation failed on device {device}: {exc}"
                        ) from exc
                device_pair_count += int(query.shape[0] * references.shape[0])
            if gpu_best_reported is None or gpu_best_positions is None:  # pragma: no cover
                raise GPUCertificationError(
                    f"coverage GPU device {device} received no reference rows"
                )
            try:
                with cp.cuda.Device(device):
                    seed_positions = np.asarray(
                        cp.asnumpy(gpu_best_positions), dtype=np.int64
                    )
                    seed_reported_finite = np.asarray(
                        cp.asnumpy(cp.isfinite(gpu_best_reported)), dtype=bool
                    )
            except Exception as exc:
                raise GPUCertificationError(
                    f"coverage GPU seed compaction failed on device {device}: {exc}"
                ) from exc
            if (
                seed_positions.shape != (query_count,)
                or not np.all(seed_reported_finite)
                or np.any(seed_positions < 0)
                or np.any(seed_positions >= device_references.shape[0])
            ):
                raise GPUCertificationError(
                    "coverage GPU could not establish a finite certified seed for every query"
                )
            device_candidate_transfer_count += int(seed_positions.size)
            device_cpu_seed_count += int(seed_positions.size)
            best = np.full(query_count, np.inf, dtype=np.float64)
            for query_offset, query_row in enumerate(query):
                seed_position = int(seed_positions[query_offset])
                seed_distance = _canonical_gpu_distances(
                    query_row,
                    device_references[seed_position : seed_position + 1],
                )
                best[query_offset] = float(seed_distance[0])

            # The first pass establishes one exact host upper bound per query.
            # The second pass reuses only bounded GPU tiles to screen against
            # that bound, so CPU canonical refinement is limited to pairs whose
            # conservative lower interval still reaches the current best.
            for tile_start in range(0, device_references.shape[0], reference_tile):
                tile_stop = min(tile_start + reference_tile, device_references.shape[0])
                references = device_references[tile_start:tile_stop]
                requested_host = int(
                    8
                    * (
                        4 * query.shape[0] * dimension
                        + 4 * references.shape[0] * dimension
                        + 2 * query.shape[0] * references.shape[0]
                    )
                    + 65_536
                )
                with resource_budget.acquire(
                    "entropy exact GPU coverage candidate tile", max(1, requested_host)
                ):
                    try:
                        with cp.cuda.Device(device):
                            gpu_query = cp.asarray(
                                query, dtype=getattr(cp, "float64", np.float64)
                            )
                            gpu_reference = cp.asarray(
                                references, dtype=getattr(cp, "float64", np.float64)
                            )
                            _reported_device, lower_device, finite_device = _distance_intervals(
                                cp,
                                gpu_query,
                                gpu_reference,
                            )
                            candidate_device = cp.flatnonzero(
                                (~finite_device)
                                | (
                                    lower_device
                                    <= cp.asarray(
                                        best,
                                        dtype=getattr(cp, "float64", np.float64),
                                    )[:, None]
                                )
                            )
                            candidate_flat = np.asarray(
                                cp.asnumpy(candidate_device), dtype=np.int64
                            )
                            del (
                                gpu_query,
                                gpu_reference,
                                _reported_device,
                                lower_device,
                                finite_device,
                                candidate_device,
                            )
                    except Exception as exc:
                        raise GPUCertificationError(
                            f"coverage GPU candidate compaction failed on device {device}: {exc}"
                        ) from exc
                device_pair_count += int(query.shape[0] * references.shape[0])
                device_candidate_transfer_count += int(candidate_flat.size)
                if candidate_flat.size:
                    candidate_query, candidate_reference = np.divmod(
                        candidate_flat, references.shape[0]
                    )
                    for query_offset in np.unique(candidate_query):
                        query_offset = int(query_offset)
                        positions = candidate_reference[candidate_query == query_offset]
                        seed_position = int(seed_positions[query_offset])
                        positions = np.unique(
                            positions[(positions + tile_start) != seed_position]
                        )
                        if positions.size == 0:
                            continue
                        canonical = _canonical_gpu_distances(
                            query[query_offset], references[positions]
                        )
                        best[query_offset] = min(
                            best[query_offset], float(np.min(canonical))
                        )
                        device_cpu_refined_count += int(positions.size)
            device_distances[query_start:query_stop] = best
            _coverage_log_progress(
                backend="exact_tiled_gpu",
                completed=query_stop,
                total=total_rows,
                started=started,
                device=device,
            )
            if progress_callback is not None:
                progress_callback(query_stop, total_rows)
        return (
            device_distances,
            device_pair_count,
            device_candidate_transfer_count,
            device_cpu_refined_count,
            device_cpu_seed_count,
        )

    ranges = tuple(
        item
        for item in contiguous_gpu_ranges(int(source_indices.size), devices)
        if item[1] != item[2]
    )
    global_distances = np.full(total_rows, np.inf, dtype=np.float64)
    if len(ranges) == 1:
        results = [run_device(*ranges[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [executor.submit(run_device, *item) for item in ranges]
            results = [future.result() for future in futures]
    pair_count = 0
    candidate_transfer_count = 0
    cpu_refined_count = 0
    cpu_seed_count = 0
    for result, worker_pairs, worker_candidates, worker_refined, worker_seeds in results:
        global_distances = np.minimum(global_distances, result)
        pair_count += worker_pairs
        candidate_transfer_count += worker_candidates
        cpu_refined_count += worker_refined
        cpu_seed_count += worker_seeds
    if operational_metrics is not None:
        operational_metrics.update(
            {
                "gpu_evaluated_pairs": pair_count,
                "gpu_screened_pairs": pair_count // 2,
                "gpu_distance_passes": 2,
                "candidate_transfer_count": candidate_transfer_count,
                "cpu_refined_pairs": cpu_refined_count,
                "cpu_seed_pairs": cpu_seed_count,
                "cpu_certified_pairs": cpu_seed_count + cpu_refined_count,
                "gpu_tile_workspace_bytes": minimum_tile_bytes,
                "gpu_tile_workspace_limit_bytes": min(device_limits.values()),
            }
        )
    return global_distances


def nearest_selected_atomic_distances(
    pool: EntropyPool,
    selected_candidate_ids: Sequence[str],
    *,
    chunk_size: int = 4096,
    backend: str = "cpu",
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_gpu_support_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    max_cpu_pair_work: int = _COVERAGE_CPU_PAIR_LIMIT,
    progress_callback: Any | None = None,
    operational_metrics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact nearest-selected distances with explicit bounded backends.

    ``cpu_oracle`` is intentionally limited to small reference cases.  The
    default CPU path is tiled and admitted against both memory and pair work;
    the GPU path partitions selected references across the explicitly requested
    visible devices and certifies every retained boundary against the canonical
    NumPy float64 distance primitive.
    """

    descriptors = np.asarray(pool.descriptors, dtype=np.float64)
    if descriptors.ndim != 2 or descriptors.shape[0] == 0 or descriptors.shape[1] == 0:
        raise ValueError("coverage descriptors must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(descriptors)):
        raise ValueError("coverage descriptors must be finite")
    if isinstance(chunk_size, bool) or int(chunk_size) < 1:
        raise ValueError("coverage chunk_size must be positive")
    if isinstance(max_cpu_pair_work, bool) or int(max_cpu_pair_work) < 1:
        raise ValueError("max_cpu_pair_work must be positive")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("coverage progress_callback must be callable")
    normalized_backend = str(backend).strip().lower()
    if normalized_backend == "exact_cpu":
        normalized_backend = "cpu"
    if normalized_backend not in {"cpu", "cpu_oracle", "exact_tiled_gpu"}:
        raise ValueError(
            "coverage backend must be 'cpu', 'cpu_oracle', or 'exact_tiled_gpu'"
        )
    source_indices = _coverage_source_indices(pool, selected_candidate_ids)
    total_rows, dimension = descriptors.shape
    pair_work = int(total_rows) * int(source_indices.size)
    if normalized_backend == "cpu_oracle" and pair_work > _COVERAGE_CPU_ORACLE_PAIR_LIMIT:
        raise ResourceCapacityError(
            "coverage CPU oracle is deliberately limited to small reference cases: "
            f"N={total_rows}, S={source_indices.size}, pairs={pair_work}, "
            f"oracle_limit={_COVERAGE_CPU_ORACLE_PAIR_LIMIT}; use backend='cpu'",
            operation="entropy coverage CPU oracle",
            requested_bytes=None,
            available_bytes=None,
        )
    effective_backend = normalized_backend
    if (
        normalized_backend == "cpu"
        and pair_work <= _COVERAGE_CPU_ORACLE_PAIR_LIMIT
        and dimension <= 16
    ):
        effective_backend = "cpu_oracle"
    started = time.perf_counter()
    logger.info(
        "Exact coverage start: N=%d, S=%d, d'=%d, backend=%s, devices=%s, "
        "query_tile=%d, reference_tile=%s, estimated_pairs=%d",
        total_rows,
        source_indices.size,
        dimension,
        effective_backend,
        (gpu_devices if gpu_devices is not None else (gpu_device,))
        if effective_backend == "exact_tiled_gpu"
        else "cpu",
        min(int(chunk_size), total_rows),
        int(gpu_tile_size) if effective_backend == "exact_tiled_gpu" else min(256, source_indices.size),
        pair_work,
    )
    runtime_budget = resource_budget or build_resource_budget()
    if effective_backend == "cpu_oracle":
        distances = _coverage_cpu_oracle(
            descriptors,
            source_indices,
            chunk_size=int(chunk_size),
            started=started,
            progress_callback=progress_callback,
        )
    elif effective_backend == "cpu_tiled":  # pragma: no cover - defensive alias
        distances = _coverage_cpu_tiled(
            descriptors,
            source_indices,
            chunk_size=int(chunk_size),
            resource_budget=runtime_budget,
            max_pair_work=int(max_cpu_pair_work),
            started=started,
            progress_callback=progress_callback,
        )
    elif effective_backend == "cpu":
        distances = _coverage_cpu_tiled(
            descriptors,
            source_indices,
            chunk_size=int(chunk_size),
            resource_budget=runtime_budget,
            max_pair_work=int(max_cpu_pair_work),
            started=started,
            progress_callback=progress_callback,
        )
    else:
        distances = _coverage_gpu_tiled(
            descriptors,
            source_indices,
            chunk_size=int(chunk_size),
            gpu_device=int(gpu_device),
            gpu_devices=gpu_devices,
            gpu_tile_size=int(gpu_tile_size),
            max_gpu_support_bytes=max_gpu_support_bytes,
            resource_budget=runtime_budget,
            started=started,
            progress_callback=progress_callback,
            operational_metrics=operational_metrics,
        )
    # A selected row is its own exact reference.  Keep this explicit so the
    # identity contract remains visible even if a future backend changes its
    # reduction order.
    distances[source_indices] = 0.0
    if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("coverage nearest distances are not finite and non-negative")
    _coverage_log_progress(
        backend=effective_backend,
        completed=total_rows,
        total=total_rows,
        started=started,
    )
    logger.info(
        "Exact coverage completed: backend=%s, N=%d, S=%d, elapsed=%.3fs, "
        "throughput=%.1f rows/s",
        effective_backend,
        total_rows,
        source_indices.size,
        max(0.0, time.perf_counter() - started),
        total_rows / max(time.perf_counter() - started, np.finfo(float).tiny),
    )
    if operational_metrics is not None:
        operational_metrics.update(
            {
                "backend": effective_backend,
                "devices": (
                    tuple(normalize_gpu_devices(gpu_devices, gpu_device=gpu_device))
                    if effective_backend == "exact_tiled_gpu"
                    else ()
                ),
                "rows": total_rows,
                "selected_reference_rows": int(source_indices.size),
                "dimension": int(dimension),
                "estimated_pair_work": pair_work,
                "query_tile": min(int(chunk_size), total_rows),
                "reference_tile": (
                    int(gpu_tile_size)
                    if effective_backend == "exact_tiled_gpu"
                    else min(256, int(source_indices.size))
                ),
                "elapsed_seconds": max(0.0, time.perf_counter() - started),
            }
        )
    return distances, source_indices


def _categorical_summary(
    values: Sequence[str],
    selected: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    all_values = [str(value) if str(value).strip() else "unavailable" for value in values]
    selected_values = [str(value) if str(value).strip() else "unavailable" for value in selected]

    def counts(items: Sequence[str], denominator: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in sorted(set(items)):
            count = items.count(item)
            result[item] = {
                "count": count,
                "denominator": denominator,
                "percentage": (100.0 * count / denominator if denominator else None),
            }
        return result

    return {
        "label": label,
        "unit": "candidate count",
        "population": "all candidates versus selected training candidates",
        "weighting": "unweighted candidate counts",
        "input_denominator": len(all_values),
        "selected_denominator": len(selected_values),
        "input": counts(all_values, len(all_values)),
        "selected": counts(selected_values, len(selected_values)),
    }


def _candidate_metadata(
    candidates: Sequence[Any] | None, selected_indices: Sequence[int]
) -> dict[str, Any]:
    if candidates is None:
        return {}
    infos = [getattr(candidate, "info", {}) for candidate in candidates]
    infos = [info if isinstance(info, Mapping) else {} for info in infos]
    selected_set = set(int(index) for index in selected_indices)

    def value(info: Mapping[str, Any], keys: Sequence[str]) -> str:
        for key in keys:
            item = info.get(key)
            if item is not None and str(item).strip():
                return str(item)
        return "unavailable"

    generator = [value(info, ("generator", "configurational_type", "source")) for info in infos]
    perturbation = [value(info, ("perturbation_type", "perturbation_family")) for info in infos]
    magnetic = [
        value(info, ("magnetic_state_id", "magnetic_state", "spin_state")) for info in infos
    ]
    compositions: list[str] = []
    for candidate in candidates:
        symbols = list(candidate.get_chemical_symbols())
        counts: dict[str, int] = {}
        for symbol in symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
        compositions.append(
            "".join(f"{key}{counts[key] if counts[key] != 1 else ''}" for key in sorted(counts))
        )

    def selected_values(values: Sequence[str]) -> list[str]:
        return [item for index, item in enumerate(values) if index in selected_set]

    return {
        "composition": _categorical_summary(
            compositions, selected_values(compositions), label="chemical formula"
        ),
        "generator_family": _categorical_summary(
            generator, selected_values(generator), label="generator family"
        ),
        "perturbation_family": _categorical_summary(
            perturbation, selected_values(perturbation), label="perturbation family"
        ),
        "magnetic_state": _categorical_summary(
            magnetic, selected_values(magnetic), label="magnetic state"
        ),
    }


def build_entropy_diagnostics(
    pool: EntropyPool,
    calibration: BandwidthCalibrationResult,
    graph: SparseAtomicKernelGraph | StreamedKernelExecutionSummary,
    contributions: SparseCandidateContributions | OutOfCoreSparseCandidateContributions,
    entropy_result: EntropySelectionResult,
    local_representation: Any,
    *,
    test_candidate_ids: Sequence[str] = (),
    candidates: Sequence[Any] | None = None,
    selected_indices: Sequence[int] | None = None,
    chunk_size: int = 4096,
    coverage_backend: str = "cpu",
    kernel_support_execution: str | None = None,
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_gpu_support_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
    max_cpu_pair_work: int = _COVERAGE_CPU_PAIR_LIMIT,
) -> EntropyScientificDiagnostics:
    """Build the closure record from validated scientific pipeline records."""

    graph_pool_fingerprint = getattr(graph, "pool_fingerprint", pool.fingerprint)
    if graph_pool_fingerprint != pool.fingerprint or contributions.pool_fingerprint not in {
        "",
        pool.fingerprint,
    }:
        raise ValueError("diagnostics upstream pool identities do not match")
    graph_bandwidth_fingerprint = getattr(
        graph, "bandwidth_fingerprint", calibration.selected.fingerprint
    )
    if graph_bandwidth_fingerprint != calibration.selected.fingerprint:
        raise ValueError("diagnostics graph bandwidth identity does not match calibration")
    expected_rule = calibration.kernel_bandwidth_rule
    graph_rule = normalize_kernel_bandwidth_rule(
        getattr(graph, "kernel_bandwidth_rule", expected_rule)
    )
    if graph_rule != expected_rule:
        raise ValueError("diagnostics graph kernel bandwidth rule does not match calibration")
    contribution_rule = normalize_kernel_bandwidth_rule(
        getattr(contributions, "kernel_bandwidth_rule", graph_rule)
    )
    if contribution_rule != expected_rule:
        raise ValueError(
            "diagnostics contribution kernel bandwidth rule does not match calibration"
        )
    if contributions.fingerprint != entropy_result.contributions_fingerprint:
        raise ValueError("diagnostics contribution identity does not match selection result")
    if tuple(entropy_result.history.acquisition_order) != tuple(entropy_result.acquisition_order):
        raise ValueError("diagnostics selection history order does not match selection result")
    if isinstance(graph, SparseAtomicKernelGraph):
        if contributions.graph_fingerprint != graph.fingerprint:
            raise ValueError("diagnostics contribution graph identity does not match graph")
        operator_fingerprint = contributions.kernel_operator_fingerprint
    else:
        if contributions.kernel_operator_fingerprint != graph.kernel_operator_fingerprint:
            raise ValueError("diagnostics contribution operator identity does not match execution")
        operator_fingerprint = graph.kernel_operator_fingerprint
    coverage_metrics: dict[str, Any] = {}
    distances, _ = nearest_selected_atomic_distances(
        pool,
        entropy_result.selected_candidate_ids,
        chunk_size=chunk_size,
        backend=(
            kernel_support_execution
            if kernel_support_execution is not None
            else coverage_backend
        ),
        gpu_device=gpu_device,
        gpu_devices=gpu_devices,
        gpu_tile_size=gpu_tile_size,
        max_gpu_support_bytes=max_gpu_support_bytes,
        resource_budget=resource_budget,
        max_cpu_pair_work=max_cpu_pair_work,
        operational_metrics=coverage_metrics,
    )
    selected_set = set(entropy_result.selected_candidate_ids)
    owners = np.asarray(pool.row_candidate_indices, dtype=np.int64)
    all_rows = np.arange(len(pool.rows), dtype=np.int64)
    residual_rows = all_rows[
        ~np.isin(owners, [pool.candidate_ids.index(value) for value in selected_set])
    ]
    coverage = (
        _coverage_summary(
            pool, distances, all_rows, population="all_pool_target_rows", owners=owners
        ),
        _coverage_summary(
            pool,
            distances[residual_rows],
            residual_rows,
            population="residual_unacquired_candidate_rows",
            owners=owners[residual_rows],
        ),
    )
    source_support = (
        graph.support_sizes.astype(np.float64)
        if isinstance(graph, SparseAtomicKernelGraph)
        else np.full(graph.atomic_row_count, graph.source_support_mean, dtype=np.float64)
    )
    if not isinstance(graph, SparseAtomicKernelGraph) and source_support.size:
        source_support[0] = graph.source_support_min
        source_support[-1] = graph.source_support_max
    candidate_support = contributions.support_sizes.astype(np.float64)
    source_masses = (
        np.add.reduceat(graph.values, graph.source_indptr[:-1])
        if isinstance(graph, SparseAtomicKernelGraph)
        else np.asarray([1.0], dtype=np.float64)
    )
    candidate_masses = (
        np.add.reduceat(contributions.values, contributions.candidate_indptr[:-1])
        if contributions.n_candidates
        else np.empty(0, dtype=np.float64)
    )
    graph_diagnostics = GraphDiagnostics(
        atomic_row_count=(
            graph.n_sources
            if isinstance(graph, SparseAtomicKernelGraph)
            else graph.atomic_row_count
        ),
        edge_count=graph.edge_count,
        edge_per_row=graph.edge_count / graph.atomic_row_count,
        edge_density=(
            graph.density
            if isinstance(graph, SparseAtomicKernelGraph)
            else graph.edge_count / float(graph.atomic_row_count * graph.atomic_row_count)
        ),
        self_edge_count=(
            int(
                np.count_nonzero(
                    graph.target_indices
                    == np.repeat(np.arange(graph.n_sources), graph.support_sizes.astype(int))
                )
            )
            if isinstance(graph, SparseAtomicKernelGraph)
            else graph.self_edge_count
        ),
        source_mass_deviation=(
            float(np.max(np.abs(source_masses - 1.0)))
            if isinstance(graph, SparseAtomicKernelGraph)
            else graph.source_mass_max_deviation
        ),
        source_support=_distribution(
            source_support,
            population=(
                "source-major graph source rows"
                if isinstance(graph, SparseAtomicKernelGraph)
                else "streamed source rows (implicit atomic support)"
            ),
            weighting="unweighted source rows",
            denominator="N atomic source rows",
            unit="graph support entries per source row",
        ),
        candidate_entry_count=contributions.entry_count,
        candidate_support=_distribution(
            candidate_support,
            population="candidate PMF supports",
            weighting="unweighted candidate rows",
            denominator="M candidate PMFs",
            unit="candidate PMF support entries",
        ),
        candidate_pmf_max_deviation=float(np.max(np.abs(candidate_masses - 1.0))),
        csr_array_bytes=(graph.array_bytes if isinstance(graph, SparseAtomicKernelGraph) else 0),
        candidate_csr_array_bytes=contributions.array_bytes,
        indexed_workspace_bytes=None,
        measured_peak_memory_bytes=(
            None if isinstance(graph, SparseAtomicKernelGraph) else graph.measured_peak_memory_bytes
        ),
        estimated_peak_memory_bytes=(
            None
            if isinstance(graph, SparseAtomicKernelGraph)
            else graph.estimated_peak_memory_bytes
        ),
        backend=calibration.backend,
        metric=calibration.selected.metric,
        backend_version=calibration.backend_version,
        kernel_bandwidth_rule=calibration.kernel_bandwidth_rule,
        kernel_bandwidth_rule_version=calibration.selected.kernel_bandwidth_rule_version,
        atomic_graph_materialized=isinstance(graph, SparseAtomicKernelGraph),
        atomic_graph_csr_bytes=(
            graph.array_bytes if isinstance(graph, SparseAtomicKernelGraph) else 0
        ),
        kernel_operator_fingerprint=operator_fingerprint,
        contribution_spool_bytes=(
            None if isinstance(graph, SparseAtomicKernelGraph) else graph.contribution_spool_bytes
        ),
        candidate_storage=(
            "memory" if isinstance(graph, SparseAtomicKernelGraph) else graph.candidate_storage
        ),
        candidate_storage_path=(
            None if isinstance(graph, SparseAtomicKernelGraph) else graph.candidate_storage_path
        ),
        candidate_resident_bytes=(
            contributions.array_bytes
            if isinstance(graph, SparseAtomicKernelGraph)
            else graph.candidate_resident_bytes
        ),
        operational_metrics={
            "calibration": dict(calibration.operational_metrics),
            "contributions": dict(getattr(graph, "operational_metrics", {})),
            "coverage": coverage_metrics,
        },
    )
    raw = np.asarray(local_representation.raw_descriptors, dtype=np.float64)
    whitened = np.asarray(local_representation.descriptors, dtype=np.float64)
    row_ids = tuple(
        f"{row.candidate_id}:{int(row.atom_index)}" for row in local_representation.rows
    )
    provenance = _candidate_metadata(
        candidates, selected_indices or entropy_result.selected_indices
    )
    selected_candidate_indices = {
        pool.candidate_ids.index(value) for value in entropy_result.selected_candidate_ids
    }
    selected_atomic_count = int(np.count_nonzero(np.isin(owners, list(selected_candidate_indices))))
    provenance["population_counts"] = {
        "candidate": {
            "input": len(pool.candidate_ids),
            "selected_training": len(selected_candidate_indices),
            "input_denominator": len(pool.candidate_ids),
            "selected_training_denominator": len(selected_candidate_indices),
        },
        "atomic_rows": {
            "input": len(pool.rows),
            "selected_training": selected_atomic_count,
            "input_denominator": len(pool.rows),
            "selected_training_denominator": selected_atomic_count,
        },
    }
    if str(local_representation.config.magnetic_mode) != "non_soc":
        provenance["magnetic_state"] = {
            "available": False,
            "reason": "structural-only representation; magnetic coverage is not defined",
        }
    elif "magnetic_state" not in provenance:
        provenance["magnetic_state"] = {
            "available": False,
            "reason": "non_soc representation has no candidate magnetic provenance metadata",
        }
    result = recompute_final_entropy_metrics(
        pool.probabilities,
        entropy_result.final_normalized_support,
        beta=entropy_result.beta,
        selected_count=entropy_result.budget,
    )
    prefix_coverage = _build_prefix_coverage_manifest(
        pool.probabilities,
        beta=entropy_result.beta,
        history=entropy_result.history,
        contributions=contributions,
        final_q=entropy_result.final_normalized_support,
        final_forward_kl=result["forward_kl"],
        tolerance=DIAGNOSTICS_NUMERICAL_TOLERANCE,
        pool_fingerprint=pool.fingerprint,
    )
    whitening_manifest = _whitening_manifest_with_diagnostics(
        local_representation.transform.to_manifest()
    )
    whitening_manifest.update(
        {
            "reference_measure_version": pool.reference_measure_version,
            "reference_weights_fingerprint": pool.reference_weights_fingerprint,
            "effective_weighted_sample_mass": float(
                1.0 / np.sum(pool.probabilities * pool.probabilities, dtype=np.float64)
            ),
            "weighted_mean_fingerprint": whitening_manifest.get("mean_fingerprint"),
            "weighted_mean_covariance_fingerprint": local_representation.transform.covariance_fingerprint,
        }
    )
    return EntropyScientificDiagnostics(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        candidate_ids=tuple(pool.candidate_ids),
        structure_ids=tuple(local_representation.structure_ids),
        row_ids=row_ids,
        row_candidate_ids=tuple(row.candidate_id for row in local_representation.rows),
        candidate_count=len(pool.candidate_ids),
        atomic_row_count=len(pool.rows),
        descriptor_dimension_before=int(raw.shape[1]),
        descriptor_dimension_after=int(whitened.shape[1]),
        descriptor_backend=str(local_representation.config.to_dict().get("backend", "unknown")),
        magnetic_mode=str(local_representation.config.magnetic_mode),
        representation_fingerprint=pool.representation_fingerprint,
        pool_fingerprint=pool.fingerprint,
        transform_fingerprint=pool.transform_fingerprint,
        whitening_manifest_json=json.dumps(
            whitening_manifest,
            sort_keys=True,
            separators=(",", ":"),
        ),
        radius_summary=_distribution(
            calibration.radii,
            population="calibrated source-row radii r_k,a",
            weighting="unweighted source rows",
            denominator="N atomic source rows",
            unit="whitened Euclidean distance",
        ),
        bandwidth_summary=_distribution(
            calibration.bandwidths,
            population="frozen source-row bandwidths h_a=c*r_k,a",
            weighting="unweighted source rows",
            denominator="N atomic source rows",
            unit="whitened Euclidean distance",
        ),
        calibration=calibration,
        graph=graph_diagnostics,
        selected_candidate_ids=tuple(entropy_result.selected_candidate_ids),
        acquisition_order=tuple(entropy_result.acquisition_order),
        test_candidate_ids=tuple(str(value) for value in test_candidate_ids),
        target_probabilities=tuple(float(value) for value in pool.probabilities),
        final_q=tuple(float(value) for value in entropy_result.final_normalized_support),
        beta=entropy_result.beta,
        final_objective=result["objective"],
        final_cross_entropy=result["cross_entropy"],
        final_shannon_entropy=result["shannon_entropy"],
        final_forward_kl=result["forward_kl"],
        final_support_mass=result["support_mass"],
        target_mass=result["target_mass"],
        q_mass=result["q_mass"],
        mass_discrepancy=abs(
            result["support_mass"] - (entropy_result.beta + entropy_result.budget)
        ),
        coverage=coverage,
        provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        reference_measure_version=pool.reference_measure_version,
        reference_measure_config_json=json.dumps(
            pool.reference_measure_config, sort_keys=True, separators=(",", ":")
        ),
        reference_weights_fingerprint=pool.reference_weights_fingerprint,
        reference_measure_diagnostics_json=json.dumps(
            pool.reference_measure_diagnostics, sort_keys=True, separators=(",", ":")
        ),
        kernel_operator_fingerprint=operator_fingerprint,
        pool=pool,
        bandwidths=calibration.selected,
        graph_record=(graph if isinstance(graph, SparseAtomicKernelGraph) else None),
        contributions=contributions,
        entropy_result=entropy_result,
        graph_source_fingerprint=operator_fingerprint,
        contributions_source_fingerprint=contributions.fingerprint,
        selection_history=entropy_result.history,
        prefix_coverage_json=json.dumps(prefix_coverage, sort_keys=True, separators=(",", ":")),
    )


# Explicit aliases make the contract discoverable to callers that use the
# shorter terminology from the issue and preserve one implementation.
ScientificDiagnostics = EntropyScientificDiagnostics
EntropyDiagnostics = EntropyScientificDiagnostics
build_scientific_diagnostics = build_entropy_diagnostics
recompute_entropy_metrics = recompute_final_entropy_metrics

__all__ = [
    "DIAGNOSTICS_NUMERICAL_TOLERANCE",
    "DIAGNOSTICS_SCHEMA_VERSION",
    "PREFIX_COVERAGE_SCHEMA_VERSION",
    "QUANTILE_CONVENTION",
    "AtomicCoverageDiagnostics",
    "DiagnosticDistribution",
    "EntropyDiagnostics",
    "EntropyScientificDiagnostics",
    "GraphDiagnostics",
    "ScientificDiagnostics",
    "build_entropy_diagnostics",
    "build_scientific_diagnostics",
    "calibration_attempt_manifest",
    "calibration_from_manifest",
    "calibration_manifest",
    "validate_prefix_coverage_manifest",
    "nearest_selected_atomic_distances",
    "recompute_entropy_metrics",
    "recompute_final_entropy_metrics",
    "validate_calibration_manifest",
    "weighted_quantile",
]
