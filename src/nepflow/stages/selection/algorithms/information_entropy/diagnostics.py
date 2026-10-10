"""Immutable scientific diagnostics for finite-pool entropy selection.

This module deliberately sits after the scientific selector.  It consumes the
validated pool, calibration, graph, contribution and selection records; it
does not refit, retune, reorder, or acquire anything.  The JSON form contains
only scientific values and identities, so presentation and operational timing
cannot change a run fingerprint.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from nepflow.io.hashing import sha256_bytes, sha256_canonical_json

from .models import (
    SPARSE_NUMERICAL_TOLERANCE,
    BandwidthCalibrationResult,
    CalibrationAttempt,
    EntropyPool,
    EntropySelectionHistory,
    EntropySelectionResult,
    FrozenBandwidths,
    SparseAtomicKernelGraph,
    SparseCandidateContributions,
)

DIAGNOSTICS_SCHEMA_VERSION = "entropy-scientific-diagnostics-v1"
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


def _tuple_floats(values: Sequence[Any], *, name: str) -> tuple[float, ...]:
    result = tuple(_finite(value, name=name) for value in values)
    return result


def _array_digest(values: Sequence[Any], *, dtype: str = "<f8") -> str:
    array = np.asarray(values, dtype=np.dtype(dtype))
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


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
    numerical_tolerance: float = DIAGNOSTICS_NUMERICAL_TOLERANCE

    def __post_init__(self) -> None:
        for name in (
            "atomic_row_count",
            "edge_count",
            "self_edge_count",
            "candidate_entry_count",
            "csr_array_bytes",
            "candidate_csr_array_bytes",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"graph {name} must be non-negative")
            object.__setattr__(self, name, value)
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
        ):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"graph {name} must be non-negative when supplied")
        if not self.backend.strip() or not self.metric.strip() or not self.backend_version.strip():
            raise ValueError("graph backend identity is required")

    def to_manifest(self) -> dict[str, Any]:
        return {
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
            },
            "backend": self.backend,
            "metric": self.metric,
            "backend_version": self.backend_version,
            "numerical_tolerance": self.numerical_tolerance,
        }

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
    target_probabilities: tuple[float, ...]
    final_q: tuple[float, ...]
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
    scientific_fingerprint: str = ""
    tolerance: float = DIAGNOSTICS_NUMERICAL_TOLERANCE
    # These are validated, immutable upstream records.  They are intentionally
    # excluded from ``to_manifest`` and from the scientific fingerprint.
    pool: EntropyPool | None = field(default=None, repr=False, compare=False)
    bandwidths: FrozenBandwidths | None = field(default=None, repr=False, compare=False)
    graph_record: SparseAtomicKernelGraph | None = field(default=None, repr=False, compare=False)
    contributions: SparseCandidateContributions | None = field(
        default=None, repr=False, compare=False
    )
    entropy_result: EntropySelectionResult | None = field(default=None, repr=False, compare=False)
    calibration_manifest_json: str = field(default="", repr=False, compare=False)
    graph_source_fingerprint: str | None = field(default=None, repr=False, compare=False)
    contributions_source_fingerprint: str | None = field(default=None, repr=False, compare=False)
    selection_history: EntropySelectionHistory | None = field(
        default=None, repr=False, compare=False
    )
    selection_history_json: str = field(default="", repr=False, compare=False)

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
        target = _tuple_floats(self.target_probabilities, name="target probability")
        final_q = _tuple_floats(self.final_q, name="final q")
        if len(target) != len(rows) or len(final_q) != len(rows):
            raise ValueError("diagnostics p and q arrays must match atomic row count")
        if any(value <= 0.0 for value in target + final_q):
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
                "calibration": (
                    None
                    if calibration_record is None
                    else calibration_record.get("calibration_fingerprint")
                ),
                "graph": self.graph_source_fingerprint
                if self.graph_source_fingerprint is not None
                else (None if self.graph_record is None else self.graph_record.fingerprint),
                "contributions": self.contributions_source_fingerprint
                if self.contributions_source_fingerprint is not None
                else (None if self.contributions is None else self.contributions.fingerprint),
            },
            "whitening_manifest": json.loads(self.whitening_manifest_json),
            "radius_summary": self.radius_summary.to_manifest(),
            "bandwidth_summary": self.bandwidth_summary.to_manifest(),
            "calibration": self._calibration_manifest(),
            "graph": self.graph.to_manifest(),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "acquisition_order": list(self.acquisition_order),
            "test_candidate_ids": list(self.test_candidate_ids),
            "selection_history": self._selection_history_manifest(),
            "p": list(self.target_probabilities),
            "q": list(self.final_q),
            "beta": self.beta,
            "final_objective": self.final_objective,
            "final_cross_entropy": self.final_cross_entropy,
            "final_shannon_entropy": self.final_shannon_entropy,
            "final_forward_kl": self.final_forward_kl,
            "coverage": [item.to_manifest() for item in self.coverage],
            "provenance": json.loads(self.provenance_json),
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

    @property
    def fingerprint(self) -> str:
        return self.scientific_fingerprint

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
    def selection_history_records(self) -> list[dict[str, Any]]:
        return self._selection_history_manifest()

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

    def to_manifest(self) -> dict[str, Any]:
        """Return the complete structured diagnostic payload."""

        calibration_record = self._calibration_manifest()
        if calibration_record is None:
            raise ValueError("entropy diagnostics require the complete calibration trace")
        return {
            "schema_version": self.schema_version,
            "scientific_fingerprint": self.scientific_fingerprint,
            "source_fingerprints": {
                "representation": self.representation_fingerprint,
                "pool": self.pool_fingerprint,
                "transform": self.transform_fingerprint,
                "calibration": (
                    None
                    if calibration_record is None
                    else calibration_record.get("calibration_fingerprint")
                ),
                "graph": self.graph_source_fingerprint
                if self.graph_source_fingerprint is not None
                else (None if self.graph_record is None else self.graph_record.fingerprint),
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
            "bandwidth": {
                "radii": self.radius_summary.to_manifest(),
                "source_bandwidths": self.bandwidth_summary.to_manifest(),
                "relation": "h_a=c*r_{k,a}",
                "selected_k": calibration_record["selected"]["k"],
                "selected_c": calibration_record["selected"]["c"],
                "calibration": calibration_record,
            },
            "graph": self.graph.to_manifest(),
            "objective": {
                "beta": self.beta,
                "target_probabilities": list(self.target_probabilities),
                "final_q": list(self.final_q),
                "target_probability_sha256": _array_digest(self.target_probabilities),
                "final_q_sha256": _array_digest(self.final_q),
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

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "EntropyScientificDiagnostics":
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
        if any(
            not isinstance(source.get(name), str) or not str(source.get(name)).strip()
            for name in (
                "representation",
                "pool",
                "transform",
                "calibration",
                "graph",
                "contributions",
            )
        ):
            raise ValueError("entropy diagnostics source fingerprints are incomplete")
        whitening = representation.get("whitening")
        if (
            not isinstance(whitening, Mapping)
            or whitening.get("fingerprint") != source["transform"]
        ):
            raise ValueError("entropy diagnostics whitening identity does not match its source")
        if _array_digest(objective.get("target_probabilities", ())) != objective.get(
            "target_probability_sha256"
        ):
            raise ValueError("entropy diagnostics target probability artifact is corrupt")
        if _array_digest(objective.get("final_q", ())) != objective.get("final_q_sha256"):
            raise ValueError("entropy diagnostics final q artifact is corrupt")
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
            target_probabilities=tuple(objective["target_probabilities"]),
            final_q=tuple(objective["final_q"]),
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
            scientific_fingerprint=str(value.get("scientific_fingerprint", "")),
            tolerance=float(objective.get("tolerance", DIAGNOSTICS_NUMERICAL_TOLERANCE)),
            calibration_manifest_json=calibration_json,
            graph_source_fingerprint=(
                None if source.get("graph") is None else str(source.get("graph"))
            ),
            contributions_source_fingerprint=(
                None if source.get("contributions") is None else str(source.get("contributions"))
            ),
            selection_history_json=json.dumps(history_value, sort_keys=True, separators=(",", ":")),
        )
        if value.get("scientific_fingerprint") != diagnostics.scientific_fingerprint:
            raise ValueError("entropy diagnostics scientific fingerprint does not match payload")
        return diagnostics


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

    return {
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
    }


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
        csr_array_bytes=int(memory["csr_array_bytes"]),
        candidate_csr_array_bytes=int(memory.get("candidate_csr_array_bytes", 0)),
        indexed_workspace_bytes=(
            None
            if memory.get("indexed_workspace_bytes") is None
            else int(memory["indexed_workspace_bytes"])
        ),
        measured_peak_memory_bytes=(
            None
            if memory.get("measured_peak_memory_bytes") is None
            else int(memory["measured_peak_memory_bytes"])
        ),
        estimated_peak_memory_bytes=(
            None
            if memory.get("estimated_peak_memory_bytes") is None
            else int(memory["estimated_peak_memory_bytes"])
        ),
        backend=str(value["backend"]),
        metric=str(value["metric"]),
        backend_version=str(value["backend_version"]),
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
            weighting="equal-candidate-weighted p_i, renormalized within this conditional population",
            denominator="sum of authoritative p_i over target rows in this population",
            unit="whitened Euclidean distance",
            weights=weights,
            original_indices=row_indices,
        )
    else:
        summary = _distribution(
            distances,
            population=population,
            weighting="equal-candidate-weighted p_i, conditional population empty",
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


def nearest_selected_atomic_distances(
    pool: EntropyPool,
    selected_candidate_ids: Sequence[str],
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact nearest-selected distances with bounded indexed queries."""

    selected = set(str(value) for value in selected_candidate_ids)
    if not selected:
        raise ValueError("coverage requires at least one selected candidate")
    if not selected.issubset(set(pool.candidate_ids)):
        raise ValueError("coverage selected candidate IDs are absent from the pool")
    source_mask = np.isin(
        pool.row_candidate_indices, [pool.candidate_ids.index(value) for value in selected]
    )
    source_indices = np.flatnonzero(source_mask)
    if source_indices.size == 0:
        raise ValueError("coverage selected candidates own no atomic rows")
    if isinstance(chunk_size, bool) or int(chunk_size) < 1:
        raise ValueError("coverage chunk_size must be positive")
    from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

    tree = cKDTree(
        np.asarray(pool.descriptors[source_indices], dtype=np.float64), compact_nodes=True
    )
    distances = np.empty(pool.descriptors.shape[0], dtype=np.float64)
    for start in range(0, pool.descriptors.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), pool.descriptors.shape[0])
        distances[start:stop] = tree.query(pool.descriptors[start:stop], k=1, workers=1)[0]
    if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("coverage nearest distances are not finite and non-negative")
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
    graph: SparseAtomicKernelGraph,
    contributions: SparseCandidateContributions,
    entropy_result: EntropySelectionResult,
    local_representation: Any,
    *,
    test_candidate_ids: Sequence[str] = (),
    candidates: Sequence[Any] | None = None,
    selected_indices: Sequence[int] | None = None,
    chunk_size: int = 4096,
) -> EntropyScientificDiagnostics:
    """Build the closure record from validated scientific pipeline records."""

    if graph.pool_fingerprint != pool.fingerprint or contributions.pool_fingerprint not in {
        "",
        pool.fingerprint,
    }:
        raise ValueError("diagnostics upstream pool identities do not match")
    if graph.bandwidth_fingerprint != calibration.selected.fingerprint:
        raise ValueError("diagnostics graph bandwidth identity does not match calibration")
    if contributions.graph_fingerprint != graph.fingerprint:
        raise ValueError("diagnostics contribution graph identity does not match graph")
    distances, _ = nearest_selected_atomic_distances(
        pool, entropy_result.selected_candidate_ids, chunk_size=chunk_size
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
    source_support = graph.support_sizes.astype(np.float64)
    candidate_support = contributions.support_sizes.astype(np.float64)
    source_masses = (
        np.add.reduceat(graph.values, graph.source_indptr[:-1])
        if graph.n_sources
        else np.empty(0, dtype=np.float64)
    )
    candidate_masses = (
        np.add.reduceat(contributions.values, contributions.candidate_indptr[:-1])
        if contributions.n_candidates
        else np.empty(0, dtype=np.float64)
    )
    graph_diagnostics = GraphDiagnostics(
        atomic_row_count=graph.n_sources,
        edge_count=graph.edge_count,
        edge_per_row=graph.edge_count / graph.n_sources,
        edge_density=graph.density,
        self_edge_count=int(
            np.count_nonzero(
                graph.target_indices
                == np.repeat(np.arange(graph.n_sources), source_support.astype(int))
            )
        ),
        source_mass_deviation=float(np.max(np.abs(source_masses - 1.0))),
        source_support=_distribution(
            source_support,
            population="source-major graph source rows",
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
        csr_array_bytes=graph.array_bytes,
        candidate_csr_array_bytes=contributions.array_bytes,
        indexed_workspace_bytes=None,
        measured_peak_memory_bytes=None,
        estimated_peak_memory_bytes=None,
        backend=calibration.backend,
        metric=calibration.selected.metric,
        backend_version=calibration.backend_version,
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
            _whitening_manifest_with_diagnostics(local_representation.transform.to_manifest()),
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
        pool=pool,
        bandwidths=calibration.selected,
        graph_record=graph,
        contributions=contributions,
        entropy_result=entropy_result,
        graph_source_fingerprint=graph.fingerprint,
        contributions_source_fingerprint=contributions.fingerprint,
        selection_history=entropy_result.history,
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
    "nearest_selected_atomic_distances",
    "recompute_entropy_metrics",
    "recompute_final_entropy_metrics",
    "validate_calibration_manifest",
    "weighted_quantile",
]
