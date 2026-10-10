"""Validation specific to the generation phase's typed inputs."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase.neighborlist import neighbor_list

from nepflow.config.models import (
    ALL_SOURCES,
    GENERATION_SOURCE_SCOPE_FIELDS,
    SUPPORTED_CONFIGURATIONAL_SOURCES,
    SUPPORTED_SURFACE_MILLER_INDICES,
    CompositionConfig,
    GenerationConfig,
)
from nepflow.errors import ConfigurationError

from .perturbations.surfaces import (
    _POLARITY_POLICIES,
    _STOICHIOMETRY_POLICIES,
    _SURFACE_PLANNER_VERSION,
    SurfaceConstructionError,
    _composition_details,
    measure_surface_bulk_core,
    measure_surface_geometry,
)

_LIQUID_METHOD = "ase_langevin_lj"
_LIQUID_FIDELITY = "geometry_disorder_only_not_material_specific"


@dataclass(frozen=True, slots=True)
class CandidateValidationIssue:
    """Structured failure evidence for one generated candidate."""

    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", dict(self.evidence))


@dataclass(frozen=True, slots=True)
class _PairDistanceSearchResult:
    """Outcome of a bounded pair-distance search.

    ``distance=None`` with ``available=True`` means that no distinct atom pair
    lies within the search cutoff. It is a valid result, not an unavailable
    distance measurement.
    """

    distance: float | None
    available: bool


_RELAXED_GEOMETRY_FAMILIES = frozenset(
    {"volume_profile", "liquid", "surface", "grain_boundary", "grain-boundary"}
)
_INTERSTITIAL_FAMILIES = frozenset({"interstitial", "gas_interstitial", "vacancy_interstitial"})
_SUBSTITUTION_FAMILIES = frozenset({"substitution", "antisite"})
_NEUTRAL_COMPOSITION_FAMILIES = frozenset(
    {"unperturbed", "volume_profile", "elastic_stress", "rattled", "liquid"}
)


def validate_generated_candidate(
    candidate: Any,
    reference: Any | None,
    settings: Any,
    family: str,
) -> CandidateValidationIssue | None:
    """Validate one generated candidate against its family policy.

    ``reference`` is normally the task supercell, not the smaller source base.
    The function returns structured evidence instead of raising so a worker can
    retain one rejection while continuing to process the remaining candidates.
    """

    try:
        symbols = list(candidate.get_chemical_symbols())
    except Exception as exc:
        return CandidateValidationIssue(
            "invalid_species",
            {"error": f"{type(exc).__name__}: {exc}"},
        )
    if not symbols:
        return CandidateValidationIssue("empty_candidate", {"atom_count": 0})

    try:
        positions = np.asarray(candidate.get_positions(), dtype=float)
    except Exception as exc:
        return CandidateValidationIssue(
            "invalid_positions",
            {"error": f"{type(exc).__name__}: {exc}"},
        )
    if positions.shape != (len(symbols), 3):
        return CandidateValidationIssue(
            "invalid_positions",
            {"shape": list(positions.shape), "expected_shape": [len(symbols), 3]},
        )
    if not np.isfinite(positions).all():
        return CandidateValidationIssue("nonfinite_positions", {})

    try:
        cell = np.asarray(candidate.cell, dtype=float)
    except Exception as exc:
        return CandidateValidationIssue(
            "invalid_cell",
            {"error": f"{type(exc).__name__}: {exc}"},
        )
    if cell.shape != (3, 3):
        return CandidateValidationIssue(
            "invalid_cell",
            {"shape": list(cell.shape), "expected_shape": [3, 3]},
        )
    if not np.isfinite(cell).all():
        return CandidateValidationIssue("nonfinite_cell", {})

    pbc = _pbc_flags(candidate)
    if pbc is None:
        return CandidateValidationIssue("invalid_pbc", {})
    periodic_vectors = cell[pbc]
    required_rank = int(np.count_nonzero(pbc))
    periodic_rank = int(np.linalg.matrix_rank(periodic_vectors, tol=1.0e-10))
    if required_rank and periodic_rank < required_rank:
        return CandidateValidationIssue(
            "singular_periodic_cell",
            {"periodic_rank": periodic_rank, "required_rank": required_rank},
        )

    invalid_symbols = _invalid_symbols(symbols)
    if invalid_symbols:
        return CandidateValidationIssue("invalid_species", {"symbols": invalid_symbols})

    if reference is not None:
        issue = _validate_family_state(candidate, reference, settings, family)
        if issue is not None:
            return issue

    minimum_distance = _minimum_pair_distance_threshold(settings, family)
    distance_search = _bounded_pair_distance_search(candidate, pbc, minimum_distance)
    measured_distance = distance_search.distance
    if len(candidate) >= 2 and not distance_search.available:
        return CandidateValidationIssue("pair_distance_unavailable", {})
    if measured_distance is not None and measured_distance <= 1.0e-12:
        return CandidateValidationIssue(
            "overlapping_atoms",
            {"measured_distance": measured_distance, "threshold": 1.0e-12},
        )
    if (
        minimum_distance is not None
        and measured_distance is not None
        and measured_distance < minimum_distance
    ):
        return CandidateValidationIssue(
            "too_close_atoms",
            {"measured_distance": measured_distance, "threshold": minimum_distance},
        )
    return None


def _pbc_flags(candidate: Any) -> np.ndarray | None:
    try:
        values = np.asarray(candidate.pbc, dtype=bool)
    except Exception:
        return None
    if values.ndim == 0:
        values = np.repeat(values, 3)
    values = values.reshape(-1)
    return values if values.size == 3 else None


def _invalid_symbols(symbols: list[Any]) -> list[str]:
    try:
        from ase.data import atomic_numbers
    except ImportError:
        atomic_numbers = {}
    return [
        str(symbol)
        for symbol in symbols
        if not isinstance(symbol, str) or atomic_numbers.get(symbol, 0) <= 0
    ]


def _minimum_pair_distance_threshold(settings: Any, family: str) -> float | None:
    if family in _RELAXED_GEOMETRY_FAMILIES:
        return None
    if family == "rattled":
        return float(settings.rattle_d_min)
    if family in {"interstitial", "vacancy_interstitial"}:
        return max(float(settings.interstitial_d_min), float(settings.defect_defect_d_min))
    if family in {"gas_interstitial", "gas_in_vacancy"}:
        value = settings.gas_interstitial_d_min
        return max(
            float(settings.interstitial_d_min if value is None else value),
            float(settings.defect_defect_d_min),
        )
    return float(settings.rattle_d_min)


def _minimum_pair_distance(candidate: Any, pbc: np.ndarray) -> float | None:
    """Return a bounded-search minimum, retaining the legacy helper shape."""

    result = _bounded_pair_distance_search(candidate, pbc, None)
    return result.distance if result.available else None


def _bounded_pair_distance_search(
    candidate: Any,
    pbc: np.ndarray,
    threshold: float | None,
) -> _PairDistanceSearchResult:
    """Search only the periodic neighbour list needed for this validation.

    ASE's neighbour-list cell bins avoid constructing an ``N x N`` distance
    matrix. The cutoff is advanced by one representable float so pairs exactly
    on the configured boundary are discovered; the validation comparison still
    uses the original strict threshold, preserving the previous acceptance
    contract. A relaxed family searches only for the overlap guard.
    """

    if len(candidate) < 2:
        return _PairDistanceSearchResult(None, True)
    overlap_threshold = 1.0e-12
    cutoff = overlap_threshold
    if threshold is not None:
        try:
            numeric_threshold = float(threshold)
        except (TypeError, ValueError, OverflowError):
            return _PairDistanceSearchResult(None, False)
        if not math.isfinite(numeric_threshold):
            return _PairDistanceSearchResult(None, False)
        cutoff = max(cutoff, numeric_threshold)
    search_cutoff = float(np.nextafter(cutoff, np.inf))
    try:
        indices, neighbours, distances = neighbor_list(
            "ijd",
            candidate,
            search_cutoff,
            self_interaction=False,
        )
        del indices, neighbours
        values = np.asarray(distances, dtype=float)
    except Exception:
        return _PairDistanceSearchResult(None, False)
    if values.ndim != 1 or not np.isfinite(values).all():
        return _PairDistanceSearchResult(None, False)
    if values.size == 0:
        return _PairDistanceSearchResult(None, True)
    return _PairDistanceSearchResult(float(np.min(values)), True)


def _validate_family_state(
    candidate: Any,
    reference: Any,
    settings: Any,
    family: str,
) -> CandidateValidationIssue | None:
    if family == "liquid":
        issue = _validate_liquid_state(candidate)
        if issue is not None:
            return issue
    if family == "surface":
        issue = _validate_surface_state(candidate, reference, settings)
        if issue is not None:
            return issue
    if family == "grain_boundary":
        issue = _validate_grain_boundary_state(candidate)
        if issue is not None:
            return issue
    try:
        reference_symbols = list(reference.get_chemical_symbols())
    except Exception:
        return None
    info = getattr(candidate, "info", {})
    requested_interstitials = _metadata_integer(
        info, "requested_n_interstitials", "requested_n_gas_interstitials"
    )
    realised_interstitials = _metadata_integer(
        info,
        "realised_n_interstitials",
        "n_interstitials",
        "realised_n_gas_interstitials",
        "n_gas_interstitials",
    )
    if family in _INTERSTITIAL_FAMILIES and requested_interstitials is not None:
        if realised_interstitials is None:
            return CandidateValidationIssue(
                "missing_realised_defect_count",
                {"requested": requested_interstitials},
            )
        if requested_interstitials != realised_interstitials:
            return CandidateValidationIssue(
                "partial_interstitial_placement",
                {"requested": requested_interstitials, "realised": realised_interstitials},
            )

    requested_vacancies = _metadata_integer(info, "requested_n_vacancies")
    realised_vacancies = _metadata_integer(info, "realised_n_vacancies", "n_vacancies")
    if requested_vacancies is not None:
        if realised_vacancies is None:
            return CandidateValidationIssue(
                "missing_realised_defect_count",
                {"requested": requested_vacancies},
            )
        if requested_vacancies != realised_vacancies:
            return CandidateValidationIssue(
                "defect_count_mismatch",
                {"requested": requested_vacancies, "realised": realised_vacancies},
            )
    realised_gas = _metadata_integer(info, "realised_n_gas_atoms", "n_gas_atoms")
    requested_gas = _metadata_integer(info, "requested_n_gas_atoms")
    if family == "gas_in_vacancy" and requested_gas is not None:
        if realised_gas is None:
            return CandidateValidationIssue(
                "missing_realised_defect_count", {"requested": requested_gas}
            )
        if requested_gas != realised_gas:
            return CandidateValidationIssue(
                "defect_count_mismatch",
                {"requested": requested_gas, "realised": realised_gas},
            )

    requested_substitutions = _metadata_integer(info, "requested_n_substitutions")
    realised_substitutions = _metadata_integer(info, "realised_n_substitutions", "n_substitutions")
    if family == "substitution" and requested_substitutions is not None:
        if realised_substitutions is None:
            return CandidateValidationIssue(
                "missing_realised_defect_count",
                {"requested": requested_substitutions},
            )
        if requested_substitutions != realised_substitutions:
            return CandidateValidationIssue(
                "partial_defect_realisation",
                {"requested": requested_substitutions, "realised": realised_substitutions},
            )

    requested_antisites = _metadata_integer(info, "requested_n_antisites")
    realised_antisites = _metadata_integer(info, "realised_n_antisites", "n_antisites")
    if family == "antisite" and requested_antisites is not None:
        if realised_antisites is None:
            return CandidateValidationIssue(
                "missing_realised_defect_count",
                {"requested": requested_antisites},
            )
        if requested_antisites != realised_antisites:
            return CandidateValidationIssue(
                "partial_defect_realisation",
                {"requested": requested_antisites, "realised": realised_antisites},
            )

    expected_count: int | None = None
    if family == "vacancy" and realised_vacancies is not None:
        expected_count = len(reference_symbols) - realised_vacancies
    elif family in {"interstitial", "gas_interstitial"} and realised_interstitials is not None:
        expected_count = len(reference_symbols) + realised_interstitials
    elif family == "vacancy_interstitial":
        if realised_vacancies is not None and realised_interstitials is not None:
            expected_count = len(reference_symbols) - realised_vacancies + realised_interstitials
    elif family == "gas_in_vacancy":
        if realised_vacancies is not None and realised_gas is not None:
            expected_count = len(reference_symbols) - realised_vacancies + realised_gas
    elif family in _SUBSTITUTION_FAMILIES:
        expected_count = len(reference_symbols)
    elif family in _NEUTRAL_COMPOSITION_FAMILIES:
        expected_count = len(reference_symbols)

    if expected_count is not None and len(candidate) != expected_count:
        return CandidateValidationIssue(
            "defect_count_mismatch",
            {"expected_atom_count": expected_count, "realised_atom_count": len(candidate)},
        )

    candidate_counts = Counter(candidate.get_chemical_symbols())
    reference_counts = Counter(reference_symbols)
    if family in _NEUTRAL_COMPOSITION_FAMILIES and candidate_counts != reference_counts:
        return CandidateValidationIssue(
            "composition_mismatch",
            {
                "expected_counts": dict(reference_counts),
                "realised_counts": dict(candidate_counts),
            },
        )
    if family == "vacancy" and any(
        candidate_counts[element] > count for element, count in reference_counts.items()
    ):
        return CandidateValidationIssue(
            "composition_mismatch",
            {
                "expected_maximum_counts": dict(reference_counts),
                "realised_counts": dict(candidate_counts),
            },
        )
    if family in {"interstitial", "gas_interstitial"} and any(
        candidate_counts[element] < count for element, count in reference_counts.items()
    ):
        return CandidateValidationIssue(
            "composition_mismatch",
            {
                "expected_minimum_counts": dict(reference_counts),
                "realised_counts": dict(candidate_counts),
            },
        )
    if family == "substitution":
        source = info.get("source_species")
        target = info.get("target_species")
        if isinstance(source, str) and isinstance(target, str):
            expected_counts = reference_counts.copy()
            amount = realised_substitutions or 0
            expected_counts[source] -= amount
            expected_counts[target] += amount
            if expected_counts[source] < 0 or candidate_counts != +expected_counts:
                return CandidateValidationIssue(
                    "composition_mismatch",
                    {
                        "expected_counts": dict(+expected_counts),
                        "realised_counts": dict(candidate_counts),
                    },
                )
    if family == "antisite" and candidate_counts != reference_counts:
        return CandidateValidationIssue(
            "composition_mismatch",
            {
                "expected_counts": dict(reference_counts),
                "realised_counts": dict(candidate_counts),
            },
        )
    return None


def _validate_liquid_state(candidate: Any) -> CandidateValidationIssue | None:
    """Validate the reproducibility contract for an ASE liquid snapshot."""

    info = getattr(candidate, "info", {})
    required = (
        "liquid_method",
        "liquid_fidelity",
        "liquid_temperature_k",
        "liquid_timestep_fs",
        "liquid_friction",
        "liquid_equilibration_steps",
        "liquid_steps_between_snapshots",
        "liquid_configuration_index",
        "liquid_snapshot_index",
        "liquid_snapshot_step",
        "liquid_effective_child_seed",
        "parent_structure_id",
        "source_composition",
        "liquid_source_parent_structure_id",
        "liquid_source_composition",
    )
    missing = [key for key in required if key not in info]
    if missing:
        return CandidateValidationIssue("missing_liquid_provenance", {"fields": missing})
    if info["liquid_method"] != _LIQUID_METHOD or info["liquid_fidelity"] != _LIQUID_FIDELITY:
        return CandidateValidationIssue(
            "invalid_liquid_method",
            {
                "method": info["liquid_method"],
                "fidelity": info["liquid_fidelity"],
            },
        )
    try:
        temperature = float(info["liquid_temperature_k"])
        timestep = float(info["liquid_timestep_fs"])
        friction = float(info["liquid_friction"])
        equilibration_steps = int(info["liquid_equilibration_steps"])
        spacing = int(info["liquid_steps_between_snapshots"])
        configuration_index = int(info["liquid_configuration_index"])
        snapshot_index = int(info["liquid_snapshot_index"])
        snapshot_step = int(info["liquid_snapshot_step"])
        effective_seed = int(info["liquid_effective_child_seed"])
    except (TypeError, ValueError, OverflowError):
        return CandidateValidationIssue("invalid_liquid_provenance", {})
    if (
        not math.isfinite(temperature)
        or temperature < 0.0
        or not math.isfinite(timestep)
        or timestep <= 0.0
        or not math.isfinite(friction)
        or friction < 0.0
        or equilibration_steps < 0
        or spacing <= 0
        or configuration_index < 0
        or snapshot_index < 0
        or effective_seed < 0
        or snapshot_step != equilibration_steps + ((snapshot_index + 1) * spacing)
    ):
        return CandidateValidationIssue("invalid_liquid_provenance", {})
    if (
        not isinstance(info["liquid_source_parent_structure_id"], str)
        or not info["liquid_source_parent_structure_id"].strip()
    ):
        return CandidateValidationIssue("invalid_liquid_source", {})
    if info["liquid_source_composition"] is None:
        return CandidateValidationIssue("invalid_liquid_source", {})
    if info["source_composition"] is None or not isinstance(info["parent_structure_id"], str):
        return CandidateValidationIssue("invalid_liquid_source", {})
    if "liquid_target_temperature_k" in info:
        try:
            target_temperature = float(info["liquid_target_temperature_k"])
        except (TypeError, ValueError, OverflowError):
            return CandidateValidationIssue(
                "invalid_liquid_provenance", {"field": "liquid_target_temperature_k"}
            )
        if target_temperature != temperature:
            return CandidateValidationIssue(
                "invalid_liquid_provenance", {"field": "liquid_target_temperature_k"}
            )
    if "liquid_snapshot_spacing_steps" in info:
        try:
            snapshot_spacing = int(info["liquid_snapshot_spacing_steps"])
        except (TypeError, ValueError, OverflowError):
            return CandidateValidationIssue(
                "invalid_liquid_provenance", {"field": "liquid_snapshot_spacing_steps"}
            )
        if snapshot_spacing != spacing:
            return CandidateValidationIssue(
                "invalid_liquid_provenance", {"field": "liquid_snapshot_spacing_steps"}
            )
    if "liquid_trajectory_index" in info:
        try:
            trajectory_index = int(info["liquid_trajectory_index"])
        except (TypeError, ValueError, OverflowError):
            return CandidateValidationIssue(
                "invalid_liquid_provenance", {"field": "liquid_trajectory_index"}
            )
        if trajectory_index != configuration_index:
            return CandidateValidationIssue(
                "invalid_liquid_provenance", {"field": "liquid_trajectory_index"}
            )
    for alias in ("liquid_random_seed", "random_seed"):
        if alias in info:
            try:
                alias_seed = int(info[alias])
            except (TypeError, ValueError, OverflowError):
                return CandidateValidationIssue("invalid_liquid_seed", {"field": alias})
            if alias_seed != effective_seed:
                return CandidateValidationIssue("invalid_liquid_seed", {"field": alias})
    return None


def _validate_surface_state(
    candidate: Any,
    reference: Any,
    settings: Any,
) -> CandidateValidationIssue | None:
    """Validate surface-specific provenance and two-dimensional periodicity."""

    info = getattr(candidate, "info", {})
    required = (
        "parent_structure_id",
        "surface_state",
        "surface_miller_index",
        "surface_termination",
        "surface_termination_descriptor",
        "surface_termination_identity",
        "surface_layers",
        "surface_vacuum",
        "surface_requested_vacuum",
        "surface_requested_vacuum_angstrom",
        "surface_realized_vacuum",
        "surface_realized_vacuum_angstrom",
        "surface_slab_thickness",
        "surface_material_thickness",
        "surface_half_depth",
        "surface_min_half_depth",
        "surface_in_plane_lengths",
        "surface_in_plane_angle_degrees",
        "surface_in_plane_area",
        "surface_shortest_in_plane_translation",
        "surface_projected_coordinates",
        "surface_normal_period",
        "surface_bulk_environment_radius",
        "surface_bulk_environment_distance_tolerance",
        "surface_min_bulk_core_atoms",
        "surface_bulk_core_eligible_atom_count",
        "surface_bulk_core_atom_count",
        "surface_bulk_core_atom_indices",
        "surface_reference_basis",
        "surface_reference_basis_cell",
        "surface_parent_to_reference_transformation",
        "surface_normal",
        "surface_target_n_atoms",
        "surface_target_tolerance",
        "surface_max_n_atoms",
        "surface_planner_min_in_plane_repeat",
        "surface_planner_min_in_plane_dimensions",
        "surface_max_in_plane_repeat",
        "surface_max_normal_repeat",
        "surface_planner_repeat",
        "surface_normal_repeat",
        "surface_realized_atom_count",
        "surface_atom_count_delta",
        "surface_target_band",
        "surface_material_shape_score",
        "surface_excess_vacuum",
        "surface_planner_tie_break",
        "surface_backend",
        "surface_backend_version",
        "surface_planner_version",
        "surface_in_plane_repeat",
        "surface_stoichiometry_change",
        "surface_symmetric",
        "surface_symmetric_requested",
        "surface_symmetry_status",
        "surface_stoichiometry_policy",
        "surface_stoichiometry_policy_result",
        "surface_polarity_policy",
        "surface_polarity",
        "surface_polarity_status",
        "surface_polarity_policy_result",
        "surface_parent_species_counts",
        "surface_parent_species_fractions",
        "surface_slab_species_counts",
        "surface_slab_species_fractions",
        "surface_atom_count_change",
        "surface_stoichiometry_changed",
    )
    missing = [key for key in required if key not in info]
    if missing:
        return CandidateValidationIssue("missing_surface_provenance", {"fields": missing})
    if info["surface_state"] != "pristine":
        return CandidateValidationIssue(
            "invalid_surface_state",
            {"value": info["surface_state"], "expected": "pristine"},
        )
    if info["surface_planner_version"] != _SURFACE_PLANNER_VERSION:
        return CandidateValidationIssue(
            "unsupported_surface_planner_version",
            {"value": info["surface_planner_version"]},
        )
    pbc = _pbc_flags(candidate)
    if pbc is None or tuple(bool(value) for value in pbc) != (True, True, False):
        return CandidateValidationIssue(
            "invalid_surface_pbc",
            {"expected": [True, True, False], "realised": None if pbc is None else pbc.tolist()},
        )
    try:
        miller = tuple(int(value) for value in info["surface_miller_index"])
    except (TypeError, ValueError):
        return CandidateValidationIssue("invalid_surface_miller_index", {})
    if len(miller) != 3 or not any(miller) or miller not in SUPPORTED_SURFACE_MILLER_INDICES:
        return CandidateValidationIssue("invalid_surface_miller_index", {"value": miller})
    if info["surface_reference_basis"] != "parent_stored_cell":
        return CandidateValidationIssue(
            "invalid_surface_reference_basis",
            {"value": info["surface_reference_basis"]},
        )
    try:
        reference_cell = np.asarray(info["surface_reference_basis_cell"], dtype=float)
        transform = np.asarray(info["surface_parent_to_reference_transformation"], dtype=float)
    except (TypeError, ValueError):
        return CandidateValidationIssue("invalid_surface_reference_basis", {})
    if (
        reference_cell.shape != (3, 3)
        or not np.isfinite(reference_cell).all()
        or abs(float(np.linalg.det(reference_cell))) <= 1.0e-10
    ):
        return CandidateValidationIssue("invalid_surface_reference_basis", {})
    if (
        transform.shape != (3, 3)
        or not np.isfinite(transform).all()
        or not np.allclose(transform, np.eye(3), rtol=0.0, atol=1.0e-12)
    ):
        return CandidateValidationIssue("invalid_surface_reference_transformation", {})
    try:
        vacuum = float(info["surface_vacuum"])
        requested_vacuum_alias = float(info["surface_requested_vacuum"])
        requested_vacuum = float(info["surface_requested_vacuum_angstrom"])
        realized_vacuum_alias = float(info["surface_realized_vacuum"])
        realized_vacuum = float(info["surface_realized_vacuum_angstrom"])
        normal = np.asarray(info["surface_normal"], dtype=float)
    except (TypeError, ValueError):
        return CandidateValidationIssue("invalid_surface_geometry_provenance", {})
    if (
        not math.isfinite(requested_vacuum)
        or requested_vacuum <= 0.0
        or not math.isfinite(vacuum)
        or not math.isclose(vacuum, requested_vacuum, rel_tol=1.0e-6, abs_tol=1.0e-6)
        or not math.isfinite(requested_vacuum_alias)
        or not math.isclose(
            requested_vacuum_alias, requested_vacuum, rel_tol=1.0e-6, abs_tol=1.0e-6
        )
        or not math.isclose(
            requested_vacuum, float(settings.surface_vacuum), rel_tol=1.0e-6, abs_tol=1.0e-6
        )
        or not math.isfinite(realized_vacuum)
        or not math.isfinite(realized_vacuum_alias)
        or not math.isclose(realized_vacuum_alias, realized_vacuum, rel_tol=1.0e-6, abs_tol=1.0e-6)
        or realized_vacuum + 1.0e-6 < requested_vacuum
        or normal.shape != (3,)
        or not np.isfinite(normal).all()
        or not math.isclose(float(np.linalg.norm(normal)), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
    ):
        return CandidateValidationIssue(
            "invalid_surface_geometry_provenance",
            {
                "requested_vacuum": requested_vacuum,
                "realized_vacuum": realized_vacuum,
            },
        )
    cell = np.asarray(candidate.cell, dtype=float)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        return CandidateValidationIssue("invalid_surface_geometry", {})
    cross = np.cross(cell[0], cell[1])
    area = float(np.linalg.norm(cross))
    if area <= 1.0e-12:
        return CandidateValidationIssue("invalid_surface_geometry", {})
    measured_normal = cross / area
    projected = np.asarray(candidate.get_positions(), dtype=float) @ measured_normal
    measured_thickness = float(np.max(projected) - np.min(projected)) if len(projected) else -1.0
    measured_period = abs(float(np.linalg.det(cell))) / area if cell.shape == (3, 3) else 0.0
    measured_vacuum = measured_period - measured_thickness
    if (
        not math.isfinite(measured_vacuum)
        or not math.isclose(measured_vacuum, realized_vacuum, rel_tol=1.0e-6, abs_tol=1.0e-6)
        or abs(float(np.dot(measured_normal, normal))) < 1.0 - 1.0e-6
    ):
        return CandidateValidationIssue(
            "invalid_surface_geometry_measurement",
            {
                "measured_vacuum": measured_vacuum,
                "recorded_vacuum": realized_vacuum,
            },
        )
    try:
        geometry = measure_surface_geometry(candidate)
        bulk_core = measure_surface_bulk_core(candidate, reference, geometry, settings)
    except SurfaceConstructionError as exc:
        return CandidateValidationIssue("invalid_surface_geometry", {"error": str(exc)})
    if geometry.realized_vacuum + 1.0e-6 < requested_vacuum:
        return CandidateValidationIssue(
            "insufficient_surface_vacuum",
            {
                "requested_vacuum": requested_vacuum,
                "realized_vacuum": geometry.realized_vacuum,
            },
        )
    if geometry.half_depth + 1.0e-6 < float(settings.surface_min_half_depth):
        return CandidateValidationIssue(
            "insufficient_surface_depth",
            {
                "minimum_half_depth": float(settings.surface_min_half_depth),
                "realized_half_depth": geometry.half_depth,
            },
        )
    if bulk_core.bulk_core_atom_count < int(settings.surface_min_bulk_core_atoms):
        return CandidateValidationIssue(
            "insufficient_surface_bulk_core",
            {
                "required_bulk_core_atoms": int(settings.surface_min_bulk_core_atoms),
                "realized_bulk_core_atoms": bulk_core.bulk_core_atom_count,
                "eligible_bulk_core_atoms": bulk_core.eligible_atom_count,
            },
        )
    if len(candidate) > int(settings.surface_max_n_atoms):
        return CandidateValidationIssue(
            "surface_atom_count_limit",
            {
                "maximum_atom_count": int(settings.surface_max_n_atoms),
                "realized_atom_count": len(candidate),
            },
        )
    try:
        stored_lengths = tuple(float(value) for value in info["surface_in_plane_lengths"])
        stored_angle = float(info["surface_in_plane_angle_degrees"])
        stored_area = float(info["surface_in_plane_area"])
        stored_shortest = float(info["surface_shortest_in_plane_translation"])
        stored_projections = tuple(float(value) for value in info["surface_projected_coordinates"])
        stored_period = float(info["surface_normal_period"])
        stored_normal = tuple(float(value) for value in info["surface_normal"])
        stored_slab_thickness = float(info["surface_slab_thickness"])
        stored_thickness = float(info["surface_material_thickness"])
        stored_half_depth = float(info["surface_half_depth"])
        stored_min_half_depth = float(info["surface_min_half_depth"])
        stored_radius = float(info["surface_bulk_environment_radius"])
        stored_tolerance = float(info["surface_bulk_environment_distance_tolerance"])
        stored_min_core = int(info["surface_min_bulk_core_atoms"])
        stored_eligible = int(info["surface_bulk_core_eligible_atom_count"])
        stored_core = int(info["surface_bulk_core_atom_count"])
        stored_core_indices = tuple(int(value) for value in info["surface_bulk_core_atom_indices"])
        stored_in_plane_repeat = tuple(int(value) for value in info["surface_in_plane_repeat"])
        stored_target = int(info["surface_target_n_atoms"])
        stored_target_tolerance = float(info["surface_target_tolerance"])
        stored_max_atoms = int(info["surface_max_n_atoms"])
        stored_min_in_plane = tuple(
            int(value) for value in info["surface_planner_min_in_plane_repeat"]
        )
        stored_min_dimensions = tuple(
            float(value) for value in info["surface_planner_min_in_plane_dimensions"]
        )
        stored_max_in_plane = tuple(int(value) for value in info["surface_max_in_plane_repeat"])
        stored_max_normal = int(info["surface_max_normal_repeat"])
        stored_planner_repeat = tuple(int(value) for value in info["surface_planner_repeat"])
        stored_normal_repeat = int(info["surface_normal_repeat"])
        stored_realized_atoms = int(info["surface_realized_atom_count"])
        stored_atom_delta = float(info["surface_atom_count_delta"])
        stored_target_band = bool(info["surface_target_band"])
        stored_shape_score = float(info["surface_material_shape_score"])
        stored_excess_vacuum = float(info["surface_excess_vacuum"])
        stored_tie_break = tuple(float(value) for value in info["surface_planner_tie_break"])
        stored_descriptor = str(info["surface_termination_descriptor"])
        stored_identity = str(info["surface_termination_identity"])
        stored_symmetric = bool(info["surface_symmetric"])
        stored_symmetric_requested = bool(info["surface_symmetric_requested"])
        stored_symmetry_status = str(info["surface_symmetry_status"])
        stored_stoichiometry_policy = str(info["surface_stoichiometry_policy"])
        stored_stoichiometry_result = str(info["surface_stoichiometry_policy_result"])
        stored_polarity_policy = str(info["surface_polarity_policy"])
        stored_polarity = str(info["surface_polarity"])
        stored_polarity_status = str(info["surface_polarity_status"])
        stored_polarity_result = str(info["surface_polarity_policy_result"])
        stored_parent_counts = {
            str(key): int(value)
            for key, value in dict(info["surface_parent_species_counts"]).items()
        }
        stored_parent_fractions = {
            str(key): float(value)
            for key, value in dict(info["surface_parent_species_fractions"]).items()
        }
        stored_slab_counts = {
            str(key): int(value) for key, value in dict(info["surface_slab_species_counts"]).items()
        }
        stored_slab_fractions = {
            str(key): float(value)
            for key, value in dict(info["surface_slab_species_fractions"]).items()
        }
        stored_count_delta = {
            str(key): int(value) for key, value in dict(info["surface_atom_count_change"]).items()
        }
        stored_stoichiometry_changed = bool(info["surface_stoichiometry_changed"])
    except (TypeError, ValueError):
        return CandidateValidationIssue("invalid_surface_geometry_provenance", {})
    geometry_matches = (
        len(stored_normal) == 3
        and np.allclose(stored_normal, geometry.normal, rtol=1.0e-6, atol=1.0e-6)
        and len(stored_lengths) == 2
        and np.allclose(stored_lengths, geometry.in_plane_lengths, rtol=1.0e-6, atol=1.0e-6)
        and math.isclose(
            stored_angle,
            geometry.in_plane_angle_degrees,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        )
        and math.isclose(stored_area, geometry.in_plane_area, rel_tol=1.0e-6, abs_tol=1.0e-6)
        and math.isclose(
            stored_shortest,
            geometry.shortest_in_plane_translation,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        )
        and len(stored_projections) == len(geometry.projected_coordinates)
        and np.allclose(
            stored_projections,
            geometry.projected_coordinates,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        and math.isclose(stored_period, geometry.normal_period, rel_tol=1.0e-6, abs_tol=1.0e-6)
        and math.isclose(
            stored_slab_thickness,
            geometry.material_thickness,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        )
        and math.isclose(
            stored_thickness, geometry.material_thickness, rel_tol=1.0e-6, abs_tol=1.0e-6
        )
        and math.isclose(stored_half_depth, geometry.half_depth, rel_tol=1.0e-6, abs_tol=1.0e-6)
        and math.isclose(
            stored_min_half_depth,
            float(settings.surface_min_half_depth),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        )
        and math.isclose(
            stored_radius, bulk_core.environment_radius, rel_tol=1.0e-6, abs_tol=1.0e-6
        )
        and math.isclose(
            stored_tolerance, bulk_core.distance_tolerance, rel_tol=1.0e-6, abs_tol=1.0e-6
        )
        and stored_min_core == int(settings.surface_min_bulk_core_atoms)
        and stored_eligible == bulk_core.eligible_atom_count
        and stored_core == bulk_core.bulk_core_atom_count
        and stored_core_indices == bulk_core.bulk_core_atom_indices
    )
    if not geometry_matches:
        return CandidateValidationIssue(
            "surface_geometry_provenance_mismatch",
            {
                "realized_thickness": geometry.material_thickness,
                "realized_bulk_core_atoms": bulk_core.bulk_core_atom_count,
            },
        )
    try:
        (
            expected_parent_counts,
            expected_parent_fractions,
            expected_slab_counts,
            expected_slab_fractions,
            expected_count_delta,
            expected_composition_delta,
        ) = _composition_details(reference, candidate)
        expected_stoichiometry_changed = bool(expected_composition_delta)
        expected_stoichiometry_policy = _STOICHIOMETRY_POLICIES[
            str(settings.surface_stoichiometry_policy).strip().lower()
        ]
        expected_polarity_policy = _POLARITY_POLICIES[
            str(settings.surface_polarity_policy).strip().lower()
        ]
    except (KeyError, TypeError, ValueError):
        return CandidateValidationIssue("invalid_surface_chemistry_provenance", {})
    chemistry_matches = (
        bool(stored_descriptor)
        and stored_descriptor in stored_identity
        and stored_symmetry_status in {"symmetric", "asymmetric", "unknown"}
        and stored_symmetric == (stored_symmetry_status == "symmetric")
        and stored_symmetric_requested == bool(settings.surface_symmetric)
        and stored_stoichiometry_policy == expected_stoichiometry_policy
        and stored_stoichiometry_result == "accepted"
        and stored_polarity_policy == expected_polarity_policy
        and stored_polarity == stored_polarity_status
        and stored_parent_counts == expected_parent_counts
        and stored_slab_counts == expected_slab_counts
        and stored_count_delta == expected_count_delta
        and stored_stoichiometry_changed == expected_stoichiometry_changed
        and set(stored_parent_fractions) == set(expected_parent_fractions)
        and np.allclose(
            [stored_parent_fractions[key] for key in sorted(expected_parent_fractions)],
            [expected_parent_fractions[key] for key in sorted(expected_parent_fractions)],
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        and set(stored_slab_fractions) == set(expected_slab_fractions)
        and np.allclose(
            [stored_slab_fractions[key] for key in sorted(expected_slab_fractions)],
            [expected_slab_fractions[key] for key in sorted(expected_slab_fractions)],
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        and info["surface_stoichiometry_change"] == expected_composition_delta
        and stored_polarity_result == "accepted"
    )
    if not chemistry_matches:
        return CandidateValidationIssue("surface_chemistry_provenance_mismatch", {})
    if settings.surface_symmetric and stored_symmetry_status != "symmetric":
        return CandidateValidationIssue("surface_symmetry_policy_violation", {})
    if expected_stoichiometry_policy == "reject" and expected_stoichiometry_changed:
        return CandidateValidationIssue("surface_stoichiometry_policy_violation", {})
    if expected_polarity_policy == "reject_known_polar" and stored_polarity == "polar":
        return CandidateValidationIssue("surface_polarity_policy_violation", {})
    if expected_polarity_policy == "require_known_nonpolar" and stored_polarity != "nonpolar":
        return CandidateValidationIssue("surface_polarity_policy_violation", {})
    target_atoms = (
        int(settings.target_n_atoms)
        if settings.surface_target_n_atoms is None
        else int(settings.surface_target_n_atoms)
    )
    if target_atoms <= 0:
        return CandidateValidationIssue("surface_planner_provenance_mismatch", {})
    expected_delta = abs(len(candidate) - target_atoms) / target_atoms
    dimensions = (*geometry.in_plane_lengths, geometry.material_thickness)
    expected_shape_score = max(dimensions) / min(dimensions) - 1.0
    expected_excess_vacuum = max(0.0, geometry.realized_vacuum - float(settings.surface_vacuum))
    expected_band = expected_delta <= float(settings.surface_target_tolerance) + 1.0e-12
    expected_repeat = stored_planner_repeat
    expected_tie_break = (
        expected_shape_score,
        float(len(candidate)),
        expected_excess_vacuum,
        float(expected_repeat[0]) if len(expected_repeat) > 0 else -1.0,
        float(expected_repeat[1]) if len(expected_repeat) > 1 else -1.0,
        float(expected_repeat[2]) if len(expected_repeat) > 2 else -1.0,
    )
    planner_matches = (
        target_atoms > 0
        and stored_target == target_atoms
        and math.isclose(
            stored_target_tolerance,
            float(settings.surface_target_tolerance),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        )
        and stored_max_atoms == int(settings.surface_max_n_atoms)
        and stored_min_in_plane == tuple(int(value) for value in settings.surface_in_plane_repeat)
        and len(stored_min_dimensions) == 2
        and np.allclose(
            stored_min_dimensions,
            tuple(float(value) for value in settings.surface_min_in_plane_dimensions),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        and stored_max_in_plane
        == tuple(int(value) for value in settings.surface_max_in_plane_repeat)
        and stored_max_normal == int(settings.surface_max_normal_repeat)
        and len(expected_repeat) == 3
        and all(value > 0 for value in expected_repeat)
        and stored_normal_repeat == expected_repeat[2]
        and expected_repeat[0] >= int(settings.surface_in_plane_repeat[0])
        and expected_repeat[1] >= int(settings.surface_in_plane_repeat[1])
        and expected_repeat[0] <= int(settings.surface_max_in_plane_repeat[0])
        and expected_repeat[1] <= int(settings.surface_max_in_plane_repeat[1])
        and expected_repeat[2] <= int(settings.surface_max_normal_repeat)
        and stored_in_plane_repeat == expected_repeat[:2]
        and stored_realized_atoms == len(candidate)
        and len(candidate) <= int(settings.surface_max_n_atoms)
        and math.isclose(stored_atom_delta, expected_delta, rel_tol=1.0e-6, abs_tol=1.0e-6)
        and stored_target_band == expected_band
        and math.isclose(stored_shape_score, expected_shape_score, rel_tol=1.0e-6, abs_tol=1.0e-6)
        and math.isclose(
            stored_excess_vacuum,
            expected_excess_vacuum,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        )
        and len(stored_tie_break) == len(expected_tie_break)
        and np.allclose(stored_tie_break, expected_tie_break, rtol=1.0e-6, atol=1.0e-6)
    )
    if not planner_matches:
        return CandidateValidationIssue(
            "surface_planner_provenance_mismatch",
            {
                "realized_atom_count": len(candidate),
                "target_atom_count": target_atoms,
                "atom_count_delta": expected_delta,
            },
        )
    if len(stored_in_plane_repeat) != 2 or any(value <= 0 for value in stored_in_plane_repeat):
        return CandidateValidationIssue("invalid_surface_repeat", {"value": stored_in_plane_repeat})
    return None


def _validate_grain_boundary_state(candidate: Any) -> CandidateValidationIssue | None:
    """Validate relationship, periodicity, and overlap-removal provenance."""

    info = getattr(candidate, "info", {})
    required = (
        "parent_structure_id",
        "grain_boundary_relationship",
        "grain_boundary_rotation_axis",
        "grain_boundary_misorientation_angle",
        "grain_boundary_sigma",
        "grain_boundary_plane",
        "grain_boundary_expand_times",
        "grain_boundary_overlap_tolerance",
        "grain_boundary_removed_atom_count",
        "grain_boundary_removed_species",
        "grain_boundary_cell_lengths",
    )
    missing = [key for key in required if key not in info]
    if missing:
        return CandidateValidationIssue("missing_grain_boundary_provenance", {"fields": missing})
    pbc = _pbc_flags(candidate)
    if pbc is None or tuple(bool(value) for value in pbc) != (True, True, True):
        return CandidateValidationIssue(
            "invalid_grain_boundary_pbc",
            {"expected": [True, True, True], "realised": None if pbc is None else pbc.tolist()},
        )
    try:
        axis = tuple(int(value) for value in info["grain_boundary_rotation_axis"])
        plane = tuple(int(value) for value in info["grain_boundary_plane"])
        sigma = int(info["grain_boundary_sigma"])
        angle = float(info["grain_boundary_misorientation_angle"])
        expand_times = int(info["grain_boundary_expand_times"])
        tolerance = float(info["grain_boundary_overlap_tolerance"])
        removed_count = int(info["grain_boundary_removed_atom_count"])
        cell_lengths = tuple(float(value) for value in info["grain_boundary_cell_lengths"])
    except (TypeError, ValueError):
        return CandidateValidationIssue("invalid_grain_boundary_provenance", {})
    if axis != (0, 0, 1) or plane != (2, 1, 0) or sigma != 5:
        return CandidateValidationIssue(
            "unsupported_grain_boundary_relationship",
            {"axis": axis, "plane": plane, "sigma": sigma},
        )
    if not math.isclose(angle, 36.86989764584402, rel_tol=1.0e-9, abs_tol=1.0e-8):
        return CandidateValidationIssue("unsupported_grain_boundary_relationship", {"angle": angle})
    if expand_times <= 0 or not 0.0 <= tolerance <= 1.0:
        return CandidateValidationIssue(
            "invalid_grain_boundary_provenance",
            {"expand_times": expand_times, "overlap_tolerance": tolerance},
        )
    removed_species = info["grain_boundary_removed_species"]
    if not isinstance(removed_species, dict) or any(
        not isinstance(value, int) or value < 0 for value in removed_species.values()
    ):
        return CandidateValidationIssue("invalid_overlap_removal_accounting", {})
    if removed_count != sum(removed_species.values()):
        return CandidateValidationIssue(
            "invalid_overlap_removal_accounting",
            {"removed_count": removed_count, "removed_species": removed_species},
        )
    if (
        removed_count < 0
        or len(cell_lengths) != 3
        or any(not math.isfinite(value) or value <= 0.0 for value in cell_lengths)
    ):
        return CandidateValidationIssue("invalid_grain_boundary_geometry", {})
    return None


def _metadata_integer(info: Any, *keys: str) -> int | None:
    for key in keys:
        if key in info:
            try:
                value = int(info[key])
            except (TypeError, ValueError):
                return None
            return value if value >= 0 else None
    return None


def validate_composition_config(config: CompositionConfig) -> CompositionConfig:
    """Validate the supported unary/binary/ternary composition space."""

    if not config.elements:
        raise ConfigurationError("generation requires at least one composition element")
    if not math.isfinite(config.composition_step) or not 0.0 < config.composition_step <= 1.0:
        raise ConfigurationError("composition.composition_step must be in (0, 1]")
    steps = round(1.0 / config.composition_step)
    if steps < 1 or not math.isclose(
        steps * config.composition_step,
        1.0,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ConfigurationError(
            "composition.composition_step must divide the unit interval exactly"
        )
    if set(config.elements) & set(config.gas_elements):
        raise ConfigurationError("composition.elements and gas_elements must be disjoint")
    return config


def validate_generation_config(config: GenerationConfig) -> GenerationConfig:
    """Validate generator-facing settings without changing their defaults."""

    if not config.crystal_structures:
        raise ConfigurationError("generation.crystal_structures must not be empty")
    if config.target_n_atoms <= 0:
        raise ConfigurationError("generation.target_n_atoms must be positive")
    if (
        not math.isfinite(config.composition_tolerance)
        or not 0.0 <= config.composition_tolerance <= 1.0
    ):
        raise ConfigurationError("generation.composition_tolerance must be in [0, 1]")
    if config.n_workers < 0:
        raise ConfigurationError("generation.n_workers must be non-negative")
    if config.n_surfaces < 0:
        raise ConfigurationError("generation.n_surfaces must be non-negative")
    if config.surface_layers < 0:
        raise ConfigurationError("generation.surface_layers must be non-negative")
    if config.surface_thickness is None and config.surface_layers == 0:
        raise ConfigurationError(
            "generation.surface_layers must be positive when surface_thickness is not set"
        )
    if not math.isfinite(config.surface_min_half_depth) or config.surface_min_half_depth <= 0.0:
        raise ConfigurationError("generation.surface_min_half_depth must be positive and finite")
    if (
        not math.isfinite(config.surface_bulk_environment_radius)
        or config.surface_bulk_environment_radius <= 0.0
    ):
        raise ConfigurationError(
            "generation.surface_bulk_environment_radius must be positive and finite"
        )
    if config.surface_min_bulk_core_atoms <= 0:
        raise ConfigurationError("generation.surface_min_bulk_core_atoms must be positive")
    if (
        not math.isfinite(config.surface_bulk_environment_distance_tolerance)
        or config.surface_bulk_environment_distance_tolerance < 0.0
    ):
        raise ConfigurationError(
            "generation.surface_bulk_environment_distance_tolerance must be finite and non-negative"
        )
    if config.surface_target_n_atoms is not None and (
        isinstance(config.surface_target_n_atoms, bool) or config.surface_target_n_atoms <= 0
    ):
        raise ConfigurationError("generation.surface_target_n_atoms must be positive when set")
    if not math.isfinite(config.surface_target_tolerance) or config.surface_target_tolerance < 0.0:
        raise ConfigurationError(
            "generation.surface_target_tolerance must be finite and non-negative"
        )
    if config.surface_max_n_atoms <= 0:
        raise ConfigurationError("generation.surface_max_n_atoms must be positive")
    if config.surface_max_terminations < 0:
        raise ConfigurationError("generation.surface_max_terminations must be non-negative")
    if config.surface_termination_policy not in {"all", "first"}:
        raise ConfigurationError("generation.surface_termination_policy must be 'all' or 'first'")
    for index in config.surface_miller_indices:
        if (
            len(index) != 3
            or any(not isinstance(value, int) or isinstance(value, bool) for value in index)
            or not any(index)
        ):
            raise ConfigurationError(
                "generation.surface_miller_indices must contain non-zero triples"
            )
        if tuple(index) not in SUPPORTED_SURFACE_MILLER_INDICES:
            raise ConfigurationError(
                f"generation.surface_miller_indices contains unsupported orientation {tuple(index)}"
            )
    if len(config.surface_in_plane_repeat) != 2 or any(
        value <= 0 for value in config.surface_in_plane_repeat
    ):
        raise ConfigurationError(
            "generation.surface_in_plane_repeat must contain two positive integers"
        )
    if len(config.surface_max_in_plane_repeat) != 2 or any(
        value <= 0 for value in config.surface_max_in_plane_repeat
    ):
        raise ConfigurationError(
            "generation.surface_max_in_plane_repeat must contain two positive integers"
        )
    if any(
        lower > upper
        for lower, upper in zip(config.surface_in_plane_repeat, config.surface_max_in_plane_repeat)
    ):
        raise ConfigurationError(
            "generation.surface_max_in_plane_repeat must not be below surface_in_plane_repeat"
        )
    if config.surface_max_normal_repeat <= 0:
        raise ConfigurationError("generation.surface_max_normal_repeat must be positive")
    if len(config.surface_min_in_plane_dimensions) != 2 or any(
        not math.isfinite(value) or value < 0.0 for value in config.surface_min_in_plane_dimensions
    ):
        raise ConfigurationError(
            "generation.surface_min_in_plane_dimensions must be finite and non-negative"
        )
    if config.surface_thickness is not None and (
        not math.isfinite(config.surface_thickness) or config.surface_thickness <= 0.0
    ):
        raise ConfigurationError("generation.surface_thickness must be positive and finite")
    if not math.isfinite(config.surface_vacuum) or config.surface_vacuum <= 0.0:
        raise ConfigurationError("generation.surface_vacuum must be positive and finite")
    if str(config.surface_stoichiometry_policy).strip().lower() not in {
        "allow",
        "reject",
        "reject_changed",
        "require_stoichiometric",
    }:
        raise ConfigurationError(
            "generation.surface_stoichiometry_policy must be 'allow' or 'reject'"
        )
    if str(config.surface_polarity_policy).strip().lower() not in {
        "allow",
        "reject_known_polar",
        "reject_polar",
        "require_known_nonpolar",
        "require_nonpolar",
    }:
        raise ConfigurationError(
            "generation.surface_polarity_policy must be 'allow', "
            "'reject_known_polar', or 'require_known_nonpolar'"
        )
    if config.n_grain_boundaries < 0:
        raise ConfigurationError("generation.n_grain_boundaries must be non-negative")
    if tuple(config.grain_boundary_rotation_axis) != (0, 0, 1):
        raise ConfigurationError(
            "only the supported Sigma-5 [001] grain-boundary axis is available"
        )
    if tuple(config.grain_boundary_plane) != (2, 1, 0):
        raise ConfigurationError(
            "only the supported Sigma-5 (210) grain-boundary plane is available"
        )
    if config.grain_boundary_sigma != 5:
        raise ConfigurationError("only Sigma 5 grain boundaries are supported")
    if not math.isclose(
        config.grain_boundary_misorientation_angle,
        36.86989764584402,
        rel_tol=1.0e-9,
        abs_tol=1.0e-8,
    ):
        raise ConfigurationError("unsupported grain-boundary misorientation angle")
    if config.grain_boundary_expand_times <= 0:
        raise ConfigurationError("generation.grain_boundary_expand_times must be positive")
    if (
        not math.isfinite(config.grain_boundary_min_thickness)
        or config.grain_boundary_min_thickness < 0.0
    ):
        raise ConfigurationError(
            "generation.grain_boundary_min_thickness must be finite and non-negative"
        )
    if not 0.0 <= config.grain_boundary_overlap_tolerance <= 1.0:
        raise ConfigurationError("generation.grain_boundary_overlap_tolerance must be in [0, 1]")
    if config.use_liquid and (config.n_liquid_configurations > 0 and config.n_liquid_snapshots > 0):
        if config.liquid_timestep_fs <= 0.0:
            raise ConfigurationError("generation.liquid_timestep_fs must be positive")
        if config.liquid_steps_between_snapshots <= 0:
            raise ConfigurationError("generation.liquid_steps_between_snapshots must be positive")
    _validate_source_scopes(config)
    return config


def _validate_source_scopes(config: GenerationConfig) -> None:
    for field_name in GENERATION_SOURCE_SCOPE_FIELDS:
        scope = getattr(config, field_name)
        name = f"generation.{field_name}"
        if not isinstance(scope, tuple) or not scope:
            raise ConfigurationError(f"{name} must contain at least one explicit source")
        if any(not isinstance(source, str) or not source.strip() for source in scope):
            raise ConfigurationError(f"{name} contains a blank source")
        normalized = tuple(source.strip().lower() for source in scope)
        if normalized != scope:
            raise ConfigurationError(f"{name} must use normalized lowercase source names")
        if len(set(scope)) != len(scope):
            raise ConfigurationError(f"{name} must not contain duplicate sources")
        if ALL_SOURCES in scope and len(scope) != 1:
            raise ConfigurationError(f"{name} cannot combine 'all' with named sources")
        unknown = set(scope) - SUPPORTED_CONFIGURATIONAL_SOURCES - {ALL_SOURCES}
        if unknown:
            raise ConfigurationError(
                f"{name} contains unsupported sources: {', '.join(sorted(unknown))}"
            )


__all__ = [
    "CandidateValidationIssue",
    "validate_composition_config",
    "validate_generated_candidate",
    "validate_generation_config",
]
