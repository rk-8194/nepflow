"""Validation specific to the generation phase's typed inputs."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nepflow.config.models import (
    ALL_SOURCES,
    GENERATION_SOURCE_SCOPE_FIELDS,
    SUPPORTED_CONFIGURATIONAL_SOURCES,
    CompositionConfig,
    GenerationConfig,
)
from nepflow.errors import ConfigurationError

_LIQUID_METHOD = "ase_langevin_lj"
_LIQUID_FIDELITY = "geometry_disorder_only_not_material_specific"


@dataclass(frozen=True, slots=True)
class CandidateValidationIssue:
    """Structured failure evidence for one generated candidate."""

    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", dict(self.evidence))


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
        issue = _validate_family_state(candidate, reference, family)
        if issue is not None:
            return issue

    minimum_distance = _minimum_pair_distance_threshold(settings, family)
    measured_distance = _minimum_pair_distance(candidate, pbc)
    if len(candidate) >= 2 and measured_distance is None:
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
    if len(candidate) < 2:
        return None
    try:
        distances = np.asarray(candidate.get_all_distances(mic=bool(np.any(pbc))), dtype=float)
        upper = distances[np.triu_indices(len(candidate), k=1)]
    except Exception:
        return None
    if upper.size == 0:
        return None
    minimum = float(np.min(upper))
    return minimum if np.isfinite(minimum) else None


def _validate_family_state(
    candidate: Any,
    reference: Any,
    family: str,
) -> CandidateValidationIssue | None:
    if family == "liquid":
        issue = _validate_liquid_state(candidate)
        if issue is not None:
            return issue
    if family == "surface":
        issue = _validate_surface_state(candidate)
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
    if (
        info["liquid_method"] != _LIQUID_METHOD
        or info["liquid_fidelity"] != _LIQUID_FIDELITY
    ):
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
    if info["source_composition"] is None or not isinstance(
        info["parent_structure_id"], str
    ):
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


def _validate_surface_state(candidate: Any) -> CandidateValidationIssue | None:
    """Validate surface-specific provenance and two-dimensional periodicity."""

    info = getattr(candidate, "info", {})
    required = (
        "parent_structure_id",
        "surface_miller_index",
        "surface_termination",
        "surface_layers",
        "surface_vacuum",
        "surface_in_plane_repeat",
        "surface_stoichiometry_change",
    )
    missing = [key for key in required if key not in info]
    if missing:
        return CandidateValidationIssue("missing_surface_provenance", {"fields": missing})
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
    if len(miller) != 3 or not any(miller):
        return CandidateValidationIssue("invalid_surface_miller_index", {"value": miller})
    try:
        vacuum = float(info["surface_vacuum"])
    except (TypeError, ValueError):
        return CandidateValidationIssue("invalid_surface_vacuum", {})
    if not math.isfinite(vacuum) or vacuum <= 0.0:
        return CandidateValidationIssue("invalid_surface_vacuum", {"value": vacuum})
    repeat = info["surface_in_plane_repeat"]
    try:
        repeat_values = tuple(int(value) for value in repeat)
    except (TypeError, ValueError):
        return CandidateValidationIssue("invalid_surface_repeat", {})
    if len(repeat_values) != 2 or any(value <= 0 for value in repeat_values):
        return CandidateValidationIssue("invalid_surface_repeat", {"value": repeat})
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
    if config.surface_max_terminations < 0:
        raise ConfigurationError("generation.surface_max_terminations must be non-negative")
    if config.surface_termination_policy not in {"all", "first"}:
        raise ConfigurationError("generation.surface_termination_policy must be 'all' or 'first'")
    for index in config.surface_miller_indices:
        if len(index) != 3 or not any(index):
            raise ConfigurationError(
                "generation.surface_miller_indices must contain non-zero triples"
            )
    if len(config.surface_in_plane_repeat) != 2 or any(
        value <= 0 for value in config.surface_in_plane_repeat
    ):
        raise ConfigurationError(
            "generation.surface_in_plane_repeat must contain two positive integers"
        )
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
        raise ConfigurationError(
            "generation.grain_boundary_overlap_tolerance must be in [0, 1]"
        )
    if config.use_liquid and (
        config.n_liquid_configurations > 0 and config.n_liquid_snapshots > 0
    ):
        if config.liquid_timestep_fs <= 0.0:
            raise ConfigurationError("generation.liquid_timestep_fs must be positive")
        if config.liquid_steps_between_snapshots <= 0:
            raise ConfigurationError(
                "generation.liquid_steps_between_snapshots must be positive"
            )
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
