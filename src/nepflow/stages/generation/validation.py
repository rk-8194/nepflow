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
        return float(settings.interstitial_d_min)
    if family in {"gas_interstitial", "gas_in_vacancy"}:
        value = settings.gas_interstitial_d_min
        return float(settings.interstitial_d_min if value is None else value)
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
    if config.n_workers < 0:
        raise ConfigurationError("generation.n_workers must be non-negative")
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
