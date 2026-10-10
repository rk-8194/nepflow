"""Independent, identity-safe test holdout policies for selection.

The entropy training objective consumes the complete local-row pool.  The
holdout policies in this module are separate decision procedures: neither
re-runs the entropy objective nor reduces local rows to one candidate mean.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nepflow.config.models import TEST_ATOM_WEIGHTINGS, TEST_SELECTION_POLICIES
from nepflow.resources.budget import ResourceBudgetService

from .algorithms.information_entropy.neighbours import IndexedCPUNeighbourIndex
from .representations import LocalEnvironmentRepresentation

logger = logging.getLogger(__name__)

TEST_SELECTION_VERSION = "local-environment-holdout-v1"
REPRESENTATIVE_POLICY_VERSION = "representative-stratified-signature-v1"
EXTRAPOLATIVE_POLICY_VERSION = "extrapolative-local-novelty-v1"
NOVELTY_DIVERSITY_WEIGHT = 0.25
SIGNATURE_QUANTILES = (0.0, 0.25, 0.75, 1.0)


class TestSelectionPolicyError(ValueError):
    """Raised when a requested test policy cannot satisfy its contract."""


@dataclass(frozen=True, slots=True)
class TestSelectionResult:
    """Typed result of one independently declared test-set policy."""

    indices: tuple[int, ...]
    policy: str
    version: str
    test_min_dist: float | None = None
    min_train_test_dist: float | None = None
    mean_train_test_dist: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "test_indices": list(self.indices),
            "test_min_dist": self.test_min_dist,
            "min_train_test_dist": self.min_train_test_dist,
            "mean_train_test_dist": self.mean_train_test_dist,
            "test_selection_policy": self.policy,
            "test_selection_version": self.version,
            "test_selection_provenance": dict(self.provenance),
        }


def _validate_policy(policy: str) -> str:
    value = str(policy).strip().lower()
    if value not in TEST_SELECTION_POLICIES:
        raise TestSelectionPolicyError(
            "unsupported test_selection_policy: "
            f"{value!r}; choose one of {sorted(TEST_SELECTION_POLICIES)}"
        )
    return value


def _validate_weighting(weighting: str) -> str:
    value = str(weighting).strip().lower()
    if value not in TEST_ATOM_WEIGHTINGS:
        raise TestSelectionPolicyError(
            f"unsupported test_atom_weighting: {value!r}; "
            f"choose one of {sorted(TEST_ATOM_WEIGHTINGS)}"
        )
    return value


def _row_indices(representation: LocalEnvironmentRepresentation) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    candidate_positions = {
        candidate_id: position
        for position, candidate_id in enumerate(representation.candidate_ids)
    }
    for row_index, row in enumerate(representation.rows):
        position = candidate_positions.get(row.candidate_id)
        if position is None:
            raise TestSelectionPolicyError(
                f"local representation contains unknown candidate {row.candidate_id!r}"
            )
        if row.structure_id != representation.structure_ids[position]:
            raise TestSelectionPolicyError(
                f"local representation structure identity disagrees for {row.candidate_id!r}"
            )
        grouped[row.candidate_id].append(row_index)
    result = {
        candidate_id: np.asarray(indices, dtype=np.int64)
        for candidate_id, indices in grouped.items()
    }
    missing = [
        candidate_id
        for candidate_id in representation.candidate_ids
        if candidate_id not in result
    ]
    if missing:
        raise TestSelectionPolicyError(
            f"local representation has no atomic rows for candidate {missing[0]!r}"
        )
    return result


def _candidate_formula(structure: Any) -> str:
    getter = getattr(structure, "get_chemical_formula", None)
    if callable(getter):
        try:
            return str(getter(mode="hill"))
        except TypeError:
            return str(getter())
    symbols = getattr(structure, "get_chemical_symbols", lambda: ())()
    counts: dict[str, int] = defaultdict(int)
    for symbol in symbols:
        counts[str(symbol)] += 1
    return ";".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))


def _candidate_family(structure: Any) -> str:
    info = getattr(structure, "info", {})
    if not isinstance(info, Mapping):
        return "unknown"
    for name in ("structure_family", "family", "source_family", "generation_family"):
        value = info.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def _distribution_signature(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] == 0:
        raise TestSelectionPolicyError("candidate local environments must be non-empty 2D arrays")
    return np.asarray(
        np.quantile(values, SIGNATURE_QUANTILES, axis=0), dtype=np.float64
    ).reshape(-1)


def _signature_spread(signature: np.ndarray, dimension: int) -> float:
    lower = signature[dimension : 2 * dimension]
    upper = signature[2 * dimension : 3 * dimension]
    outer_lower = signature[:dimension]
    outer_upper = signature[3 * dimension :]
    return float(np.linalg.norm(upper - lower) + 0.5 * np.linalg.norm(outer_upper - outer_lower))


def _select_identity_representatives(
    eligible: Sequence[int],
    candidate_ids: Sequence[str],
    structure_ids: Sequence[str],
    *,
    signatures: Mapping[str, np.ndarray] | None = None,
    scores: Mapping[str, float] | None = None,
) -> tuple[list[int], dict[str, str], dict[str, int]]:
    by_structure: dict[str, list[int]] = defaultdict(list)
    for index in eligible:
        by_structure[str(structure_ids[index])].append(int(index))
    selected: list[int] = []
    exclusions: dict[str, str] = {}
    for structure_id, indices in sorted(by_structure.items()):
        if scores is not None:
            index = min(
                indices,
                key=lambda value: (-float(scores[candidate_ids[value]]), candidate_ids[value]),
            )
        elif signatures is not None:
            index = min(
                indices,
                key=lambda value: (tuple(signatures[candidate_ids[value]]), candidate_ids[value]),
            )
        else:
            index = min(indices, key=lambda value: candidate_ids[value])
        selected.append(index)
        for duplicate in indices:
            if duplicate != index:
                exclusions[candidate_ids[duplicate]] = (
                    f"duplicate physical structure identity {structure_id}"
                )
    counts = {
        "candidate_groups": len(by_structure),
        "duplicate_physical_candidates": len(exclusions),
    }
    return sorted(selected), exclusions, counts


def _eligible_candidates(
    representation: LocalEnvironmentRepresentation,
    train_indices: Sequence[int],
    anchor_indices: Sequence[int],
    *,
    candidate_ids: Sequence[str],
    structure_ids: Sequence[str],
) -> tuple[list[int], dict[str, str]]:
    candidate_count = len(representation.candidate_ids)
    if len(candidate_ids) != candidate_count or len(structure_ids) != candidate_count:
        raise TestSelectionPolicyError(
            "test identity sequences must match the local candidate pool"
        )
    if tuple(candidate_ids) != representation.candidate_ids:
        raise TestSelectionPolicyError("test candidate IDs do not match local representation order")
    if tuple(structure_ids) != representation.structure_ids:
        raise TestSelectionPolicyError("test structure IDs do not match local representation order")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise TestSelectionPolicyError("test candidate IDs must be unique")
    train = {int(index) for index in train_indices}
    anchors = {int(index) for index in anchor_indices}
    invalid = (train | anchors) - set(range(candidate_count))
    if invalid:
        raise TestSelectionPolicyError(
            "training and anchor indices must be within the local candidate pool"
        )
    train_structures = {str(structure_ids[index]) for index in train}
    eligible: list[int] = []
    exclusions: dict[str, str] = {}
    for index in range(candidate_count):
        candidate_id = str(candidate_ids[index])
        if index in anchors:
            exclusions[candidate_id] = "mandatory training anchor"
        elif index in train:
            exclusions[candidate_id] = "training candidate"
        elif str(structure_ids[index]) in train_structures:
            exclusions[candidate_id] = "physical structure identity is in training"
        else:
            eligible.append(index)
    return eligible, exclusions


def _allocate_strata(
    strata: Mapping[tuple[str, str, int], list[int]],
    target: int,
    weights: Mapping[tuple[str, str, int], float],
) -> dict[tuple[str, str, int], int]:
    quotas = {key: 0 for key in strata}
    total_weight = sum(weights.values())
    if target <= 0 or not strata or total_weight <= 0.0:
        return quotas
    ideals = {key: target * weights[key] / total_weight for key in strata}
    while sum(quotas.values()) < target:
        available = [key for key in strata if quotas[key] < len(strata[key])]
        if not available:
            break
        key = max(
            available,
            key=lambda item: (
                ideals[item] - quotas[item],
                weights[item],
                tuple(str(value) for value in item),
            ),
        )
        quotas[key] += 1
    return quotas


def _representative_holdout(
    representation: LocalEnvironmentRepresentation,
    structures: Sequence[Any],
    eligible: Sequence[int],
    candidate_ids: Sequence[str],
    structure_ids: Sequence[str],
    *,
    target: int,
    atom_weighting: str,
    signature_bins: int,
    exclusions: dict[str, str],
) -> TestSelectionResult:
    rows = _row_indices(representation)
    signatures = {
        candidate_ids[index]: _distribution_signature(
            representation.descriptors[rows[candidate_ids[index]]]
        )
        for index in eligible
    }
    dimension = representation.descriptors.shape[1]
    signature_scores = {
        candidate_id: _signature_spread(signature, dimension)
        for candidate_id, signature in signatures.items()
    }
    ordered, duplicate_exclusions, identity_counts = _select_identity_representatives(
        eligible,
        candidate_ids,
        structure_ids,
        signatures=signatures,
    )
    exclusions.update(duplicate_exclusions)
    order = sorted(
        ordered,
        key=lambda index: (signature_scores[candidate_ids[index]], candidate_ids[index]),
    )
    bins = max(1, int(signature_bins))
    strata: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    candidate_strata: dict[int, tuple[str, str, int]] = {}
    for rank, index in enumerate(order):
        bin_index = min(bins - 1, rank * bins // max(1, len(order)))
        key = (
            _candidate_formula(structures[index]),
            _candidate_family(structures[index]),
            bin_index,
        )
        strata[key].append(index)
        candidate_strata[index] = key
    weights = {
        key: float(
            sum(
                len(rows[candidate_ids[index]]) if atom_weighting == "atom" else 1
                for index in values
            )
        )
        for key, values in strata.items()
    }
    quotas = _allocate_strata(strata, min(int(target), len(ordered)), weights)
    chosen: list[int] = []
    for key in sorted(strata):
        values = sorted(
            strata[key],
            key=lambda index: (tuple(signatures[candidate_ids[index]]), candidate_ids[index]),
        )
        quota = quotas[key]
        for slot in range(quota):
            position = min(len(values) - 1, slot * len(values) // quota)
            chosen.append(values[position])
    chosen = sorted(set(chosen))
    deficiency = max(0, int(target) - len(chosen))
    if deficiency:
        logger.warning(
            "Representative test holdout could not satisfy target: requested=%d, selected=%d, "
            "identity_groups=%d",
            target,
            len(chosen),
            len(ordered),
        )
    population_weight = sum(weights.values())
    selected_weights = {
        key: float(
            sum(
                len(rows[candidate_ids[index]]) if atom_weighting == "atom" else 1
                for index in chosen
                if candidate_strata[index] == key
            )
        )
        for key in strata
    }
    population_fractions = {
        "/".join((key[0], key[1], str(key[2]))): weight / population_weight
        for key, weight in weights.items()
    }
    selected_total_weight = sum(selected_weights.values())
    selected_fractions = {
        "/".join((key[0], key[1], str(key[2]))): (
            selected_weights[key] / selected_total_weight
            if selected_total_weight
            else 0.0
        )
        for key in strata
    }
    stratum_l1_distance = sum(
        abs(
            population_fractions.get("/".join((key[0], key[1], str(key[2]))), 0.0)
            - selected_fractions.get("/".join((key[0], key[1], str(key[2]))), 0.0)
        )
        for key in strata
    )
    selected_spreads = [signature_scores[candidate_ids[index]] for index in chosen]
    eligible_spreads = list(signature_scores.values())
    provenance = {
        "policy_version": REPRESENTATIVE_POLICY_VERSION,
        "representation": "whitened local-environment distribution signatures",
        "signature_quantiles": list(SIGNATURE_QUANTILES),
        "signature_bins": bins,
        "atom_weighting": atom_weighting,
        "eligible_candidate_count": len(eligible),
        "eligible_identity_group_count": len(ordered),
        "selected_count": len(chosen),
        "target_count": int(target),
        "count_deficiency": deficiency,
        "filtered_candidate_count": len(exclusions),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "population_stratum_fractions": population_fractions,
        "selected_stratum_fractions": selected_fractions,
        "stratum_distribution_l1_distance": float(stratum_l1_distance),
        "eligible_signature_spread_mean": float(np.mean(eligible_spreads)),
        "selected_signature_spread_mean": (
            float(np.mean(selected_spreads)) if selected_spreads else None
        ),
        "selected_candidate_ids": [candidate_ids[index] for index in chosen],
        "selected_structure_ids": [structure_ids[index] for index in chosen],
        **identity_counts,
        "label_holdout": True,
        "preprocessing_scope": "full_candidate_pool_representation_and_calibration",
    }
    return TestSelectionResult(
        tuple(chosen),
        "representative",
        TEST_SELECTION_VERSION,
        provenance=provenance,
    )


def _extrapolative_holdout(
    representation: LocalEnvironmentRepresentation,
    eligible: Sequence[int],
    candidate_ids: Sequence[str],
    structure_ids: Sequence[str],
    train_indices: Sequence[int],
    *,
    target: int,
    quantile: float,
    beta: float,
    exclusions: dict[str, str],
    resource_budget: ResourceBudgetService,
    chunk_size: int,
) -> TestSelectionResult:
    rows = _row_indices(representation)
    training_candidate_ids = [candidate_ids[index] for index in train_indices]
    train_row_indices = np.concatenate(
        [rows[candidate_id] for candidate_id in training_candidate_ids]
    )
    if train_row_indices.size == 0:
        raise TestSelectionPolicyError(
            "extrapolative test selection requires at least one training local environment"
        )
    index = IndexedCPUNeighbourIndex(
        representation.descriptors[train_row_indices],
        resource_budget=resource_budget,
    )
    profiles: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    for candidate_index in eligible:
        candidate_id = candidate_ids[candidate_index]
        distances = index.query_nearest_distances(
            representation.descriptors[rows[candidate_id]],
            chunk_size=chunk_size,
            context=f"extrapolative novelty for {candidate_id}",
        )
        profile = np.asarray(
            [
                float(np.mean(distances)),
                float(np.quantile(distances, quantile)),
                float(np.max(distances)),
            ],
            dtype=np.float64,
        )
        profiles[candidate_id] = profile
        scores[candidate_id] = float((1.0 - beta) * profile[0] + beta * profile[1])
    ordered, duplicate_exclusions, identity_counts = _select_identity_representatives(
        eligible,
        candidate_ids,
        structure_ids,
        scores=scores,
    )
    exclusions.update(duplicate_exclusions)
    desired = min(int(target), len(ordered))
    selected: list[int] = []
    remaining = set(ordered)
    profile_values = np.asarray([profiles[candidate_ids[index]] for index in ordered])
    scale = np.ptp(profile_values, axis=0)
    scale[scale <= 1.0e-15] = 1.0
    while remaining and len(selected) < desired:
        if not selected:
            choice = min(
                remaining,
                key=lambda index: (-scores[candidate_ids[index]], candidate_ids[index]),
            )
        else:
            selected_profiles = np.asarray(
                [profiles[candidate_ids[index]] for index in selected], dtype=np.float64
            )

            def rank_key(index: int) -> tuple[float, float, str]:
                profile = profiles[candidate_ids[index]]
                diversity = float(
                    np.min(np.linalg.norm((selected_profiles - profile) / scale, axis=1))
                )
                novelty = scores[candidate_ids[index]]
                return (
                    -((1.0 - NOVELTY_DIVERSITY_WEIGHT) * novelty + diversity),
                    -novelty,
                    candidate_ids[index],
                )

            choice = min(remaining, key=rank_key)
        selected.append(choice)
        remaining.remove(choice)
    selected = sorted(selected)
    deficiency = max(0, int(target) - len(selected))
    if deficiency:
        logger.warning(
            "Extrapolative test holdout could not satisfy target: requested=%d, selected=%d, "
            "identity_groups=%d",
            target,
            len(selected),
            len(ordered),
        )
    selected_scores = [scores[candidate_ids[index]] for index in selected]
    eligible_scores = list(scores.values())
    provenance = {
        "policy_version": EXTRAPOLATIVE_POLICY_VERSION,
        "representation": "candidate-owned whitened local rows",
        "novelty_metric": "mean_and_upper_quantile_nearest_training_environment_distance",
        "novelty_quantile": float(quantile),
        "novelty_beta": float(beta),
        "diversity_weight": NOVELTY_DIVERSITY_WEIGHT,
        "selection_rule": "novelty_ranked_with_local_novelty_profile_diversity",
        "eligible_candidate_count": len(eligible),
        "eligible_identity_group_count": len(ordered),
        "selected_count": len(selected),
        "target_count": int(target),
        "count_deficiency": deficiency,
        "filtered_candidate_count": len(exclusions),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "eligible_novelty_mean": float(np.mean(eligible_scores)),
        "eligible_novelty_quantile": float(np.quantile(eligible_scores, quantile)),
        "selected_novelty_mean": float(np.mean(selected_scores)) if selected_scores else None,
        "selected_novelty_max": float(np.max(selected_scores)) if selected_scores else None,
        "selected_novelty_quantile": (
            float(np.quantile(selected_scores, quantile)) if selected_scores else None
        ),
        "selected_candidate_ids": [candidate_ids[index] for index in selected],
        "selected_structure_ids": [structure_ids[index] for index in selected],
        **identity_counts,
        "holdout_purpose": "out_of_domain_stress_test",
        "accuracy_interpretation": "not_population_average_accuracy",
        "label_holdout": True,
        "preprocessing_scope": "full_candidate_pool_representation_and_calibration",
    }
    return TestSelectionResult(
        tuple(selected),
        "extrapolative",
        TEST_SELECTION_VERSION,
        provenance=provenance,
    )


def select_local_test_holdout(
    representation: LocalEnvironmentRepresentation,
    structures: Sequence[Any],
    train_indices: Sequence[int],
    *,
    anchor_indices: Sequence[int] = (),
    target_count: int,
    policy: str,
    candidate_ids: Sequence[str] | None = None,
    structure_ids: Sequence[str] | None = None,
    atom_weighting: str = "candidate",
    signature_bins: int = 4,
    novelty_quantile: float = 0.95,
    novelty_beta: float = 0.5,
    resource_budget: ResourceBudgetService | None = None,
    chunk_size: int = 1024,
) -> TestSelectionResult:
    """Select an entropy holdout from local rows with explicit provenance."""

    selected_policy = _validate_policy(policy)
    if selected_policy == "candidate_mean_fps_legacy":
        raise TestSelectionPolicyError(
            "candidate_mean_fps_legacy is dispatched by the FPS-compatible strategy, "
            "not by the local-environment holdout selector"
        )
    if len(structures) != len(representation.candidate_ids):
        raise TestSelectionPolicyError("structures must match the local candidate pool")
    weighting = _validate_weighting(atom_weighting)
    if int(target_count) < 0:
        raise TestSelectionPolicyError("target_count must be non-negative")
    if int(signature_bins) < 1:
        raise TestSelectionPolicyError("signature_bins must be positive")
    if not 0.0 < float(novelty_quantile) <= 1.0:
        raise TestSelectionPolicyError("novelty_quantile must be in (0, 1]")
    if not 0.0 <= float(novelty_beta) <= 1.0:
        raise TestSelectionPolicyError("novelty_beta must be in [0, 1]")
    if candidate_ids is None:
        candidate_ids = representation.candidate_ids
    if structure_ids is None:
        structure_ids = representation.structure_ids
    _row_indices(representation)
    eligible, exclusions = _eligible_candidates(
        representation,
        train_indices,
        anchor_indices,
        candidate_ids=candidate_ids,
        structure_ids=structure_ids,
    )
    if int(target_count) == 0 or not eligible:
        version = (
            REPRESENTATIVE_POLICY_VERSION
            if selected_policy == "representative"
            else EXTRAPOLATIVE_POLICY_VERSION
        )
        return TestSelectionResult(
            (),
            selected_policy,
            TEST_SELECTION_VERSION,
            provenance={
                "policy_version": version,
                "eligible_candidate_count": len(eligible),
                "selected_count": 0,
                "target_count": int(target_count),
                "count_deficiency": max(0, int(target_count) - len(eligible)),
                "filtered_candidate_count": len(exclusions),
                "exclusion_reasons": dict(sorted(exclusions.items())),
                "selected_candidate_ids": [],
                "selected_structure_ids": [],
                "label_holdout": True,
                "preprocessing_scope": "full_candidate_pool_representation_and_calibration",
            },
        )
    if selected_policy == "representative":
        return _representative_holdout(
            representation,
            structures,
            eligible,
            candidate_ids,
            structure_ids,
            target=int(target_count),
            atom_weighting=weighting,
            signature_bins=int(signature_bins),
            exclusions=exclusions,
        )
    active_budget = resource_budget
    if active_budget is None:
        raise TestSelectionPolicyError(
            "extrapolative test selection requires the shared runtime resource budget"
        )
    return _extrapolative_holdout(
        representation,
        eligible,
        candidate_ids,
        structure_ids,
        train_indices,
        target=int(target_count),
        quantile=float(novelty_quantile),
        beta=float(novelty_beta),
        exclusions=exclusions,
        resource_budget=active_budget,
        chunk_size=int(chunk_size),
    )


__all__ = [
    "EXTRAPOLATIVE_POLICY_VERSION",
    "REPRESENTATIVE_POLICY_VERSION",
    "TEST_SELECTION_VERSION",
    "TestSelectionPolicyError",
    "TestSelectionResult",
    "select_local_test_holdout",
]
