"""Selection strategies and anchor resolution over canonical primitives."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from ase.io import read as ase_read

from nepflow.config.models import SelectionConfig
from nepflow.domain.identities import calculate_structure_id

from .sampling import (
    calculate_cross_distance_stats,
    build_composition_aware_candidate_set,
    composition_aware_attempt_schedule,
    composition_projection_bins,
    select_best_sampling_attempt,
    select_farthest_points_for_target,
)


logger = logging.getLogger("nepflow.selection.strategy")


def setting(settings: SelectionConfig | Mapping[str, Any], name: str) -> Any:
    """Read one setting from the typed config or a test mapping."""

    if isinstance(settings, Mapping):
        legacy_names = {
            "target_train_count": "target_train",
            "target_test_count": "target_test",
            "target_tolerance": "tolerance",
            "max_search_iterations": "max_iterations",
        }
        if name in settings:
            return settings[name]
        if name == "descriptor_type" and "mean_descriptor" in settings:
            return "structure" if settings["mean_descriptor"] else "atomic"
        return settings[legacy_names.get(name, name)]
    return getattr(settings, name)


def select_training_set(
    representations: np.ndarray,
    structures: list,
    settings: SelectionConfig | Mapping[str, Any],
    *,
    ase_structures: list | None = None,
    seed_indices: list[int] | None = None,
    single_element_elastic_indices: list[int] | None = None,
    elastic_indices: list[int] | None = None,
) -> tuple[list[int], float]:
    """Select a training set while preserving mandatory anchor identity."""

    logger.info("")
    logger.info("Step 3: Selecting training set via FPS")

    seed_indices = sorted(set(seed_indices or []))
    single_element_elastic_indices = sorted(set(single_element_elastic_indices or []))
    elastic_indices = sorted(set(elastic_indices or []))
    anchor_indices = sorted(
        set(seed_indices + single_element_elastic_indices + elastic_indices)
    )
    target_train = setting(settings, "target_train_count")
    if len(anchor_indices) > target_train:
        raise ValueError(
            "Preselected anchor count exceeds target_train_count: "
            "seed anchors=%d, single-element elastic stress anchors=%d, "
            "all elastic stress anchors=%d, unique anchors=%d, target_train_count=%d"
            % (
                len(seed_indices),
                len(single_element_elastic_indices),
                len(elastic_indices),
                len(anchor_indices),
                target_train,
            )
        )

    if seed_indices:
        logger.info("  Preselected %d seed structures as training anchors", len(seed_indices))
    if single_element_elastic_indices:
        logger.info(
            "  Preselected %d single-element elastic stress structures as training anchors",
            len(single_element_elastic_indices),
        )
    if elastic_indices:
        logger.info(
            "  Preselected %d elastic stress structures as training anchors",
            len(elastic_indices),
        )
    if setting(settings, "composition_aware_fps"):
        logger.info("  Composition-aware FPS enabled for training selection")

    if target_train >= len(structures):
        logger.info(
            "  target_train_count (%d) >= total (%d), selecting all for training",
            target_train,
            len(structures),
        )
        return list(range(len(structures))), 0.0

    anchor_set = set(anchor_indices)
    remaining_indices = [
        index for index in range(len(structures)) if index not in anchor_set
    ]
    remaining_target = target_train - len(anchor_indices)
    if remaining_target == 0:
        return anchor_indices, 0.0

    if setting(settings, "composition_aware_fps"):
        selected, minimum = select_composition_aware_training_set(
            representations,
            structures,
            settings,
            ase_structures or [],
            anchor_indices,
            remaining_indices,
            remaining_target,
        )
    else:
        selected, minimum = select_plain_fps_training_set(
            representations,
            structures,
            settings,
            remaining_indices,
            remaining_target,
        )
        selected = sorted(anchor_indices + selected)

    logger.info(
        "  Training set: %d structures (min_distance=%.6f)",
        len(selected),
        minimum,
    )
    return selected, minimum


def select_plain_fps_training_set(
    representations: np.ndarray,
    structures: list,
    settings: SelectionConfig | Mapping[str, Any],
    remaining_indices: list[int],
    remaining_target: int,
) -> tuple[list[int], float]:
    """Select a target-sized remainder using the canonical FPS service."""

    remaining_representations = representations[remaining_indices]
    remaining_structures = [structures[index] for index in remaining_indices]
    fps_indices, minimum = select_farthest_points_for_target(
        remaining_representations,
        remaining_structures,
        setting(settings, "descriptor_type") == "structure",
        remaining_target,
        setting(settings, "target_tolerance"),
        setting(settings, "max_search_iterations"),
        label="train",
    )
    return [int(remaining_indices[index]) for index in fps_indices], minimum


def select_composition_aware_training_set(
    representations: np.ndarray,
    structures: list,
    settings: SelectionConfig | Mapping[str, Any],
    ase_structures: list,
    anchor_indices: list[int],
    remaining_indices: list[int],
    remaining_target: int,
) -> tuple[list[int], float]:
    """Run adaptive composition-aware selection and preserve descriptor floors."""

    if not ase_structures:
        logger.warning(
            "  Composition-aware FPS requested, but ASE structures were not provided; "
            "falling back to descriptor-only FPS"
        )
        selected, minimum = select_plain_fps_training_set(
            representations,
            structures,
            settings,
            remaining_indices,
            remaining_target,
        )
        return sorted(anchor_indices + selected), minimum

    candidate_bins = {
        index: composition_projection_bins(ase_structures[index])
        for index in range(len(ase_structures))
    }
    total_binary_bins = len({
        bin_key
        for index in remaining_indices + anchor_indices
        for bin_key in candidate_bins[index]["binary"]
    })
    total_ternary_bins = len({
        bin_key
        for index in remaining_indices + anchor_indices
        for bin_key in candidate_bins[index]["ternary"]
    })
    if total_binary_bins == 0 and total_ternary_bins == 0:
        logger.info(
            "  No binary or ternary composition projections found; "
            "falling back to descriptor-only FPS"
        )
        selected, minimum = select_plain_fps_training_set(
            representations,
            structures,
            settings,
            remaining_indices,
            remaining_target,
        )
        return sorted(anchor_indices + selected), minimum

    from collections import Counter

    binary_counts: Counter = Counter()
    ternary_counts: Counter = Counter()
    for index in anchor_indices:
        binary_counts.update(candidate_bins[index]["binary"])
        ternary_counts.update(candidate_bins[index]["ternary"])

    logger.info(
        "  Composition-aware fill: %d remaining slots, %d binary bins, %d ternary bins",
        remaining_target,
        total_binary_bins,
        total_ternary_bins,
    )
    logger.info(
        "  Anchor composition footprint: %d occupied binary bins, %d occupied ternary bins",
        len(binary_counts),
        len(ternary_counts),
    )

    attempt_schedule = composition_aware_attempt_schedule(
        setting(settings, "composition_aware_fps_frontier_fraction"),
        setting(settings, "composition_aware_fps_ternary_weight"),
        setting(settings, "composition_aware_fps_adaptive_retries"),
    )
    attempt_results: list[dict] = []
    for attempt_number, (frontier_fraction, ternary_weight) in enumerate(
        attempt_schedule,
        start=1,
    ):
        logger.info(
            "  Attempt %d/%d: frontier_fraction=%.3f, ternary_weight=%.3f",
            attempt_number,
            len(attempt_schedule),
            frontier_fraction,
            ternary_weight,
        )
        attempt_result = build_composition_aware_candidate_set(
            representations,
            candidate_bins,
            anchor_indices,
            remaining_indices,
            remaining_target,
            frontier_fraction,
            ternary_weight,
        )
        attempt_result["attempt_number"] = attempt_number
        attempt_results.append(attempt_result)
        logger.info(
            "    Coverage: binary occ=%.3f, binary entropy=%.3f, "
            "ternary occ=%.3f, ternary entropy=%.3f",
            attempt_result["binary_occupied_bin_fraction"],
            attempt_result["binary_normalized_entropy"],
            attempt_result["ternary_occupied_bin_fraction"],
            attempt_result["ternary_normalized_entropy"],
        )
        logger.info(
            "    Descriptor quality: min_dist=%.6f, mean_nn=%.6f",
            attempt_result["train_min_dist"],
            attempt_result["train_mean_nn_dist"],
        )

    best_attempt = select_best_sampling_attempt(
        attempt_results,
        setting(settings, "composition_aware_fps_descriptor_floor_fraction"),
    )
    logger.info(
        "  Selected attempt %d with frontier_fraction=%.3f, ternary_weight=%.3f",
        best_attempt["attempt_number"],
        best_attempt["frontier_fraction"],
        best_attempt["ternary_weight"],
    )
    logger.info(
        "  Composition-aware selection filled %d structures; touched %d/%d "
        "binary bins and %d/%d ternary bins",
        len(best_attempt["selected_indices"]) - len(anchor_indices),
        best_attempt["occupied_binary_bins"],
        total_binary_bins,
        best_attempt["occupied_ternary_bins"],
        total_ternary_bins,
    )
    return best_attempt["selected_indices"], best_attempt["train_min_dist"]


def select_test_set(
    representations: np.ndarray,
    structures: list,
    train_indices: list[int],
    settings: SelectionConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Select the test set from candidates farthest from training structures."""

    logger.info("")
    logger.info("Step 4: Selecting test set via FPS (from remaining structures)")
    train_set = set(train_indices)
    remaining_indices = np.array(
        [index for index in range(len(structures)) if index not in train_set]
    )
    logger.info("  %d candidates remaining after training selection", len(remaining_indices))

    target_test = setting(settings, "target_test_count")
    if len(remaining_indices) == 0:
        logger.warning("  No structures remain for test set")
        return {
            "test_indices": [],
            "test_min_dist": 0.0,
            "min_train_test_dist": float("inf"),
            "mean_train_test_dist": float("inf"),
        }

    if target_test >= len(remaining_indices):
        logger.info(
            "  target_test_count (%d) >= remaining (%d), using all remaining for test",
            target_test,
            len(remaining_indices),
        )
        test_indices = remaining_indices.tolist()
        minimum, mean_distance = calculate_cross_distance_stats(
            representations,
            train_indices,
            test_indices,
        )
        return {
            "test_indices": test_indices,
            "test_min_dist": 0.0,
            "min_train_test_dist": minimum,
            "mean_train_test_dist": mean_distance,
        }

    from scipy.spatial.distance import cdist

    distances_to_train = cdist(
        representations[remaining_indices],
        representations[train_indices],
    ).min(axis=1)
    pool_size = max(
        target_test,
        int(len(remaining_indices) * setting(settings, "test_pool_factor")),
    )
    top_k = np.argsort(distances_to_train)[::-1][:pool_size]
    pool_indices = remaining_indices[top_k]
    logger.info(
        "  Pre-filtered to %d candidates farthest from training "
        "(top %.0f%%, min d_train in pool: %.6f, max: %.6f)",
        len(pool_indices),
        setting(settings, "test_pool_factor") * 100,
        distances_to_train[top_k[-1]],
        distances_to_train[top_k[0]],
    )

    test_local, test_min_dist = select_farthest_points_for_target(
        representations[pool_indices],
        [structures[index] for index in pool_indices],
        setting(settings, "descriptor_type") == "structure",
        target_test,
        setting(settings, "target_tolerance"),
        setting(settings, "max_search_iterations"),
        label="test",
    )
    test_indices = [int(pool_indices[index]) for index in test_local]
    minimum, mean_distance = calculate_cross_distance_stats(
        representations,
        train_indices,
        test_indices,
    )
    logger.info(
        "  Test set: %d structures (min_distance=%.6f)",
        len(test_indices),
        test_min_dist,
    )
    logger.info(
        "  Train<->test nearest-neighbour distance - min: %.6f, mean: %.6f",
        minimum,
        mean_distance,
    )
    return {
        "test_indices": test_indices,
        "test_min_dist": test_min_dist,
        "min_train_test_dist": minimum,
        "mean_train_test_dist": mean_distance,
    }


def is_elastic_stress(atoms: Any) -> bool:
    """Return whether a structure is tagged as an elastic-stress sample."""

    return str(atoms.info.get("perturbation_type", "")) == "elastic_stress"


def is_single_element_structure(atoms: Any) -> bool:
    """Return whether composition metadata or symbols identify one element."""

    composition = atoms.info.get("composition")
    if isinstance(composition, dict):
        try:
            positive = [
                element
                for element, fraction in composition.items()
                if float(fraction) > 0.0
            ]
        except (TypeError, ValueError):
            positive = []
        if positive:
            return len(positive) == 1

    try:
        return len(set(atoms.get_chemical_symbols())) == 1
    except Exception:
        return False


def resolve_seed_indices(project_dir: Path, reference_ase: list) -> list[int]:
    """Resolve seed anchors by exact physical structure identity."""

    seeds_path = project_dir / "structures" / "seeds" / "base_structures.xyz"
    if not seeds_path.exists():
        raise FileNotFoundError(
            f"Seed structures requested, but file not found: {seeds_path}"
        )

    logger.info("")
    logger.info("Step 2b: Loading seed structures")
    seed_ase = ase_read(str(seeds_path), index=":", format="extxyz")
    if not isinstance(seed_ase, list):
        seed_ase = [seed_ase]
    if len(seed_ase) == 0:
        raise ValueError(
            f"Seed inclusion enabled, but no structures were found in {seeds_path}"
        )

    reference_index: dict[str, int] = {}
    for index, atoms in enumerate(reference_ase):
        reference_index.setdefault(calculate_structure_id(atoms), index)

    seed_indices: list[int] = []
    missing_seeds: list[str] = []
    for seed in seed_ase:
        physical_hash = calculate_structure_id(seed)
        reference_index_for_seed = reference_index.get(physical_hash)
        if reference_index_for_seed is None:
            missing_seeds.append(physical_hash)
            continue
        seed_indices.append(reference_index_for_seed)

    if missing_seeds:
        raise ValueError(
            f"Seed inclusion enabled, but {len(missing_seeds)}/{len(seed_ase)} "
            "seed structures had no physically identical generated candidate"
        )

    seed_indices = sorted(set(seed_indices))
    logger.info(
        "  Matched %d/%d seed structures by physical hash",
        len(seed_indices),
        len(seed_ase),
    )
    return seed_indices


def find_elastic_stress_indices(reference_ase: list) -> list[int]:
    """Return all generated candidates carrying elastic-stress provenance."""

    logger.info("")
    logger.info("Step 2d: Loading elastic stress structures")
    indices = [
        index for index, atoms in enumerate(reference_ase)
        if is_elastic_stress(atoms)
    ]
    if not indices:
        raise ValueError(
            "Elastic stress structure inclusion enabled, but no generated "
            "structures with perturbation_type=elastic_stress were found. "
            "Run the generate stage with elastic_stress_enabled=true."
        )
    logger.info(
        "  Matched %d elastic stress structures in generated candidates",
        len(indices),
    )
    return indices


def find_single_element_elastic_stress_indices(reference_ase: list) -> list[int]:
    """Return unary candidates carrying elastic-stress provenance."""

    logger.info("")
    logger.info("Step 2c: Loading single-element elastic stress structures")
    indices = [
        index
        for index, atoms in enumerate(reference_ase)
        if is_elastic_stress(atoms) and is_single_element_structure(atoms)
    ]
    if not indices:
        raise ValueError(
            "Single-element elastic stress structure inclusion enabled, but no "
            "generated structures with perturbation_type=elastic_stress and "
            "single-element composition were found. Run the generate stage with "
            "elastic_stress_enabled=true and check unary seed generation."
        )
    logger.info(
        "  Matched %d single-element elastic stress structures in generated candidates",
        len(indices),
    )
    return indices


__all__ = [
    "find_elastic_stress_indices",
    "find_single_element_elastic_stress_indices",
    "is_elastic_stress",
    "is_single_element_structure",
    "resolve_seed_indices",
    "select_composition_aware_training_set",
    "select_plain_fps_training_set",
    "select_test_set",
    "select_training_set",
    "setting",
]
