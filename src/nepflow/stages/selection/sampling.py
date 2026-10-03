"""Pure distance, composition-geometry, and sampling services."""

from __future__ import annotations

import logging
import math
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
from NepTrainKit.core.io import farthest_point_sampling


logger = logging.getLogger("nepflow.selection.sampling")

COMPOSITION_AWARE_BINARY_BINS = 20
COMPOSITION_AWARE_TERNARY_RESOLUTION = 18
COMPOSITION_AWARE_NOVELTY_FLOOR_FRACTION = 0.90


def select_farthest_points(
    representations: np.ndarray,
    structures: list,
    mean_descriptor: bool,
    min_dist: float,
) -> list[int]:
    """Return sorted frame indices selected by the accepted FPS policy."""

    selected_indices = farthest_point_sampling(
        representations,
        n_samples=len(structures),
        min_dist=min_dist,
    )
    if not mean_descriptor:
        atoms_per_frame = [structure.num_atoms for structure in structures]
        comes_from = []
        for index, atom_count in enumerate(atoms_per_frame):
            comes_from.extend([index] * atom_count)
        return sorted(set(comes_from[index] for index in selected_indices))
    return sorted(selected_indices)


def _selected_count(
    representations: np.ndarray,
    structures: list,
    mean_descriptor: bool,
    min_dist: float,
) -> int:
    """Return the number of frames selected at one distance threshold."""

    return len(
        select_farthest_points(
            representations,
            structures,
            mean_descriptor,
            min_dist,
        )
    )


def select_farthest_points_for_target(
    representations: np.ndarray,
    structures: list,
    mean_descriptor: bool,
    target: int,
    tolerance: int,
    max_iterations: int,
    label: str = "",
) -> tuple[list[int], float]:
    """Binary-search a minimum distance to hit the accepted target count."""

    prefix = f"[{label}] " if label else ""
    lo, hi = 0.0, None
    distance = 0.01
    while True:
        count = _selected_count(
            representations,
            structures,
            mean_descriptor,
            distance,
        )
        logger.info(
            "  %sProbing min_distance=%.6f -> %s structures",
            prefix,
            distance,
            count,
        )
        if count <= target:
            hi = distance
            break
        distance *= 2

    if hi == 0.0:
        indices = select_farthest_points(
            representations,
            structures,
            mean_descriptor,
            0.0,
        )
        return indices, 0.0

    best_dist = hi
    best_indices = select_farthest_points(
        representations,
        structures,
        mean_descriptor,
        hi,
    )

    for iteration in range(max_iterations):
        mid = (lo + hi) / 2
        count = _selected_count(
            representations,
            structures,
            mean_descriptor,
            mid,
        )
        logger.info(
            "  %sIteration %s: min_distance=%.6f -> %s structures "
            "(target=%s+/-%s)",
            prefix,
            iteration + 1,
            mid,
            count,
            target,
            tolerance,
        )

        if abs(count - target) <= tolerance:
            best_dist = mid
            best_indices = select_farthest_points(
                representations,
                structures,
                mean_descriptor,
                mid,
            )
            break

        if count > target:
            lo = mid
        else:
            hi = mid
            best_dist = mid
            best_indices = select_farthest_points(
                representations,
                structures,
                mean_descriptor,
                mid,
            )
    else:
        logger.info(
            "  %sSearch did not converge within tolerance; "
            "using min_distance=%.6f",
            prefix,
            best_dist,
        )

    return best_indices, best_dist


def calculate_cross_distance_stats(
    representations: np.ndarray,
    indices_a: list[int],
    indices_b: list[int],
) -> tuple[float, float]:
    """Return (minimum, mean) nearest-neighbour distance from B to A."""

    from scipy.spatial.distance import cdist

    if not indices_a or not indices_b:
        return float("inf"), float("inf")
    distances = cdist(representations[indices_b], representations[indices_a])
    nearest_neighbour_distances = distances.min(axis=1)
    return (
        float(nearest_neighbour_distances.min()),
        float(nearest_neighbour_distances.mean()),
    )


def extract_composition_fractions(atoms: Any) -> dict[str, float]:
    """Return normalized positive composition fractions for one structure."""

    composition = atoms.info.get("composition")
    if isinstance(composition, dict):
        fractions: dict[str, float] = {}
        for element, value in composition.items():
            try:
                fraction = float(value)
            except (TypeError, ValueError):
                continue
            if fraction > 0.0:
                fractions[str(element)] = fraction
        if fractions:
            total = sum(fractions.values())
            if total > 0.0:
                return {
                    element: fraction / total
                    for element, fraction in sorted(fractions.items())
                }

    counts = Counter(atoms.get_chemical_symbols())
    total_atoms = sum(counts.values())
    if total_atoms <= 0:
        return {}
    return {
        element: count / total_atoms
        for element, count in sorted(counts.items())
    }


def normalize_subset_fractions(
    fractions: dict[str, float],
    subset: tuple[str, ...],
) -> tuple[float, ...]:
    """Normalize fractions onto one binary or ternary element subset."""

    subset_total = sum(fractions[element] for element in subset)
    if subset_total <= 0.0:
        return tuple(0.0 for _ in subset)
    return tuple(fractions[element] / subset_total for element in subset)


def binary_bin_index(value: float, bins: int) -> int:
    """Map a binary fraction to a deterministic composition bin."""

    clamped = min(max(value, 0.0), 1.0)
    if math.isclose(clamped, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return bins - 1
    return min(int(clamped * bins), bins - 1)


def largest_remainder_integer_partition(
    values: tuple[float, ...],
    total: int,
) -> tuple[int, ...]:
    """Convert barycentric fractions into deterministic integer coordinates."""

    scaled = [value * total for value in values]
    floors = [math.floor(value) for value in scaled]
    remainder = total - sum(floors)
    if remainder > 0:
        ranked = sorted(
            enumerate(scaled),
            key=lambda item: (item[1] - floors[item[0]], -item[0]),
            reverse=True,
        )
        for idx, _ in ranked[:remainder]:
            floors[idx] += 1
    return tuple(int(value) for value in floors)


def ternary_bin_index(
    barycentric: tuple[float, float, float],
    resolution: int,
) -> tuple[int, int, int]:
    """Map ternary barycentric fractions to integer simplex coordinates."""

    return largest_remainder_integer_partition(barycentric, resolution)


def composition_projection_bins(atoms: Any) -> dict[str, list[tuple]]:
    """Return binary and ternary composition projections for one structure."""

    fractions = extract_composition_fractions(atoms)
    active_elements = tuple(
        sorted(element for element, fraction in fractions.items() if fraction > 0.0)
    )
    binary_bins: list[tuple] = []
    ternary_bins: list[tuple] = []

    for subset in combinations(active_elements, 2):
        normalized = normalize_subset_fractions(fractions, subset)
        binary_bins.append(
            subset + (binary_bin_index(normalized[1], COMPOSITION_AWARE_BINARY_BINS),)
        )

    for subset in combinations(active_elements, 3):
        normalized = normalize_subset_fractions(fractions, subset)
        ternary_bins.append(
            subset + ternary_bin_index(
                normalized,
                COMPOSITION_AWARE_TERNARY_RESOLUTION,
            )
        )

    return {"binary": binary_bins, "ternary": ternary_bins}


def bin_deficit_reward(
    bin_count: int,
    subset_total_count: int,
    total_bins: int,
) -> float:
    """Return the normalized reward for an underrepresented composition bin."""

    if total_bins <= 0:
        return 0.0
    target_level = max(1, int(math.floor(subset_total_count / total_bins)) + 1)
    deficit = max(0, target_level - bin_count)
    if deficit <= 0:
        return 0.0
    return deficit / target_level


def composition_sparsity_reward(
    candidate_bins: dict[str, list[tuple]],
    binary_counts: Counter,
    ternary_counts: Counter,
    binary_subset_totals: Counter,
    ternary_subset_totals: Counter,
    ternary_weight: float,
) -> float:
    """Score how much a candidate fills sparse binary/ternary bins."""

    weighted_rewards = 0.0
    total_weight = 0.0
    for bin_key in candidate_bins["binary"]:
        subset = bin_key[:2]
        weighted_rewards += bin_deficit_reward(
            binary_counts[bin_key],
            binary_subset_totals[subset],
            COMPOSITION_AWARE_BINARY_BINS,
        )
        total_weight += 1.0
    for bin_key in candidate_bins["ternary"]:
        if ternary_weight <= 0.0:
            continue
        subset = bin_key[:3]
        weighted_rewards += ternary_weight * bin_deficit_reward(
            ternary_counts[bin_key],
            ternary_subset_totals[subset],
            ternary_bin_count(COMPOSITION_AWARE_TERNARY_RESOLUTION),
        )
        total_weight += ternary_weight
    if total_weight <= 0.0:
        return 0.0
    return weighted_rewards / total_weight


def min_max_normalize(values: dict[int, float]) -> dict[int, float]:
    """Normalize a mapping while keeping ties deterministic at zero."""

    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return {key: 0.0 for key in values}
    scale = high - low
    return {key: (value - low) / scale for key, value in values.items()}


def descriptor_distance(
    representations: np.ndarray,
    index_a: int,
    index_b: int,
) -> float:
    """Return Euclidean distance between two representation rows."""

    vector_a = representations[index_a]
    vector_b = representations[index_b]
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(vector_a, vector_b))
    )


def nearest_descriptor_distance(
    representations: np.ndarray,
    index: int,
    selected_indices: list[int],
) -> float:
    """Return the nearest selected representation distance."""

    if not selected_indices:
        return 0.0
    return float(
        min(
            descriptor_distance(representations, index, other_index)
            for other_index in selected_indices
        )
    )


def flatten_single_column_distances(distance_matrix: Any) -> list[float]:
    """Flatten scipy-like or simple nested one-column distance values."""

    if hasattr(distance_matrix, "ravel"):
        return [float(value) for value in distance_matrix.ravel()]

    flattened: list[float] = []
    for row in distance_matrix:
        if isinstance(row, (list, tuple)):
            if not row:
                continue
            flattened.append(float(row[0]))
        else:
            flattened.append(float(row))
    return flattened


def initialize_nearest_distances(
    representations: np.ndarray,
    candidate_indices: list[int],
    selected_indices: list[int],
) -> dict[int, float]:
    """Calculate nearest distances with a scipy fast path and pure fallback."""

    if not selected_indices:
        return {idx: 0.0 for idx in candidate_indices}
    try:
        from scipy.spatial.distance import cdist

        nearest = cdist(
            representations[candidate_indices],
            representations[selected_indices],
        ).min(axis=1)
        return {
            idx: float(distance)
            for idx, distance in zip(candidate_indices, nearest)
        }
    except Exception:
        return {
            idx: nearest_descriptor_distance(representations, idx, selected_indices)
            for idx in candidate_indices
        }


def update_nearest_distances(
    representations: np.ndarray,
    candidate_indices: list[int],
    new_selected_index: int,
    current_nearest: dict[int, float],
) -> None:
    """Update candidate novelty distances after adding one selected row."""

    if not candidate_indices:
        return
    try:
        from scipy.spatial.distance import cdist

        distance_matrix = cdist(
            representations[candidate_indices],
            representations[[new_selected_index]],
        )
        distances = flatten_single_column_distances(distance_matrix)
        for idx, distance in zip(candidate_indices, distances):
            current_nearest[idx] = min(current_nearest[idx], float(distance))
    except Exception:
        for idx in candidate_indices:
            distance = descriptor_distance(
                representations,
                idx,
                new_selected_index,
            )
            current_nearest[idx] = min(current_nearest[idx], distance)


def calculate_min_distance(
    representations: np.ndarray,
    selected_indices: list[int],
) -> float:
    """Return the minimum pairwise distance, including zero duplicates."""

    if len(selected_indices) < 2:
        return 0.0
    best = math.inf
    for position, index in enumerate(selected_indices):
        for other in selected_indices[position + 1:]:
            best = min(best, descriptor_distance(representations, index, other))
    return float(best if best < math.inf else 0.0)


def calculate_positive_min_distance(
    representations: np.ndarray,
    selected_indices: list[int],
) -> float:
    """Return the minimum strictly positive pairwise distance."""

    if len(selected_indices) < 2:
        return 0.0
    best = math.inf
    for position, index in enumerate(selected_indices):
        for other in selected_indices[position + 1:]:
            distance = descriptor_distance(representations, index, other)
            if distance > 1e-12:
                best = min(best, distance)
    return float(best if best < math.inf else 0.0)


def composition_aware_attempt_schedule(
    frontier_fraction: float,
    ternary_weight: float,
    adaptive_retries: int,
) -> list[tuple[float, float]]:
    """Return the accepted deterministic composition-attempt schedule."""

    attempts = [
        (frontier_fraction, ternary_weight),
        (max(0.05, 0.5 * frontier_fraction), ternary_weight),
        (min(0.25, 2.0 * frontier_fraction), ternary_weight),
        (frontier_fraction, 2.0 * ternary_weight),
    ]
    return attempts[:max(1, adaptive_retries)]


def calculate_composition_coverage_metrics(
    selected_indices: list[int],
    candidate_bins: dict[int, dict[str, list[tuple]]],
) -> dict[str, float]:
    """Calculate binary/ternary occupied-bin and entropy metrics."""

    binary_subset_counts: dict[tuple[str, str], Counter] = {}
    ternary_subset_counts: dict[tuple[str, str, str], Counter] = {}

    for index in selected_indices:
        for bin_key in candidate_bins[index]["binary"]:
            subset = bin_key[:2]
            binary_subset_counts.setdefault(subset, Counter())[bin_key] += 1
        for bin_key in candidate_bins[index]["ternary"]:
            subset = bin_key[:3]
            ternary_subset_counts.setdefault(subset, Counter())[bin_key] += 1

    binary_occupied = [
        len(counts) / COMPOSITION_AWARE_BINARY_BINS
        for counts in binary_subset_counts.values()
    ]
    binary_entropy = [
        normalized_entropy(list(counts.values()), COMPOSITION_AWARE_BINARY_BINS)
        for counts in binary_subset_counts.values()
    ]
    ternary_total_bins = ternary_bin_count(COMPOSITION_AWARE_TERNARY_RESOLUTION)
    ternary_occupied = [
        len(counts) / ternary_total_bins
        for counts in ternary_subset_counts.values()
    ]
    ternary_entropy = [
        normalized_entropy(list(counts.values()), ternary_total_bins)
        for counts in ternary_subset_counts.values()
    ]

    return {
        "binary_occupied_bin_fraction": mean(binary_occupied),
        "binary_normalized_entropy": mean(binary_entropy),
        "ternary_occupied_bin_fraction": mean(ternary_occupied),
        "ternary_normalized_entropy": mean(ternary_entropy),
    }


def normalized_entropy(counts: list[int], total_bins: int) -> float:
    """Return entropy normalized by the number of available bins."""

    total = sum(counts)
    if total <= 0 or total_bins <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(total_bins)


def ternary_bin_count(resolution: int) -> int:
    """Return the number of integer bins in a ternary simplex."""

    return (resolution + 1) * (resolution + 2) // 2


def mean(values: list[float]) -> float:
    """Return a zero-valued mean for an empty metric collection."""

    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_mean_nearest_distance(
    representations: np.ndarray,
    selected_indices: list[int],
) -> float:
    """Return each selected row's nearest-neighbour distance mean."""

    if len(selected_indices) < 2:
        return 0.0
    nearest_distances: list[float] = []
    for position, index in enumerate(selected_indices):
        candidates = [
            other
            for other_position, other in enumerate(selected_indices)
            if position != other_position
        ]
        nearest_distances.append(
            nearest_descriptor_distance(representations, index, candidates)
        )
    return mean(nearest_distances)


def composition_aware_attempt_score(attempt_result: dict) -> float:
    """Score composition coverage without consulting stage state."""

    return (
        0.35 * attempt_result["binary_occupied_bin_fraction"]
        + 0.20 * attempt_result["binary_normalized_entropy"]
        + 0.30 * attempt_result["ternary_occupied_bin_fraction"]
        + 0.15 * attempt_result["ternary_normalized_entropy"]
    )


def select_best_sampling_attempt(
    attempt_results: list[dict],
    descriptor_floor_fraction: float,
) -> dict:
    """Select the best coverage attempt subject to descriptor-quality floors."""

    baseline = attempt_results[0]
    baseline_score = composition_aware_attempt_score(baseline)
    best_attempt = baseline
    best_score = baseline_score
    baseline_min_reference = max(
        baseline["train_positive_min_dist"],
        baseline["train_min_dist"],
    )
    min_distance_floor = baseline_min_reference * descriptor_floor_fraction
    mean_nn_floor = baseline["train_mean_nn_dist"] * descriptor_floor_fraction
    logger.info(
        "  Baseline descriptor floors: positive_min_dist>=%.6f, mean_nn>=%.6f",
        min_distance_floor,
        mean_nn_floor,
    )
    logger.info(
        "    Attempt %d accepted as baseline (score=%.3f)",
        baseline["attempt_number"],
        baseline_score,
    )

    for attempt in attempt_results[1:]:
        attempt_score = composition_aware_attempt_score(attempt)
        attempt_min_reference = max(
            attempt["train_positive_min_dist"],
            attempt["train_min_dist"],
        )
        eligible = (
            attempt_min_reference >= min_distance_floor
            and attempt["train_mean_nn_dist"] >= mean_nn_floor
        )
        logger.info(
            "    Attempt %d %s descriptor floors (score=%.3f)",
            attempt["attempt_number"],
            "passed" if eligible else "failed",
            attempt_score,
        )
        if not eligible:
            continue

        better = False
        if attempt_score > best_score + 1e-12:
            better = True
        elif math.isclose(attempt_score, best_score, rel_tol=0.0, abs_tol=1e-12):
            if attempt["train_min_dist"] > best_attempt["train_min_dist"] + 1e-12:
                better = True
            elif math.isclose(
                attempt["train_min_dist"],
                best_attempt["train_min_dist"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                if attempt["train_mean_nn_dist"] > best_attempt["train_mean_nn_dist"] + 1e-12:
                    better = True
                elif math.isclose(
                    attempt["train_mean_nn_dist"],
                    best_attempt["train_mean_nn_dist"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    better = attempt["attempt_number"] < best_attempt["attempt_number"]
        if better:
            best_attempt = attempt
            best_score = attempt_score
            logger.info(
                "    Attempt %d became current best",
                attempt["attempt_number"],
            )

    return best_attempt


def build_composition_aware_candidate_set(
    representations: np.ndarray,
    candidate_bins: dict[int, dict[str, list[tuple]]],
    anchor_indices: list[int],
    remaining_indices: list[int],
    remaining_target: int,
    frontier_fraction: float,
    ternary_weight: float,
) -> dict:
    """Build one deterministic composition-aware candidate set."""

    selected_indices = list(anchor_indices)
    remaining_pool = list(remaining_indices)
    binary_counts: Counter = Counter()
    ternary_counts: Counter = Counter()
    binary_subset_totals: Counter = Counter()
    ternary_subset_totals: Counter = Counter()
    for index in anchor_indices:
        binary_counts.update(candidate_bins[index]["binary"])
        ternary_counts.update(candidate_bins[index]["ternary"])
        binary_subset_totals.update(
            bin_key[:2] for bin_key in candidate_bins[index]["binary"]
        )
        ternary_subset_totals.update(
            bin_key[:3] for bin_key in candidate_bins[index]["ternary"]
        )

    logger.info(
        "    Initializing descriptor distances for %d candidates against %d anchors",
        len(remaining_pool),
        len(selected_indices),
    )
    current_nearest = initialize_nearest_distances(
        representations,
        remaining_pool,
        selected_indices,
    )
    logger.info("    Initial descriptor-distance initialization complete")

    for iteration in range(remaining_target):
        if not remaining_pool:
            break

        descriptor_ranked = sorted(
            remaining_pool,
            key=lambda index: (-current_nearest[index], index),
        )
        frontier_size = max(1, int(math.ceil(frontier_fraction * len(remaining_pool))))
        frontier = descriptor_ranked[:frontier_size]
        positive_frontier = [
            index for index in frontier if current_nearest[index] > 1e-12
        ]
        if positive_frontier:
            frontier = positive_frontier
        best_frontier_novelty = max(current_nearest[index] for index in frontier)
        if best_frontier_novelty > 1e-12:
            novelty_floor = COMPOSITION_AWARE_NOVELTY_FLOOR_FRACTION * best_frontier_novelty
            novelty_frontier = [
                index
                for index in frontier
                if current_nearest[index] >= novelty_floor
            ]
            if novelty_frontier:
                frontier = novelty_frontier
        composition_reward = {
            index: composition_sparsity_reward(
                candidate_bins[index],
                binary_counts,
                ternary_counts,
                binary_subset_totals,
                ternary_subset_totals,
                ternary_weight,
            )
            for index in frontier
        }
        best_index = min(
            frontier,
            key=lambda index: (
                -composition_reward[index],
                -current_nearest[index],
                index,
            ),
        )

        selected_indices.append(best_index)
        binary_counts.update(candidate_bins[best_index]["binary"])
        ternary_counts.update(candidate_bins[best_index]["ternary"])
        binary_subset_totals.update(
            bin_key[:2] for bin_key in candidate_bins[best_index]["binary"]
        )
        ternary_subset_totals.update(
            bin_key[:3] for bin_key in candidate_bins[best_index]["ternary"]
        )
        remaining_pool.remove(best_index)
        current_nearest.pop(best_index, None)
        update_nearest_distances(
            representations,
            remaining_pool,
            best_index,
            current_nearest,
        )

        if (iteration + 1) % 50 == 0 or iteration + 1 == remaining_target:
            logger.info(
                "    Attempt progress: %d/%d selected (remaining candidates: %d)",
                iteration + 1,
                remaining_target,
                len(remaining_pool),
            )

    selected_indices = sorted(selected_indices)
    coverage = calculate_composition_coverage_metrics(selected_indices, candidate_bins)
    coverage.update(
        {
            "selected_indices": selected_indices,
            "train_min_dist": calculate_min_distance(
                representations,
                selected_indices,
            ),
            "train_positive_min_dist": calculate_positive_min_distance(
                representations,
                selected_indices,
            ),
            "train_mean_nn_dist": calculate_mean_nearest_distance(
                representations,
                selected_indices,
            ),
            "frontier_fraction": frontier_fraction,
            "ternary_weight": ternary_weight,
            "occupied_binary_bins": len(binary_counts),
            "occupied_ternary_bins": len(ternary_counts),
        }
    )
    return coverage


__all__ = [
    "COMPOSITION_AWARE_BINARY_BINS",
    "COMPOSITION_AWARE_NOVELTY_FLOOR_FRACTION",
    "COMPOSITION_AWARE_TERNARY_RESOLUTION",
    "_selected_count",
    "binary_bin_index",
    "build_composition_aware_candidate_set",
    "calculate_cross_distance_stats",
    "composition_aware_attempt_schedule",
    "composition_aware_attempt_score",
    "calculate_composition_coverage_metrics",
    "calculate_mean_nearest_distance",
    "calculate_min_distance",
    "calculate_positive_min_distance",
    "composition_projection_bins",
    "composition_sparsity_reward",
    "descriptor_distance",
    "extract_composition_fractions",
    "flatten_single_column_distances",
    "initialize_nearest_distances",
    "largest_remainder_integer_partition",
    "mean",
    "min_max_normalize",
    "nearest_descriptor_distance",
    "normalize_subset_fractions",
    "normalized_entropy",
    "select_best_sampling_attempt",
    "select_farthest_points",
    "select_farthest_points_for_target",
    "ternary_bin_count",
    "ternary_bin_index",
    "update_nearest_distances",
]
