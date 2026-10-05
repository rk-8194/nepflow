"""Pure distance, composition-geometry, and sampling services."""

from __future__ import annotations

import logging
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Any

import numpy as np

from .sampling_composition import (
    build_composition_aware_candidate_set,
    composition_aware_attempt_score,
    select_best_sampling_attempt,
)

logger = logging.getLogger(__name__)

COMPOSITION_AWARE_BINARY_BINS = 20
COMPOSITION_AWARE_TERNARY_RESOLUTION = 18
COMPOSITION_AWARE_NOVELTY_FLOOR_FRACTION = 0.90
SQRT3_OVER_2 = math.sqrt(3.0) / 2.0


@dataclass(frozen=True)
class StructureComposition:
    """Composition data shared by selection diagnostics."""

    structure_index: int
    formula: str
    total_atoms: int
    unique_elements: tuple[str, ...]
    element_counts: dict[str, int]
    element_fractions: dict[str, float]


@dataclass(frozen=True)
class BinaryProjection:
    """One normalized binary composition coordinate in ``[0, 1]``."""
    subset: tuple[str, str]
    structure_index: int
    normalized_fraction_b: float


@dataclass(frozen=True)
class TernaryProjection:
    """One normalized ternary composition point in barycentric order."""
    subset: tuple[str, str, str]
    structure_index: int
    barycentric: tuple[float, float, float]


@dataclass(frozen=True)
class CoverageSummary:
    """Histogram, entropy, concentration, and novelty diagnostics for a subset."""
    subset_label: str
    subset_size: int
    dimensions: int
    structure_count: int
    total_bins: int
    occupied_bins: int
    occupied_bin_fraction: float
    normalized_entropy: float
    max_bin_fraction: float
    gini: float
    nn_distance_mean: float | None
    nn_distance_p95: float | None


@dataclass(frozen=True)
class PairFrequencyPoint:
    """Frequency of one binary composition fraction across structures."""
    pair: tuple[str, str]
    b_fraction: float
    frequency: int
    distinct_elements: int


def select_farthest_points(
    representations: np.ndarray,
    structures: list,
    mean_descriptor: bool,
    min_dist: float,
) -> list[int]:
    """Select frame indices using descriptor-space farthest-point sampling.

    ``representations`` has one row per descriptor atom or frame, depending on
    ``mean_descriptor``.  ``structures`` supplies atom counts for mapping
    atomic rows back to frame indices.  Returned indices are sorted unique
    frame positions; distances use the descriptor's native units.
    """

    from NepTrainKit.core.io import farthest_point_sampling

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
    """Binary-search a descriptor distance for a target FPS frame count.

    ``target`` is a frame count and ``tolerance`` is an allowed count error.
    The returned distance is in representation units and the indices are
    sorted frame positions.  The search is monotonic only under the accepted
    FPS implementation; failure to converge returns the best observed point.
    """

    prefix = f"[{label}] " if label else ""
    lo = 0.0
    hi = _find_target_distance_upper_bound(
        representations,
        structures,
        mean_descriptor,
        target,
        prefix,
    )

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
            "  %sIteration %s: min_distance=%.6f -> %s structures (target=%s+/-%s)",
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
            "  %sSearch did not converge within tolerance; using min_distance=%.6f",
            prefix,
            best_dist,
        )

    return best_indices, best_dist


def _find_target_distance_upper_bound(
    representations: np.ndarray,
    structures: list,
    mean_descriptor: bool,
    target: int,
    prefix: str,
) -> float:
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
            return distance
        distance *= 2


def calculate_cross_distance_stats(
    representations: np.ndarray,
    indices_a: list[int],
    indices_b: list[int],
) -> tuple[float, float]:
    """Return minimum and mean nearest distance from B to A.

    ``representations`` has shape ``(n_rows, n_features)`` and indices refer
    to its rows.  Distances are Euclidean in descriptor units; empty inputs
    return ``(inf, inf)`` to preserve the absence of cross-comparisons.
    """

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
                    element: fraction / total for element, fraction in sorted(fractions.items())
                }

    counts = Counter(atoms.get_chemical_symbols())
    total_atoms = sum(counts.values())
    if total_atoms <= 0:
        return {}
    return {element: count / total_atoms for element, count in sorted(counts.items())}


def composition_from_atoms(atoms: Any, structure_index: int) -> StructureComposition:
    """Create a stable composition record from an ASE-like structure."""

    counts = Counter(atoms.get_chemical_symbols())
    total_atoms = sum(counts.values())
    if total_atoms == 0:
        raise ValueError(f"Structure {structure_index} contains no atoms")
    unique_elements = tuple(sorted(counts))
    return StructureComposition(
        structure_index=structure_index,
        formula=atoms.get_chemical_formula(),
        total_atoms=total_atoms,
        unique_elements=unique_elements,
        element_counts=dict(counts),
        element_fractions={element: counts[element] / total_atoms for element in unique_elements},
    )


def normalize_subset_fractions(
    fractions: dict[str, float],
    subset: tuple[str, ...],
) -> tuple[float, ...]:
    """Normalize composition fractions onto a binary or ternary subset.

    The result follows ``subset`` ordering and sums to one when the subset has
    positive support; unsupported subsets return zeros.
    """

    subset_total = sum(fractions[element] for element in subset)
    if subset_total <= 0.0:
        return tuple(0.0 for _ in subset)
    return tuple(fractions[element] / subset_total for element in subset)


def normalize_composition_subset(
    composition: StructureComposition,
    subset: tuple[str, ...],
) -> tuple[float, ...]:
    """Normalize a composition onto a binary or ternary projection."""

    return normalize_subset_fractions(composition.element_fractions, subset)


def collect_binary_projections(
    compositions: list[StructureComposition],
) -> dict[tuple[str, str], list[BinaryProjection]]:
    """Collect every binary projection, including projections of higher-order structures."""

    projections: dict[tuple[str, str], list[BinaryProjection]] = {}
    for composition in compositions:
        for subset in combinations(composition.unique_elements, 2):
            normalized = normalize_composition_subset(composition, subset)
            projections.setdefault(subset, []).append(
                BinaryProjection(subset, composition.structure_index, normalized[1])
            )
    return projections


def collect_ternary_projections(
    compositions: list[StructureComposition],
) -> dict[tuple[str, str, str], list[TernaryProjection]]:
    """Collect every ternary projection, including quaternary structures."""

    projections: dict[tuple[str, str, str], list[TernaryProjection]] = {}
    for composition in compositions:
        for subset in combinations(composition.unique_elements, 3):
            projections.setdefault(subset, []).append(
                TernaryProjection(
                    subset,
                    composition.structure_index,
                    _as_ternary(normalize_composition_subset(composition, subset)),
                )
            )
    return projections


def binary_bin_index(value: float, bins: int) -> int:
    """Map a binary fraction to a deterministic composition bin."""

    clamped = min(max(value, 0.0), 1.0)
    if math.isclose(clamped, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return bins - 1
    return min(int(clamped * bins), bins - 1)


def largest_remainder_integer_partition(
    values: tuple[float, float, float],
    total: int,
) -> tuple[int, int, int]:
    """Convert barycentric fractions to deterministic simplex coordinates.

    ``values`` are ordered fractions summing approximately to one and ``total``
    is the integer resolution.  The returned non-negative coordinates sum to
    ``total``; ties are resolved by original component order.
    """

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
    return int(floors[0]), int(floors[1]), int(floors[2])


def _as_ternary(values: tuple[float, ...]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("ternary composition requires exactly three fractions")
    return values[0], values[1], values[2]


def ternary_bin_index(
    barycentric: tuple[float, float, float],
    resolution: int,
) -> tuple[int, int, int]:
    """Map ternary barycentric fractions to integer simplex coordinates."""

    return largest_remainder_integer_partition(barycentric, resolution)


def ternary_bin_center(
    index: tuple[int, int, int],
    resolution: int,
) -> tuple[float, float, float]:
    """Return the normalized barycentric center represented by one bin."""
    return index[0] / resolution, index[1] / resolution, index[2] / resolution


def barycentric_to_cartesian(
    barycentric: tuple[float, float, float],
) -> tuple[float, float]:
    """Map ``(a,b,c)`` barycentric fractions to a unit equilateral triangle."""
    _a, b, c = barycentric
    return b + 0.5 * c, c * SQRT3_OVER_2


def occupied_bin_fraction(counts: list[int], total_bins: int) -> tuple[int, float]:
    """Return occupied-bin count and occupied fraction."""
    occupied = sum(1 for count in counts if count > 0)
    return occupied, occupied / total_bins if total_bins > 0 else 0.0


def max_bin_fraction(counts: list[int]) -> float:
    """Return the largest bin count divided by the total count."""
    total = sum(counts)
    return max(counts) / total if total > 0 else 0.0


def gini(values: list[int]) -> float:
    """Return the count-distribution Gini coefficient in ``[0, 1]``."""
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    weighted_sum = sum(
        (2 * idx - n - 1) * value for idx, value in enumerate(sorted_values, start=1)
    )
    return weighted_sum / (n * total)


def nearest_neighbor_distances(points: list[tuple[float, ...]]) -> list[float]:
    """Return each point's Euclidean nearest-neighbour distance."""
    if len(points) < 2:
        return []
    distances: list[float] = []
    for index, point in enumerate(points):
        best = math.inf
        for other_index, other in enumerate(points):
            if index == other_index:
                continue
            best = min(
                best,
                math.sqrt(sum((a - b) ** 2 for a, b in zip(point, other))),
            )
        distances.append(best)
    return distances


def nearest_representation_distances(representations: np.ndarray) -> np.ndarray:
    """Return each representation row's nearest other-row distance."""

    if len(representations) < 2:
        return np.array([], dtype=float)
    from scipy.spatial.distance import cdist

    distances = cdist(representations, representations)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def percentile95(values: list[float]) -> float | None:
    """Return linearly interpolated 95th percentile or ``None`` when empty."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_binary_subset(
    subset: tuple[str, str],
    projections: list[BinaryProjection],
    bins: int,
) -> tuple[CoverageSummary, list[int]]:
    """Summarize one binary projection using deterministic histogram bins."""
    bin_counts = [0] * bins
    coordinates: list[tuple[float]] = []
    for projection in projections:
        bin_counts[binary_bin_index(projection.normalized_fraction_b, bins)] += 1
        coordinates.append((projection.normalized_fraction_b,))
    occupied, occupied_fraction = occupied_bin_fraction(bin_counts, bins)
    nn_distances = nearest_neighbor_distances(coordinates)
    return CoverageSummary(
        subset_label="-".join(subset),
        subset_size=2,
        dimensions=1,
        structure_count=len(projections),
        total_bins=bins,
        occupied_bins=occupied,
        occupied_bin_fraction=occupied_fraction,
        normalized_entropy=normalized_entropy(bin_counts, bins),
        max_bin_fraction=max_bin_fraction(bin_counts),
        gini=gini(bin_counts),
        nn_distance_mean=statistics.fmean(nn_distances) if nn_distances else None,
        nn_distance_p95=percentile95(nn_distances),
    ), bin_counts


def summarize_ternary_subset(
    subset: tuple[str, str, str],
    projections: list[TernaryProjection],
    resolution: int,
) -> tuple[CoverageSummary, dict[tuple[int, int, int], int]]:
    """Summarize one ternary projection on the integer simplex grid."""
    bin_counts: dict[tuple[int, int, int], int] = {}
    coordinates: list[tuple[float, float]] = []
    for projection in projections:
        bin_index = ternary_bin_index(projection.barycentric, resolution)
        bin_counts[bin_index] = bin_counts.get(bin_index, 0) + 1
        coordinates.append(barycentric_to_cartesian(projection.barycentric))
    total_bins = ternary_bin_count(resolution)
    dense_counts = list(bin_counts.values()) + [0] * (total_bins - len(bin_counts))
    occupied, occupied_fraction = occupied_bin_fraction(dense_counts, total_bins)
    nn_distances = nearest_neighbor_distances(coordinates)
    return CoverageSummary(
        subset_label="-".join(subset),
        subset_size=3,
        dimensions=2,
        structure_count=len(projections),
        total_bins=total_bins,
        occupied_bins=occupied,
        occupied_bin_fraction=occupied_fraction,
        normalized_entropy=normalized_entropy(dense_counts, total_bins),
        max_bin_fraction=max_bin_fraction(dense_counts),
        gini=gini(dense_counts),
        nn_distance_mean=statistics.fmean(nn_distances) if nn_distances else None,
        nn_distance_p95=percentile95(nn_distances),
    ), bin_counts


def collect_pair_frequency_points(
    compositions: list[StructureComposition],
) -> dict[tuple[str, str], list[PairFrequencyPoint]]:
    """Build pair-frequency report points from canonical composition records."""

    pair_fraction_counts: dict[tuple[str, str], Counter[float]] = {}
    pair_fraction_distinct: dict[tuple[str, str], dict[float, set[int]]] = {}
    for composition in compositions:
        distinct_count = len(composition.unique_elements)
        for index, element_a in enumerate(composition.unique_elements):
            for element_b in composition.unique_elements[index + 1 :]:
                pair = (element_a, element_b)
                fraction = composition.element_fractions[element_b]
                pair_fraction_counts.setdefault(pair, Counter())[fraction] += 1
                pair_fraction_distinct.setdefault(pair, {}).setdefault(fraction, set()).add(
                    distinct_count
                )
    points: dict[tuple[str, str], list[PairFrequencyPoint]] = {}
    for pair, counts in pair_fraction_counts.items():
        points[pair] = [
            PairFrequencyPoint(pair, fraction, frequency, distinct)
            for fraction, frequency in sorted(counts.items())
            for distinct in sorted(pair_fraction_distinct[pair][fraction])
        ]
    return points


def fraction_label(value: float) -> str:
    """Render a composition fraction as a simple rational when possible."""
    fraction = Fraction(value).limit_denominator()
    if math.isclose(float(fraction), value, rel_tol=0.0, abs_tol=1e-12):
        return f"{fraction.numerator}/{fraction.denominator}"
    return f"{value:.6f}"


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
            subset
            + ternary_bin_index(
                _as_ternary(normalized),
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
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(vector_a, vector_b)))


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
        return {idx: float(distance) for idx, distance in zip(candidate_indices, nearest)}
    except (ImportError, TypeError, ValueError, FloatingPointError):
        # SciPy's cdist is an optional acceleration.  These are the expected
        # dependency/input failures; the scalar implementation is equivalent
        # and remains fully identity-preserving.
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
    except (ImportError, TypeError, ValueError, FloatingPointError):
        # Keep the same explicitly equivalent fallback as the initial
        # distance calculation; arbitrary programming or resource failures
        # must remain visible.
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
        for other in selected_indices[position + 1 :]:
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
        for other in selected_indices[position + 1 :]:
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
    return attempts[: max(1, adaptive_retries)]


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
        len(counts) / COMPOSITION_AWARE_BINARY_BINS for counts in binary_subset_counts.values()
    ]
    binary_entropy = [
        normalized_entropy(list(counts.values()), COMPOSITION_AWARE_BINARY_BINS)
        for counts in binary_subset_counts.values()
    ]
    ternary_total_bins = ternary_bin_count(COMPOSITION_AWARE_TERNARY_RESOLUTION)
    ternary_occupied = [
        len(counts) / ternary_total_bins for counts in ternary_subset_counts.values()
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
        nearest_distances.append(nearest_descriptor_distance(representations, index, candidates))
    return mean(nearest_distances)


__all__ = [
    "BinaryProjection",
    "COMPOSITION_AWARE_BINARY_BINS",
    "COMPOSITION_AWARE_NOVELTY_FLOOR_FRACTION",
    "COMPOSITION_AWARE_TERNARY_RESOLUTION",
    "CoverageSummary",
    "PairFrequencyPoint",
    "SQRT3_OVER_2",
    "StructureComposition",
    "TernaryProjection",
    "binary_bin_index",
    "build_composition_aware_candidate_set",
    "calculate_cross_distance_stats",
    "composition_aware_attempt_schedule",
    "composition_aware_attempt_score",
    "calculate_composition_coverage_metrics",
    "calculate_mean_nearest_distance",
    "calculate_min_distance",
    "calculate_positive_min_distance",
    "barycentric_to_cartesian",
    "collect_binary_projections",
    "collect_pair_frequency_points",
    "collect_ternary_projections",
    "composition_from_atoms",
    "composition_projection_bins",
    "composition_sparsity_reward",
    "descriptor_distance",
    "extract_composition_fractions",
    "fraction_label",
    "flatten_single_column_distances",
    "initialize_nearest_distances",
    "largest_remainder_integer_partition",
    "mean",
    "min_max_normalize",
    "nearest_descriptor_distance",
    "nearest_neighbor_distances",
    "nearest_representation_distances",
    "normalize_subset_fractions",
    "normalize_composition_subset",
    "normalized_entropy",
    "occupied_bin_fraction",
    "percentile95",
    "select_best_sampling_attempt",
    "select_farthest_points",
    "select_farthest_points_for_target",
    "ternary_bin_count",
    "ternary_bin_center",
    "ternary_bin_index",
    "summarize_binary_subset",
    "summarize_ternary_subset",
    "gini",
    "max_bin_fraction",
    "update_nearest_distances",
]
