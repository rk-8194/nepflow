"""Pure farthest-point sampling services for selection."""

from __future__ import annotations

import logging

import numpy as np
from NepTrainKit.core.io import farthest_point_sampling


logger = logging.getLogger("nepflow.selection.sampling")


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


__all__ = [
    "calculate_cross_distance_stats",
    "select_farthest_points",
    "select_farthest_points_for_target",
]
