"""Shared farthest-point-sampling helpers."""

from __future__ import annotations

import logging

import numpy as np
from NepTrainKit.core.io import farthest_point_sampling

logger = logging.getLogger("nepflow.common.fps")


def fps_run(
    descriptors: np.ndarray,
    structures: list,
    mean_descriptor: bool,
    min_dist: float,
) -> list[int]:
    """Run FPS and return frame indices."""
    selected_indices = farthest_point_sampling(
        descriptors,
        n_samples=len(structures),
        min_dist=min_dist,
    )
    if not mean_descriptor:
        atoms_per_frame = [s.num_atoms for s in structures]
        comesfrom = []
        for i, n in enumerate(atoms_per_frame):
            comesfrom.extend([i] * n)
        return sorted(set(comesfrom[i] for i in selected_indices))
    return sorted(selected_indices)


def fps_count(
    descriptors: np.ndarray,
    structures: list,
    mean_descriptor: bool,
    min_dist: float,
) -> int:
    """Run FPS and return count of selected frame indices."""
    return len(fps_run(descriptors, structures, mean_descriptor, min_dist))


def fps_target_count(
    descriptors: np.ndarray,
    structures: list,
    mean_descriptor: bool,
    target: int,
    tolerance: int,
    max_iterations: int,
    label: str = "",
) -> tuple[list[int], float]:
    """Binary search on min_distance to hit target structure count."""
    prefix = f"[{label}] " if label else ""
    lo, hi = 0.0, None

    dist = 0.01
    while True:
        count = fps_count(descriptors, structures, mean_descriptor, dist)
        logger.info(f"  {prefix}Probing min_distance={dist:.6f} -> {count} structures")
        if count <= target:
            hi = dist
            break
        dist *= 2

    if hi == 0.0:
        indices = fps_run(descriptors, structures, mean_descriptor, 0.0)
        return indices, 0.0

    best_dist = hi
    best_indices = fps_run(descriptors, structures, mean_descriptor, hi)

    for i in range(max_iterations):
        mid = (lo + hi) / 2
        count = fps_count(descriptors, structures, mean_descriptor, mid)
        logger.info(
            f"  {prefix}Iteration {i+1}: min_distance={mid:.6f} -> {count} structures "
            f"(target={target}+/-{tolerance})"
        )

        if abs(count - target) <= tolerance:
            best_dist = mid
            best_indices = fps_run(descriptors, structures, mean_descriptor, mid)
            break

        if count > target:
            lo = mid
        else:
            hi = mid
            best_dist = mid
            best_indices = fps_run(descriptors, structures, mean_descriptor, mid)
    else:
        logger.info(
            f"  {prefix}Search did not converge within tolerance; "
            f"using min_distance={best_dist:.6f}"
        )

    return best_indices, best_dist


def cross_distance_stats(
    descriptors: np.ndarray,
    indices_a: list[int],
    indices_b: list[int],
) -> tuple[float, float]:
    """Return (min, mean) nearest-neighbour distance from B to A."""
    from scipy.spatial.distance import cdist

    if not indices_a or not indices_b:
        return float("inf"), float("inf")
    d = cdist(descriptors[indices_b], descriptors[indices_a])
    nn_dists = d.min(axis=1)
    return float(nn_dists.min()), float(nn_dists.mean())
