"""Composition-aware candidate construction and attempt scoring."""

from __future__ import annotations

from collections import Counter
import logging
import math
from typing import Any

import numpy as np


logger = logging.getLogger("nepflow.selection.sampling")


def _sampling_module() -> Any:
    """Load the FPS/metric module at call time to keep this boundary acyclic."""

    from . import sampling

    return sampling


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

        if _sampling_attempt_is_better(attempt, attempt_score, best_attempt, best_score):
            best_attempt = attempt
            best_score = attempt_score
            logger.info(
                "    Attempt %d became current best",
                attempt["attempt_number"],
            )

    return best_attempt


def _sampling_attempt_is_better(
    attempt: dict,
    attempt_score: float,
    best_attempt: dict,
    best_score: float,
) -> bool:
    if attempt_score > best_score + 1e-12:
        return True
    if not math.isclose(attempt_score, best_score, rel_tol=0.0, abs_tol=1e-12):
        return False
    if attempt["train_min_dist"] > best_attempt["train_min_dist"] + 1e-12:
        return True
    if not math.isclose(
        attempt["train_min_dist"],
        best_attempt["train_min_dist"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    if attempt["train_mean_nn_dist"] > best_attempt["train_mean_nn_dist"] + 1e-12:
        return True
    if not math.isclose(
        attempt["train_mean_nn_dist"],
        best_attempt["train_mean_nn_dist"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    return attempt["attempt_number"] < best_attempt["attempt_number"]


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

    sampling = _sampling_module()
    selected_indices = list(anchor_indices)
    remaining_pool = list(remaining_indices)
    binary_counts: Counter = Counter()
    ternary_counts: Counter = Counter()
    binary_subset_totals: Counter = Counter()
    ternary_subset_totals: Counter = Counter()
    for index in anchor_indices:
        _update_composition_counts(
            candidate_bins[index],
            binary_counts,
            ternary_counts,
            binary_subset_totals,
            ternary_subset_totals,
        )

    logger.info(
        "    Initializing descriptor distances for %d candidates against %d anchors",
        len(remaining_pool),
        len(selected_indices),
    )
    current_nearest = sampling.initialize_nearest_distances(
        representations,
        remaining_pool,
        selected_indices,
    )
    logger.info("    Initial descriptor-distance initialization complete")

    for iteration in range(remaining_target):
        if not remaining_pool:
            break
        best_index = _select_composition_candidate(
            remaining_pool,
            current_nearest,
            candidate_bins,
            binary_counts,
            ternary_counts,
            binary_subset_totals,
            ternary_subset_totals,
            frontier_fraction,
            ternary_weight,
        )
        selected_indices.append(best_index)
        _update_composition_counts(
            candidate_bins[best_index],
            binary_counts,
            ternary_counts,
            binary_subset_totals,
            ternary_subset_totals,
        )
        remaining_pool.remove(best_index)
        current_nearest.pop(best_index, None)
        sampling.update_nearest_distances(
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
    coverage = sampling.calculate_composition_coverage_metrics(
        selected_indices,
        candidate_bins,
    )
    coverage.update(
        {
            "selected_indices": selected_indices,
            "train_min_dist": sampling.calculate_min_distance(
                representations,
                selected_indices,
            ),
            "train_positive_min_dist": sampling.calculate_positive_min_distance(
                representations,
                selected_indices,
            ),
            "train_mean_nn_dist": sampling.calculate_mean_nearest_distance(
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


def _select_composition_candidate(
    remaining_pool: list[int],
    current_nearest: dict[int, float],
    candidate_bins: dict[int, dict[str, list[tuple]]],
    binary_counts: Counter,
    ternary_counts: Counter,
    binary_subset_totals: Counter,
    ternary_subset_totals: Counter,
    frontier_fraction: float,
    ternary_weight: float,
) -> int:
    sampling = _sampling_module()
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
        novelty_floor = (
            sampling.COMPOSITION_AWARE_NOVELTY_FLOOR_FRACTION
            * best_frontier_novelty
        )
        novelty_frontier = [
            index for index in frontier if current_nearest[index] >= novelty_floor
        ]
        if novelty_frontier:
            frontier = novelty_frontier
    composition_reward = {
        index: sampling.composition_sparsity_reward(
            candidate_bins[index],
            binary_counts,
            ternary_counts,
            binary_subset_totals,
            ternary_subset_totals,
            ternary_weight,
        )
        for index in frontier
    }
    return min(
        frontier,
        key=lambda index: (
            -composition_reward[index],
            -current_nearest[index],
            index,
        ),
    )


def _update_composition_counts(
    bins: dict[str, list[tuple]],
    binary_counts: Counter,
    ternary_counts: Counter,
    binary_subset_totals: Counter,
    ternary_subset_totals: Counter,
) -> None:
    binary_counts.update(bins["binary"])
    ternary_counts.update(bins["ternary"])
    binary_subset_totals.update(bin_key[:2] for bin_key in bins["binary"])
    ternary_subset_totals.update(bin_key[:3] for bin_key in bins["ternary"])


__all__ = [
    "build_composition_aware_candidate_set",
    "composition_aware_attempt_score",
    "select_best_sampling_attempt",
]
