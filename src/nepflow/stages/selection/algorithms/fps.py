"""Farthest-point-sampling selection behind the common algorithm boundary."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Mapping

from ..sampling import (
    build_composition_aware_candidate_set,
    composition_aware_attempt_schedule,
    composition_projection_bins,
    select_best_sampling_attempt,
)
from .base import SelectionAlgorithmRequest, SelectionAlgorithmResult

logger = logging.getLogger(__name__)


def _option(options: Mapping[str, Any], name: str) -> Any:
    try:
        return options[name]
    except KeyError as exc:
        raise ValueError(f"FPS option {name!r} is required") from exc


def _select_plain(
    request: SelectionAlgorithmRequest,
    remaining_indices: list[int],
    remaining_target: int,
) -> tuple[list[int], float]:
    remaining_representations = request.representations[remaining_indices]
    remaining_structures = [request.candidate_structures[index] for index in remaining_indices]
    selector = _option(request.options, "fps_target_selector")
    fps_indices, minimum = selector(
        remaining_representations,
        remaining_structures,
        _option(request.options, "descriptor_type") == "structure",
        remaining_target,
        _option(request.options, "target_tolerance"),
        _option(request.options, "max_search_iterations"),
        label="train",
    )
    return [int(remaining_indices[index]) for index in fps_indices], minimum


def _select_composition_aware(
    request: SelectionAlgorithmRequest,
    remaining_indices: list[int],
    remaining_target: int,
) -> tuple[list[int], float]:
    ase_structures = list(request.options.get("ase_structures", ()))
    anchor_indices = list(request.anchor_indices)
    if not ase_structures:
        logger.warning(
            "  Composition-aware FPS requested, but ASE structures were not provided; "
            "falling back to descriptor-only FPS"
        )
        selected, minimum = _select_plain(request, remaining_indices, remaining_target)
        return sorted(anchor_indices + selected), minimum

    candidate_bins = {
        index: composition_projection_bins(ase_structures[index])
        for index in range(len(ase_structures))
    }
    total_binary_bins = len(
        {
            bin_key
            for index in remaining_indices + anchor_indices
            for bin_key in candidate_bins[index]["binary"]
        }
    )
    total_ternary_bins = len(
        {
            bin_key
            for index in remaining_indices + anchor_indices
            for bin_key in candidate_bins[index]["ternary"]
        }
    )
    if total_binary_bins == 0 and total_ternary_bins == 0:
        logger.info(
            "  No binary or ternary composition projections found; "
            "falling back to descriptor-only FPS"
        )
        selected, minimum = _select_plain(request, remaining_indices, remaining_target)
        return sorted(anchor_indices + selected), minimum

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
        _option(request.options, "composition_aware_fps_frontier_fraction"),
        _option(request.options, "composition_aware_fps_ternary_weight"),
        _option(request.options, "composition_aware_fps_adaptive_retries"),
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
            request.representations,
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
        _option(request.options, "composition_aware_fps_descriptor_floor_fraction"),
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


class FPSSelectionAlgorithm:
    """Canonical FPS implementation of the common selection contract."""

    algorithm_id = "fps"
    algorithm_version = "fps-v1"

    def select(self, request: SelectionAlgorithmRequest) -> SelectionAlgorithmResult:
        if request.algorithm_id != self.algorithm_id:
            raise ValueError(
                f"Request algorithm {request.algorithm_id!r} does not match FPS algorithm"
            )

        anchor_indices = sorted(set(request.anchor_indices))
        if len(anchor_indices) > request.target_count:
            raise ValueError(
                "Preselected anchor count exceeds target_train_count: "
                f"unique anchors={len(anchor_indices)}, target_train_count={request.target_count}"
            )

        if request.target_count >= len(request.candidate_structures):
            selected = list(range(len(request.candidate_structures)))
            minimum = 0.0
        else:
            eligible_indices = request.options.get("eligible_indices")
            if eligible_indices is None:
                eligible_indices = range(len(request.candidate_structures))
            remaining_indices = [index for index in eligible_indices if index not in anchor_indices]
            remaining_target = request.target_count - len(anchor_indices)
            if remaining_target == 0:
                selected, minimum = anchor_indices, 0.0
            elif _option(request.options, "composition_aware_fps"):
                selected, minimum = _select_composition_aware(
                    request,
                    remaining_indices,
                    remaining_target,
                )
            else:
                selected, minimum = _select_plain(request, remaining_indices, remaining_target)
                selected = sorted(anchor_indices + selected)

        selected_ids = tuple(request.candidate_ids[index] for index in selected)
        return SelectionAlgorithmResult(
            algorithm_id=self.algorithm_id,
            algorithm_version=self.algorithm_version,
            selected_indices=tuple(selected),
            selected_candidate_ids=selected_ids,
            minimum_distance=float(minimum),
        )


__all__ = ["FPSSelectionAlgorithm"]
