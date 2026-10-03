"""Canonical selection stage and supporting services."""

from .representations import (
    compute_descriptors_batched,
    compute_structure_descriptors,
    descriptor_cache_path,
    descriptor_manifest_path,
    load_or_calculate_representations,
)
from .sampling import (
    binary_bin_index,
    build_composition_aware_candidate_set,
    calculate_composition_coverage_metrics,
    calculate_cross_distance_stats,
    calculate_mean_nearest_distance,
    calculate_min_distance,
    calculate_positive_min_distance,
    composition_aware_attempt_schedule,
    composition_aware_attempt_score,
    composition_projection_bins,
    descriptor_distance,
    select_farthest_points,
    select_farthest_points_for_target,
)
from .models import SelectionResult
from .stage import SelectionStage
from .strategy import (
    select_composition_aware_training_set,
    select_test_set,
    select_training_set,
)

__all__ = [
    "calculate_cross_distance_stats",
    "binary_bin_index",
    "build_composition_aware_candidate_set",
    "calculate_composition_coverage_metrics",
    "composition_aware_attempt_schedule",
    "composition_aware_attempt_score",
    "composition_projection_bins",
    "compute_descriptors_batched",
    "compute_structure_descriptors",
    "descriptor_distance",
    "descriptor_cache_path",
    "descriptor_manifest_path",
    "load_or_calculate_representations",
    "calculate_mean_nearest_distance",
    "calculate_min_distance",
    "calculate_positive_min_distance",
    "select_composition_aware_training_set",
    "select_farthest_points",
    "select_farthest_points_for_target",
    "select_test_set",
    "select_training_set",
    "SelectionResult",
    "SelectionStage",
]
