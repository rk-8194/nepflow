"""Canonical selection representation and sampling services."""

from .representations import (
    compute_descriptors_batched,
    compute_structure_descriptors,
    descriptor_cache_path,
    descriptor_manifest_path,
    load_or_calculate_representations,
)
from .sampling import (
    calculate_cross_distance_stats,
    select_farthest_points,
    select_farthest_points_for_target,
)

__all__ = [
    "calculate_cross_distance_stats",
    "compute_descriptors_batched",
    "compute_structure_descriptors",
    "descriptor_cache_path",
    "descriptor_manifest_path",
    "load_or_calculate_representations",
    "select_farthest_points",
    "select_farthest_points_for_target",
]
