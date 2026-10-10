"""Memory-safety regressions for exact entropy calibration."""

import numpy as np
import pytest

from nepflow.stages.selection.algorithms.information_entropy.bandwidth import (
    BandwidthCalibrationCapacityError,
    build_entropy_pool,
    calibrate_bandwidth,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    EntropyBandwidthSettings,
)
from nepflow.stages.selection.algorithms.information_entropy.neighbours import (
    IndexedCPUNeighbourIndex,
)


def test_radius_support_is_ephemeral_across_repeated_queries() -> None:
    descriptors = np.asarray([[0.0], [1.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    index = IndexedCPUNeighbourIndex(descriptors)

    first_targets, first_distances = index.radius_support(0, 2.0)
    second_targets, second_distances = index.radius_support(1, 3.0)

    np.testing.assert_array_equal(first_targets, [0, 1, 2])
    np.testing.assert_allclose(first_distances, [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(second_targets, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(second_distances, [1.0, 0.0, 0.0, 1.0, 2.0])
    assert not hasattr(index, "_support_cache")
    assert index.memory_metrics()["support_cache_bytes"] == 0
    assert index.memory_metrics()["support_cache_peak_bytes"] == 0


def test_calibration_capacity_failure_is_not_an_invalid_pair() -> None:
    descriptors = np.arange(5, dtype=np.float64).reshape(-1, 1)
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=["a", "b", "c", "d", "e"],
        candidate_ids=["a", "b", "c", "d", "e"],
    )
    settings = EntropyBandwidthSettings(
        mode="automatic",
        k_candidates=(1,),
        c_candidates=(8.0,),
        max_radius_query_bytes=3500,
    )

    with pytest.raises(BandwidthCalibrationCapacityError, match=r"N=5.*d'=1.*k=1, c=8"):
        calibrate_bandwidth(pool, settings)


def test_neighbour_cache_has_a_deterministic_byte_budget() -> None:
    descriptors = np.arange(12, dtype=np.float64).reshape(-1, 1)
    index = IndexedCPUNeighbourIndex(descriptors, max_neighbour_cache_bytes=1)

    index.query_neighbours(1)

    assert index.neighbour_cache_bytes == 0
    assert index.memory_metrics()["support_cache_bytes"] == 0
