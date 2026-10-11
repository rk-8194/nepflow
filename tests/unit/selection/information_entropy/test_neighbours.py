"""Exact distinct-location neighbour reference tests."""

import logging

import numpy as np
import pytest

from nepflow.resources.budget import build_resource_budget
from nepflow.stages.selection.algorithms.information_entropy.neighbours import (
    IndexedCPUNeighbourIndex,
    compute_exact_neighbours,
    compute_indexed_cpu_neighbours,
)


def test_distinct_location_ties_use_smallest_original_representative() -> None:
    descriptors = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )

    result = compute_exact_neighbours(descriptors, 2, chunk_size=1)

    assert result.neighbour_indices[0].tolist() == [1, 2]
    np.testing.assert_array_equal(result.neighbour_indices[3], result.neighbour_indices[0])
    np.testing.assert_allclose(result.radii, [1.0, 2.0, 2.0, 1.0])
    assert result.row_to_location[0] == result.row_to_location[3]
    assert result.unique_locations.shape == (3, 2)


def test_chunk_boundaries_do_not_change_exact_answers() -> None:
    descriptors = np.asarray(
        [[0.0], [0.5], [1.0], [2.0], [3.0]],
        dtype=np.float64,
    )

    first = compute_exact_neighbours(descriptors, 2, chunk_size=1)
    second = compute_exact_neighbours(descriptors, 2, chunk_size=64)

    np.testing.assert_array_equal(first.neighbour_indices, second.neighbour_indices)
    np.testing.assert_array_equal(first.neighbour_distances, second.neighbour_distances)
    np.testing.assert_array_equal(first.radii, second.radii)


@pytest.mark.parametrize(
    "descriptors",
    [
        np.empty((0, 2), dtype=np.float64),
        np.empty((2, 0), dtype=np.float64),
        np.asarray([[0.0, np.nan]], dtype=np.float64),
        np.asarray([[0.0, np.inf]], dtype=np.float64),
    ],
)
def test_invalid_descriptor_pools_fail_explicitly(descriptors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        compute_exact_neighbours(descriptors, 1)


@pytest.mark.parametrize("bad_k", [0, -1, True, 1.5])
def test_invalid_k_fails_explicitly(bad_k: object) -> None:
    descriptors = np.asarray([[0.0], [1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="k"):
        compute_exact_neighbours(descriptors, bad_k)  # type: ignore[arg-type]


def test_all_coincident_or_insufficient_locations_fail() -> None:
    coincident = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="distinct"):
        compute_exact_neighbours(coincident, 1)

    insufficient = np.asarray([[0.0], [1.0], [0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="distinct"):
        compute_exact_neighbours(insufficient, 2)


def test_neighbour_progress_reports_each_small_completed_query(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger="nepflow.stages.selection.algorithms.information_entropy.neighbours",
    )
    descriptors = np.arange(5, dtype=np.float64).reshape(-1, 1)

    compute_exact_neighbours(descriptors, 1)

    progress = [
        record.getMessage()
        for record in caplog.records
        if "Exact neighbour search progress:" in record.getMessage()
    ]
    assert len(progress) == len(descriptors)
    assert all(f"{index}/5" in message for index, message in enumerate(progress, 1))
    assert "100.0%" in progress[-1]
    assert all("estimated remaining=" in message for message in progress)


def test_neighbour_progress_reaches_one_percent_thresholds_for_large_pool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger="nepflow.stages.selection.algorithms.information_entropy.neighbours",
    )
    descriptors = np.arange(101, dtype=np.float64).reshape(-1, 1)

    compute_exact_neighbours(descriptors, 1)

    progress = [
        record.getMessage()
        for record in caplog.records
        if "Exact neighbour search progress:" in record.getMessage()
    ]
    assert len(progress) == 100
    completed = [int(message.split(": ", 1)[1].split("/", 1)[0]) for message in progress]
    assert completed == sorted(set(completed))
    assert completed[-1] == len(descriptors)
    assert "Exact neighbour search started: N=101, distinct locations=101, k=1" in caplog.text
    assert "Exact neighbour search completed: 101/101 (100.0%)" in caplog.text


def _indexed_against_exhaustive(
    descriptors: np.ndarray,
    *,
    k: int,
    chunk_size: int,
    worker_count: int,
) -> tuple[object, dict[str, object]]:
    index = IndexedCPUNeighbourIndex(
        descriptors,
        resource_budget=build_resource_budget(worker_cap=worker_count),
    )
    try:
        actual = compute_indexed_cpu_neighbours(
            descriptors,
            k,
            index=index,
            chunk_size=chunk_size,
        )
        metrics = index.memory_metrics()
        return actual, metrics
    finally:
        index.close()


@pytest.mark.parametrize("chunk_size", [1, 3, 32])
@pytest.mark.parametrize("worker_count", [1, 2])
def test_indexed_certification_preserves_exhaustive_42d_oracle(
    chunk_size: int,
    worker_count: int,
) -> None:
    rng = np.random.default_rng(20261011)
    descriptors = rng.normal(size=(37, 42)).astype(np.float64)
    descriptors[11] = descriptors[3]
    descriptors[24] = descriptors[3]

    expected = compute_exact_neighbours(descriptors, 8, chunk_size=1)
    actual, metrics = _indexed_against_exhaustive(
        descriptors,
        k=8,
        chunk_size=chunk_size,
        worker_count=worker_count,
    )

    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.distances, expected.distances)
    np.testing.assert_array_equal(actual.radii, expected.radii)
    assert actual.representation_fingerprint == expected.representation_fingerprint
    assert actual.row_ids == expected.row_ids
    assert int(metrics["knn_fast_path_certified_count"]) > 0


@pytest.mark.parametrize("dimension", [1, 7, 42, 128])
def test_fp64_certification_bound_is_stable_and_positive(dimension: int) -> None:
    query = np.zeros(dimension, dtype=np.float64)
    target = np.ones(dimension, dtype=np.float64)

    bound = IndexedCPUNeighbourIndex._euclidean_relative_error_bound(
        query,
        target,
        dimension=dimension,
    )

    assert bound is not None
    assert np.isfinite(bound)
    assert bound > 0.0
    expected_operation_count = 3 * dimension + 8
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    numerator = expected_operation_count * unit_roundoff
    expected = numerator / (1.0 - numerator)
    assert bound >= expected
    if dimension == 42:
        assert 1.0e-14 < bound < 2.0e-14
        assert bound != np.finfo(np.float64).tiny


def test_certification_gap_guard_is_scale_aware_and_fail_closed() -> None:
    query = np.zeros(42, dtype=np.float64)
    kth_target = np.zeros(42, dtype=np.float64)
    kth_target[0] = 1.0

    one_ulp_target = kth_target.copy()
    one_ulp_target[0] = np.nextafter(1.0, 2.0)
    assert not IndexedCPUNeighbourIndex._certified_knn_boundary(
        query,
        kth_target,
        one_ulp_target,
        kth_target,
        one_ulp_target,
        1.0,
        float(one_ulp_target[0]),
        1.0,
        float(one_ulp_target[0]),
        dimension=42,
    )

    separated_target = kth_target.copy()
    separated_target[0] = 1.0 + 1.0e-10
    assert IndexedCPUNeighbourIndex._certified_knn_boundary(
        query,
        kth_target,
        separated_target,
        kth_target,
        separated_target,
        1.0,
        float(separated_target[0]),
        1.0,
        float(separated_target[0]),
        dimension=42,
    )

    scale = 1.0e100
    scaled_kth_target = np.zeros(42, dtype=np.float64)
    scaled_kth_target[0] = scale
    scaled_near_target = scaled_kth_target.copy()
    scaled_near_target[0] = np.nextafter(scale, np.inf)
    assert not IndexedCPUNeighbourIndex._certified_knn_boundary(
        query,
        scaled_kth_target,
        scaled_near_target,
        scaled_kth_target,
        scaled_near_target,
        scale,
        float(scaled_near_target[0]),
        scale,
        float(scaled_near_target[0]),
        dimension=42,
    )


def test_indexed_certification_skips_radius_queries_for_a_separated_cloud() -> None:
    positions = np.cumsum(
        np.random.default_rng(20261011).uniform(1.0, 10.0, size=40), dtype=np.float64
    )
    descriptors = positions.reshape(-1, 1)

    actual, metrics = _indexed_against_exhaustive(
        descriptors,
        k=3,
        chunk_size=8,
        worker_count=1,
    )
    expected = compute_exact_neighbours(descriptors, 3, chunk_size=1)

    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.distances, expected.distances)
    assert metrics["knn_fast_path_certified_count"] == len(descriptors)
    assert metrics["knn_fast_path_ambiguous_count"] == 0
    assert metrics["radius_query_preflights"] == 0
    assert metrics["radius_queries"] == 0


def test_indexed_certification_fingerprint_is_chunk_and_worker_invariant() -> None:
    rng = np.random.default_rng(20261011)
    descriptors = rng.normal(size=(19, 42)).astype(np.float64)
    first, _ = _indexed_against_exhaustive(
        descriptors,
        k=4,
        chunk_size=1,
        worker_count=1,
    )
    second, _ = _indexed_against_exhaustive(
        descriptors,
        k=4,
        chunk_size=8,
        worker_count=2,
    )

    assert first.fingerprint == second.fingerprint
    assert first.representation_fingerprint == second.representation_fingerprint
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.distances, second.distances)
    np.testing.assert_array_equal(first.radii, second.radii)


def test_indexed_certification_falls_back_for_equal_and_ulp_boundaries() -> None:
    ulp = np.nextafter(np.float64(1.0), np.float64(2.0))
    descriptors = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [ulp, 0.0],
        ],
        dtype=np.float64,
    )
    expected = compute_exact_neighbours(descriptors, 2, chunk_size=1)
    actual, metrics = _indexed_against_exhaustive(
        descriptors,
        k=2,
        chunk_size=1,
        worker_count=1,
    )

    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.distances, expected.distances)
    np.testing.assert_array_equal(actual.radii, expected.radii)
    assert actual.representation_fingerprint == expected.representation_fingerprint
    assert actual.row_ids == expected.row_ids
    assert metrics["knn_fast_path_ambiguous_count"] > 0
    assert metrics["knn_fast_path_fallback_count"] > 0
    assert metrics["radius_query_preflights"] > 0


def test_indexed_certification_handles_duplicates_and_symmetric_cloud() -> None:
    descriptors = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float64,
    )
    expected = compute_exact_neighbours(descriptors, 3, chunk_size=1)
    actual, _metrics = _indexed_against_exhaustive(
        descriptors,
        k=3,
        chunk_size=2,
        worker_count=2,
    )

    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.distances, expected.distances)
    np.testing.assert_array_equal(actual.row_to_location, expected.row_to_location)
    assert actual.indices[0].tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "descriptors",
    [
        np.asarray([[0.0], [np.nextafter(0.0, 1.0)], [1.0]], dtype=np.float64),
        np.asarray([[0.0], [1.0e200], [-1.0e200]], dtype=np.float64),
    ],
)
def test_indexed_certification_fails_explicitly_on_distance_underflow_or_overflow(
    descriptors: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        compute_exact_neighbours(descriptors, 1, chunk_size=1)
    with pytest.raises(ValueError):
        _indexed_against_exhaustive(
            descriptors,
            k=1,
            chunk_size=1,
            worker_count=1,
        )


def test_indexed_certification_preserves_wide_radius_k() -> None:
    rng = np.random.default_rng(7)
    descriptors = rng.normal(scale=0.05, size=(64, 42)).astype(np.float64)
    expected = compute_exact_neighbours(descriptors, 8, chunk_size=1)
    actual, metrics = _indexed_against_exhaustive(
        descriptors,
        k=8,
        chunk_size=5,
        worker_count=2,
    )

    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.distances, expected.distances)
    np.testing.assert_array_equal(actual.radii, expected.radii)
    assert metrics["query_workspace_peak_bytes"] > 0
