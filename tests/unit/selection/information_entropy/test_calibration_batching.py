"""Focused regressions for CSR grouping and bounded calibration batches."""

import numpy as np
import pytest

from nepflow.stages.selection.algorithms.information_entropy.bandwidth import (
    build_entropy_pool,
    calculate_frozen_bandwidths,
    calibrate_bandwidth,
)
from nepflow.stages.selection.algorithms.information_entropy.kernels import (
    evaluate_leave_one_out_objective,
    evaluate_leave_one_out_objectives,
)
from nepflow.stages.selection.algorithms.information_entropy.models import EntropyBandwidthSettings
from nepflow.stages.selection.algorithms.information_entropy.neighbours import (
    IndexedCPUNeighbourIndex,
    RadiusQueryCapacityError,
    compute_neighbours,
    compute_radius_support,
)


def test_location_grouping_is_compact_stable_and_complete() -> None:
    descriptors = np.asarray(
        [[2.0], [0.0], [1.0], [0.0], [2.0], [3.0]],
        dtype=np.float64,
    )
    index = IndexedCPUNeighbourIndex(descriptors)

    np.testing.assert_array_equal(index.rows_by_location, [1, 3, 2, 0, 4, 5])
    np.testing.assert_array_equal(index.location_indptr, [0, 2, 3, 5, 6])
    assert index.rows_by_location.dtype == np.dtype(np.int64)
    assert index.location_indptr.dtype == np.dtype(np.int64)
    assert index.rows_by_location.flags.writeable is False
    assert index.location_indptr.flags.writeable is False
    assert sorted(index.rows_by_location.tolist()) == list(range(len(descriptors)))
    assert index.memory_metrics()["grouping_bytes"] == (
        index.rows_by_location.nbytes + index.location_indptr.nbytes
    )


def test_index_memory_limit_accounts_for_compact_grouping() -> None:
    descriptors = np.arange(6, dtype=np.float64).reshape(-1, 1)
    reference = IndexedCPUNeighbourIndex(descriptors)

    with pytest.raises(ValueError, match="max_index_bytes"):
        IndexedCPUNeighbourIndex(descriptors, max_index_bytes=reference.index_bytes - 1)


def test_indexed_support_matches_exact_reference_for_unsorted_duplicates() -> None:
    descriptors = np.asarray(
        [[2.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]],
        dtype=np.float64,
    )
    indexed = IndexedCPUNeighbourIndex(descriptors)

    indexed_targets, indexed_distances = indexed.radius_support(0, 2.0)
    reference_targets, reference_distances = compute_radius_support(
        descriptors,
        0,
        2.0,
        backend="exact_cpu",
        chunk_size=1,
    )

    np.testing.assert_array_equal(indexed_targets, reference_targets)
    np.testing.assert_array_equal(indexed_distances, reference_distances)


def test_batched_loo_matches_each_single_pair_reference() -> None:
    descriptors = np.asarray(
        [[2.0], [0.0], [1.0], [0.0], [2.0], [3.0]],
        dtype=np.float64,
    )
    probabilities = np.full(len(descriptors), 1.0 / len(descriptors), dtype=np.float64)
    bandwidths = np.asarray(
        [
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        ],
        dtype=np.float64,
    )
    index = IndexedCPUNeighbourIndex(descriptors)

    batched = evaluate_leave_one_out_objectives(
        descriptors,
        probabilities,
        bandwidths,
        backend="exact_indexed_cpu",
        index=index,
    )
    references = tuple(
        evaluate_leave_one_out_objective(
            descriptors,
            probabilities,
            candidate_bandwidths,
            backend="exact_cpu",
        )
        for candidate_bandwidths in bandwidths
    )

    for actual, expected in zip(batched, references, strict=True):
        assert actual.valid == expected.valid
        if expected.objective is None:
            assert actual.objective is None
        else:
            assert actual.objective == pytest.approx(expected.objective)
        np.testing.assert_allclose(actual.probabilities, expected.probabilities)
        assert actual.reason == expected.reason


def test_batched_loo_reduces_repeated_indexed_radius_queries() -> None:
    descriptors = np.arange(8, dtype=np.float64).reshape(-1, 1)
    probabilities = np.full(len(descriptors), 1.0 / len(descriptors), dtype=np.float64)
    bandwidths = np.asarray(
        [
            np.full(len(descriptors), 1.5, dtype=np.float64),
            np.full(len(descriptors), 2.5, dtype=np.float64),
            np.full(len(descriptors), 4.5, dtype=np.float64),
        ],
        dtype=np.float64,
    )

    batched_index = IndexedCPUNeighbourIndex(descriptors)
    evaluate_leave_one_out_objectives(
        descriptors,
        probabilities,
        bandwidths,
        backend="exact_indexed_cpu",
        index=batched_index,
    )
    batched_queries = batched_index.radius_queries

    reference_index = IndexedCPUNeighbourIndex(descriptors)
    for candidate_bandwidths in bandwidths:
        evaluate_leave_one_out_objective(
            descriptors,
            probabilities,
            candidate_bandwidths,
            backend="exact_indexed_cpu",
            index=reference_index,
        )
    reference_queries = reference_index.radius_queries

    assert batched_queries < reference_queries


def test_batched_workspace_capacity_fails_before_support_queries() -> None:
    descriptors = np.arange(4, dtype=np.float64).reshape(-1, 1)
    probabilities = np.full(len(descriptors), 0.25, dtype=np.float64)
    bandwidths = np.ones((2, len(descriptors)), dtype=np.float64)
    index = IndexedCPUNeighbourIndex(descriptors)

    with pytest.raises(RadiusQueryCapacityError, match="batched calibration workspace"):
        evaluate_leave_one_out_objectives(
            descriptors,
            probabilities,
            bandwidths,
            backend="exact_indexed_cpu",
            index=index,
            max_calibration_work_bytes=1,
        )
    assert index.radius_queries == 0


def test_calibration_batch_partition_preserves_attempts_and_winner() -> None:
    descriptors = np.asarray(
        [[0.0], [0.75], [1.5], [3.0], [5.0], [8.0]],
        dtype=np.float64,
    )
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=["a", "a", "b", "c", "c", "d"],
        candidate_ids=["a", "b", "c", "d"],
    )
    common = dict(
        mode="automatic",
        k_candidates=(1, 2),
        c_candidates=(1.5, 2.0, 4.0),
    )
    one_at_a_time = calibrate_bandwidth(
        pool,
        EntropyBandwidthSettings(**common, calibration_batch_size=1),
    )
    partitioned = calibrate_bandwidth(
        pool,
        EntropyBandwidthSettings(**common, calibration_batch_size=2),
    )

    assert [
        (attempt.k, attempt.c, attempt.status, attempt.reason)
        for attempt in one_at_a_time.attempts
    ] == [
        (attempt.k, attempt.c, attempt.status, attempt.reason)
        for attempt in partitioned.attempts
    ]
    for first, second in zip(one_at_a_time.attempts, partitioned.attempts, strict=True):
        assert first.objective == pytest.approx(second.objective)
        assert first.bandwidth_fingerprint == second.bandwidth_fingerprint
    assert one_at_a_time.selected.k == partitioned.selected.k
    assert one_at_a_time.selected.c == partitioned.selected.c
    assert one_at_a_time.objective == pytest.approx(partitioned.objective)
    np.testing.assert_array_equal(
        one_at_a_time.selected.bandwidths,
        partitioned.selected.bandwidths,
    )


def test_scientifically_invalid_c_does_not_invalidate_other_values_in_batch() -> None:
    descriptors = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=["a", "b", "c"],
        candidate_ids=["a", "b", "c"],
    )

    result = calibrate_bandwidth(
        pool,
        EntropyBandwidthSettings(
            mode="automatic",
            k_candidates=(1,),
            c_candidates=(1.0, 2.0),
            calibration_batch_size=2,
        ),
    )

    assert [attempt.status for attempt in result.attempts] == ["invalid", "valid"]
    assert result.attempts[0].reason is not None
    assert result.selected.c == 2.0


def test_indexed_frozen_bandwidths_match_reference_backend_with_duplicates() -> None:
    descriptors = np.asarray(
        [[2.0], [0.0], [1.0], [0.0], [2.0], [3.0]],
        dtype=np.float64,
    )
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=["a", "a", "b", "b", "c", "c"],
        candidate_ids=["a", "b", "c"],
    )
    indexed = calculate_frozen_bandwidths(
        pool,
        1,
        2.0,
        backend="exact_indexed_cpu",
    )
    reference = calculate_frozen_bandwidths(pool, 1, 2.0, backend="exact_cpu")

    np.testing.assert_array_equal(indexed.radii, reference.radii)
    np.testing.assert_array_equal(indexed.bandwidths, reference.bandwidths)


def test_index_grouping_is_used_by_neighbour_queries() -> None:
    descriptors = np.asarray([[2.0], [0.0], [1.0], [0.0], [2.0]], dtype=np.float64)
    index = IndexedCPUNeighbourIndex(descriptors)
    indexed = compute_neighbours(descriptors, 1, backend="exact_indexed_cpu", index=index)

    assert indexed.radii.shape == (len(descriptors),)
    assert index.memory_metrics()["grouping_build_seconds"] >= 0.0
