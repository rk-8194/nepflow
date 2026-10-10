"""Sparse atomic-kernel graph and candidate-contribution tests."""

from dataclasses import replace

import numpy as np
import pytest

from nepflow.stages.selection.algorithms.information_entropy.bandwidth import build_entropy_pool
from nepflow.stages.selection.algorithms.information_entropy.kernels import (
    aggregate_candidate_contributions,
    build_sparse_atomic_kernel_graph,
    evaluate_leave_one_out_objective,
    normalized_kernel_matrix,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    EntropyPool,
    FrozenBandwidths,
    SparseAtomicKernelGraph,
)


def _pool(descriptors: list[float], owners: list[str], candidates: list[str]) -> EntropyPool:
    return build_entropy_pool(
        np.asarray(descriptors, dtype=np.float64).reshape(-1, 1),
        row_candidate_ids=owners,
        candidate_ids=candidates,
    )


def _bandwidths(pool: EntropyPool, values: list[float]) -> FrozenBandwidths:
    bandwidths = np.asarray(values, dtype=np.float64)
    return FrozenBandwidths(
        radii=np.ones(len(values), dtype=np.float64),
        bandwidths=bandwidths,
        k=1,
        c=1.0,
        neighbour_fingerprint="neighbours",
        pool_fingerprint=pool.fingerprint,
        transform_fingerprint=pool.transform_fingerprint,
        fingerprint="bandwidths",
    )


def _dense_from_graph(graph: SparseAtomicKernelGraph) -> np.ndarray:
    dense = np.zeros((graph.n_targets, graph.n_sources), dtype=np.float64)
    for source_index, row in enumerate(graph.iter_source_rows()):
        dense[row.target_indices, source_index] = row.values
    return dense


def test_sparse_graph_matches_dense_reference_and_preserves_source_orientation() -> None:
    pool = _pool([0.0, 1.0, 3.0, 6.0], ["a", "b", "a", "b"], ["a", "b"])
    frozen = _bandwidths(pool, [2.0, 3.0, 4.0, 1.5])

    graph = build_sparse_atomic_kernel_graph(pool, frozen, chunk_size=1)
    dense = normalized_kernel_matrix(pool.descriptors, frozen.bandwidths, chunk_size=3)

    np.testing.assert_array_equal(_dense_from_graph(graph), dense)
    assert graph.source_bandwidth_orientation == "h_a"
    assert all(
        source_index in row.target_indices
        for source_index, row in enumerate(graph.iter_source_rows())
    )
    np.testing.assert_allclose(np.sum(graph.values), graph.n_sources)
    assert graph.target_indices.flags.writeable is False
    assert graph.values.flags.writeable is False


def test_source_bandwidths_allow_support_beyond_k_and_isolated_self_edges() -> None:
    pool = _pool([0.0, 1.0, 2.0, 10.0], ["a", "a", "b", "b"], ["a", "b"])
    frozen = _bandwidths(pool, [4.0, 4.0, 4.0, 1.0])
    graph = build_sparse_atomic_kernel_graph(pool, frozen)

    assert graph.support_sizes[0] > 2
    isolated = graph.source_row(3)
    np.testing.assert_array_equal(isolated.target_indices, [3])
    np.testing.assert_array_equal(isolated.values, [1.0])


def test_coincident_rows_are_retained_and_candidate_contributions_use_row_multiplicity() -> None:
    pool = _pool([0.0, 0.0, 3.0, 6.0], ["a", "b", "a", "b"], ["a", "b"])
    frozen = _bandwidths(pool, [3.0, 3.0, 3.0, 3.0])
    graph = build_sparse_atomic_kernel_graph(pool, frozen)
    contributions = aggregate_candidate_contributions(graph)
    dense = _dense_from_graph(graph)

    assert graph.n_sources == 4
    assert graph.edge_count >= 4
    assert contributions.candidate_ids == ("a", "b")
    np.testing.assert_array_equal(contributions.candidate_source_counts, [2, 2])
    for candidate_index, candidate_id in enumerate(pool.candidate_ids):
        source_indices = np.flatnonzero(pool.row_candidate_indices == candidate_index)
        expected = np.sum(dense[:, source_indices], axis=1) / len(source_indices)
        actual = np.zeros(graph.n_targets, dtype=np.float64)
        row = contributions.candidate_row(candidate_id)
        actual[row.target_indices] = row.values
        np.testing.assert_allclose(actual, expected)
        assert np.sum(row.values) == pytest.approx(1.0)
        assert np.all(np.diff(row.target_indices) > 0)


def test_candidate_progress_and_graph_progress_do_not_change_results() -> None:
    pool = _pool([0.0, 1.0, 3.0, 6.0], ["a", "b", "a", "b"], ["a", "b"])
    frozen = _bandwidths(pool, [2.0, 3.0, 4.0, 1.5])
    graph_without_callback = build_sparse_atomic_kernel_graph(pool, frozen, chunk_size=1)
    source_progress: list[tuple[int, int]] = []
    graph_with_callback = build_sparse_atomic_kernel_graph(
        pool,
        frozen,
        chunk_size=99,
        progress_callback=lambda completed, total: source_progress.append((completed, total)),
    )
    candidate_progress: list[tuple[int, int]] = []
    contributions_without_callback = aggregate_candidate_contributions(graph_without_callback)
    contributions_with_callback = aggregate_candidate_contributions(
        graph_with_callback,
        progress_callback=lambda completed, total: candidate_progress.append((completed, total)),
    )

    assert source_progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert candidate_progress == [(1, 2), (2, 2)]
    assert graph_without_callback.fingerprint == graph_with_callback.fingerprint
    np.testing.assert_array_equal(
        graph_without_callback.target_indices, graph_with_callback.target_indices
    )
    np.testing.assert_array_equal(graph_without_callback.values, graph_with_callback.values)
    assert contributions_without_callback.fingerprint == contributions_with_callback.fingerprint
    np.testing.assert_array_equal(
        contributions_without_callback.target_indices,
        contributions_with_callback.target_indices,
    )
    np.testing.assert_array_equal(
        contributions_without_callback.values,
        contributions_with_callback.values,
    )


def test_graph_columns_reproduce_reference_loo_probabilities() -> None:
    pool = _pool([0.0, 1.0, 3.0], ["a", "b", "c"], ["a", "b", "c"])
    frozen = _bandwidths(pool, [2.0, 3.0, 2.0])
    graph = build_sparse_atomic_kernel_graph(pool, frozen, chunk_size=1)
    objective = evaluate_leave_one_out_objective(
        pool.descriptors,
        pool.probabilities,
        frozen.bandwidths,
        chunk_size=2,
    )

    numerator = np.zeros(graph.n_targets, dtype=np.float64)
    for source_index, row in enumerate(graph.iter_source_rows()):
        mask = row.target_indices != source_index
        numerator[row.target_indices[mask]] += pool.probabilities[source_index] * row.values[mask]
    reconstructed = numerator / (1.0 - pool.probabilities)

    assert objective.valid
    np.testing.assert_array_equal(reconstructed, objective.probabilities)


def test_budget_is_checked_before_graph_arrays_are_allocated() -> None:
    pool = _pool([0.0, 1.0, 2.0, 3.0], ["a", "b", "a", "b"], ["a", "b"])
    frozen = _bandwidths(pool, [10.0, 10.0, 10.0, 10.0])
    with pytest.raises(ValueError, match=r"required_edges=.*max_edges"):
        build_sparse_atomic_kernel_graph(pool, frozen, max_edges=1)

    graph = build_sparse_atomic_kernel_graph(pool, frozen)
    with pytest.raises(ValueError, match=r"required_entries=.*max_entries"):
        aggregate_candidate_contributions(graph, max_entries=1)


def test_identity_and_csr_validation_reject_stale_or_corrupt_inputs() -> None:
    pool = _pool([0.0, 1.0, 2.0, 3.0], ["a", "b", "a", "b"], ["a", "b"])
    frozen = _bandwidths(pool, [10.0, 10.0, 10.0, 10.0])
    graph = build_sparse_atomic_kernel_graph(pool, frozen)
    stale_pool = replace(pool, transform_fingerprint="different-transform")

    with pytest.raises(ValueError, match="transform"):
        build_sparse_atomic_kernel_graph(stale_pool, frozen)

    bad_targets = graph.target_indices.copy()
    bad_targets[0], bad_targets[1] = bad_targets[1], bad_targets[0]
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(graph, target_indices=bad_targets)

    bad_values = graph.values.copy()
    bad_values[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        replace(graph, values=bad_values)


def test_candidate_aggregation_rejects_pool_mismatch() -> None:
    pool = _pool([0.0, 1.0, 2.0, 3.0], ["a", "b", "a", "b"], ["a", "b"])
    frozen = _bandwidths(pool, [10.0, 10.0, 10.0, 10.0])
    graph = build_sparse_atomic_kernel_graph(pool, frozen)
    other_pool = _pool([0.0, 1.0, 2.0, 4.0], ["a", "b", "a", "b"], ["a", "b"])

    with pytest.raises(ValueError, match="different entropy pool"):
        aggregate_candidate_contributions(graph, other_pool)
