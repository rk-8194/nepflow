"""Exact streamed candidate-PMF construction tests."""

import numpy as np
import pytest

from nepflow.stages.selection.algorithms.information_entropy.bandwidth import build_entropy_pool
from nepflow.stages.selection.algorithms.information_entropy.kernels import (
    aggregate_candidate_contributions,
    build_sparse_atomic_kernel_graph,
    build_streamed_candidate_contributions,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    EntropyPool,
    FrozenBandwidths,
)


def _pool() -> EntropyPool:
    return build_entropy_pool(
        np.asarray([[0.0], [2.0], [0.0], [5.0], [1.0]], dtype=np.float64),
        row_candidate_ids=["b", "a", "c", "b", "a"],
        candidate_ids=["a", "b", "c"],
    )


def _bandwidths(pool: EntropyPool) -> FrozenBandwidths:
    return FrozenBandwidths(
        radii=np.ones(len(pool.rows), dtype=np.float64),
        bandwidths=np.full(len(pool.rows), 4.0, dtype=np.float64),
        k=1,
        c=1.0,
        neighbour_fingerprint="reference-neighbours",
        pool_fingerprint=pool.fingerprint,
        transform_fingerprint=pool.transform_fingerprint,
        fingerprint="reference-bandwidths",
    )


def _dense(contributions, target_count: int) -> np.ndarray:
    result = np.zeros((len(contributions.candidate_ids), target_count), dtype=np.float64)
    for candidate_index, row in enumerate(contributions.iter_candidates()):
        result[candidate_index, row.target_indices] = row.values
    return result


def test_streamed_candidate_contributions_match_bounded_graph_reference() -> None:
    pool = _pool()
    frozen = _bandwidths(pool)
    graph = build_sparse_atomic_kernel_graph(pool, frozen)
    reference = aggregate_candidate_contributions(graph)
    direct, summary = build_streamed_candidate_contributions(pool, frozen)

    np.testing.assert_array_equal(direct.candidate_indptr, reference.candidate_indptr)
    np.testing.assert_array_equal(direct.target_indices, reference.target_indices)
    np.testing.assert_allclose(direct.values, reference.values, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(_dense(direct, len(pool.rows)), _dense(reference, len(pool.rows)))
    assert summary.atomic_graph_materialized is False
    assert summary.atomic_graph_csr_bytes == 0
    assert summary.E == graph.edge_count
    assert summary.Q == direct.entry_count
    assert summary.source_query_count == len(pool.rows)
    assert summary.kernel_operator_fingerprint == direct.kernel_operator_fingerprint


def test_streamed_candidate_capacity_fails_before_final_csr() -> None:
    pool = _pool()
    with pytest.raises(ValueError, match=r"Q=.*max_entries"):
        build_streamed_candidate_contributions(pool, _bandwidths(pool), max_entries=1)
