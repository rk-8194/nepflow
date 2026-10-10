"""Focused regressions for the exact indexed information-entropy backend."""

import numpy as np
import pytest

from nepflow.stages.selection.algorithms.information_entropy.bandwidth import (
    build_entropy_pool,
    calculate_frozen_bandwidths,
)
from nepflow.stages.selection.algorithms.information_entropy.kernels import (
    build_sparse_atomic_kernel_graph,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    DEFAULT_NEIGHBOUR_BACKEND_ID,
    INDEXED_NEIGHBOUR_BACKEND_ID,
    EntropyBandwidthSettings,
)
from nepflow.stages.selection.algorithms.information_entropy.neighbours import (
    IndexedCPUNeighbourIndex,
    compute_exact_neighbours,
    compute_neighbours,
)


def test_indexed_neighbours_match_reference_for_unsorted_duplicates_and_ties() -> None:
    descriptors = np.asarray(
        [[2.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]],
        dtype=np.float64,
    )
    reference = compute_exact_neighbours(descriptors, 2, chunk_size=1)
    indexed = compute_neighbours(descriptors, 2, backend=INDEXED_NEIGHBOUR_BACKEND_ID)

    np.testing.assert_array_equal(indexed.neighbour_indices, reference.neighbour_indices)
    np.testing.assert_array_equal(indexed.neighbour_distances, reference.neighbour_distances)
    np.testing.assert_array_equal(indexed.radii, reference.radii)
    np.testing.assert_array_equal(indexed.row_to_location, reference.row_to_location)
    assert indexed.backend == INDEXED_NEIGHBOUR_BACKEND_ID
    assert indexed.backend_version
    assert indexed.backend_fingerprint


def test_indexed_radius_support_expands_duplicates_and_excludes_boundary() -> None:
    descriptors = np.asarray([[0.0], [1.0], [1.0], [2.0]], dtype=np.float64)
    index = IndexedCPUNeighbourIndex(descriptors)

    targets, distances = index.radius_support(0, 2.0)

    np.testing.assert_array_equal(targets, [0, 1, 2])
    np.testing.assert_allclose(distances, [0.0, 1.0, 1.0])
    assert 3 not in targets


def test_backend_dispatch_rejects_unknown_backend_without_reference_fallback() -> None:
    descriptors = np.asarray([[0.0], [1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="unavailable"):
        compute_neighbours(descriptors, 1, backend="not-a-backend")


def test_indexed_backend_flows_through_bandwidth_and_sparse_graph() -> None:
    descriptors = np.asarray([[0.0], [0.7], [1.8], [4.0], [8.0]], dtype=np.float64)
    owners = ["a", "a", "b", "b", "c"]
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=owners,
        candidate_ids=["a", "b", "c"],
    )
    indexed = calculate_frozen_bandwidths(
        pool,
        1,
        2.0,
        backend=INDEXED_NEIGHBOUR_BACKEND_ID,
    )
    reference = calculate_frozen_bandwidths(pool, 1, 2.0, backend="exact_cpu")

    assert indexed.backend == INDEXED_NEIGHBOUR_BACKEND_ID
    assert indexed.backend_version
    assert indexed.backend_fingerprint
    np.testing.assert_array_equal(indexed.radii, reference.radii)
    np.testing.assert_array_equal(indexed.bandwidths, reference.bandwidths)
    indexed_graph = build_sparse_atomic_kernel_graph(pool, indexed)
    reference_graph = build_sparse_atomic_kernel_graph(pool, reference)
    np.testing.assert_array_equal(indexed_graph.target_indices, reference_graph.target_indices)
    np.testing.assert_allclose(indexed_graph.values, reference_graph.values)


def test_entropy_bandwidth_settings_default_to_indexed_backend() -> None:
    assert EntropyBandwidthSettings().backend == DEFAULT_NEIGHBOUR_BACKEND_ID
