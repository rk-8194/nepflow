"""Operational entropy timing counters must remain reference-consistent."""

import numpy as np

from nepflow.stages.selection.algorithms.information_entropy.bandwidth import (
    build_entropy_pool,
)
from nepflow.stages.selection.algorithms.information_entropy.kernels import (
    build_streamed_candidate_contributions,
)
from nepflow.stages.selection.algorithms.information_entropy.models import FrozenBandwidths
from nepflow.stages.selection.algorithms.information_entropy.neighbours import (
    IndexedCPUNeighbourIndex,
    compute_exact_neighbours,
    compute_indexed_cpu_neighbours_many,
)


def test_indexed_phase_counters_cover_exact_query_and_checkpoint(tmp_path) -> None:
    descriptors = np.asarray(
        [[0.0], [1.0], [2.0], [0.0], [3.0], [4.0]],
        dtype=np.float64,
    )
    index = IndexedCPUNeighbourIndex(descriptors)
    indexed = compute_indexed_cpu_neighbours_many(
        descriptors,
        (1, 2),
        index=index,
        chunk_size=2,
        checkpoint_path=tmp_path,
        checkpoint_every_locations=1,
        resume=False,
    )
    reference = compute_exact_neighbours(descriptors, 2, chunk_size=1)

    np.testing.assert_array_equal(indexed[2].indices, reference.indices)
    np.testing.assert_array_equal(indexed[2].radii, reference.radii)
    metrics = index.memory_metrics()
    assert metrics["knn_tree_query_seconds"] >= 0.0
    assert metrics["radius_preflight_count"] >= len(index.unique_locations)
    assert metrics["radius_retrieval_count"] >= len(index.unique_locations)
    assert metrics["canonical_distance_verification_count"] >= len(index.unique_locations)
    assert metrics["checkpoint_serialization_count"] == len(index.unique_locations) // 2 + (
        1 if len(index.unique_locations) % 2 else 0
    )
    assert metrics["checkpoint_serialization_bytes"] > 0
    assert metrics["query_neighbours_many_projection_count"] == 2


def test_streamed_summary_exposes_support_density_and_pmf_operations() -> None:
    descriptors = np.asarray([[0.0], [1.0], [2.0], [0.0]], dtype=np.float64)
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=["a", "a", "b", "b"],
        candidate_ids=["a", "b"],
    )
    frozen = FrozenBandwidths(
        radii=np.ones(len(descriptors), dtype=np.float64),
        bandwidths=np.full(len(descriptors), 2.5, dtype=np.float64),
        k=1,
        c=1.0,
        neighbour_fingerprint="reference-neighbours",
        pool_fingerprint=pool.fingerprint,
        transform_fingerprint=pool.transform_fingerprint,
        fingerprint="reference-bandwidths",
    )
    contributions, summary = build_streamed_candidate_contributions(pool, frozen)

    assert contributions.entry_count == summary.candidate_entry_count
    operations = summary.operational_metrics
    assert operations["source_count"] == len(descriptors)
    assert operations["support_query_count"] == len(descriptors)
    assert operations["source_support_density_mean"] > 0.0
    assert operations["candidate_support_density_mean"] > 0.0
    assert operations["phase_seconds"]["candidate_reduction"] >= 0.0
    assert operations["process_peak_rss_bytes"] is None or operations[
        "process_peak_rss_bytes"
    ] >= 0
