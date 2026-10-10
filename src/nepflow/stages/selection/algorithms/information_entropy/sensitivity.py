"""Opt-in scientific sensitivity comparisons for entropy selection.

None of these helpers is called by production selection.  Every function
requires explicit inputs and returns labelled numerical differences; no
parameter is silently retuned and no result is fed back into the selector.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from nepflow.io.hashing import sha256_canonical_json

from .kernels import normalized_kernel_matrix
from .models import EntropyPool, SparseCandidateContributions
from .neighbours import compute_neighbours
from .selector import full_greedy, lazy_greedy


def _dense_contributions(
    pool: EntropyPool,
    bandwidths: Sequence[float],
) -> tuple[np.ndarray, SparseCandidateContributions]:
    """Build the explicitly bounded dense reference candidate PMFs."""

    matrix = normalized_kernel_matrix(
        pool.descriptors,
        np.asarray(bandwidths, dtype=np.float64),
        max_dense_entries=max(1, pool.descriptors.shape[0] ** 2),
    )
    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    indptr = [0]
    owners = np.asarray(pool.row_candidate_indices, dtype=np.int64)
    for candidate_index in range(len(pool.candidate_ids)):
        source = np.flatnonzero(owners == candidate_index)
        q = np.asarray(np.mean(matrix[:, source], axis=1, dtype=np.float64))
        targets = np.flatnonzero(q > 0.0).astype(np.int64)
        rows.append(targets)
        values.append(np.asarray(q[targets], dtype=np.float64))
        indptr.append(indptr[-1] + len(targets))
    target_indices = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    contribution_values = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    fingerprint = sha256_canonical_json(
        {
            "schema": "dense-reference-candidate-contributions-v1",
            "pool": pool.fingerprint,
            "indptr": indptr,
            "targets": target_indices.tolist(),
            "values": contribution_values.tolist(),
        }
    )
    return matrix, SparseCandidateContributions(
        candidate_indptr=np.asarray(indptr, dtype=np.int64),
        target_indices=target_indices,
        values=contribution_values,
        candidate_ids=pool.candidate_ids,
        graph_fingerprint="dense-reference",
        fingerprint=fingerprint,
        row_count=len(pool.rows),
        candidate_source_counts=np.bincount(owners, minlength=len(pool.candidate_ids)),
        pool_fingerprint=pool.fingerprint,
    )


def compare_beta_sensitivity(
    pool: EntropyPool,
    contributions: SparseCandidateContributions,
    beta_values: Sequence[float],
    *,
    K: int,
    anchors: Sequence[str] = (),
    method: str = "full_greedy",
) -> dict[str, Any]:
    """Compare real selections at explicitly supplied positive beta values."""

    values = tuple(float(value) for value in beta_values)
    if not values or any(value <= 0.0 or not np.isfinite(value) for value in values):
        raise ValueError("beta_values must contain explicit finite positive values")
    optimizer = full_greedy if method == "full_greedy" else lazy_greedy
    if method not in {"full_greedy", "lazy_greedy"}:
        raise ValueError("method must be full_greedy or lazy_greedy")
    runs = []
    for beta in values:
        result = optimizer(pool, contributions, beta=beta, K=K, anchors=anchors)
        runs.append(
            {
                "beta": beta,
                "selected_candidate_ids": list(result.selected_candidate_ids),
                "acquisition_order": list(result.acquisition_order),
                "objective_history": list(result.objective_history),
                "forward_kl_nats": result.final_forward_kl,
                "pool_fingerprint": pool.fingerprint,
                "contributions_fingerprint": contributions.fingerprint,
            }
        )
    return {
        "schema_version": "entropy-beta-sensitivity-v1",
        "method": method,
        "K": int(K),
        "runs": runs,
        "selection_order_invariant": len({tuple(run["acquisition_order"]) for run in runs}) == 1,
    }


def compare_sparse_dense(
    pool: EntropyPool,
    contributions: SparseCandidateContributions,
    bandwidths: Sequence[float],
    *,
    beta: float,
    K: int,
    anchors: Sequence[str] = (),
) -> dict[str, Any]:
    """Compare sparse PMFs and greedy science with a bounded dense oracle."""

    matrix, dense = _dense_contributions(pool, bandwidths)
    sparse_result = full_greedy(pool, contributions, beta=beta, K=K, anchors=anchors)
    dense_result = full_greedy(pool, dense, beta=beta, K=K, anchors=anchors)
    deviations = []
    for index in range(len(pool.candidate_ids)):
        sparse = contributions.candidate_row(index)
        dense_row = dense.candidate_row(index)
        sparse_values = np.zeros(len(pool.rows), dtype=np.float64)
        dense_values = np.zeros(len(pool.rows), dtype=np.float64)
        sparse_values[sparse.target_indices] = sparse.values
        dense_values[dense_row.target_indices] = dense_row.values
        deviations.append(float(np.max(np.abs(sparse_values - dense_values))))
    return {
        "schema_version": "entropy-sparse-dense-sensitivity-v1",
        "reference": "bounded-dense-normalized-wendland-v1",
        "pool_fingerprint": pool.fingerprint,
        "sparse_contributions_fingerprint": contributions.fingerprint,
        "dense_contributions_fingerprint": dense.fingerprint,
        "max_candidate_pmf_deviation": max(deviations, default=0.0),
        "objective_deviation": abs(sparse_result.final_objective - dense_result.final_objective),
        "kl_deviation": abs(sparse_result.final_forward_kl - dense_result.final_forward_kl),
        "sparse_acquisition_order": list(sparse_result.acquisition_order),
        "dense_acquisition_order": list(dense_result.acquisition_order),
        "dense_matrix_entries": int(matrix.size),
    }


def compare_exact_neighbour_backends(
    descriptors: np.ndarray,
    k: int,
    *,
    row_ids: Sequence[str] | None = None,
    representation_fingerprint: str = "sensitivity",
    max_neighbour_entries: int | None = None,
) -> dict[str, Any]:
    """Compare the exact reference and exact indexed CPU neighbour backends."""

    values = np.asarray(descriptors, dtype=np.float64)
    ids = tuple(row_ids or (f"row:{index}" for index in range(len(values))))
    reference = compute_neighbours(
        values,
        k,
        backend="exact_cpu",
        representation_fingerprint=representation_fingerprint,
        row_ids=ids,
        max_neighbour_entries=max_neighbour_entries,
    )
    indexed = compute_neighbours(
        values,
        k,
        backend="exact_indexed_cpu",
        representation_fingerprint=representation_fingerprint,
        row_ids=ids,
        max_neighbour_entries=max_neighbour_entries,
    )
    return {
        "schema_version": "entropy-exact-backend-sensitivity-v1",
        "reference_backend": reference.backend,
        "indexed_backend": indexed.backend,
        "reference_fingerprint": reference.fingerprint,
        "indexed_fingerprint": indexed.fingerprint,
        "radii_max_abs_deviation": float(np.max(np.abs(reference.radii - indexed.radii))),
        "locations_equal": bool(
            np.array_equal(reference.unique_locations, indexed.unique_locations)
        ),
        "row_mapping_equal": bool(
            np.array_equal(reference.row_to_location, indexed.row_to_location)
        ),
    }


def compare_duplicate_or_thinning(
    baseline: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Package explicit duplicate/candidate/family variant outputs for review.

    The helper does not infer invariance.  Callers supply actual production
    outputs for each variant and the result preserves the labelled measures.
    """

    required = {"variant_id", "candidate_count", "atomic_row_count", "selected_ids"}
    for item in (baseline, *variants):
        missing = sorted(required.difference(item))
        if missing:
            raise ValueError(f"sensitivity variant is missing fields: {missing}")
    return {
        "schema_version": "entropy-duplicate-thinning-sensitivity-v1",
        "baseline": dict(baseline),
        "variants": [dict(item) for item in variants],
        "selection_changed": any(
            tuple(item["selected_ids"]) != tuple(baseline["selected_ids"]) for item in variants
        ),
    }


# Friendly aliases used by validation notebooks and issue fixtures.
beta_sensitivity = compare_beta_sensitivity
sparse_dense_sensitivity = compare_sparse_dense
exact_backend_sensitivity = compare_exact_neighbour_backends
duplicate_thinning_sensitivity = compare_duplicate_or_thinning

__all__ = [
    "beta_sensitivity",
    "compare_beta_sensitivity",
    "compare_duplicate_or_thinning",
    "compare_exact_neighbour_backends",
    "compare_sparse_dense",
    "duplicate_thinning_sensitivity",
    "exact_backend_sensitivity",
    "sparse_dense_sensitivity",
]
