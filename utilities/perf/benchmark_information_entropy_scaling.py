"""Bounded benchmark for the production streamed entropy-contribution path.

The descriptor fixture is generated once in ordered pool-row order and reused
for every comparison.  No ``N x N`` array is allocated; the independent
correctness oracle is enabled only when ``N`` is below ``--oracle-limit``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from nepflow.resources.budget import build_resource_budget
from nepflow.stages.selection.algorithms.information_entropy.bandwidth import (
    _freeze_from_neighbours,
    _pool_row_ids,
    build_entropy_pool,
)
from nepflow.stages.selection.algorithms.information_entropy.kernels import (
    aggregate_candidate_contributions,
    build_sparse_atomic_kernel_graph,
    build_streamed_candidate_contributions,
    evaluate_leave_one_out_objective,
    evaluate_leave_one_out_objectives,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    EntropyBandwidthSettings,
    INDEXED_NEIGHBOUR_BACKEND_ID,
    NEIGHBOUR_BACKEND_ID,
    TILED_NEIGHBOUR_BACKEND_ID,
)
from nepflow.stages.selection.algorithms.information_entropy.neighbours import (
    ExactNeighbourResult,
    IndexedCPUNeighbourIndex,
    TiledCPUNeighbourIndex,
    build_neighbour_index,
    compute_neighbours,
    compute_radius_support,
)
from nepflow.stages.selection.algorithms.information_entropy.selector import lazy_greedy


CASE_NAMES = (
    "high_duplication",
    "low_duplication",
    "exact_tie_near_tie",
    "sparse",
    "wide_support",
)
BACKENDS = (INDEXED_NEIGHBOUR_BACKEND_ID, TILED_NEIGHBOUR_BACKEND_ID, NEIGHBOUR_BACKEND_ID)
PRODUCTION_K_VALUES = EntropyBandwidthSettings().k_candidates
PRODUCTION_C_VALUES = EntropyBandwidthSettings().c_candidates
LOO_NUMERICAL_TOLERANCE = 1.0e-12


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar diagnostics to native JSON scalar types."""

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _snapshot_metrics(index: Any) -> dict[str, Any]:
    """Take a detached lifetime snapshot of an optional neighbour index."""

    if index is None:
        return {}
    return dict(index.memory_metrics())


def _metric_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    """Return deltas for cumulative counters/timers, excluding static gauges."""

    cumulative_names = {
        "cache_hit",
        "cache_miss",
        "cache_eviction",
        "cache_admission_rejected",
        "index_queries",
        "radius_queries",
        "radius_query_preflights",
        "radius_preflight_calls",
        "distance_evaluations",
        "unique_distance_evaluations",
        "atomic_distance_evaluations",
        "pilot_support_queries",
        "pilot_support_count_total",
        "pilot_support_samples",
    }
    result: dict[str, Any] = {}
    for key, value in after.items():
        if not (
            key in cumulative_names
            or key.endswith("_seconds")
            or key.endswith("_count")
            or key.endswith("_calls")
            or key.endswith("_bytes")
        ):
            continue
        previous = before.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(previous, bool) or not isinstance(previous, (int, float)):
            previous = 0
        difference = value - previous
        result[key] = max(0, difference) if isinstance(value, int) else max(0.0, difference)
    if "radius_preflight_calls" in result:
        result["radius_preflight_count"] = result["radius_preflight_calls"]
    if "radius_preflight_count_count" in result:
        result["radius_preflight_count"] = result["radius_preflight_count_count"]
    return result


def _exclusive_knn_phase_seconds(metrics: dict[str, Any]) -> dict[str, float]:
    """Name the non-overlapping kNN phases used by the profiling rollup."""

    phase_names = (
        "knn_tree_query",
        "radius_preflight_count",
        "radius_retrieval",
        "canonical_distance_verification",
        "checkpoint_serialization",
        "checkpoint_fsync",
        "checkpoint_digest",
        "checkpoint_read",
    )
    raw_phases = {
        phase: float(metrics.get(f"{phase}_seconds", 0.0) or 0.0) for phase in phase_names
    }
    publication = raw_phases["checkpoint_serialization"]
    fsync = raw_phases["checkpoint_fsync"]
    digest = raw_phases["checkpoint_digest"]
    phases = {
        phase: value
        for phase, value in raw_phases.items()
        if phase not in {"radius_preflight_count", "checkpoint_serialization"}
    }
    phases["radius_preflight"] = raw_phases["radius_preflight_count"]
    phases["checkpoint_serialization"] = max(0.0, publication - fsync - digest)
    return phases


def _fixture(
    rows: int,
    dimension: int,
    candidates: int,
    case: str,
    k_max: int,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if dimension != 42:
        raise ValueError("the scaling fixture is intentionally fixed at d'=42")
    rng = np.random.default_rng(20261011)
    unique_count = min(rows, max(k_max + 2, candidates + 2, 16))
    if case == "high_duplication":
        unique_count = min(unique_count, max(k_max + 2, rows // 8))
        unique = rng.normal(size=(unique_count, dimension))
    elif case == "low_duplication":
        unique_count = rows
        unique = rng.normal(size=(unique_count, dimension))
    elif case == "exact_tie_near_tie":
        unique = np.zeros((unique_count, dimension), dtype=np.float64)
        unique[:, 0] = np.arange(unique_count, dtype=np.float64)
        unique[1::2, 1] = 1.0e-12
    elif case == "sparse":
        unique = rng.normal(scale=10.0, size=(unique_count, dimension))
    elif case == "wide_support":
        unique = rng.normal(scale=0.05, size=(unique_count, dimension))
    else:
        raise ValueError(f"unknown fixture case: {case!r}")
    descriptors = (
        unique.copy()
        if case == "low_duplication"
        else unique[np.arange(rows, dtype=np.int64) % unique_count]
    )
    candidate_ids = tuple(f"candidate-{index:05d}" for index in range(candidates))
    owners = tuple(candidate_ids[index % candidates] for index in range(rows))
    return np.ascontiguousarray(descriptors, dtype=np.float64), candidate_ids, owners


def _scale_for_case(case: str) -> float:
    return {
        "high_duplication": 1.5,
        "low_duplication": 1.5,
        "exact_tie_near_tie": 1.5,
        "sparse": 1.05,
        "wide_support": 8.0,
    }[case]


def _budget(worker_count: int, memory_bytes: int | None, scratch_bytes: int | None):
    kwargs: dict[str, Any] = {"worker_cap": worker_count}
    if memory_bytes is not None:
        kwargs.update(execution_mode="explicit", memory_budget_bytes=memory_bytes)
    if scratch_bytes is not None:
        kwargs["scratch_budget_bytes"] = scratch_bytes
    return build_resource_budget(**kwargs)


def _prepare(
    pool: Any,
    backend: str,
    worker_count: int,
    *,
    k_max: int,
    c: float,
    chunk_size: int,
    memory_bytes: int | None,
    scratch_bytes: int | None,
    max_radius_query_bytes: int | None,
    max_index_bytes: int | None,
    max_neighbour_cache_bytes: int | None,
    checkpoint_enabled: bool,
) -> tuple[
    Any,
    IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None,
    Any,
    ExactNeighbourResult,
    dict[str, Any],
]:
    service = _budget(worker_count, memory_bytes, scratch_bytes)
    index = None
    index_build_started = time.perf_counter()
    if backend in {INDEXED_NEIGHBOUR_BACKEND_ID, TILED_NEIGHBOUR_BACKEND_ID}:
        index = build_neighbour_index(
            pool.descriptors,
            backend=backend,
            tile_size=chunk_size,
            max_index_bytes=max_index_bytes,
            max_radius_query_bytes=max_radius_query_bytes,
            max_neighbour_cache_bytes=max_neighbour_cache_bytes,
            resource_budget=service,
        )
    index_build_wall_seconds = (
        max(0.0, time.perf_counter() - index_build_started) if index is not None else 0.0
    )
    index_after_build = _snapshot_metrics(index)
    before_knn = dict(index_after_build)
    knn_checkpoint_metrics: dict[str, Any] = {}
    knn_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nepflow-entropy-knn-bench-") as temporary:
        checkpoint_root = Path(temporary) if checkpoint_enabled else None
        preserve_for_restart = bool(checkpoint_enabled)
        knn_options: dict[str, Any] = {
            "checkpoint_path": checkpoint_root,
            "checkpoint_every_locations": chunk_size,
            "checkpoint_metrics": knn_checkpoint_metrics,
        }
        if preserve_for_restart:
            knn_options["preserve_checkpoint"] = True
        neighbours = compute_neighbours(
            pool.descriptors,
            k_max,
            backend=backend,
            index=index,
            chunk_size=chunk_size,
            representation_fingerprint=pool.representation_fingerprint,
            row_ids=_pool_row_ids(pool),
            resource_budget=service,
            **knn_options,
        )
        knn_query_wall_seconds = max(0.0, time.perf_counter() - knn_started)
        after_initial_knn = _snapshot_metrics(index)
        if preserve_for_restart:
            # The restart call replaces its output dictionary; keep the first
            # pass's checkpoint write metrics and collect restart I/O separately.
            restart_checkpoint_metrics: dict[str, Any] = {}
            if index is not None:
                index._evict_neighbour_cache()
            restart_started = time.perf_counter()
            compute_neighbours(
                pool.descriptors,
                k_max,
                backend=backend,
                index=index,
                chunk_size=chunk_size,
                representation_fingerprint=pool.representation_fingerprint,
                row_ids=_pool_row_ids(pool),
                resource_budget=service,
                checkpoint_path=checkpoint_root,
                checkpoint_every_locations=chunk_size,
                checkpoint_metrics=restart_checkpoint_metrics,
                resume=True,
                preserve_checkpoint=False,
            )
            restart_cost_seconds = max(0.0, time.perf_counter() - restart_started)
            knn_checkpoint_metrics["restart_cost_seconds"] = restart_cost_seconds
            knn_checkpoint_metrics["restart_cost_measured"] = True
            knn_checkpoint_metrics["restart_bytes_read"] = int(
                restart_checkpoint_metrics.get("bytes_read", 0) or 0
            )
    # The diagnostic restart is deliberately excluded from the production kNN
    # wall time and from the full-selection estimate.
    knn_wall_seconds = knn_query_wall_seconds
    after_knn = after_initial_knn
    lifetime_index_metrics = _snapshot_metrics(index)
    frozen_started = time.perf_counter()
    frozen = _freeze_from_neighbours(pool, neighbours, c)
    frozen_wall_seconds = max(0.0, time.perf_counter() - frozen_started)
    preparation_wall_seconds = (
        index_build_wall_seconds + knn_wall_seconds + frozen_wall_seconds
    )
    knn_delta = _metric_delta(after_knn, before_knn)
    checkpoint_metrics = dict(knn_checkpoint_metrics)
    checkpoint_metrics.setdefault("enabled", bool(checkpoint_enabled))
    checkpoint_metrics.setdefault("bytes_written", 0)
    checkpoint_metrics.setdefault("array_bytes_written", 0)
    checkpoint_metrics.setdefault("physical_bytes_written", checkpoint_metrics["bytes_written"])
    checkpoint_metrics.setdefault("fsync_seconds", 0.0)
    checkpoint_metrics.setdefault("digest_seconds", 0.0)
    checkpoint_metrics.setdefault("write_seconds", 0.0)
    checkpoint_metrics.setdefault("peak_retained_bytes", 0)
    checkpoint_metrics.setdefault("peak_scratch_bytes", checkpoint_metrics["peak_retained_bytes"])
    checkpoint_metrics.setdefault("retained_bytes", 0)
    checkpoint_metrics.setdefault("commits", 0)
    checkpoint_metrics["physical_bytes_written"] = int(
        checkpoint_metrics.get("physical_bytes_written", checkpoint_metrics.get("bytes_written", 0))
        or 0
    )
    checkpoint_metrics["chunk_count"] = int(checkpoint_metrics.get("commits", 0) or 0)
    checkpoint_metrics["peak_scratch_bytes"] = max(
        int(checkpoint_metrics.get("peak_scratch_bytes", 0) or 0),
        int(checkpoint_metrics.get("peak_retained_bytes", 0) or 0),
    )
    checkpoint_metrics.setdefault("restart_cost_seconds", None)
    checkpoint_metrics.setdefault("restart_cost_measured", False)
    knn_phase_metrics = dict(knn_delta)
    for key in (
        "checkpoint_serialization_seconds",
        "checkpoint_fsync_seconds",
        "checkpoint_digest_seconds",
    ):
        knn_phase_metrics.setdefault(
            key,
            float(checkpoint_metrics.get(key, 0.0) or 0.0)
            if key != "checkpoint_serialization_seconds"
            else float(checkpoint_metrics.get("write_seconds", 0.0) or 0.0),
        )
    preparation = {
        "wall_seconds": preparation_wall_seconds,
        "stage_wall_seconds": {
            "index_grouping_and_tree_construction": index_build_wall_seconds,
            "index_grouping": float(index_after_build.get("grouping_build_seconds", 0.0) or 0.0),
            "tree_construction": float(index_after_build.get("tree_build_seconds", 0.0) or 0.0),
            "exact_knn_query": knn_wall_seconds,
            "frozen_bandwidth_materialization": frozen_wall_seconds,
        },
        "exclusive_phase_seconds": {
            "index_grouping": float(index_after_build.get("grouping_build_seconds", 0.0) or 0.0),
            "tree_construction": float(index_after_build.get("tree_build_seconds", 0.0) or 0.0),
            **_exclusive_knn_phase_seconds(knn_phase_metrics),
        },
        "knn_wall_seconds": knn_wall_seconds,
        "knn_phase_counters": {
            key: value
            for key, value in knn_delta.items()
            if key.endswith("_count")
            or key.endswith("_calls")
            or key in {"index_queries", "radius_queries", "distance_evaluations"}
        },
        "knn_rates": {
            "index_queries_per_second": float(knn_delta.get("index_queries", 0) or 0)
            / knn_wall_seconds
            if knn_wall_seconds > 0.0
            else 0.0,
            "radius_queries_per_second": float(knn_delta.get("radius_queries", 0) or 0)
            / knn_wall_seconds
            if knn_wall_seconds > 0.0
            else 0.0,
            "locations_per_second": int(neighbours.unique_locations.shape[0]) / knn_wall_seconds
            if knn_wall_seconds > 0.0
            else 0.0,
        },
        "knn_per_run_metrics": knn_phase_metrics,
        "cumulative_index_metrics": lifetime_index_metrics,
        "knn_checkpoint": checkpoint_metrics,
        "selection_prepare_semantics": (
            "index construction, exact kNN, and frozen bandwidth materialization; "
            "does not include streamed candidate contributions"
        ),
    }
    return service, index, frozen, neighbours, preparation


def _run_stream(
    pool: Any,
    frozen: Any,
    index: IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None,
    service: Any,
    *,
    chunk_size: int,
    max_entries: int,
    max_radius_query_bytes: int | None,
    checkpoint_enabled: bool,
    label: str,
) -> dict[str, Any]:
    index_before = _snapshot_metrics(index)
    with tempfile.TemporaryDirectory(prefix="nepflow-entropy-bench-") as temporary:
        checkpoint_path = Path(temporary) if checkpoint_enabled else None
        started = time.perf_counter()
        contributions, summary = build_streamed_candidate_contributions(
            pool,
            frozen,
            chunk_size=chunk_size,
            max_entries=max_entries,
            max_radius_query_bytes=max_radius_query_bytes,
            resource_budget=service,
            index=index,
            storage="memory",
            checkpoint_path=checkpoint_path,
            checkpoint_name=f"{label}-candidate-aggregation",
            resume=False,
        )
        elapsed = max(0.0, time.perf_counter() - started)
    operations = dict(summary.operational_metrics)
    index_after = _snapshot_metrics(index)
    cumulative_index_metrics = {
        key: operations.pop(key)
        for key in tuple(index_after)
        if key in operations
    }
    operations["cumulative_index_metrics"] = cumulative_index_metrics
    operations["per_run_index_metrics"] = _metric_delta(index_after, index_before)
    operations["completion_seconds"] = elapsed
    operations["candidate_throughput_per_second"] = (
        len(pool.candidate_ids) / elapsed if elapsed > 0.0 else 0.0
    )
    return {
        "elapsed_seconds": elapsed,
        "streamed_contribution_wall_seconds": elapsed,
        "elapsed_semantics": "streamed candidate contributions only",
        "throughput_candidates_per_second": operations["candidate_throughput_per_second"],
        "summary": summary.to_manifest(),
        "operations": operations,
        "contributions_fingerprint": contributions.fingerprint,
        "candidate_ids": list(contributions.candidate_ids),
        "contributions": contributions,
    }


def _correctness_checks(
    pool: Any,
    frozen: Any,
    neighbours: ExactNeighbourResult,
    contributions: Any,
    *,
    chunk_size: int,
    oracle_limit: int,
    max_radius_query_bytes: int | None,
    backend: str,
    index: IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None,
    resource_budget: Any,
) -> dict[str, Any]:
    """Run independent bounded checks for the complete production calibration grid."""

    n_rows = int(pool.descriptors.shape[0])
    calibration_started = time.perf_counter()
    if n_rows > oracle_limit:
        return {"status": "skipped", "reason": f"N={n_rows} exceeds oracle_limit={oracle_limit}"}
    reference_by_k: dict[int, ExactNeighbourResult] = {}
    production_by_k: dict[int, ExactNeighbourResult] = {}
    location_count = int(np.unique(neighbours.row_to_location).size)
    supported_k = tuple(k for k in PRODUCTION_K_VALUES if k <= location_count - 1)
    for candidate_k in supported_k:
        reference_by_k[candidate_k] = compute_neighbours(
            pool.descriptors,
            candidate_k,
            backend=NEIGHBOUR_BACKEND_ID,
            chunk_size=chunk_size,
            representation_fingerprint=pool.representation_fingerprint,
            row_ids=_pool_row_ids(pool),
        )
        production_by_k[candidate_k] = compute_neighbours(
            pool.descriptors,
            candidate_k,
            backend=backend,
            index=index,
            chunk_size=chunk_size,
            representation_fingerprint=pool.representation_fingerprint,
            row_ids=_pool_row_ids(pool),
            resource_budget=resource_budget,
        )
    reference = reference_by_k.get(max(supported_k, default=1), neighbours)
    indices_equal = bool(np.array_equal(neighbours.indices, reference.indices))
    radii_equal = bool(
        np.array_equal(neighbours.radii, reference.radii)
        and np.array_equal(neighbours.distances, reference.distances)
    )
    strict_support = True
    for source_index, scale in enumerate(frozen.bandwidths):
        targets, distances = compute_radius_support(
            pool.descriptors,
            source_index,
            float(scale),
            backend=NEIGHBOUR_BACKEND_ID,
            chunk_size=chunk_size,
            max_radius_query_bytes=max_radius_query_bytes,
        )
        strict_support = strict_support and bool(
            np.any(targets == source_index)
            and np.all(distances < float(scale))
            and np.all(np.isfinite(distances))
        )

    pair_results: list[dict[str, Any]] = []
    production_winner: tuple[float, int, float, str] | None = None
    reference_winner: tuple[float, int, float, str] | None = None
    for candidate_k in PRODUCTION_K_VALUES:
        for candidate_c in PRODUCTION_C_VALUES:
            pair_key = f"k={candidate_k},c={candidate_c:g}"
            if candidate_k not in supported_k:
                pair_results.append(
                    {
                        "pair": pair_key,
                        "status": "unsupported",
                        "production_valid": False,
                        "reference_valid": False,
                        "production_reason": (
                            f"k={candidate_k} requires at least {candidate_k} other distinct "
                            f"locations; fixture has {location_count - 1}"
                        ),
                        "reference_reason": "same fixture capacity rejection",
                    }
                )
                continue
            production_frozen = _freeze_from_neighbours(
                pool, production_by_k[candidate_k], candidate_c
            )
            reference_frozen = _freeze_from_neighbours(
                pool, reference_by_k[candidate_k], candidate_c
            )
            production_batch = evaluate_leave_one_out_objectives(
                pool.descriptors,
                pool.probabilities,
                np.stack([production_frozen.bandwidths]),
                chunk_size=chunk_size,
                backend=backend,
                index=index,
                max_radius_query_bytes=max_radius_query_bytes,
                resource_budget=resource_budget,
            )[0]
            reference_independent_batch = evaluate_leave_one_out_objectives(
                pool.descriptors,
                pool.probabilities,
                np.stack([reference_frozen.bandwidths]),
                chunk_size=chunk_size,
                backend=NEIGHBOUR_BACKEND_ID,
                max_radius_query_bytes=max_radius_query_bytes,
            )[0]
            production_scalar = evaluate_leave_one_out_objective(
                pool.descriptors,
                pool.probabilities,
                production_frozen.bandwidths,
                chunk_size=chunk_size,
                backend=backend,
                index=index,
                max_radius_query_bytes=max_radius_query_bytes,
                resource_budget=resource_budget,
            )
            reference_scalar = evaluate_leave_one_out_objective(
                pool.descriptors,
                pool.probabilities,
                reference_frozen.bandwidths,
                chunk_size=chunk_size,
                backend=NEIGHBOUR_BACKEND_ID,
                max_radius_query_bytes=max_radius_query_bytes,
            )
            objective_match = (
                production_batch.valid == reference_independent_batch.valid
                and production_batch.reason == reference_independent_batch.reason
                and (
                    production_batch.objective is None
                    and reference_independent_batch.objective is None
                    or production_batch.objective is not None
                    and reference_independent_batch.objective is not None
                    and np.isclose(
                        production_batch.objective,
                        reference_independent_batch.objective,
                        rtol=0.0,
                        atol=LOO_NUMERICAL_TOLERANCE,
                    )
                )
            )
            scalar_batch_match = (
                production_batch.valid == production_scalar.valid
                and production_batch.reason == production_scalar.reason
                and (
                    production_batch.objective is None
                    and production_scalar.objective is None
                    or production_batch.objective is not None
                    and production_scalar.objective is not None
                    and np.isclose(
                        production_batch.objective,
                        production_scalar.objective,
                        rtol=0.0,
                        atol=LOO_NUMERICAL_TOLERANCE,
                    )
                )
            )
            reference_scalar_batch_match = (
                reference_independent_batch.valid == reference_scalar.valid
                and reference_independent_batch.reason == reference_scalar.reason
                and (
                    reference_independent_batch.objective is None
                    and reference_scalar.objective is None
                    or reference_independent_batch.objective is not None
                    and reference_scalar.objective is not None
                    and np.isclose(
                        reference_independent_batch.objective,
                        reference_scalar.objective,
                        rtol=0.0,
                        atol=LOO_NUMERICAL_TOLERANCE,
                    )
                )
            )
            neighbour_match = bool(
                np.array_equal(
                    production_by_k[candidate_k].indices,
                    reference_by_k[candidate_k].indices,
                )
                and np.array_equal(
                    production_by_k[candidate_k].distances,
                    reference_by_k[candidate_k].distances,
                )
            )
            bandwidth_arrays_equal = bool(
                np.array_equal(production_frozen.radii, reference_frozen.radii)
                and np.array_equal(production_frozen.bandwidths, reference_frozen.bandwidths)
            )
            pair_passed = bool(
                neighbour_match
                and bandwidth_arrays_equal
                and objective_match
                and scalar_batch_match
                and reference_scalar_batch_match
            )
            if production_batch.valid and production_batch.objective is not None:
                candidate = (
                    float(production_batch.objective),
                    candidate_k,
                    candidate_c,
                    production_frozen.fingerprint,
                )
                if production_winner is None or candidate[:3] < production_winner[:3]:
                    production_winner = candidate
            if (
                reference_independent_batch.valid
                and reference_independent_batch.objective is not None
            ):
                candidate = (
                    float(reference_independent_batch.objective),
                    candidate_k,
                    candidate_c,
                    reference_frozen.fingerprint,
                )
                if reference_winner is None or candidate[:3] < reference_winner[:3]:
                    reference_winner = candidate
            pair_results.append(
                {
                    "pair": pair_key,
                    "status": "passed" if pair_passed else "failed",
                    "production_valid": production_batch.valid,
                    "reference_valid": reference_independent_batch.valid,
                    "production_reason": production_batch.reason,
                    "reference_reason": reference_independent_batch.reason,
                    "production_scalar_reason": production_scalar.reason,
                    "reference_scalar_reason": reference_scalar.reason,
                    "production_objective": production_batch.objective,
                    "reference_objective": reference_independent_batch.objective,
                    "production_scalar_objective": production_scalar.objective,
                    "reference_scalar_objective": reference_scalar.objective,
                    "scalar_vs_batched_match": scalar_batch_match,
                    "reference_scalar_vs_batched_match": reference_scalar_batch_match,
                    "objective_match": objective_match,
                    "neighbour_match": neighbour_match,
                    "scientific_values_equal": bandwidth_arrays_equal,
                    "bandwidth_fingerprint_strings_equal": (
                        production_frozen.fingerprint == reference_frozen.fingerprint
                    ),
                    "production_bandwidth_fingerprint": production_frozen.fingerprint,
                    "reference_bandwidth_fingerprint": reference_frozen.fingerprint,
                    "bandwidth_arrays_equal": bandwidth_arrays_equal,
                }
            )

    pmfs_normalized = True
    for candidate_index in range(contributions.n_candidates):
        start = int(contributions.candidate_indptr[candidate_index])
        stop = int(contributions.candidate_indptr[candidate_index + 1])
        pmfs_normalized = pmfs_normalized and bool(
            stop > start
            and np.all(np.diff(contributions.target_indices[start:stop]) > 0)
            and np.isclose(
                np.sum(contributions.values[start:stop], dtype=np.float64),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
    reference_graph = build_sparse_atomic_kernel_graph(
        pool,
        frozen,
        chunk_size=chunk_size,
        max_edges=max(1, n_rows * n_rows),
    )
    reference_contributions = aggregate_candidate_contributions(reference_graph)
    pmfs_match_reference = bool(
        np.array_equal(contributions.candidate_indptr, reference_contributions.candidate_indptr)
        and np.array_equal(contributions.target_indices, reference_contributions.target_indices)
        and np.allclose(
            contributions.values,
            reference_contributions.values,
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    selection = lazy_greedy(
        pool,
        contributions,
        beta=1.0,
        budget=min(3, len(pool.candidate_ids)),
    )
    reference_selection = lazy_greedy(
        pool,
        reference_contributions,
        beta=1.0,
        budget=min(3, len(pool.candidate_ids)),
    )
    frozen_identity = bool(
        np.array_equal(frozen.bandwidths, np.asarray(frozen.c * frozen.radii, dtype=np.float64))
        and bool(frozen.fingerprint)
    )
    duplicate_rows = bool(np.unique(neighbours.row_to_location).size < n_rows)
    duplicate_mapping_valid = True
    for location in np.unique(reference.row_to_location):
        rows = np.flatnonzero(reference.row_to_location == location)
        duplicate_mapping_valid = duplicate_mapping_valid and bool(
            np.all(reference.row_to_location[rows] == location)
            and np.all(pool.descriptors[rows] == pool.descriptors[rows[0]])
        )
    boundary_radius = float(reference_by_k.get(1, reference).distances[0, 0])
    boundary_targets, boundary_distances = compute_radius_support(
        pool.descriptors,
        0,
        boundary_radius,
        backend=NEIGHBOUR_BACKEND_ID,
        chunk_size=chunk_size,
        max_radius_query_bytes=max_radius_query_bytes,
    )
    strict_boundary = bool(
        np.any(boundary_targets == 0)
        and np.all(boundary_distances < boundary_radius)
        and not np.any(np.isclose(boundary_distances, boundary_radius, rtol=0.0, atol=0.0))
    )
    grid_complete = len(pair_results) == len(PRODUCTION_K_VALUES) * len(PRODUCTION_C_VALUES)
    all_pairs_passed = grid_complete and all(
        pair.get("status") == "passed" for pair in pair_results
    )
    winner_match = bool(
        production_winner is not None
        and reference_winner is not None
        and production_winner[1:3] == reference_winner[1:3]
    )
    checks_passed = all(
        (
            indices_equal,
            radii_equal,
            strict_support,
            pmfs_normalized,
            pmfs_match_reference,
            frozen_identity,
            selection.selected_candidate_ids == reference_selection.selected_candidate_ids,
            duplicate_mapping_valid,
            strict_boundary,
            all_pairs_passed,
            winner_match,
        )
    )
    return {
        "status": "passed" if checks_passed else "failed",
        "calibration_wall_seconds": max(0.0, time.perf_counter() - calibration_started),
        "neighbour_indices_equal": indices_equal,
        "neighbour_radii_equal": radii_equal,
        "strict_radius_support": strict_support,
        "calibration_grid": {
            "k_values": list(PRODUCTION_K_VALUES),
            "c_values": list(PRODUCTION_C_VALUES),
            "pair_count": len(pair_results),
            "fingerprint_comparison_semantics": (
                "scientific_values_equal and bandwidth_arrays_equal compare exact "
                "float64 bandwidth/radius data. bandwidth_fingerprint_strings_equal "
                "compares full provenance-bearing fingerprint strings, which can "
                "differ across exact backends with identical scientific values."
            ),
            "all_16_pairs_tested": grid_complete,
            "all_16_pairs_passed": all_pairs_passed,
            "pairs": pair_results,
            "winner_match": winner_match,
            "production_winner": None
            if production_winner is None
            else {
                "objective": production_winner[0],
                "k": production_winner[1],
                "c": production_winner[2],
                "fingerprint": production_winner[3],
            },
            "reference_winner": None
            if reference_winner is None
            else {
                "objective": reference_winner[0],
                "k": reference_winner[1],
                "c": reference_winner[2],
                "fingerprint": reference_winner[3],
            },
            "numerical_tolerance": LOO_NUMERICAL_TOLERANCE,
        },
        "adversarial_checks": {
            "duplicate_locations_present": duplicate_rows,
            "duplicate_location_mapping": duplicate_mapping_valid,
            "equal_or_near_tie_neighbour_identity": indices_equal and radii_equal,
            "strict_radius_boundary": strict_boundary,
            "selected_candidate_ids": list(selection.selected_candidate_ids),
            "selected_candidate_ids_match_reference": (
                selection.selected_candidate_ids == reference_selection.selected_candidate_ids
            ),
        },
        "candidate_pmfs_normalized": pmfs_normalized,
        "candidate_pmfs_match_reference": pmfs_match_reference,
        "frozen_bandwidth_identity": frozen_identity,
        "selected_ids": list(selection.selected_candidate_ids),
        "selected_ids_match_reference": (
            selection.selected_candidate_ids == reference_selection.selected_candidate_ids
        ),
        "scientific_fingerprints": {
            "pool": pool.fingerprint,
            "representation": pool.representation_fingerprint,
            "frozen_bandwidth": frozen.fingerprint,
            "contributions": contributions.fingerprint,
            "selection_state": selection.state_fingerprint,
        },
    }


def _profiling_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    exclusive_totals: dict[str, float] = {}
    nested_totals: dict[str, float] = {}
    phase_counts: dict[str, int] = {}
    kNN_checkpoint: dict[str, float | int] = {
        "physical_bytes_written": 0,
        "array_bytes_written": 0,
        "fsync_seconds": 0.0,
        "digest_seconds": 0.0,
        "peak_retained_bytes": 0,
        "peak_scratch_bytes": 0,
        "chunk_count": 0,
        "restart_cost_seconds": 0.0,
        "restart_bytes_read": 0,
    }
    contribution_checkpoint: dict[str, float | int] = {
        "bytes": 0,
        "seconds": 0.0,
        "writes": 0,
    }
    stage_wall: dict[str, float] = {
        "index_grouping_and_tree_construction": 0.0,
        "exact_knn_query": 0.0,
        "frozen_bandwidth_materialization": 0.0,
        "streamed_candidate_contributions": 0.0,
        "loo_calibration": 0.0,
    }
    seen_preparation: set[tuple[Any, ...]] = set()
    seen_calibration: set[str] = set()

    def add_phase(name: str, seconds: float, *, count: int | None = None) -> None:
        exclusive_totals[name] = exclusive_totals.get(name, 0.0) + max(0.0, seconds)
        if count is not None:
            phase_counts[name] = phase_counts.get(name, 0) + int(count)

    for result in results:
        payload = result.get("result", {})
        operations = payload.get("operations", {})
        configuration = (
            result.get("backend"),
            result.get("worker_count"),
            result.get("checkpoint_enabled"),
        )
        preparation = payload.get("preparation", {})
        if configuration not in seen_preparation:
            seen_preparation.add(configuration)
            stage = preparation.get("stage_wall_seconds", {})
            stage_wall["index_grouping_and_tree_construction"] += float(
                stage.get("index_grouping_and_tree_construction", 0.0) or 0.0
            )
            stage_wall["exact_knn_query"] += float(stage.get("exact_knn_query", 0.0) or 0.0)
            stage_wall["frozen_bandwidth_materialization"] += float(
                stage.get("frozen_bandwidth_materialization", 0.0) or 0.0
            )
            prep_phases = {
                str(key): float(value or 0.0)
                for key, value in preparation.get("exclusive_phase_seconds", {}).items()
            }
            knn_wall = float(preparation.get("knn_wall_seconds", 0.0) or 0.0)
            known_knn = sum(prep_phases.values())
            prep_phases["knn_unattributed"] = max(0.0, knn_wall - known_knn)
            for phase, seconds in prep_phases.items():
                add_phase(f"preparation.{phase}", seconds)
            checkpoint = preparation.get("knn_checkpoint", {})
            kNN_checkpoint["physical_bytes_written"] += int(
                checkpoint.get("physical_bytes_written", 0) or 0
            )
            kNN_checkpoint["array_bytes_written"] += int(
                checkpoint.get("array_bytes_written", 0) or 0
            )
            kNN_checkpoint["fsync_seconds"] += float(checkpoint.get("fsync_seconds", 0.0) or 0.0)
            kNN_checkpoint["digest_seconds"] += float(
                checkpoint.get("digest_seconds", 0.0) or 0.0
            )
            kNN_checkpoint["peak_retained_bytes"] = max(
                int(kNN_checkpoint["peak_retained_bytes"]),
                int(checkpoint.get("peak_retained_bytes", 0) or 0),
            )
            kNN_checkpoint["peak_scratch_bytes"] = max(
                int(kNN_checkpoint["peak_scratch_bytes"]),
                int(checkpoint.get("peak_scratch_bytes", 0) or 0),
            )
            kNN_checkpoint["chunk_count"] += int(checkpoint.get("chunk_count", 0) or 0)
            kNN_checkpoint["restart_cost_seconds"] += float(
                checkpoint.get("restart_cost_seconds", 0.0) or 0.0
            )
            kNN_checkpoint["restart_bytes_read"] += int(
                checkpoint.get("restart_bytes_read", 0) or 0
            )
        if str(result.get("backend")) not in seen_calibration:
            seen_calibration.add(str(result.get("backend")))
            correctness = result.get("correctness", {})
            calibration_seconds = float(correctness.get("calibration_wall_seconds", 0.0) or 0.0)
            stage_wall["loo_calibration"] += calibration_seconds
            add_phase("loo_calibration", calibration_seconds)
        phase_seconds = operations.get("phase_seconds", {})
        phase_counts_payload = operations.get("phase_counts", {})
        for phase, seconds in phase_seconds.items():
            add_phase(
                f"contributions.{phase}",
                float(seconds),
                count=int(phase_counts_payload.get(phase, 0) or 0),
            )
        per_run_index = operations.get("per_run_index_metrics", {})
        for phase in (
            "knn_tree_query",
            "radius_preflight_count",
            "radius_retrieval",
            "canonical_distance_verification",
        ):
            seconds = float(per_run_index.get(f"{phase}_seconds", 0.0) or 0.0)
            nested_name = {
                "knn_tree_query": "kNN tree traversal",
                "radius_preflight_count": "radius preflight",
                "radius_retrieval": "radius materialisation",
                "canonical_distance_verification": "canonical distance/tie verification",
            }[phase]
            nested_totals[nested_name] = nested_totals.get(nested_name, 0.0) + seconds
        support_seconds = float(operations.get("support_query_seconds", 0.0) or 0.0)
        add_phase(
            "contributions.strict_kernel_support",
            support_seconds,
            count=int(operations.get("support_query_count", 0) or 0),
        )
        stage_wall["streamed_candidate_contributions"] += float(
            payload.get("streamed_contribution_wall_seconds", 0.0) or 0.0
        )
        contribution = operations.get("contribution_checkpoint", {})
        contribution_checkpoint["bytes"] += int(contribution.get("bytes", 0) or 0)
        contribution_checkpoint["seconds"] += float(contribution.get("seconds", 0.0) or 0.0)
        contribution_checkpoint["writes"] += int(contribution.get("writes", 0) or 0)
    dominant = sorted(exclusive_totals.items(), key=lambda item: (-item[1], item[0]))
    return {
        "exclusive_phase_seconds": exclusive_totals,
        "nested_inclusive_phase_seconds": nested_totals,
        "dominant_phases_by_aggregate_seconds": [
            {"phase": phase, "seconds": seconds} for phase, seconds in dominant[:5]
        ],
        "phase_counts": phase_counts,
        "stage_wall_seconds": stage_wall,
        "knn_checkpoint_io": kNN_checkpoint,
        "contribution_checkpoint_io": contribution_checkpoint,
        "phase_definitions": {
            "preparation.exact_knn_query": (
                "inclusive kNN query wall time; child phases are reported separately "
                "and are not added to this chart"
            ),
            "preparation.knn_unattributed": (
                "kNN wall time not covered by the exclusive child timers"
            ),
            "contributions.strict_kernel_support": (
                "strict radius support query timer from the kernel provider"
            ),
            "contributions.source_normalization_and_accumulation": (
                "source-column normalization and accumulation only"
            ),
            "contributions.checkpoint_serialization": (
                "candidate-contribution checkpoint writes only"
            ),
            "preparation.checkpoint_serialization": "kNN checkpoint publication only",
            "nested_inclusive_phase_seconds": (
                "diagnostic child timers; intentionally excluded from the exclusive ranking"
            ),
            "loo_calibration": "independent 16-pair correctness/calibration oracle timing",
        },
        "interpretation": (
            "Dominant phases are summed from exclusive child timers. Inclusive stage wall times "
            "and contribution-versus-kNN checkpoint I/O are reported separately."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--dimension", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--case", choices=CASE_NAMES, default="low_duplication")
    parser.add_argument("--backend", choices=("all", *BACKENDS), default="all")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 8, 64])
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-entries", type=int, default=10_000_000)
    parser.add_argument("--max-radius-query-bytes", type=int, default=None)
    parser.add_argument("--max-index-bytes", type=int, default=None)
    parser.add_argument("--max-neighbour-cache-bytes", type=int, default=None)
    parser.add_argument("--memory-budget-bytes", type=int, default=None)
    parser.add_argument("--scratch-budget-bytes", type=int, default=None)
    parser.add_argument("--oracle-limit", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("information_entropy_scaling.json"))
    args = parser.parse_args()
    if args.rows < 2 or args.candidates < 1 or args.candidates > args.rows:
        parser.error("rows must be >= 2 and candidates must be in [1, rows]")
    if args.k_max < 1 or args.k_max >= args.rows:
        parser.error("k-max must be positive and smaller than rows")
    if any(worker < 1 for worker in args.workers):
        parser.error("worker counts must be positive")

    descriptors, candidate_ids, owners = _fixture(
        args.rows, args.dimension, args.candidates, args.case, args.k_max
    )
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=owners,
        candidate_ids=candidate_ids,
    )
    backends = BACKENDS if args.backend == "all" else (args.backend,)
    results: list[dict[str, Any]] = []
    correctness_cache: dict[str, dict[str, Any]] = {}
    for backend in backends:
        for worker_count in args.workers:
            for checkpoint_enabled in (False, True):
                service, index, frozen, neighbours, preparation = _prepare(
                    pool,
                    backend,
                    worker_count,
                    k_max=args.k_max,
                    c=_scale_for_case(args.case),
                    chunk_size=args.chunk_size,
                    memory_bytes=args.memory_budget_bytes,
                    scratch_bytes=args.scratch_budget_bytes,
                    max_radius_query_bytes=args.max_radius_query_bytes,
                    max_index_bytes=args.max_index_bytes,
                    max_neighbour_cache_bytes=args.max_neighbour_cache_bytes,
                    checkpoint_enabled=checkpoint_enabled,
                )
                cold = _run_stream(
                    pool,
                    frozen,
                    index,
                    service,
                    chunk_size=args.chunk_size,
                    max_entries=args.max_entries,
                    max_radius_query_bytes=args.max_radius_query_bytes,
                    checkpoint_enabled=checkpoint_enabled,
                    label=f"{backend}-{worker_count}-cold",
                )
                warm = _run_stream(
                    pool,
                    frozen,
                    index,
                    service,
                    chunk_size=args.chunk_size,
                    max_entries=args.max_entries,
                    max_radius_query_bytes=args.max_radius_query_bytes,
                    checkpoint_enabled=checkpoint_enabled,
                    label=f"{backend}-{worker_count}-warm",
                )
                for run in (cold, warm):
                    run["operations"]["contribution_checkpoint"] = {
                        "writes": int(run["operations"].get("checkpoint_writes", 0) or 0),
                        "bytes": int(run["operations"].get("checkpoint_bytes", 0) or 0),
                        "seconds": float(run["operations"].get("checkpoint_seconds", 0.0) or 0.0),
                    }
                    run["operations"]["support_query"] = {
                        "seconds": float(
                            run["operations"].get("support_query_seconds", 0.0) or 0.0
                        ),
                        "count": int(run["operations"].get("support_query_count", 0) or 0),
                        "entries": int(run["operations"].get("support_entries", 0) or 0),
                    }
                    run["preparation"] = preparation
                    run["full_selection_wall_seconds"] = float(
                        preparation["wall_seconds"] + run["streamed_contribution_wall_seconds"]
                    )
                    run["full_selection_elapsed_semantics"] = (
                        "preparation plus streamed candidate contributions; "
                        "contribution checkpoint I/O remains nested in the stream stage"
                    )
                if cold["contributions_fingerprint"] != warm["contributions_fingerprint"]:
                    raise AssertionError("warm-cache contribution fingerprint changed")
                checks = correctness_cache.get(backend)
                if checks is None:
                    checks = _correctness_checks(
                        pool,
                        frozen,
                        neighbours,
                        cold["contributions"],
                        chunk_size=args.chunk_size,
                        oracle_limit=args.oracle_limit,
                        max_radius_query_bytes=args.max_radius_query_bytes,
                        backend=backend,
                        index=index,
                        resource_budget=service,
                    )
                    correctness_cache[backend] = checks
                workload = {
                    "N": int(pool.descriptors.shape[0]),
                    "M": len(pool.candidate_ids),
                    "d_prime": int(pool.descriptors.shape[1]),
                    "Q": int(cold["summary"]["candidate_entry_count"]),
                    "k_max": int(args.k_max),
                    "unique_locations": int(neighbours.unique_locations.shape[0]),
                }
                limits = {
                    "chunk_size": args.chunk_size,
                    "max_entries": args.max_entries,
                    "max_radius_query_bytes": args.max_radius_query_bytes,
                    "max_index_bytes": args.max_index_bytes,
                    "max_neighbour_cache_bytes": args.max_neighbour_cache_bytes,
                    "memory_budget_bytes": args.memory_budget_bytes,
                    "scratch_budget_bytes": args.scratch_budget_bytes,
                }
                common = {
                    "backend": backend,
                    "worker_count": worker_count,
                    "checkpoint_enabled": checkpoint_enabled,
                    "knn_checkpoint_enabled": checkpoint_enabled,
                    "contribution_checkpoint_enabled": checkpoint_enabled,
                    "exact_workload": workload,
                    "operational_limits": limits,
                    "pool_fingerprint": pool.fingerprint,
                    "frozen_bandwidth_fingerprint": frozen.fingerprint,
                    "correctness": checks,
                }
                for cache_state, run in (("cold", cold), ("warm", warm)):
                    results.append(
                        {
                            **common,
                            "cache_state": cache_state,
                            "cold_cache": cache_state == "cold",
                            "result": {
                                key: value
                                for key, value in run.items()
                                if key != "contributions"
                            },
                        }
                    )
                if index is not None:
                    index.close()

    output = {
        "schema": "nepflow-information-entropy-scaling-benchmark-v3",
        "fixture": {
            "case": args.case,
            "seed": 20261011,
            "ordered_descriptor_fixture": True,
            "N": int(pool.descriptors.shape[0]),
            "M": len(pool.candidate_ids),
            "d_prime": int(pool.descriptors.shape[1]),
            "k_max": int(args.k_max),
            "backend_set": list(backends),
            "worker_counts": list(args.workers),
            "pool_fingerprint": pool.fingerprint,
        },
        "results": results,
        "profiling_summary": _profiling_summary(results),
        "scientific_note": (
            "Measurements and correctness diagnostics are never used to select science parameters."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        output,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
