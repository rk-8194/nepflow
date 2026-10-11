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
) -> tuple[
    Any,
    IndexedCPUNeighbourIndex | TiledCPUNeighbourIndex | None,
    Any,
    ExactNeighbourResult,
]:
    service = _budget(worker_count, memory_bytes, scratch_bytes)
    index = None
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
    neighbours = compute_neighbours(
        pool.descriptors,
        k_max,
        backend=backend,
        index=index,
        chunk_size=chunk_size,
        representation_fingerprint=pool.representation_fingerprint,
        row_ids=_pool_row_ids(pool),
        resource_budget=service,
    )
    frozen = _freeze_from_neighbours(pool, neighbours, c)
    return service, index, frozen, neighbours


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
    operations["completion_seconds"] = elapsed
    operations["candidate_throughput_per_second"] = (
        len(pool.candidate_ids) / elapsed if elapsed > 0.0 else 0.0
    )
    return {
        "elapsed_seconds": elapsed,
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
) -> dict[str, Any]:
    """Run independent bounded checks, including every configured LOO pair."""

    n_rows = int(pool.descriptors.shape[0])
    if n_rows > oracle_limit:
        return {"status": "skipped", "reason": f"N={n_rows} exceeds oracle_limit={oracle_limit}"}
    reference = compute_neighbours(
        pool.descriptors,
        neighbours.k,
        backend=NEIGHBOUR_BACKEND_ID,
        chunk_size=chunk_size,
        representation_fingerprint=pool.representation_fingerprint,
        row_ids=_pool_row_ids(pool),
    )
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

    location_count = int(np.unique(neighbours.row_to_location).size)
    k_values = tuple(range(1, min(neighbours.k, location_count - 1) + 1))
    c_values = tuple(dict.fromkeys((1.5, float(frozen.c))))
    loo_objectives: dict[str, float | None] = {}
    for candidate_k in k_values:
        candidate_neighbours = compute_neighbours(
            pool.descriptors,
            candidate_k,
            backend=NEIGHBOUR_BACKEND_ID,
            chunk_size=chunk_size,
            representation_fingerprint=pool.representation_fingerprint,
            row_ids=_pool_row_ids(pool),
        )
        frozen_values = tuple(
            _freeze_from_neighbours(pool, candidate_neighbours, candidate_c)
            for candidate_c in c_values
        )
        batched = evaluate_leave_one_out_objectives(
            pool.descriptors,
            pool.probabilities,
            np.stack([value.bandwidths for value in frozen_values]),
            chunk_size=chunk_size,
            backend=NEIGHBOUR_BACKEND_ID,
            max_radius_query_bytes=max_radius_query_bytes,
        )
        for candidate_c, frozen_value, objective in zip(
            c_values, frozen_values, batched, strict=True
        ):
            scalar = evaluate_leave_one_out_objective(
                pool.descriptors,
                pool.probabilities,
                frozen_value.bandwidths,
                chunk_size=chunk_size,
                backend=NEIGHBOUR_BACKEND_ID,
                max_radius_query_bytes=max_radius_query_bytes,
            )
            pair_key = f"k={candidate_k},c={candidate_c:g}"
            loo_objectives[pair_key] = objective.objective
            if objective.valid != scalar.valid or (
                objective.objective is not None
                and (
                    scalar.objective is None
                    or not np.isclose(
                        objective.objective,
                        scalar.objective,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )
            ):
                raise AssertionError(f"batched/scalar LOO mismatch for {pair_key}")

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
    checks_passed = all(
        (
            indices_equal,
            radii_equal,
            strict_support,
            pmfs_normalized,
            pmfs_match_reference,
            frozen_identity,
            selection.selected_candidate_ids == reference_selection.selected_candidate_ids,
        )
    )
    return {
        "status": "passed" if checks_passed else "failed",
        "neighbour_indices_equal": indices_equal,
        "neighbour_radii_equal": radii_equal,
        "strict_radius_support": strict_support,
        "loo_objectives": loo_objectives,
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
    totals: dict[str, float] = {}
    for result in results:
        phase_seconds = result.get("result", {}).get("operations", {}).get(
            "phase_seconds", {}
        )
        for phase, seconds in phase_seconds.items():
            totals[phase] = totals.get(phase, 0.0) + float(seconds)
    dominant = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return {
        "dominant_phases_by_aggregate_seconds": [
            {"phase": phase, "seconds": seconds} for phase, seconds in dominant[:5]
        ],
        "interpretation": (
            "Use the largest phase as the next optimization target; support/query, "
            "canonical reduction, spool, and checkpoint timings are operational only."
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
                service, index, frozen, neighbours = _prepare(
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
        "schema": "nepflow-information-entropy-scaling-benchmark-v2",
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
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
