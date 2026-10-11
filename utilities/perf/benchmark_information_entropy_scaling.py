"""Repeatable scaling benchmark for the exact information-entropy pipeline.

Run from the repository root, for example::

    python utilities/perf/benchmark_information_entropy_scaling.py --rows 2000 --dimension 8

The benchmark reports work counters and peak Python allocation as well as
elapsed time.  It intentionally never constructs the dense kernel oracle.
Results are measurements only and are not used in scientific identities.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc

import numpy as np

from nepflow.stages.selection.algorithms.information_entropy.bandwidth import (
    build_entropy_pool,
    calibrate_bandwidth,
)
from nepflow.stages.selection.algorithms.information_entropy.kernels import (
    aggregate_candidate_contributions,
    build_sparse_atomic_kernel_graph,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    EntropyBandwidthSettings,
)
from nepflow.stages.selection.algorithms.information_entropy.neighbours import (
    build_neighbour_index,
    compute_indexed_cpu_neighbours_many,
    compute_neighbours,
    compute_tiled_cpu_neighbours_many,
)
from nepflow.stages.selection.algorithms.information_entropy.selector import lazy_greedy


def _timed(label: str, callback):
    started = time.perf_counter()
    result = callback()
    elapsed = time.perf_counter() - started
    print(f"{label}: seconds={elapsed:.6f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--dimension", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--backend", default="exact_indexed_cpu")
    args = parser.parse_args()
    if args.rows < 2 or args.dimension < 1 or args.candidates < 1:
        parser.error("rows must be >= 2, dimension and candidates must be positive")

    rng = np.random.default_rng(20261010)
    unique_count = min(args.rows, max(9, int(0.8 * args.rows)))
    unique_descriptors = rng.normal(size=(unique_count, args.dimension)).astype(np.float64)
    descriptors = unique_descriptors[np.arange(args.rows, dtype=np.int64) % unique_count]
    neighbour_orders = tuple(order for order in (1, 2, 4, 8) if order < unique_count)
    owner_indices = np.arange(args.rows, dtype=np.int64) % args.candidates
    candidate_ids = tuple(f"candidate-{index:05d}" for index in range(args.candidates))
    owners = [candidate_ids[int(index)] for index in owner_indices]
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=owners,
        candidate_ids=candidate_ids,
    )
    print(
        f"workload: N={len(pool.descriptors)}, M={len(pool.candidate_ids)}, "
        f"d'={pool.descriptors.shape[1]}, unique_locations={unique_count}, backend={args.backend}"
    )
    tracemalloc.start()
    disabled_metrics = {}

    index = _timed(
        "index_build",
        lambda: build_neighbour_index(pool.descriptors, backend=args.backend),
    )
    if index is not None:
        _timed(
            "neighbour_query_k_max",
            lambda: (
                compute_indexed_cpu_neighbours_many(
                    pool.descriptors,
                    neighbour_orders,
                    index=index,
                    representation_fingerprint=pool.representation_fingerprint,
                )
                if args.backend == "exact_indexed_cpu"
                else compute_tiled_cpu_neighbours_many(
                    pool.descriptors,
                    neighbour_orders,
                    index=index,
                    representation_fingerprint=pool.representation_fingerprint,
                )
            ),
        )
        _timed(
            "neighbour_query_cached_prefixes",
            lambda: (
                compute_indexed_cpu_neighbours_many(
                    pool.descriptors,
                    neighbour_orders,
                    index=index,
                    representation_fingerprint=pool.representation_fingerprint,
                )
                if args.backend == "exact_indexed_cpu"
                else compute_tiled_cpu_neighbours_many(
                    pool.descriptors,
                    neighbour_orders,
                    index=index,
                    representation_fingerprint=pool.representation_fingerprint,
                )
            ),
        )
        disabled_index = _timed(
            "index_build_cache_disabled",
            lambda: build_neighbour_index(
                pool.descriptors,
                backend=args.backend,
                max_neighbour_cache_bytes=1,
            ),
        )
        _timed(
            "neighbour_query_cache_disabled",
            lambda: (
                compute_indexed_cpu_neighbours_many(
                    pool.descriptors,
                    neighbour_orders,
                    index=disabled_index,
                    representation_fingerprint=pool.representation_fingerprint,
                )
                if args.backend == "exact_indexed_cpu"
                else compute_tiled_cpu_neighbours_many(
                    pool.descriptors,
                    neighbour_orders,
                    index=disabled_index,
                    representation_fingerprint=pool.representation_fingerprint,
                )
            ),
        )
        disabled_metrics = disabled_index.memory_metrics()
        distinct_locations = int(index.unique_locations.shape[0])
    else:
        neighbour_result = _timed(
            "neighbour_query",
            lambda: compute_neighbours(
                pool.descriptors,
                1,
                backend=args.backend,
                index=index,
                representation_fingerprint=pool.representation_fingerprint,
            ),
        )
        distinct_locations = int(neighbour_result.unique_locations.shape[0])
    print(
        f"neighbours: distinct_locations={distinct_locations}, "
        f"index_bytes={getattr(index, 'index_bytes', 0)}"
    )
    index_metrics = index.memory_metrics() if index is not None else {}
    print(
        "index_metrics: "
        f"grouping_seconds={index_metrics.get('grouping_build_seconds', 'unavailable')}, "
        f"grouping_bytes={index_metrics.get('grouping_bytes', 'unavailable')}, "
        f"tree_seconds={index_metrics.get('tree_build_seconds', 'unavailable')}, "
        f"radius_queries={index_metrics.get('radius_queries', 'unavailable')}, "
        f"index_queries={index_metrics.get('index_queries', 'unavailable')}, "
        f"unique_distance_evaluations="
        f"{index_metrics.get('unique_distance_evaluations', 'unavailable')}, "
        f"atomic_distance_evaluations="
        f"{index_metrics.get('atomic_distance_evaluations', 'unavailable')}, "
        f"cache_hit={index_metrics.get('cache_hit', 'unavailable')}, "
        f"cache_miss={index_metrics.get('cache_miss', 'unavailable')}, "
        f"cache_eviction={index_metrics.get('cache_eviction', 'unavailable')}, "
        f"cache_admission_rejected="
        f"{index_metrics.get('cache_admission_rejected', 'unavailable')}, "
        f"cache_bytes={index_metrics.get('cache_bytes', 'unavailable')}, "
        f"cache_peak_bytes={index_metrics.get('cache_peak_bytes', 'unavailable')}, "
        f"cache_time_saved_seconds="
        f"{index_metrics.get('cache_time_saved_seconds', 'unavailable')}, "
        f"disabled_cache_hit={disabled_metrics.get('cache_hit', 'unavailable')}, "
        f"disabled_cache_miss={disabled_metrics.get('cache_miss', 'unavailable')}, "
        f"disabled_cache_admission_rejected="
        f"{disabled_metrics.get('cache_admission_rejected', 'unavailable')}"
    )
    settings = EntropyBandwidthSettings(
        mode="automatic",
        k_candidates=neighbour_orders,
        c_candidates=(1.5, 2.0, 4.0, 8.0),
        backend=args.backend,
    )
    calibration = _timed("calibration_loo", lambda: calibrate_bandwidth(pool, settings))
    print(
        f"calibration: c_count={len(settings.c_candidates)}, "
        f"configured_c_batch_size={settings.calibration_batch_size or 'automatic'}, "
        f"selected_k={calibration.selected.k}, selected_c={calibration.selected.c}"
    )
    graph = _timed("sparse_graph", lambda: build_sparse_atomic_kernel_graph(pool, calibration))
    contributions = _timed(
        "candidate_aggregation", lambda: aggregate_candidate_contributions(graph)
    )
    selection = _timed(
        "lazy_greedy",
        lambda: lazy_greedy(pool, contributions, beta=1.0, budget=min(10, len(candidate_ids))),
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    support = graph.support_sizes
    print(
        f"work: mean_support={float(np.mean(support)):.6f}, max_support={int(np.max(support))}, "
        f"edges={graph.edge_count}, candidate_entries={contributions.entry_count}, "
        f"graph_array_bytes={graph.array_bytes}, candidate_array_bytes={contributions.array_bytes}, "
        f"selected={len(selection.selected_ids)}, peak_python_bytes={peak}"
    )


if __name__ == "__main__":
    main()
