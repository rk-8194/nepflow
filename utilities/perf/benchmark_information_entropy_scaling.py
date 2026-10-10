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
    compute_neighbours,
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
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--backend", default="exact_indexed_cpu")
    args = parser.parse_args()
    if args.rows < 2 or args.dimension < 1 or args.candidates < 1:
        parser.error("rows must be >= 2, dimension and candidates must be positive")

    rng = np.random.default_rng(20261010)
    descriptors = rng.normal(size=(args.rows, args.dimension)).astype(np.float64)
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
        f"d'={pool.descriptors.shape[1]}, backend={args.backend}"
    )
    tracemalloc.start()

    index = _timed(
        "index_build",
        lambda: build_neighbour_index(pool.descriptors, backend=args.backend),
    )
    neighbours = _timed(
        "neighbour_query",
        lambda: compute_neighbours(
            pool.descriptors,
            1,
            backend=args.backend,
            index=index,
            representation_fingerprint=pool.representation_fingerprint,
        ),
    )
    print(
        f"neighbours: distinct_locations={len(neighbours.unique_locations)}, "
        f"index_bytes={getattr(index, 'index_bytes', 0)}"
    )
    settings = EntropyBandwidthSettings(
        mode="automatic",
        k_candidates=(1, 2),
        c_candidates=(1.5, 2.0),
        backend=args.backend,
    )
    calibration = _timed("calibration_loo", lambda: calibrate_bandwidth(pool, settings))
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
