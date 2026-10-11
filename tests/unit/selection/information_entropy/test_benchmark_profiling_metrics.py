"""Regression tests for the issue #154 entropy profiling diagnostics."""

import runpy
from pathlib import Path
from typing import Any

from nepflow.stages.selection.algorithms.information_entropy.bandwidth import (
    build_entropy_pool,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    INDEXED_NEIGHBOUR_BACKEND_ID,
)

_BENCHMARK: dict[str, Any] = runpy.run_path(
    str(
        Path(__file__).resolve().parents[4]
        / "utilities"
        / "perf"
        / "benchmark_information_entropy_scaling.py"
    )
)


def _prepare_fixture(*, checkpoint_enabled: bool) -> tuple[Any, Any, Any, Any, Any, Any]:
    descriptors, candidate_ids, owners = _BENCHMARK["_fixture"](
        32, 42, 8, "low_duplication", 8
    )
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=owners,
        candidate_ids=candidate_ids,
    )
    service, index, frozen, neighbours, preparation = _BENCHMARK["_prepare"](
        pool,
        INDEXED_NEIGHBOUR_BACKEND_ID,
        1,
        k_max=8,
        c=1.5,
        chunk_size=16,
        memory_bytes=None,
        scratch_bytes=None,
        max_radius_query_bytes=None,
        max_index_bytes=None,
        max_neighbour_cache_bytes=None,
        checkpoint_enabled=checkpoint_enabled,
    )
    return pool, service, index, frozen, neighbours, preparation


def test_knn_checkpoint_initial_write_metrics_survive_diagnostic_restart() -> None:
    _, _, index, _, _, preparation = _prepare_fixture(checkpoint_enabled=True)
    try:
        checkpoint = preparation["knn_checkpoint"]
        assert checkpoint["enabled"] is True
        assert checkpoint["array_bytes_written"] > 0
        assert checkpoint["physical_bytes_written"] >= checkpoint["array_bytes_written"]
        assert checkpoint["chunk_count"] == 2
        assert checkpoint["restart_cost_measured"] is True
        assert checkpoint["restart_cost_seconds"] >= 0.0
        assert checkpoint["restart_bytes_read"] > 0
        assert preparation["knn_wall_seconds"] > 0.0
    finally:
        if index is not None:
            index.close()


def test_bandwidth_value_equivalence_is_not_reported_as_fingerprint_equality() -> None:
    pool, service, index, frozen, neighbours, _ = _prepare_fixture(
        checkpoint_enabled=False
    )
    try:
        result = _BENCHMARK["_run_stream"](
            pool,
            frozen,
            index,
            service,
            chunk_size=16,
            max_entries=100_000,
            max_radius_query_bytes=None,
            checkpoint_enabled=False,
            label="fingerprint-diagnostics",
        )
        correctness = _BENCHMARK["_correctness_checks"](
            pool,
            frozen,
            neighbours,
            result["contributions"],
            chunk_size=16,
            oracle_limit=64,
            max_radius_query_bytes=None,
            backend=INDEXED_NEIGHBOUR_BACKEND_ID,
            index=index,
            resource_budget=service,
        )
        grid = correctness["calibration_grid"]
        assert correctness["status"] == "passed"
        assert grid["all_16_pairs_passed"] is True
        assert "fingerprint_comparison_semantics" in grid
        for pair in grid["pairs"]:
            assert pair["scientific_values_equal"] is pair["bandwidth_arrays_equal"]
            assert pair["bandwidth_fingerprint_strings_equal"] == (
                pair["production_bandwidth_fingerprint"]
                == pair["reference_bandwidth_fingerprint"]
            )
            assert "scientific_fingerprint_match" not in pair
        assert any(
            pair["scientific_values_equal"]
            and not pair["bandwidth_fingerprint_strings_equal"]
            for pair in grid["pairs"]
        )
    finally:
        if index is not None:
            index.close()
