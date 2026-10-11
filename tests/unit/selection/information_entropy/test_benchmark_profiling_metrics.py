"""Regression tests for the issue #154 entropy profiling diagnostics."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

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
            kernel_support_execution="cpu",
            gpu_device=0,
            gpu_tile_size=128,
            max_gpu_support_bytes=None,
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
            kernel_support_execution="cpu",
            gpu_device=0,
            gpu_tile_size=128,
            max_gpu_support_bytes=None,
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


def test_large_oracle_skip_still_times_production_sixteen_pair_calibration() -> None:
    pool, service, index, frozen, neighbours, _ = _prepare_fixture(checkpoint_enabled=False)
    try:
        checks = _BENCHMARK["_correctness_checks"](
            pool,
            frozen,
            neighbours,
            None,
            chunk_size=16,
            oracle_limit=1,
            max_radius_query_bytes=None,
            backend=INDEXED_NEIGHBOUR_BACKEND_ID,
            index=index,
            resource_budget=service,
            kernel_support_execution="cpu",
            gpu_device=0,
            gpu_tile_size=128,
            max_gpu_support_bytes=None,
        )
        assert checks["status"] == "skipped"
        assert checks["calibration_wall_seconds"] > 0.0
        timing = checks["calibration_grid"]["production_timing"]
        assert timing["pair_count"] == 16
        assert timing["status"] == "measured"
    finally:
        if index is not None:
            index.close()


def test_knn_only_mode_skips_contributions_and_loo(monkeypatch: Any, tmp_path: Path) -> None:
    pool = SimpleNamespace(
        descriptors=np.zeros((4, 42), dtype=np.float64),
        candidate_ids=("candidate-0", "candidate-1"),
        fingerprint="pool-fingerprint",
    )
    frozen = SimpleNamespace(fingerprint="frozen-fingerprint")
    neighbours = SimpleNamespace(
        neighbour_indices=np.zeros((4, 1), dtype=np.int64),
        unique_locations=np.zeros((4, 42), dtype=np.float64),
    )
    index = SimpleNamespace(close=lambda: None)
    preparation = {
        "wall_seconds": 0.25,
        "knn_wall_seconds": 0.2,
        "stage_wall_seconds": {
            "index_grouping_and_tree_construction": 0.05,
            "exact_knn_query": 0.2,
            "frozen_bandwidth_materialization": 0.01,
        },
        "exclusive_phase_seconds": {"knn_tree_query": 0.2},
        "knn_phase_counters": {},
        "knn_per_run_metrics": {},
        "knn_checkpoint": {},
    }

    monkeypatch.setitem(_BENCHMARK, "_fixture", lambda *args: (None, (), ()))
    monkeypatch.setitem(_BENCHMARK, "build_entropy_pool", lambda *args, **kwargs: pool)
    prepare_calls: list[dict[str, Any]] = []

    def fake_prepare(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        prepare_calls.append(kwargs)
        return SimpleNamespace(), index, frozen, neighbours, preparation

    monkeypatch.setitem(
        _BENCHMARK,
        "_prepare",
        fake_prepare,
    )

    def unexpected_stage(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("kNN-only mode invoked a full-selection stage")

    monkeypatch.setitem(_BENCHMARK, "_run_stream", unexpected_stage)
    monkeypatch.setitem(_BENCHMARK, "_correctness_checks", unexpected_stage)
    output_path = tmp_path / "knn-only.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_information_entropy_scaling.py",
            "--rows",
            "4",
            "--dimension",
            "42",
            "--candidates",
            "2",
            "--backend",
            INDEXED_NEIGHBOUR_BACKEND_ID,
            "--workers",
            "1",
            "--k-max",
            "1",
            "--knn-only",
            "--output",
            str(output_path),
        ],
    )

    _BENCHMARK["main"]()
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["fixture"]["benchmark_mode"] == "knn-only"
    assert output["fixture"]["knn_only"] is True
    assert len(output["results"]) == 2
    assert all(call["materialize_frozen"] is False for call in prepare_calls)
    for result in output["results"]:
        assert result["knn_only"] is True
        assert result["contribution_checkpoint_enabled"] is False
        assert result["correctness"]["status"] == "not_run"
        assert result["result"]["knn_only_wall_seconds"] == 0.2
        assert "streamed_contribution_wall_seconds" not in result["result"]
    assert output["profiling_summary"]["stage_wall_seconds"]["loo_calibration"] == 0.0
