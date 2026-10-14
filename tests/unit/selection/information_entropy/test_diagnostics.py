"""Focused scientific closure diagnostics regression tests."""

import sys
import types

import numpy as np
import pytest

from nepflow.resources.budget import build_resource_budget
from nepflow.stages.selection.algorithms.information_entropy.diagnostics import (
    DiagnosticDistribution,
    _build_prefix_coverage_manifest,
    nearest_selected_atomic_distances,
    recompute_final_entropy_metrics,
    weighted_quantile,
)
from nepflow.stages.selection.algorithms.information_entropy.gpu_support import (
    GPUCapacityError,
    GPUCertificationError,
    GPUDeviceValidationError,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    EntropyPool,
    EntropySelectionHistory,
    EntropySelectionStep,
    SparseCandidateContributions,
    reference_weights_fingerprint,
)


def test_weighted_quantile_is_stable_by_original_row_identity() -> None:
    values = [2.0, 1.0, 1.0, 3.0]
    weights = [0.1, 0.2, 0.4, 0.3]

    assert weighted_quantile(values, 0.5, weights, original_indices=[0, 1, 2, 3]) == 1.0
    assert weighted_quantile(values, 0.99, weights, original_indices=[0, 1, 2, 3]) == 3.0


def test_final_entropy_metrics_recompute_from_p_and_q() -> None:
    probabilities = [0.25, 0.25, 0.25, 0.25]
    final_q = [0.1, 0.2, 0.3, 0.4]
    result = recompute_final_entropy_metrics(
        probabilities,
        final_q,
        beta=1.0,
        selected_count=2,
    )

    assert result["target_mass"] == pytest.approx(1.0)
    assert result["q_mass"] == pytest.approx(1.0)
    assert result["support_mass"] == pytest.approx(3.0)
    assert result["forward_kl"] == pytest.approx(
        result["cross_entropy"] - result["shannon_entropy"]
    )


def test_distribution_manifest_retains_metric_contract() -> None:
    summary = DiagnosticDistribution(
        count=2,
        minimum=1.0,
        median=1.5,
        mean=1.5,
        q95=2.0,
        q99=2.0,
        maximum=2.0,
        unit="whitened Euclidean distance",
        population="all pool rows",
        weighting="equal-candidate-weighted p_i",
        denominator="sum of p_i",
    )

    restored = DiagnosticDistribution.from_manifest(summary.to_manifest())
    assert restored == summary
    assert restored.to_manifest()["quantile_convention"]


def test_exact_coverage_tiling_matches_bounded_cpu_oracle_with_duplicates() -> None:
    descriptors = np.zeros((6, 42), dtype=np.float64)
    descriptors[3, 0] = 1.0
    descriptors[4, 0] = 2.0
    descriptors[5, 0] = 1.0
    probabilities = np.full(6, 1.0 / 6.0, dtype=np.float64)
    pool = EntropyPool(
        descriptors,
        tuple(range(6)),
        ("candidate-b", "candidate-a", "candidate-c"),
        np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        probabilities,
        "transform",
        "representation",
        "pool",
        reference_weights_fingerprint=reference_weights_fingerprint(probabilities),
    )
    budget = build_resource_budget(memory_budget_bytes=256 * 1024 * 1024)

    tiled, source_indices = nearest_selected_atomic_distances(
        pool,
        ("candidate-a",),
        backend="cpu",
        chunk_size=2,
        resource_budget=budget,
    )
    oracle, oracle_indices = nearest_selected_atomic_distances(
        pool,
        ("candidate-a",),
        backend="cpu_oracle",
        chunk_size=5,
        resource_budget=budget,
    )

    expected = np.min(
        np.linalg.norm(
            descriptors[:, None, :] - descriptors[np.array([2, 3])][None, :, :], axis=2
        ),
        axis=1,
    )
    assert source_indices.tolist() == [2, 3]
    assert oracle_indices.tolist() == [2, 3]
    np.testing.assert_array_equal(tiled, expected)
    np.testing.assert_array_equal(tiled, oracle)


def _fake_coverage_cupy(
    *,
    device_count: int = 4,
    free_bytes: int = 64 * 1024 * 1024,
    sqrt_fn: object = np.sqrt,
):
    class Device:
        def __init__(self, ordinal: int):
            self.ordinal = ordinal

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: device_count,
        memGetInfo=lambda: (free_bytes, free_bytes),
    )
    return types.SimpleNamespace(
        float64=np.float64,
        int64=np.int64,
        abs=np.abs,
        argmin=np.argmin,
        asarray=np.asarray,
        asnumpy=np.asarray,
        flatnonzero=np.flatnonzero,
        isfinite=np.isfinite,
        min=np.min,
        maximum=np.maximum,
        ones_like=np.ones_like,
        full=np.full,
        sqrt=sqrt_fn,
        sum=np.sum,
        where=np.where,
        zeros=np.zeros,
        cuda=types.SimpleNamespace(Device=Device, runtime=runtime),
    )


def _coverage_gpu_pool() -> tuple[EntropyPool, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(19)
    descriptors = generator.normal(size=(12, 42)).astype(np.float64)
    descriptors[1] = descriptors[0]
    descriptors[2] = descriptors[0] + 1.0e-14
    descriptors[3] = descriptors[4]
    descriptors[5] = descriptors[4] + 1.0e-15
    descriptors[6] = descriptors[4] + 4.0
    descriptors[7] = descriptors[4] - 4.0
    probabilities = np.full(12, 1.0 / 12.0, dtype=np.float64)
    pool = EntropyPool(
        descriptors,
        tuple(range(12)),
        ("selected", "other"),
        np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64),
        probabilities,
        "transform",
        "representation",
        "pool",
        reference_weights_fingerprint=reference_weights_fingerprint(probabilities),
    )
    source_indices = np.arange(5, dtype=np.int64)
    expected = np.sqrt(
        np.sum(
            (descriptors[:, None, :] - descriptors[source_indices][None, :, :]) ** 2,
            axis=2,
        )
    ).min(axis=1)
    return pool, source_indices, expected


@pytest.mark.parametrize("gpu_devices", [(0,), (0, 1, 2, 3)])
@pytest.mark.parametrize("gpu_tile_size", [2, 3, 5])
def test_exact_gpu_coverage_is_exact_for_42d_duplicates_and_multi_gpu_tiles(
    monkeypatch: pytest.MonkeyPatch,
    gpu_devices: tuple[int, ...],
    gpu_tile_size: int,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_coverage_cupy())
    pool, source_indices, expected = _coverage_gpu_pool()
    metrics: dict[str, object] = {}

    actual, actual_indices = nearest_selected_atomic_distances(
        pool,
        ("selected",),
        backend="exact_tiled_gpu",
        gpu_devices=gpu_devices,
        chunk_size=3,
        gpu_tile_size=gpu_tile_size,
        max_gpu_support_bytes=2 * 1024 * 1024,
        resource_budget=build_resource_budget(memory_budget_bytes=64 * 1024 * 1024),
        operational_metrics=metrics,
    )

    np.testing.assert_array_equal(actual_indices, source_indices)
    np.testing.assert_array_equal(actual, expected)
    assert metrics["backend"] == "exact_tiled_gpu"
    assert metrics["devices"] == gpu_devices
    assert metrics["gpu_evaluated_pairs"] == 2 * 12 * 5
    assert metrics["gpu_screened_pairs"] == 12 * 5
    assert metrics["gpu_distance_passes"] == 2
    assert metrics["gpu_tile_workspace_bytes"] <= metrics["gpu_tile_workspace_limit_bytes"]
    assert 0 < metrics["candidate_transfer_count"] < 2 * metrics["gpu_screened_pairs"]
    assert metrics["cpu_refined_pairs"] < metrics["gpu_screened_pairs"]
    assert metrics["cpu_certified_pairs"] == (
        metrics["cpu_seed_pairs"] + metrics["cpu_refined_pairs"]
    )


def test_exact_gpu_coverage_rejects_missing_cupy_without_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", None)
    pool, _source_indices, _expected = _coverage_gpu_pool()
    with pytest.raises(GPUDeviceValidationError, match="no CPU fallback"):
        nearest_selected_atomic_distances(
            pool,
            ("selected",),
            backend="exact_tiled_gpu",
            resource_budget=build_resource_budget(memory_budget_bytes=64 * 1024 * 1024),
        )


def test_exact_gpu_coverage_rejects_unavailable_device_and_under_budget_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_coverage_cupy(device_count=1))
    pool, _source_indices, _expected = _coverage_gpu_pool()
    with pytest.raises(GPUDeviceValidationError, match="unavailable"):
        nearest_selected_atomic_distances(
            pool,
            ("selected",),
            backend="exact_tiled_gpu",
            gpu_devices=(0, 1),
            resource_budget=build_resource_budget(memory_budget_bytes=64 * 1024 * 1024),
        )
    with pytest.raises(GPUCapacityError, match="declared device workspace"):
        nearest_selected_atomic_distances(
            pool,
            ("selected",),
            backend="exact_tiled_gpu",
            gpu_devices=(0,),
            gpu_tile_size=3,
            max_gpu_support_bytes=1,
            resource_budget=build_resource_budget(memory_budget_bytes=64 * 1024 * 1024),
        )


def test_exact_gpu_coverage_fails_closed_on_unprovable_gpu_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_sqrt(value: object) -> np.ndarray:
        return np.full(np.shape(value), np.nan, dtype=np.float64)

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        _fake_coverage_cupy(sqrt_fn=invalid_sqrt),
    )
    pool, _source_indices, _expected = _coverage_gpu_pool()
    with pytest.raises(GPUCertificationError, match="finite certified seed"):
        nearest_selected_atomic_distances(
            pool,
            ("selected",),
            backend="exact_tiled_gpu",
            gpu_devices=(0,),
            gpu_tile_size=3,
            resource_budget=build_resource_budget(memory_budget_bytes=64 * 1024 * 1024),
        )


def test_prefix_coverage_matches_sparse_union_and_preserves_signed_covered_kl() -> None:
    probabilities = np.array([0.01, 0.49, 0.49, 0.01], dtype=np.float64)
    beta = 2.0
    contributions = SparseCandidateContributions(
        candidate_indptr=np.array([0, 1, 3, 4], dtype=np.int64),
        target_indices=np.array([1, 0, 2, 3], dtype=np.int64),
        values=np.array([1.0, 0.25, 0.75, 1.0], dtype=np.float64),
        candidate_ids=("anchor", "g1", "g2"),
        graph_fingerprint="graph",
        fingerprint="contributions",
        row_count=4,
        kernel_operator_fingerprint="graph",
    )

    def metrics(selected: tuple[str, ...]) -> tuple[np.ndarray, float, float]:
        selected_sum = np.zeros(4, dtype=np.float64)
        for candidate_id in selected:
            row = contributions.candidate_row(candidate_id)
            selected_sum[row.target_indices] += row.values
        q = (beta * probabilities + selected_sum) / (beta + len(selected))
        cross_entropy = float(-np.sum(probabilities * np.log(q)))
        forward_kl = float(np.sum(probabilities * np.log(probabilities / q)))
        return q, cross_entropy, forward_kl

    _q_anchor, ce_anchor, kl_anchor = metrics(("anchor",))
    _q_g1, ce_g1, kl_g1 = metrics(("anchor", "g1"))
    q_final, ce_final, kl_final = metrics(("anchor", "g1", "g2"))
    history = EntropySelectionHistory(
        ("anchor", "g1", "g2"),
        (
            EntropySelectionStep(
                "anchor",
                None,
                0,
                "anchor",
                None,
                0.0,
                0.0,
                1,
                ce_anchor,
                kl_anchor,
                "state-anchor",
            ),
            EntropySelectionStep(
                "g1",
                None,
                1,
                "greedy",
                1.0,
                0.0,
                0.0,
                2,
                ce_g1,
                kl_g1,
                "state-g1",
            ),
            EntropySelectionStep(
                "g2",
                None,
                2,
                "greedy",
                1.0,
                0.0,
                0.0,
                3,
                ce_final,
                kl_final,
                "state-final",
            ),
        ),
    )

    manifest = _build_prefix_coverage_manifest(
        probabilities,
        beta=beta,
        history=history,
        contributions=contributions,
        final_q=q_final,
        final_forward_kl=kl_final,
        pool_fingerprint="pool",
    )
    records = manifest["records"]
    assert len(records) == 3
    assert records[0]["target_mass_covered"] == pytest.approx(0.49)
    assert records[0]["target_mass_uncovered"] == pytest.approx(0.51)
    assert records[0]["covered_row_count"] == 1
    assert records[0]["kl_uncovered"] == pytest.approx(0.51 * np.log1p(1.0 / beta))
    assert records[0]["kl_covered"] < 0.0
    assert records[1]["target_mass_covered"] == pytest.approx(0.99)
    assert records[1]["target_mass_uncovered"] == pytest.approx(0.01)
    assert records[2]["target_mass_covered"] == pytest.approx(1.0)
    assert records[2]["target_mass_uncovered"] == pytest.approx(0.0)
    for record, expected in zip(records, (kl_anchor, kl_g1, kl_final)):
        assert record["kl_total"] == pytest.approx(expected)
        assert record["kl_total"] == pytest.approx(
            record["kl_covered"] + record["kl_uncovered"]
        )


def test_prefix_coverage_labels_no_anchor_q0_baseline() -> None:
    probabilities = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    beta = 3.0
    contributions = SparseCandidateContributions(
        candidate_indptr=np.array([0, 2], dtype=np.int64),
        target_indices=np.array([0, 2], dtype=np.int64),
        values=np.array([0.4, 0.6], dtype=np.float64),
        candidate_ids=("g1",),
        graph_fingerprint="graph",
        fingerprint="contributions",
        row_count=3,
        kernel_operator_fingerprint="graph",
    )
    selected_sum = np.array([0.4, 0.0, 0.6], dtype=np.float64)
    q = (beta * probabilities + selected_sum) / (beta + 1.0)
    ce = float(-np.sum(probabilities * np.log(q)))
    kl = float(np.sum(probabilities * np.log(probabilities / q)))
    history = EntropySelectionHistory(
        ("g1",),
        (
            EntropySelectionStep(
                "g1", None, 1, "greedy", 1.0, 0.0, 0.0, 1, ce, kl, "state"
            ),
        ),
    )
    records = _build_prefix_coverage_manifest(
        probabilities,
        beta=beta,
        history=history,
        contributions=contributions,
        final_q=q,
        final_forward_kl=kl,
    )["records"]
    assert records[0]["kind"] == "baseline"
    assert records[0]["selected_count"] == 0
    assert records[0]["target_mass_covered"] == 0.0
    assert records[0]["target_mass_uncovered"] == 1.0
    assert records[0]["kl_total"] == 0.0
    assert [record["acquisition_iteration"] for record in records] == [0, 1]
