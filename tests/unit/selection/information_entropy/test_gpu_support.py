"""CPU-safe and CUDA integration evidence for exact GPU support execution."""

from __future__ import annotations

import builtins
import importlib
import sys
import threading
import types
from typing import Any

import numpy as np
import pytest

from nepflow.resources.budget import build_resource_budget
from nepflow.stages.selection.algorithms.information_entropy.bandwidth import (
    build_entropy_pool,
    calibrate_bandwidth,
)
from nepflow.stages.selection.algorithms.information_entropy.gpu_devices import (
    contiguous_gpu_ranges,
)
from nepflow.stages.selection.algorithms.information_entropy.gpu_support import (
    ExactTiledGPUSupport,
    GPUCapacityError,
    GPUDeviceValidationError,
    GPUSupportError,
)
from nepflow.stages.selection.algorithms.information_entropy.kernels import (
    build_streamed_candidate_contributions,
    evaluate_leave_one_out_objective,
    evaluate_leave_one_out_objectives,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    EntropyBandwidthSettings,
)
from nepflow.stages.selection.algorithms.information_entropy.selector import lazy_greedy


def _fake_cupy(*, device_count: int = 1, free_bytes: int = 64 * 1024 * 1024):
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
        asarray=np.asarray,
        asnumpy=np.asarray,
        copyto=np.copyto,
        zeros=np.zeros,
        ones_like=np.ones_like,
        empty=np.empty,
        abs=np.abs,
        maximum=np.maximum,
        isfinite=np.isfinite,
        flatnonzero=np.flatnonzero,
        stack=np.stack,
        sqrt=np.sqrt,
        sum=np.sum,
        cuda=types.SimpleNamespace(Device=Device, runtime=runtime),
    )


def test_gpu_support_module_is_lazy_and_cpu_import_is_cuda_free(monkeypatch: pytest.MonkeyPatch):
    module_name = "nepflow.stages.selection.algorithms.information_entropy.gpu_support"
    module = importlib.import_module(module_name)
    original_import = builtins.__import__

    def reject_cupy(name: str, *args: object, **kwargs: object):
        if name == "cupy":
            raise AssertionError("CPU import attempted to load CuPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_cupy)
    importlib.reload(module)


def test_explicit_gpu_settings_validate_without_cuda():
    with pytest.raises(ValueError, match="kernel_support_execution"):
        EntropyBandwidthSettings(kernel_support_execution="automatic_gpu")
    with pytest.raises(ValueError, match="gpu_device"):
        EntropyBandwidthSettings(kernel_support_execution="exact_tiled_gpu", gpu_device=-1)


def test_gpu_scheduler_partitions_balanced_and_empty_ranges():
    assert contiguous_gpu_ranges(7, (0, 1, 2)) == (
        (0, 0, 3),
        (1, 3, 5),
        (2, 5, 7),
    )
    assert contiguous_gpu_ranges(2, (0, 1, 2, 3)) == (
        (0, 0, 1),
        (1, 1, 2),
        (2, 2, 2),
        (3, 2, 2),
    )


def test_fake_gpu_reusable_upload_copies_bounded_tiles_without_aliasing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    provider = ExactTiledGPUSupport(
        np.zeros((4, 2), dtype=np.float64),
        tile_size=4,
        max_device_bytes=64 * 1024,
        max_host_bytes=1_000_000,
    )

    first_targets = np.ascontiguousarray(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    )
    first_view = provider._upload_reusable_tile("_gpu_target_workspace", first_targets)
    target_workspace = provider._gpu_target_workspace
    assert target_workspace is not None
    np.testing.assert_array_equal(first_view, first_targets)

    same_size = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    same_view = provider._upload_reusable_tile("_gpu_target_workspace", same_size)
    assert provider._gpu_target_workspace is target_workspace
    np.testing.assert_array_equal(same_view, same_size)

    non_contiguous = np.asarray(
        [[11.0, 12.0], [13.0, 14.0]], dtype=np.float32
    ).T
    assert not non_contiguous.flags.c_contiguous
    non_contiguous_view = provider._upload_reusable_tile(
        "_gpu_target_workspace", non_contiguous
    )
    assert provider._gpu_target_workspace is target_workspace
    np.testing.assert_array_equal(non_contiguous_view, non_contiguous)
    assert non_contiguous_view.dtype == np.dtype(np.float64)
    non_contiguous[0, 0] = -99.0
    assert non_contiguous_view[0, 0] == 11.0

    shorter = np.asarray([[9.0, 10.0]], dtype=np.float64)
    short_view = provider._upload_reusable_tile("_gpu_target_workspace", shorter)
    assert provider._gpu_target_workspace is target_workspace
    np.testing.assert_array_equal(short_view, shorter)

    larger = np.arange(12.0, dtype=np.float64).reshape(6, 2)
    large_view = provider._upload_reusable_tile("_gpu_target_workspace", larger)
    assert provider._gpu_target_workspace is not target_workspace
    np.testing.assert_array_equal(large_view, larger)

    radii = provider._upload_reusable_tile(
        "_gpu_radii_workspace", np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    )
    np.testing.assert_array_equal(radii, np.asarray([1.0, 2.0, 3.0]))
    radii_short = provider._upload_reusable_tile(
        "_gpu_radii_workspace", np.asarray([4.0], dtype=np.float64)
    )
    np.testing.assert_array_equal(radii_short, np.asarray([4.0]))
    provider.close()


@pytest.mark.cuda
def test_cuda_reusable_upload_copies_bounded_tiles() -> None:
    cp = pytest.importorskip("cupy")
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        pytest.skip("no visible CUDA device")

    provider = ExactTiledGPUSupport(
        np.zeros((4, 2), dtype=np.float64),
        tile_size=4,
        max_device_bytes=64 * 1024,
        max_host_bytes=1_000_000,
    )
    try:
        with cp.cuda.Device(provider.device):
            first = provider._upload_reusable_tile(
                "_gpu_target_workspace",
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            )
            target_workspace = provider._gpu_target_workspace
            assert target_workspace is not None
            np.testing.assert_array_equal(cp.asnumpy(first), [[1.0, 2.0], [3.0, 4.0]])
            assert int(first.device.id) == provider.device

            steady = provider._upload_reusable_tile(
                "_gpu_target_workspace",
                np.asarray([[5.0, 6.0]], dtype=np.float64),
            )
            assert provider._gpu_target_workspace is target_workspace
            np.testing.assert_array_equal(cp.asnumpy(steady), [[5.0, 6.0]])

            grown = provider._upload_reusable_tile(
                "_gpu_target_workspace",
                np.arange(12.0, dtype=np.float64).reshape(6, 2),
            )
            assert provider._gpu_target_workspace is not target_workspace
            np.testing.assert_array_equal(cp.asnumpy(grown), np.arange(12.0).reshape(6, 2))

            radii = provider._upload_reusable_tile(
                "_gpu_radii_workspace",
                np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            )
            radii_short = provider._upload_reusable_tile(
                "_gpu_radii_workspace", np.asarray([4.0], dtype=np.float64)
            )
            np.testing.assert_array_equal(cp.asnumpy(radii), [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(cp.asnumpy(radii_short), [4.0])
    finally:
        provider.close()


def test_missing_cupy_fails_explicitly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(GPUSupportError, match="requires optional CuPy"):
        ExactTiledGPUSupport(
            np.zeros((2, 2), dtype=np.float64),
            max_host_bytes=1_000_000,
        )


def test_gpu_device_and_tile_capacity_are_explicit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(device_count=1, free_bytes=1024))
    with pytest.raises(GPUDeviceValidationError, match="gpu_device=1"):
        ExactTiledGPUSupport(
            np.zeros((2, 2), dtype=np.float64),
            device=1,
            max_host_bytes=1_000_000,
        )
    with pytest.raises(GPUCapacityError, match="at least 40 bytes"):
        ExactTiledGPUSupport(
            np.zeros((2, 2), dtype=np.float64),
            max_device_bytes=39,
            max_host_bytes=1_000_000,
        )


@pytest.mark.parametrize("device_count", (1, 2, 4))
def test_fake_gpu_support_initializes_each_requested_device_capacity(
    monkeypatch: pytest.MonkeyPatch, device_count: int
):
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        _fake_cupy(device_count=device_count, free_bytes=64 * 1024 * 1024),
    )
    devices = tuple(range(device_count))
    provider = ExactTiledGPUSupport(
        np.zeros((4, 2), dtype=np.float64),
        gpu_devices=devices,
        tile_size=2,
        max_device_bytes=64 * 1024,
        max_host_bytes=1_000_000,
    )
    metrics = provider.operational_metrics()
    assert metrics["gpu_devices"] == list(devices)
    assert metrics["gpu_device_count"] == device_count
    assert metrics["gpu_total_bytes"] == 64 * 1024 * 1024
    assert all(item["gpu_total_bytes"] == 64 * 1024 * 1024 for item in metrics["gpu_per_device"])
    provider.close()


@pytest.mark.parametrize("device_count", (1, 2, 4))
def test_fake_gpu_support_rejects_insufficient_capacity_on_every_requested_device(
    monkeypatch: pytest.MonkeyPatch, device_count: int
):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(device_count=device_count))
    with pytest.raises(GPUCapacityError, match="at least 40 bytes"):
        ExactTiledGPUSupport(
            np.zeros((4, 2), dtype=np.float64),
            gpu_devices=tuple(range(device_count)),
            max_device_bytes=39,
            max_host_bytes=1_000_000,
        )


def test_fake_gpu_support_rejects_one_unavailable_requested_ordinal(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(device_count=4))
    with pytest.raises(GPUDeviceValidationError, match=r"requested=\(0, 1, 4\)"):
        ExactTiledGPUSupport(
            np.zeros((4, 2), dtype=np.float64),
            gpu_devices=(0, 1, 4),
            max_host_bytes=1_000_000,
        )


def test_fake_gpu_support_preserves_strict_ulp_duplicates_and_original_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    descriptors = np.asarray([[0.0], [1.0], [1.0], [2.0]], dtype=np.float64)
    radii = np.asarray(
        [np.nextafter(1.0, 0.0), 1.0, np.nextafter(1.0, 2.0), 2.0],
        dtype=np.float64,
    )
    provider = ExactTiledGPUSupport(
        descriptors,
        tile_size=2,
        max_device_bytes=64 * 1024,
        max_host_bytes=1_000_000,
    )
    offsets, targets, distances = provider.support_many(np.arange(4), radii)
    expected_targets: list[int] = []
    expected_distances: list[float] = []
    expected_offsets = [0]
    for source, radius in enumerate(radii):
        exact = np.sqrt(np.sum((descriptors - descriptors[source]) ** 2, axis=1))
        keep = exact < radius
        expected_targets.extend(np.flatnonzero(keep).tolist())
        expected_distances.extend(exact[keep].tolist())
        expected_offsets.append(len(expected_targets))
    np.testing.assert_array_equal(offsets, np.asarray(expected_offsets, dtype=np.int64))
    np.testing.assert_array_equal(targets, np.asarray(expected_targets, dtype=np.int64))
    np.testing.assert_array_equal(distances, np.asarray(expected_distances, dtype=np.float64))
    assert provider.operational_metrics()["gpu_cpu_rescanned_count"] >= len(expected_targets)


def test_fake_gpu_support_shared_radius_matrix_matches_isolated_queries(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    descriptors = np.asarray([[0.0], [1.0], [1.0], [2.0]], dtype=np.float64)
    sources = np.arange(descriptors.shape[0], dtype=np.int64)
    radii = np.asarray(
        [
            [np.nextafter(1.0, 0.0)] * sources.size,
            [1.0] * sources.size,
            [np.nextafter(1.0, 2.0)] * sources.size,
            [2.0] * sources.size,
        ],
        dtype=np.float64,
    )
    shared = ExactTiledGPUSupport(
        descriptors,
        tile_size=2,
        max_device_bytes=64 * 1024,
        max_host_bytes=1_000_000,
    )
    offsets, targets, distances = shared.support_many(sources, radii)
    for candidate_index in range(radii.shape[0]):
        for source_index in range(sources.size):
            begin = int(offsets[source_index])
            end = int(offsets[source_index + 1])
            keep = distances[begin:end] < radii[candidate_index, source_index]
            exact = np.sqrt(
                np.sum((descriptors - descriptors[source_index]) ** 2, axis=1)
            )
            expected = np.flatnonzero(exact < radii[candidate_index, source_index])
            np.testing.assert_array_equal(targets[begin:end][keep], expected)
    metrics = shared.operational_metrics()
    assert metrics["gpu_grouped_radius_query_count"] == 1
    assert metrics["gpu_reused_radius_scans"] == radii.shape[0] - 1
    assert metrics["gpu_launch_count"] == metrics["gpu_d2h_transfer_count"]
    assert metrics["gpu_full_tile_d2h_bytes"] == 0
    assert metrics["gpu_compact_candidate_d2h_bytes"] == metrics["gpu_d2h_bytes"]
    assert metrics["gpu_compact_candidate_d2h_bytes"] > 0
    assert metrics["gpu_distance_block_count"] > 0
    assert metrics["gpu_kernel_launch_equivalent_count"] > 0
    assert metrics["gpu_distance_block_width"] == 1
    assert metrics["gpu_active_kernel_seconds_sum"] >= 0.0
    shared.close()


def test_fake_gpu_calibration_planner_reports_shared_and_singleton_choices(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    descriptors = np.asarray(
        [[float(index), float(index % 2)] for index in range(8)], dtype=np.float64
    )
    pool = build_entropy_pool(
        descriptors,
        candidate_ids=("a", "b", "c", "d"),
        row_candidate_ids=("a", "a", "b", "b", "c", "c", "d", "d"),
    )
    common = dict(
        backend="exact_tiled_cpu",
        kernel_support_execution="exact_tiled_gpu",
        k_candidates=(1,),
        c_candidates=(1.5, 2.0, 4.0),
        chunk_size=2,
        max_calibration_work_bytes=1_000_000,
        max_radius_query_bytes=1_000_000,
        max_gpu_support_bytes=64 * 1024,
        gpu_tile_size=2,
    )
    shared = calibrate_bandwidth(
        pool,
        EntropyBandwidthSettings(**common, calibration_batch_size=3),
    )
    singleton = calibrate_bandwidth(
        pool,
        EntropyBandwidthSettings(**common, calibration_batch_size=1),
    )
    assert any(len(item["c_values"]) > 1 for item in shared.operational_metrics["c_batch_groups"])
    assert shared.operational_metrics["gpu_reused_radius_scans"] > 0
    assert all(
        len(item["c_values"]) == 1 for item in singleton.operational_metrics["c_batch_groups"]
    )
    assert singleton.operational_metrics["c_batch_reused_scans"] == 0


def test_fake_multi_gpu_support_merges_contiguous_ranges_deterministically(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(device_count=4))
    descriptors = np.asarray([[float(i), float(i % 3)] for i in range(12)], dtype=np.float64)
    sources = np.arange(descriptors.shape[0], dtype=np.int64)
    radii = np.full(sources.size, 3.5, dtype=np.float64)
    single = ExactTiledGPUSupport(
        descriptors,
        gpu_devices=(0,),
        tile_size=3,
        max_device_bytes=64 * 1024,
        max_host_bytes=1_000_000,
    )
    multi = ExactTiledGPUSupport(
        descriptors,
        gpu_devices=(0, 1, 2, 3),
        tile_size=3,
        max_device_bytes=64 * 1024,
        max_host_bytes=1_000_000,
    )
    barrier = threading.Barrier(4)
    original_worker_query = ExactTiledGPUSupport._support_many_on_device

    def synchronized_worker_query(self: object, *args: object, **kwargs: object):
        barrier.wait(timeout=5.0)
        return original_worker_query(self, *args, **kwargs)

    monkeypatch.setattr(ExactTiledGPUSupport, "_support_many_on_device", synchronized_worker_query)
    expected = single.support_many(sources, radii)
    actual = multi.support_many(sources, radii)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)
    metrics = multi.operational_metrics()
    assert metrics["gpu_device_count"] == 4
    assert metrics["gpu_wall_seconds"] > 0.0
    assert metrics["gpu_active_seconds_sum"] > 0.0
    assert len(metrics["gpu_per_device"]) == 4
    single.close()
    multi.close()


def test_scalar_loo_diagnostics_report_the_requested_gpu_devices(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(device_count=2))
    descriptors = np.asarray([[float(i), float(i % 2)] for i in range(6)], dtype=np.float64)
    probabilities = np.full(6, 1.0 / 6.0, dtype=np.float64)
    diagnostics: dict[str, Any] = {}
    budget = build_resource_budget(
        execution_mode="explicit",
        memory_budget_bytes=64 * 1024 * 1024,
        scratch_budget_bytes=8 * 1024 * 1024,
        safety_margin_fraction=0.0,
    )
    outcome = evaluate_leave_one_out_objective(
        descriptors,
        probabilities,
        np.full(6, 2.0, dtype=np.float64),
        chunk_size=2,
        max_radius_query_bytes=1_000_000,
        resource_budget=budget,
        kernel_support_execution="exact_tiled_gpu",
        gpu_devices=(0, 1),
        max_gpu_support_bytes=64 * 1024,
        diagnostics=diagnostics,
    )
    assert outcome.valid
    assert diagnostics["gpu_devices"] == [0, 1]
    assert len(diagnostics["gpu_per_device"]) == 2


def test_gpu_mode_checkpoint_resume_keeps_execution_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    descriptors = np.asarray([[float(i), float(i % 2)] for i in range(6)], dtype=np.float64)
    probabilities = np.full(6, 1.0 / 6.0, dtype=np.float64)
    bandwidths = np.full(6, 2.0, dtype=np.float64)
    budget = build_resource_budget(
        execution_mode="explicit",
        memory_budget_bytes=64 * 1024 * 1024,
        scratch_budget_bytes=8 * 1024 * 1024,
        safety_margin_fraction=0.0,
    )
    checkpoint = tmp_path / "loo"

    def interrupt(completed: int, _total: int) -> None:
        if completed >= 3:
            raise RuntimeError("test interruption")

    with pytest.raises(RuntimeError, match="test interruption"):
        evaluate_leave_one_out_objectives(
            descriptors,
            probabilities,
            bandwidths.reshape(1, -1),
            chunk_size=2,
            progress_callback=interrupt,
            max_radius_query_bytes=1_000_000,
            max_calibration_work_bytes=4_000_000,
            resource_budget=budget,
            checkpoint_path=checkpoint,
            checkpoint_every_sources=2,
            kernel_support_execution="exact_tiled_gpu",
            max_gpu_support_bytes=64 * 1024,
        )
    resumed = evaluate_leave_one_out_objectives(
        descriptors,
        probabilities,
        bandwidths.reshape(1, -1),
        chunk_size=2,
        max_radius_query_bytes=1_000_000,
        max_calibration_work_bytes=4_000_000,
        resource_budget=budget,
        checkpoint_path=checkpoint,
        checkpoint_every_sources=2,
        kernel_support_execution="exact_tiled_gpu",
        max_gpu_support_bytes=64 * 1024,
    )[0]
    reference = evaluate_leave_one_out_objectives(
        descriptors,
        probabilities,
        bandwidths.reshape(1, -1),
        chunk_size=2,
        max_radius_query_bytes=1_000_000,
        max_calibration_work_bytes=4_000_000,
        resource_budget=budget,
        kernel_support_execution="exact_tiled_gpu",
        max_gpu_support_bytes=64 * 1024,
    )[0]
    assert resumed.valid == reference.valid
    assert resumed.objective == reference.objective
    np.testing.assert_array_equal(resumed.probabilities, reference.probabilities)


@pytest.mark.cuda
def test_cuda_support_and_complete_calibration_match_cpu():
    cp = pytest.importorskip("cupy")
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        pytest.skip("no visible CUDA device")
    descriptors = np.asarray(
        [[float(i), float(i % 3), float((i * 7) % 5)] for i in range(20)],
        dtype=np.float64,
    )
    pool = build_entropy_pool(
        descriptors,
        candidate_ids=tuple(f"c{i}" for i in range(5)),
        row_candidate_ids=tuple(f"c{i // 4}" for i in range(20)),
    )
    budget = build_resource_budget(
        execution_mode="explicit",
        memory_budget_bytes=256 * 1024 * 1024,
        scratch_budget_bytes=64 * 1024 * 1024,
        safety_margin_fraction=0.0,
    )
    cpu_settings = EntropyBandwidthSettings(
        backend="exact_indexed_cpu",
        kernel_support_execution="cpu",
        chunk_size=4,
    )
    gpu_settings = EntropyBandwidthSettings(
        backend="exact_indexed_cpu",
        kernel_support_execution="exact_tiled_gpu",
        gpu_device=0,
        gpu_tile_size=4,
        max_gpu_support_bytes=32 * 1024 * 1024,
        chunk_size=4,
    )
    cpu = calibrate_bandwidth(
        pool,
        cpu_settings,
        resource_budget=budget,
        max_radius_query_bytes=4 * 1024 * 1024,
    )
    gpu = calibrate_bandwidth(
        pool,
        gpu_settings,
        resource_budget=budget,
        max_radius_query_bytes=4 * 1024 * 1024,
    )
    np.testing.assert_array_equal(cpu.selected.bandwidths, gpu.selected.bandwidths)
    assert cpu.selected.k == gpu.selected.k
    assert cpu.selected.c == gpu.selected.c
    assert cpu.calibration_fingerprint == gpu.calibration_fingerprint
    assert cpu.evaluation_count == gpu.evaluation_count == 16
    for cpu_attempt, gpu_attempt in zip(cpu.attempts, gpu.attempts, strict=True):
        assert cpu_attempt.k == gpu_attempt.k
        assert cpu_attempt.c == gpu_attempt.c
        assert cpu_attempt.status == gpu_attempt.status
        assert cpu_attempt.reason == gpu_attempt.reason
        if cpu_attempt.objective is None or gpu_attempt.objective is None:
            assert cpu_attempt.objective == gpu_attempt.objective
        else:
            assert cpu_attempt.objective == gpu_attempt.objective
    cpu_contributions, _ = build_streamed_candidate_contributions(
        pool,
        cpu.selected,
        chunk_size=4,
        max_radius_query_bytes=4 * 1024 * 1024,
        resource_budget=budget,
    )
    gpu_contributions, gpu_summary = build_streamed_candidate_contributions(
        pool,
        gpu.selected,
        chunk_size=4,
        max_radius_query_bytes=4 * 1024 * 1024,
        max_gpu_support_bytes=32 * 1024 * 1024,
        resource_budget=budget,
        kernel_support_execution="exact_tiled_gpu",
    )
    np.testing.assert_array_equal(
        cpu_contributions.candidate_indptr, gpu_contributions.candidate_indptr
    )
    np.testing.assert_array_equal(
        cpu_contributions.target_indices, gpu_contributions.target_indices
    )
    np.testing.assert_array_equal(cpu_contributions.values, gpu_contributions.values)
    assert cpu_contributions.fingerprint == gpu_contributions.fingerprint
    cpu_selection = lazy_greedy(pool, cpu_contributions, beta=1.0, budget=3)
    gpu_selection = lazy_greedy(pool, gpu_contributions, beta=1.0, budget=3)
    assert cpu_selection.selected_candidate_ids == gpu_selection.selected_candidate_ids
    assert cpu_selection.objective_history == gpu_selection.objective_history
    assert cpu_selection.marginal_gains == gpu_selection.marginal_gains
    assert gpu_summary.support_execution == "exact_tiled_gpu"
    assert gpu_summary.operational_metrics["gpu_fp64_evaluations"] > 0
