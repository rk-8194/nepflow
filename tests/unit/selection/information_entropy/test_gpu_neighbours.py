"""CPU-safe fake-CuPy tests and separately marked CUDA evidence for GPU kNN."""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nepflow.resources.budget import ResourceCapacityError, build_resource_budget
from nepflow.stages.selection.algorithms.information_entropy.checkpointing import (
    IncrementalCheckpointStore,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    GPU_NEIGHBOUR_BACKEND_ID,
    EntropyBandwidthSettings,
)
from nepflow.stages.selection.algorithms.information_entropy.neighbours import (
    compute_exact_neighbours,
    compute_neighbours,
    neighbour_backend_capabilities,
)


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
        int64=np.int64,
        asarray=np.asarray,
        asnumpy=np.asarray,
        zeros=np.zeros,
        full=np.full,
        sqrt=np.sqrt,
        sum=np.sum,
        max=np.max,
        minimum=np.minimum,
        concatenate=np.concatenate,
        where=np.where,
        nan=np.nan,
        isfinite=np.isfinite,
        all=np.all,
        argpartition=np.argpartition,
        take_along_axis=np.take_along_axis,
        cuda=types.SimpleNamespace(Device=Device, runtime=runtime),
        get_default_memory_pool=lambda: types.SimpleNamespace(free_all_blocks=lambda: None),
        get_default_pinned_memory_pool=lambda: types.SimpleNamespace(free_all_blocks=lambda: None),
    )


def _budget():
    return build_resource_budget(
        execution_mode="explicit",
        memory_budget_bytes=128 * 1024 * 1024,
        scratch_budget_bytes=16 * 1024 * 1024,
        safety_margin_fraction=0.0,
    )


def test_gpu_backend_is_explicit_and_has_distinct_capability_identity() -> None:
    settings = EntropyBandwidthSettings(backend=GPU_NEIGHBOUR_BACKEND_ID)
    assert settings.backend == GPU_NEIGHBOUR_BACKEND_ID
    capabilities = neighbour_backend_capabilities(GPU_NEIGHBOUR_BACKEND_ID)
    assert capabilities.exact
    assert capabilities.supports_radius_support
    assert (
        EntropyBandwidthSettings(
            backend=GPU_NEIGHBOUR_BACKEND_ID, kernel_support_execution="cpu"
        ).kernel_support_execution
        == "cpu"
    )


def test_gpu_knn_missing_cupy_fails_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(RuntimeError, match="requires optional CuPy"):
        compute_neighbours(
            np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64),
            1,
            backend=GPU_NEIGHBOUR_BACKEND_ID,
            resource_budget=_budget(),
        )


def test_fake_gpu_knn_matches_exhaustive_oracle_for_duplicates_and_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    descriptors = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0e-14, 0.0],
            [3.0, -1.0e-14, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    budget = _budget()
    for k in (1, 2, 4, 8):
        reference = compute_exact_neighbours(
            descriptors,
            k,
            chunk_size=2,
            resource_budget=budget,
        )
        actual = compute_neighbours(
            descriptors,
            k,
            backend=GPU_NEIGHBOUR_BACKEND_ID,
            chunk_size=2,
            gpu_device=0,
            gpu_tile_size=4,
            max_gpu_neighbour_bytes=32 * 1024,
            resource_budget=budget,
        )
        np.testing.assert_array_equal(actual.neighbour_indices, reference.neighbour_indices)
        np.testing.assert_array_equal(actual.neighbour_distances, reference.neighbour_distances)
        np.testing.assert_array_equal(actual.radii, reference.radii)
        assert actual.row_to_location.tolist() == reference.row_to_location.tolist()
        metrics = actual.backend_fingerprint
        assert metrics


def test_fake_gpu_knn_reports_bounded_transfers_and_reuses_k_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    descriptors = np.asarray([[float(i), float(i % 3)] for i in range(12)], dtype=np.float64)
    from nepflow.stages.selection.algorithms.information_entropy.gpu_neighbours import (
        ExactTiledGPUNeighbourIndex,
        compute_gpu_tiled_neighbours_many,
    )

    index = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_tile_size=3,
        max_device_bytes=4096,
        resource_budget=_budget(),
    )
    results = compute_gpu_tiled_neighbours_many(
        descriptors,
        (1, 2, 4),
        index=index,
        chunk_size=3,
        resource_budget=_budget(),
    )
    np.testing.assert_array_equal(results[1].neighbour_indices, results[4].neighbour_indices[:, :1])
    np.testing.assert_array_equal(results[2].neighbour_indices, results[4].neighbour_indices[:, :2])
    metrics = index.memory_metrics()
    assert metrics["gpu_device_workspace_peak_bytes"] <= 4096
    assert metrics["gpu_h2d_bytes"] > 0
    assert metrics["gpu_d2h_bytes"] > 0
    assert metrics["gpu_target_tiles"] > 0
    assert metrics["gpu_source_tiles"] == 4
    assert metrics["gpu_h2d_bytes"] < descriptors.nbytes * descriptors.shape[0]
    assert metrics["gpu_reference_uploads"] == 1
    assert metrics["gpu_reference_shards"] == 0
    assert metrics["gpu_resident_reference_bytes"] == descriptors.nbytes
    assert metrics["gpu_d2h_bytes"] == 12 * (6 * 16 + 16)
    assert metrics["gpu_operation_seconds_inclusive"] >= metrics["gpu_kernel_seconds"]
    second = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_tile_size=4,
        max_device_bytes=4096,
        resource_budget=_budget(),
    )
    assert second.fingerprint == index.fingerprint
    second.close()
    index.close()


def test_fake_gpu_knn_uses_first_excluded_witness_across_reference_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A distant omitted point must not create a false boundary rescan."""

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    descriptors = np.zeros((11, 42), dtype=np.float64)
    descriptors[:, 0] = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, -8, 20], dtype=np.float64)
    budget = _budget()
    reference = compute_exact_neighbours(descriptors, 8, chunk_size=4, resource_budget=budget)
    from nepflow.stages.selection.algorithms.information_entropy.gpu_neighbours import (
        ExactTiledGPUNeighbourIndex,
    )

    index = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_tile_size=4,
        max_device_bytes=64 * 1024,
        resource_budget=budget,
    )
    source_location = int(index.row_to_location[0])
    rescanned_sources: list[int] = []
    original_rescan = index._exact_rescan

    def record_rescan(query: np.ndarray, source: int, k: int):
        rescanned_sources.append(int(source))
        return original_rescan(query, source, k)

    monkeypatch.setattr(index, "_exact_rescan", record_rescan)
    _, rows, distances = index.query_neighbours(8, chunk_size=4)
    np.testing.assert_array_equal(rows, reference.neighbour_indices)
    np.testing.assert_array_equal(distances, reference.neighbour_distances)
    assert source_location not in rescanned_sources
    metrics = index.memory_metrics()
    assert metrics["gpu_false_ambiguity_avoided"] >= 1
    assert metrics["gpu_first_excluded_gap_max"] >= 12.0
    index.close()


def test_fake_gpu_knn_adaptively_resolves_kth_boundary_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ties beyond the retained prefix are CPU-canonicalized after GPU filtering."""

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    descriptors = np.zeros((7, 42), dtype=np.float64)
    descriptors[1:7, 0] = np.asarray([1, -1, 0, 0, 0, 0], dtype=np.float64)
    descriptors[3, 1] = 1.0
    descriptors[4, 1] = -1.0
    descriptors[5, 2] = 1.0
    descriptors[6, 2] = -1.0
    budget = _budget()
    reference = compute_exact_neighbours(descriptors, 2, chunk_size=3, resource_budget=budget)
    from nepflow.stages.selection.algorithms.information_entropy.gpu_neighbours import (
        ExactTiledGPUNeighbourIndex,
    )

    index = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_tile_size=3,
        max_device_bytes=64 * 1024,
        resource_budget=budget,
    )
    _, rows, distances = index.query_neighbours(2, chunk_size=3)
    np.testing.assert_array_equal(rows, reference.neighbour_indices)
    np.testing.assert_array_equal(distances, reference.neighbour_distances)
    metrics = index.memory_metrics()
    assert metrics["gpu_genuine_unresolved_boundary_count"] > 0
    assert metrics["gpu_boundary_refinements"] > 0
    assert metrics["gpu_boundary_refinement_candidate_count"] >= 4
    assert metrics["gpu_certification_rescans"] == 0
    index.close()


@pytest.mark.parametrize("device_count", (2, 4))
def test_fake_multi_gpu_knn_scheduler_preserves_order_and_device_metrics(
    monkeypatch: pytest.MonkeyPatch, device_count: int
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(device_count=device_count))
    descriptors = np.asarray(
        [[float(i), float(i % 5), float((i * 11) % 7)] for i in range(24)],
        dtype=np.float64,
    )
    reference = compute_exact_neighbours(descriptors, 4, chunk_size=3, resource_budget=_budget())
    from nepflow.stages.selection.algorithms.information_entropy.gpu_neighbours import (
        ExactTiledGPUNeighbourIndex,
    )

    barrier = threading.Barrier(device_count)
    original_query = ExactTiledGPUNeighbourIndex.query_neighbours_range

    def synchronized_query(self: Any, *args: Any, **kwargs: Any):
        barrier.wait(timeout=5.0)
        return original_query(self, *args, **kwargs)

    monkeypatch.setattr(ExactTiledGPUNeighbourIndex, "query_neighbours_range", synchronized_query)

    index = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_devices=tuple(range(device_count)),
        gpu_tile_size=3,
        max_device_bytes=32 * 1024,
        resource_budget=_budget(),
    )
    _, rows, distances = index.query_neighbours(4, chunk_size=3)
    np.testing.assert_array_equal(rows, reference.neighbour_indices)
    np.testing.assert_array_equal(distances, reference.neighbour_distances)
    metrics = index.memory_metrics()
    assert metrics["gpu_devices"] == list(range(device_count))
    assert len(metrics["gpu_per_device"]) == device_count
    assert metrics["gpu_wall_seconds"] > 0.0
    assert metrics["gpu_active_kernel_seconds_sum"] >= 0.0
    index.close()


@pytest.mark.parametrize("checkpoint_every", (1, 8, 1024, None))
def test_checkpoint_interval_does_not_shrink_multi_gpu_compute_tiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint_every: int | None,
) -> None:
    """Checkpoint cadence must not turn a configured source tile into rows."""

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(device_count=2))
    descriptors = np.asarray(
        [[float(i), float(i % 5), float((i * 11) % 7)] for i in range(40)],
        dtype=np.float64,
    )
    identity = {"schema": "gpu-tile-checkpoint-regression", "rows": list(range(40))}
    store = (
        None
        if checkpoint_every is None
        else IncrementalCheckpointStore(str(tmp_path / f"interval-{checkpoint_every}"))
    )
    from nepflow.stages.selection.algorithms.information_entropy.gpu_neighbours import (
        ExactTiledGPUNeighbourIndex,
    )

    index = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_devices=(0, 1),
        gpu_tile_size=8,
        max_device_bytes=32 * 1024,
        resource_budget=_budget(),
    )
    index.query_neighbours(
        2,
        chunk_size=4,
        checkpoint_store=store,
        checkpoint_identity=identity if store is not None else None,
        checkpoint_every_locations=1 if checkpoint_every is None else checkpoint_every,
        preserve_checkpoint=store is not None,
    )

    metrics = index.memory_metrics()
    assert metrics["gpu_effective_source_tile_size"] == 4
    assert metrics["gpu_effective_source_tile_sizes"] == {0: 4, 1: 4}
    assert metrics["gpu_checkpoint_batch_size"] == 4
    assert metrics["gpu_source_tiles"] == 10
    assert metrics["gpu_checkpoint_every_locations"] == (
        0 if checkpoint_every is None else checkpoint_every
    )
    if store is None:
        assert metrics["gpu_checkpoint_commits"] == 0
    else:
        assert 1 <= metrics["gpu_checkpoint_commits"] <= 10
        state = store.load(
            "knn",
            expected_identity=identity,
            k_max=2,
            location_count=len(index.unique_locations),
        )[1]
        assert state["cursor"] == len(index.unique_locations)
    index.close()


def test_fake_gpu_knn_adaptively_resolves_ambiguous_ties_and_wide_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    descriptors = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [np.nextafter(1.0, 2.0), 0.0],
            [1.0e150, 0.0],
            [-1.0e150, 0.0],
        ],
        dtype=np.float64,
    )
    budget = _budget()
    reference = compute_exact_neighbours(descriptors, 1, chunk_size=3, resource_budget=budget)
    from nepflow.stages.selection.algorithms.information_entropy.gpu_neighbours import (
        ExactTiledGPUNeighbourIndex,
    )

    index = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_tile_size=3,
        max_device_bytes=8192,
        resource_budget=budget,
    )
    _, rows, distances = index.query_neighbours(1, chunk_size=3)
    np.testing.assert_array_equal(rows, reference.neighbour_indices)
    np.testing.assert_array_equal(distances, reference.neighbour_distances)
    metrics = index.memory_metrics()
    assert metrics["gpu_boundary_refinements"] > 0
    assert metrics["gpu_certification_rescans"] == 0
    index.close()


def test_fake_gpu_knn_uses_bounded_reference_shards_when_pool_cannot_reside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    descriptors = np.asarray([[float(i), float(i % 3)] for i in range(12)], dtype=np.float64)
    budget = _budget()
    from nepflow.stages.selection.algorithms.information_entropy.gpu_neighbours import (
        ExactTiledGPUNeighbourIndex,
    )

    index = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_tile_size=3,
        max_device_bytes=1300,
        resource_budget=budget,
    )
    _, rows, distances = index.query_neighbours(2, chunk_size=3)
    reference = compute_exact_neighbours(descriptors, 2, chunk_size=3, resource_budget=budget)
    np.testing.assert_array_equal(rows, reference.neighbour_indices)
    np.testing.assert_array_equal(distances, reference.neighbour_distances)
    metrics = index.memory_metrics()
    assert metrics["gpu_resident_reference_bytes"] == 0
    assert metrics["gpu_reference_uploads"] == metrics["gpu_reference_shards"]
    assert metrics["gpu_reference_shards"] > 1
    index.close()


def test_fake_gpu_knn_checkpoint_resume_publishes_only_new_ranges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())

    class InterruptingStore(IncrementalCheckpointStore):
        def __init__(self, root: str) -> None:
            super().__init__(root)
            self.interrupted = False

        def write_chunk(self, *args: Any, **kwargs: Any) -> dict[str, float | int]:
            result = super().write_chunk(*args, **kwargs)
            if not self.interrupted:
                self.interrupted = True
                raise RuntimeError("intentional interruption")
            return result

    descriptors = np.asarray(
        [[float(i), float(i % 5), float((i * 11) % 7)] for i in range(13)],
        dtype=np.float64,
    )
    budget = _budget()
    store = InterruptingStore(str(tmp_path))
    identity = {"schema": "gpu-test", "ordered": list(range(len(descriptors)))}
    from nepflow.stages.selection.algorithms.information_entropy.gpu_neighbours import (
        ExactTiledGPUNeighbourIndex,
    )

    interrupted = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_tile_size=4,
        max_device_bytes=4096,
        resource_budget=budget,
    )
    with pytest.raises(RuntimeError, match="intentional interruption"):
        interrupted.query_neighbours(
            3,
            chunk_size=4,
            checkpoint_store=store,
            checkpoint_identity=identity,
            checkpoint_every_locations=1,
        )
    assert store.exists("knn")

    resumed = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_tile_size=4,
        max_device_bytes=4096,
        resource_budget=budget,
    )
    _, resumed_indices, resumed_distances = resumed.query_neighbours(
        3,
        chunk_size=4,
        checkpoint_store=store,
        checkpoint_identity=identity,
        checkpoint_every_locations=1,
        preserve_checkpoint=True,
    )
    reference = compute_exact_neighbours(descriptors, 3, chunk_size=4, resource_budget=budget)
    np.testing.assert_array_equal(resumed_indices, reference.neighbour_indices)
    np.testing.assert_array_equal(resumed_distances, reference.neighbour_distances)
    state = store.load(
        "knn",
        expected_identity=identity,
        k_max=3,
        location_count=len(resumed.unique_locations),
    )[1]
    assert state["cursor"] == len(resumed.unique_locations)
    assert state["chunk_count"] == 4


@pytest.mark.parametrize("device_count", (2, 4))
def test_fake_multi_gpu_checkpoint_resume_commits_contiguous_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    device_count: int,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(device_count=device_count))

    class InterruptingStore(IncrementalCheckpointStore):
        def __init__(self, root: str) -> None:
            super().__init__(root)
            self.interrupted = False

        def write_chunk(self, *args: Any, **kwargs: Any) -> dict[str, float | int]:
            result = super().write_chunk(*args, **kwargs)
            if not self.interrupted:
                self.interrupted = True
                raise RuntimeError("intentional multi-GPU interruption")
            return result

    descriptors = np.asarray(
        [[float(i), float(i % 5), float((i * 11) % 7)] for i in range(24)],
        dtype=np.float64,
    )
    identity = {"schema": "gpu-multi-checkpoint-test", "ordered": list(range(len(descriptors)))}
    store = InterruptingStore(str(tmp_path))
    from nepflow.stages.selection.algorithms.information_entropy.gpu_neighbours import (
        ExactTiledGPUNeighbourIndex,
    )

    interrupted = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_devices=tuple(range(device_count)),
        gpu_tile_size=4,
        max_device_bytes=32 * 1024,
        resource_budget=_budget(),
    )
    with pytest.raises(RuntimeError, match="intentional multi-GPU interruption"):
        interrupted.query_neighbours(
            3,
            chunk_size=3,
            checkpoint_store=store,
            checkpoint_identity=identity,
            checkpoint_every_locations=3,
        )
    checkpoint_state = store.load(
        "knn",
        expected_identity=identity,
        k_max=3,
        location_count=len(np.unique(descriptors, axis=0)),
    )[1]
    assert 0 < checkpoint_state["cursor"] < len(descriptors)

    resumed = ExactTiledGPUNeighbourIndex(
        descriptors,
        gpu_devices=tuple(range(device_count)),
        gpu_tile_size=4,
        max_device_bytes=32 * 1024,
        resource_budget=_budget(),
    )
    _, resumed_indices, resumed_distances = resumed.query_neighbours(
        3,
        chunk_size=3,
        checkpoint_store=store,
        checkpoint_identity=identity,
        checkpoint_every_locations=3,
        preserve_checkpoint=True,
    )
    reference = compute_exact_neighbours(descriptors, 3, chunk_size=3, resource_budget=_budget())
    np.testing.assert_array_equal(resumed_indices, reference.neighbour_indices)
    np.testing.assert_array_equal(resumed_distances, reference.neighbour_distances)
    final_state = store.load(
        "knn",
        expected_identity=identity,
        k_max=3,
        location_count=len(resumed.unique_locations),
    )[1]
    assert final_state["cursor"] == len(resumed.unique_locations)


def test_fake_multi_gpu_worker_failure_is_propagated_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(device_count=2))
    from nepflow.stages.selection.algorithms.information_entropy.gpu_neighbours import (
        ExactTiledGPUNeighbourIndex,
    )

    original_query = ExactTiledGPUNeighbourIndex.query_neighbours_range

    def failing_query(self: Any, *args: Any, **kwargs: Any):
        if self.device == 1:
            raise RuntimeError("injected device worker failure")
        return original_query(self, *args, **kwargs)

    monkeypatch.setattr(ExactTiledGPUNeighbourIndex, "query_neighbours_range", failing_query)
    index = ExactTiledGPUNeighbourIndex(
        np.asarray([[float(i), float(i % 3)] for i in range(16)], dtype=np.float64),
        gpu_devices=(0, 1),
        gpu_tile_size=4,
        max_device_bytes=32 * 1024,
        resource_budget=_budget(),
    )
    with pytest.raises(RuntimeError, match="injected device worker failure"):
        index.query_neighbours(2, chunk_size=3)


def test_fake_gpu_knn_fails_before_unbounded_result_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    descriptors = np.asarray([[float(i), float(i % 3)] for i in range(32)])
    with pytest.raises(ResourceCapacityError, match="memory allowance"):
        compute_neighbours(
            descriptors,
            4,
            backend=GPU_NEIGHBOUR_BACKEND_ID,
            gpu_tile_size=4,
            max_gpu_neighbour_bytes=4096,
            resource_budget=build_resource_budget(
                execution_mode="explicit",
                memory_budget_bytes=4096,
                scratch_budget_bytes=16 * 1024 * 1024,
                safety_margin_fraction=0.0,
            ),
        )


@pytest.mark.cuda
def test_cuda_gpu_knn_matches_cpu_oracle() -> None:
    cp = pytest.importorskip("cupy")
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        pytest.skip("no visible CUDA device")
    budget = _budget()
    rng = np.random.default_rng(20261011)
    random_case = rng.normal(size=(24, 42)).astype(np.float64)
    tie_case = np.zeros((24, 42), dtype=np.float64)
    tie_case[:, 0] = np.arange(-12, 12, dtype=np.float64)
    duplicate_case = random_case.copy()
    duplicate_case[3] = duplicate_case[0]
    duplicate_case[9] = duplicate_case[0]
    wide_case = np.zeros((24, 42), dtype=np.float64)
    wide_case[:, 0] = np.arange(-12, 12, dtype=np.float64) * 1.0e150
    ulp_case = tie_case.copy()
    ulp_case[12, 0] = np.nextafter(1.0, 2.0)
    cases = (random_case, tie_case, duplicate_case, wide_case, ulp_case)
    for descriptors in cases:
        for tile_size in (3, 7):
            for k in (1, 2, 4, 8):
                reference = compute_exact_neighbours(
                    descriptors, k, chunk_size=tile_size, resource_budget=budget
                )
                actual = compute_neighbours(
                    descriptors,
                    k,
                    backend=GPU_NEIGHBOUR_BACKEND_ID,
                    gpu_device=0,
                    gpu_tile_size=tile_size,
                    max_gpu_neighbour_bytes=1 << 20,
                    resource_budget=budget,
                )
                np.testing.assert_array_equal(actual.neighbour_indices, reference.neighbour_indices)
                np.testing.assert_array_equal(
                    actual.neighbour_distances, reference.neighbour_distances
                )
                np.testing.assert_array_equal(actual.radii, reference.radii)
