"""CPU-safe fake-CuPy tests and separately marked CUDA evidence for GPU kNN."""

from __future__ import annotations

import sys
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
    EntropyBandwidthSettings,
    GPU_NEIGHBOUR_BACKEND_ID,
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
        asarray=np.asarray,
        asnumpy=np.asarray,
        zeros=np.zeros,
        sqrt=np.sqrt,
        isfinite=np.isfinite,
        all=np.all,
        argpartition=np.argpartition,
        take_along_axis=np.take_along_axis,
        cuda=types.SimpleNamespace(Device=Device, runtime=runtime),
        get_default_memory_pool=lambda: types.SimpleNamespace(free_all_blocks=lambda: None),
        get_default_pinned_memory_pool=lambda: types.SimpleNamespace(
            free_all_blocks=lambda: None
        ),
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
    assert EntropyBandwidthSettings(
        backend=GPU_NEIGHBOUR_BACKEND_ID, kernel_support_execution="cpu"
    ).kernel_support_execution == "cpu"


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
    assert metrics["gpu_operation_seconds_inclusive"] >= metrics["gpu_kernel_seconds"]
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

        def write_chunk(
            self, *args: Any, **kwargs: Any
        ) -> dict[str, float | int]:
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
    reference = compute_exact_neighbours(
        descriptors, 3, chunk_size=4, resource_budget=budget
    )
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
    descriptors = np.asarray(
        [[float(i), float(i % 2), float((i * 7) % 5)] for i in range(20)],
        dtype=np.float64,
    )
    budget = _budget()
    reference = compute_exact_neighbours(descriptors, 4, resource_budget=budget)
    actual = compute_neighbours(
        descriptors,
        4,
        backend=GPU_NEIGHBOUR_BACKEND_ID,
        gpu_device=0,
        gpu_tile_size=4,
        resource_budget=budget,
    )
    np.testing.assert_array_equal(actual.neighbour_indices, reference.neighbour_indices)
    np.testing.assert_array_equal(actual.radii, reference.radii)
