"""Fault and integrity tests for indexed canonical kNN checkpoints."""

import json

import numpy as np
import pytest

from nepflow.resources.budget import ResourceCapacityError, build_resource_budget
from nepflow.stages.selection.algorithms.information_entropy import checkpointing
from nepflow.stages.selection.algorithms.information_entropy.checkpointing import (
    CheckpointValidationError,
    IncrementalCheckpointStore,
)


def _chunk(start: int, length: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(start * k, (start + length) * k, dtype=np.int64).reshape(length, k)
    distances = np.arange(start * k, (start + length) * k, dtype=np.float64).reshape(length, k)
    return indices, distances


def test_incremental_store_restores_contiguous_read_only_prefix(tmp_path) -> None:
    store = IncrementalCheckpointStore(tmp_path)
    identity = {"descriptors": "ordered-fixture", "backend": "indexed_cpu"}
    first_indices, first_distances = _chunk(0, 2, 2)
    second_indices, second_distances = _chunk(2, 1, 2)

    store.write_chunk(
        "knn",
        identity=identity,
        start=0,
        neighbour_locations=first_indices,
        neighbour_distances=first_distances,
        k_max=2,
        location_count=3,
    )
    store.write_chunk(
        "knn",
        identity=identity,
        start=2,
        neighbour_locations=second_indices,
        neighbour_distances=second_distances,
        k_max=2,
        location_count=3,
    )

    _, state, arrays = store.load("knn", expected_identity=identity, k_max=2, location_count=3)
    assert state["cursor"] == 3
    np.testing.assert_array_equal(
        arrays["neighbour_locations"], np.concatenate((first_indices, second_indices))
    )
    assert arrays["neighbour_locations"].flags.writeable is False
    assert arrays["neighbour_distances"].flags.writeable is False
    pointer = json.loads(
        (tmp_path / "knn.incremental.pointer.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((tmp_path / pointer["manifest"]).read_text(encoding="utf-8"))
    manifest_bytes = 0
    current = manifest
    while current is not None:
        manifest_bytes += sum(entry["bytes"] for entry in current["chunk"]["arrays"].values())
        previous = current.get("previous_manifest")
        current = (
            json.loads((tmp_path / previous).read_text(encoding="utf-8"))
            if previous is not None
            else None
        )
    assert store.metrics()["bytes_written"] == manifest_bytes


def test_failed_pointer_promotion_keeps_last_good_generation(tmp_path, monkeypatch) -> None:
    store = IncrementalCheckpointStore(tmp_path)
    identity = {"ordered": True}
    first_indices, first_distances = _chunk(0, 1, 2)
    second_indices, second_distances = _chunk(1, 1, 2)
    store.write_chunk(
        "knn",
        identity=identity,
        start=0,
        neighbour_locations=first_indices,
        neighbour_distances=first_distances,
        k_max=2,
        location_count=2,
    )

    original_replace = checkpointing.os.replace

    def fail_pointer_promotion(source, destination):
        if str(destination).endswith("knn.incremental.pointer.json"):
            raise OSError("injected pointer promotion failure")
        return original_replace(source, destination)

    monkeypatch.setattr(checkpointing.os, "replace", fail_pointer_promotion)
    with pytest.raises(OSError, match="pointer promotion"):
        store.write_chunk(
            "knn",
            identity=identity,
            start=1,
            neighbour_locations=second_indices,
            neighbour_distances=second_distances,
            k_max=2,
            location_count=2,
        )

    _, state, arrays = store.load("knn", expected_identity=identity, k_max=2, location_count=2)
    assert state["cursor"] == 1
    np.testing.assert_array_equal(arrays["neighbour_locations"], first_indices)
    removed = store.clean_stale()
    assert removed >= 2


def test_incremental_store_rejects_pointer_swap_with_cached_cursor(tmp_path) -> None:
    store = IncrementalCheckpointStore(tmp_path)
    identity = {"ordered": True}
    indices, distances = _chunk(0, 1, 1)
    store.write_chunk(
        "knn",
        identity=identity,
        start=0,
        neighbour_locations=indices,
        neighbour_distances=distances,
        k_max=1,
        location_count=2,
    )

    pointer_path = tmp_path / "knn.incremental.pointer.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    original_manifest = tmp_path / pointer["manifest"]
    alternate_name = "knn.incremental.manifest-alternate.json"
    (tmp_path / alternate_name).write_text(
        original_manifest.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    pointer["manifest"] = alternate_name
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(CheckpointValidationError, match="pointer changed"):
        store.write_chunk(
            "knn",
            identity=identity,
            start=1,
            neighbour_locations=indices,
            neighbour_distances=distances,
            k_max=1,
            location_count=2,
        )


def test_incremental_store_rejects_gap_and_corrupt_chunk(tmp_path) -> None:
    store = IncrementalCheckpointStore(tmp_path)
    identity = {"ordered": True}
    indices, distances = _chunk(0, 1, 1)
    store.write_chunk(
        "knn",
        identity=identity,
        start=0,
        neighbour_locations=indices,
        neighbour_distances=distances,
        k_max=1,
        location_count=2,
    )
    with pytest.raises(CheckpointValidationError, match="contiguous"):
        store.write_chunk(
            "knn",
            identity=identity,
            start=2,
            neighbour_locations=indices,
            neighbour_distances=distances,
            k_max=1,
            location_count=3,
        )

    pointer = json.loads(
        (tmp_path / "knn.incremental.pointer.json").read_text(encoding="utf-8")
    )
    manifest_path = tmp_path / pointer["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    array_path = tmp_path / manifest["chunk"]["arrays"]["neighbour_locations"]["name"]
    with array_path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(CheckpointValidationError, match="checksum"):
        store.load("knn", expected_identity=identity, k_max=1, location_count=2)


def test_incremental_store_admits_against_explicit_scratch_budget(tmp_path) -> None:
    budget = build_resource_budget(
        execution_mode="explicit",
        memory_budget_bytes=1024 * 1024,
        scratch_budget_bytes=1,
        reserved_headroom_bytes=0,
    )
    store = IncrementalCheckpointStore(tmp_path, resource_budget=budget)
    indices, distances = _chunk(0, 1, 1)
    with pytest.raises(ResourceCapacityError, match="scratch budget"):
        store.write_chunk(
            "knn",
            identity={"ordered": True},
            start=0,
            neighbour_locations=indices,
            neighbour_distances=distances,
            k_max=1,
            location_count=2,
        )


def test_incremental_store_compacts_large_ordered_identity(tmp_path) -> None:
    store = IncrementalCheckpointStore(tmp_path)
    row_ids = [f"row-{index}" for index in range(1025)]
    identity = {"ordered": True, "row_ids": row_ids}
    indices, distances = _chunk(0, 1, 1)

    store.write_chunk(
        "knn",
        identity=identity,
        start=0,
        neighbour_locations=indices,
        neighbour_distances=distances,
        k_max=1,
        location_count=2,
    )

    pointer = json.loads(
        (tmp_path / "knn.incremental.pointer.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((tmp_path / pointer["manifest"]).read_text(encoding="utf-8"))
    stored_identity = manifest["identity"]
    assert "row_ids" not in stored_identity
    assert stored_identity["row_ids_count"] == len(row_ids)
    assert store.load("knn", expected_identity=identity)[1]["cursor"] == 1

    with pytest.raises(CheckpointValidationError, match="ordered input"):
        store.load("knn", expected_identity={"ordered": True, "row_ids": row_ids[::-1]})


def test_incremental_store_rejects_corrupt_intermediate_cursor_metadata(tmp_path) -> None:
    store = IncrementalCheckpointStore(tmp_path)
    identity = {"ordered": True}
    first_indices, first_distances = _chunk(0, 1, 1)
    second_indices, second_distances = _chunk(1, 1, 1)
    store.write_chunk(
        "knn",
        identity=identity,
        start=0,
        neighbour_locations=first_indices,
        neighbour_distances=first_distances,
        k_max=1,
        location_count=2,
    )
    store.write_chunk(
        "knn",
        identity=identity,
        start=1,
        neighbour_locations=second_indices,
        neighbour_distances=second_distances,
        k_max=1,
        location_count=2,
    )

    pointer = json.loads(
        (tmp_path / "knn.incremental.pointer.json").read_text(encoding="utf-8")
    )
    latest = json.loads((tmp_path / pointer["manifest"]).read_text(encoding="utf-8"))
    previous_path = tmp_path / latest["previous_manifest"]
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous["cursor"] = 2
    previous_path.write_text(json.dumps(previous), encoding="utf-8")

    with pytest.raises(CheckpointValidationError, match="cursor"):
        store.load("knn", expected_identity=identity, k_max=1, location_count=2)
