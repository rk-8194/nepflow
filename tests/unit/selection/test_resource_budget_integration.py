"""Selection regression for unlimited cgroup memory detection."""

import pytest

pytest.importorskip("ase")
from ase import Atoms

import nepflow.stages.selection.representations as representation_module
from nepflow.resources.budget import ResourceSnapshot, build_resource_budget
from nepflow.stages.selection.representations import (
    LocalRepresentationConfig,
    _compute_local_raw_descriptors,
    _resolve_local_descriptor_workers,
)


def test_auto_descriptor_workers_use_allocation_and_explicit_resource_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(representation_module, "_effective_local_cpu_allocation", lambda: 64)

    assert _resolve_local_descriptor_workers(0, 64) == 64
    assert _resolve_local_descriptor_workers(0, 64, worker_cap=12) == 12
    assert _resolve_local_descriptor_workers(0, 3) == 3
    assert _resolve_local_descriptor_workers(1, 64) == 1

    with pytest.raises(ValueError, match="effective CPU allocation"):
        _resolve_local_descriptor_workers(65, 100)


def test_local_descriptors_accept_known_unlimited_cgroup_budget() -> None:
    snapshot = ResourceSnapshot(
        host_available_memory_bytes=256 * 1024**2,
        cgroup_status="unlimited",
    )
    budget = build_resource_budget(snapshot, safety_margin_fraction=0.0)
    candidate = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[8.0, 8.0, 8.0], pbc=False)

    descriptors, rows = _compute_local_raw_descriptors(
        [candidate],
        ["candidate-0"],
        ["structure-0"],
        ("Si",),
        LocalRepresentationConfig(radial_bins=1, angular_bins=1),
        1,
        resource_budget=budget,
    )

    assert descriptors.shape[0] == len(rows) == 1
