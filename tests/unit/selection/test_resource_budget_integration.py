"""Selection regression for unlimited cgroup memory detection."""

import pytest

pytest.importorskip("ase")
from ase import Atoms

from nepflow.resources.budget import ResourceSnapshot, build_resource_budget
from nepflow.stages.selection.representations import (
    LocalRepresentationConfig,
    _compute_local_raw_descriptors,
)


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
