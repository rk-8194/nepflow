"""Bounded-memory admission regressions for entropy selection workspaces."""

import pytest
from ase import Atoms

from nepflow.stages.selection.representations import (
    LocalRepresentationConfig,
    _compute_local_raw_descriptors,
)


def test_descriptor_worker_rejects_oversized_candidate_before_launch() -> None:
    candidate = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[8.0, 8.0, 8.0], pbc=False)
    config = LocalRepresentationConfig(radial_bins=1, angular_bins=1)

    with pytest.raises(ValueError, match="candidate_id=candidate-0"):
        _compute_local_raw_descriptors(
            [candidate],
            ["candidate-0"],
            ["structure-0"],
            ("Si",),
            config,
            1,
            max_local_descriptor_inflight_bytes=25,
        )
