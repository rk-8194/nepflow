"""Independent local-environment test holdout policy tests."""

from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms

from nepflow.resources.budget import ResourceSnapshot, build_resource_budget
from nepflow.stages.selection.representations import (
    LocalEnvironmentRepresentation,
    LocalEnvironmentRow,
)
from nepflow.stages.selection.test_selection import (
    TestSelectionPolicyError,
    select_local_test_holdout,
)


def _representation(
    descriptors: np.ndarray,
    owners: tuple[str, ...],
    structure_ids: tuple[str, ...] | None = None,
) -> LocalEnvironmentRepresentation:
    candidate_ids = tuple(dict.fromkeys(owners))
    structure_ids = structure_ids or candidate_ids
    rows = []
    counters = {candidate_id: 0 for candidate_id in candidate_ids}
    for candidate_id in owners:
        rows.append(
            LocalEnvironmentRow(
                candidate_id,
                structure_ids[candidate_ids.index(candidate_id)],
                counters[candidate_id],
            )
        )
        counters[candidate_id] += 1
    return LocalEnvironmentRepresentation(
        raw_descriptors=descriptors,
        descriptors=descriptors,
        rows=tuple(rows),
        candidate_ids=candidate_ids,
        structure_ids=structure_ids,
        config=cast(Any, None),
        transform=cast(Any, None),
        fingerprint="local-test-representation",
    )


def _structures(count: int) -> list[Atoms]:
    return [
        Atoms("Si", positions=[[float(index), 0.0, 0.0]], cell=[10.0] * 3, pbc=False)
        for index in range(count)
    ]


def _budget():
    return build_resource_budget(
        ResourceSnapshot(host_available_memory_bytes=1024**3),
        safety_margin_fraction=0.0,
    )


def test_extrapolative_policy_uses_local_rows_not_candidate_means() -> None:
    descriptors = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [3.0, 3.0],
            [-1.0, -3.0],
        ],
        dtype=np.float64,
    )
    representation = _representation(descriptors, ("c0", "c0", "c1", "c1", "c2", "c2"))

    result = select_local_test_holdout(
        representation,
        _structures(3),
        [0],
        target_count=1,
        policy="extrapolative",
        resource_budget=_budget(),
        candidate_ids=("c0", "c1", "c2"),
        structure_ids=("s0", "s1", "s2"),
    )

    assert result.indices == (2,)
    assert result.provenance["representation"] == "candidate-owned whitened local rows"
    assert result.provenance["holdout_purpose"] == "out_of_domain_stress_test"


def test_representative_and_extrapolative_are_reproducible_but_distinct() -> None:
    descriptors = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, -1.0], [3.0, 3.0], [-1.0, -3.0]],
        dtype=np.float64,
    )
    representation = _representation(descriptors, ("c0", "c0", "c1", "c1", "c2", "c2"))
    structures = _structures(3)
    kwargs: dict[str, Any] = {
        "representation": representation,
        "structures": structures,
        "train_indices": [0],
        "target_count": 1,
        "candidate_ids": ("c0", "c1", "c2"),
        "structure_ids": ("s0", "s1", "s2"),
    }

    representative = select_local_test_holdout(
        **kwargs,
        policy="representative",
        signature_bins=1,
    )
    repeated = select_local_test_holdout(
        **kwargs,
        policy="representative",
        signature_bins=1,
    )
    extrapolative = select_local_test_holdout(
        **kwargs,
        policy="extrapolative",
        resource_budget=_budget(),
    )

    assert representative.indices == repeated.indices
    assert representative.indices != extrapolative.indices
    assert representative.provenance["atom_weighting"] == "candidate"


def test_atom_weighting_is_explicit_and_identity_groups_are_leakage_safe() -> None:
    descriptors = np.asarray(
        [[0.0], [1.0], [2.0], [3.0], [4.0]],
        dtype=np.float64,
    )
    representation = _representation(
        descriptors,
        ("train", "a", "b", "b", "b"),
        ("physical-train", "physical-a", "physical-b"),
    )
    result = select_local_test_holdout(
        representation,
        _structures(3),
        [0],
        target_count=2,
        policy="representative",
        atom_weighting="atom",
        signature_bins=1,
        candidate_ids=("train", "a", "b"),
        structure_ids=("physical-train", "physical-a", "physical-b"),
    )

    assert len(result.indices) == 2
    assert result.provenance["atom_weighting"] == "atom"
    assert result.provenance["duplicate_physical_candidates"] == 0

    leaked = _representation(
        np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64),
        ("train", "magnetic-test", "other"),
        ("same", "same", "other"),
    )
    safe = select_local_test_holdout(
        leaked,
        _structures(3),
        [0],
        target_count=2,
        policy="representative",
        candidate_ids=("train", "magnetic-test", "other"),
        structure_ids=("same", "same", "other"),
    )
    assert safe.indices == (2,)
    assert "physical structure identity is in training" in safe.provenance[
        "exclusion_reasons"
    ].values()


def test_unsupported_policy_is_typed() -> None:
    representation = _representation(np.asarray([[0.0]], dtype=np.float64), ("c0",))
    with pytest.raises(TestSelectionPolicyError, match="unsupported test_selection_policy"):
        select_local_test_holdout(
            representation,
            _structures(1),
            [],
            target_count=1,
            policy="fps",
        )
