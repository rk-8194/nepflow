"""Phase 6 candidate identity contracts at the selection boundary."""

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("NepTrainKit")
from ase import Atoms

from nepflow.config.models import SelectionConfig
from nepflow.errors import StateError
from nepflow.stages.selection.artifacts import (
    SELECTION_ARTIFACT_SCHEMA,
    read_selection_manifest,
    write_selected_structures,
)
from nepflow.stages.selection.models import SelectionResult
from nepflow.stages.selection.persistence import (
    LEGACY_SELECTION_RUN_SCHEMA,
    candidate_ids,
    candidate_set_fingerprint,
    persist_selection_result,
    restore_selection_result,
    structure_ids,
)
from nepflow.stages.selection.representations import (
    _descriptor_manifest,
    validate_candidate_representation_identity,
)
from nepflow.state import StateStore


def _pair() -> list[Atoms]:
    first = Atoms("Fe", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4.0, pbc=True)
    second = first.copy()
    first.info.update({"candidate_id": "candidate_fm", "magnetic_state_id": "state-fm"})
    second.info.update({"candidate_id": "candidate_afm", "magnetic_state_id": "state-afm"})
    return [first, second]


def _result() -> SelectionResult:
    return SelectionResult(
        descriptors=np.ones((2, 1)),
        train_indices=[0],
        train_min_dist=0.5,
        train_seed_count=0,
        train_single_element_elastic_count=0,
        train_elastic_count=0,
        train_anchor_count=0,
        train_fps_count=1,
        test_indices=[1],
        test_min_dist=0.25,
        min_train_test_dist=1.0,
        mean_train_test_dist=1.0,
        seed_indices=[],
        single_element_elastic_indices=[],
        elastic_indices=[],
    )


def test_candidate_ids_are_unique_while_structure_ids_repeat() -> None:
    structures = _pair()

    assert candidate_ids(structures) == ["candidate_fm", "candidate_afm"]
    assert structure_ids(structures)[0] == structure_ids(structures)[1]
    assert candidate_set_fingerprint(candidate_ids(structures)) == candidate_set_fingerprint(
        list(reversed(candidate_ids(structures)))
    )


def test_selection_state_persists_and_restores_by_candidate_id(tmp_path) -> None:
    structures = _pair()
    candidates = candidate_ids(structures)
    physical = structure_ids(structures)
    settings = SelectionConfig(target_train_count=1, target_test_count=1)
    with StateStore(tmp_path / "state.db") as store:
        store.upsert_project("demo", name="demo", root_path=str(tmp_path))
        record = persist_selection_result(
            store,
            "demo",
            "demo",
            str(tmp_path),
            settings,
            candidates,
            _result(),
            candidate_structure_ids=physical,
        )

    parameters = record["parameters"]
    assert parameters["candidate_ids"] == candidates
    assert parameters["candidate_structure_ids"] == physical
    assert parameters["selected_candidate_ids"] == {
        "train": ["candidate_fm"],
        "test": ["candidate_afm"],
    }

    restored = restore_selection_result(
        record,
        np.ones((2, 1)),
        list(reversed(candidates)),
        candidate_structure_ids=list(reversed(physical)),
    )
    assert restored.train_indices == [1]
    assert restored.test_indices == [0]


def test_selection_artifact_persists_both_identity_types(tmp_path) -> None:
    structures = _pair()
    write_selected_structures(tmp_path, structures, [0], [1])

    manifest = read_selection_manifest(tmp_path)
    assert manifest["schema_version"] == SELECTION_ARTIFACT_SCHEMA
    assert manifest["train_candidate_ids"] == ["candidate_fm"]
    assert manifest["test_candidate_ids"] == ["candidate_afm"]
    assert manifest["train_structure_ids"] == manifest["test_structure_ids"]


def test_structural_descriptor_cache_identity_does_not_include_candidate_ids() -> None:
    manifest = _descriptor_manifest(
        ["same-structure", "same-structure"],
        {"filename": "nep89.txt", "sha256": "model"},
        True,
        np.ones((2, 3)),
    )

    assert manifest["structure_ids"] == ["same-structure", "same-structure"]
    assert "candidate_ids" not in manifest


def test_structural_only_representation_rejects_magnetic_identity_collision() -> None:
    with pytest.raises(StateError, match="structural-only"):
        validate_candidate_representation_identity(
            ["candidate_fm", "candidate_afm"],
            ["same-structure", "same-structure"],
        )


def test_legacy_selection_schema_is_explicitly_recognized() -> None:
    assert LEGACY_SELECTION_RUN_SCHEMA == "selection-run-v1"
