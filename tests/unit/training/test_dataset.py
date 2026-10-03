from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nepflow.dft.vasp.outputs import VaspParseResult
from nepflow.stages.training import dataset as dataset_module
from nepflow.stages.training.dataset import (
    DatasetSplit,
    build_training_dataset,
    write_nep_dataset,
)


class FakeStateStore:
    def __init__(self, results: list[VaspParseResult]) -> None:
        self.results = {
            dict(result.calculation_identity)["calculation_id"]: result
            for result in results
        }
        self.datasets = []
        self.members = []

    def get_dft_calculation(self, calculation_id: str) -> dict | None:
        result = self.results.get(calculation_id)
        if result is None:
            return None
        identity = dict(result.calculation_identity)
        return {
            "calculation_id": calculation_id,
            "identity": identity,
            "status": "completed",
            "accepted_attempt_id": f"attempt-{calculation_id}",
        }

    def list_artifacts(self, *, originating_attempt_id: str) -> list[dict]:
        calculation_id = originating_attempt_id.removeprefix("attempt-")
        return [
            {
                "artifact_type": "vasp_outcar",
                "sha256": self.results[calculation_id].source_outcar_hash,
            }
        ]

    def upsert_dataset(self, manifest, **kwargs):
        self.datasets.append(manifest)

    def upsert_structure(self, identity, **kwargs):
        return None

    def record_dataset_member(self, dataset_id, member, **kwargs):
        self.members.append((dataset_id, member))


def _result(structure_id: str, *, accepted: bool = True) -> VaspParseResult:
    identity = {
        "structure_id": structure_id,
        "incar_hash": "incar-1",
        "potcar_hash": "potcar-1",
        "calculation_id": f"calculation-{structure_id}",
    }
    return VaspParseResult(
        structure_id=structure_id,
        calculation_identity=tuple(identity.items()),
        source_outcar=f"/results/{structure_id}/OUTCAR",
        source_outcar_hash=f"hash-{structure_id}" if accepted else None,
        status="accepted" if accepted else "rejected",
        rejection_reason=None if accepted else "completed_outcar_missing",
        energy_ev=-1.0 if accepted else None,
        forces_ev_per_angstrom=np.zeros((2, 3)) if accepted else None,
        virial_ev=np.eye(3) if accepted else None,
        positions_angstrom=np.zeros((2, 3)) if accepted else None,
        lattice_angstrom=np.eye(3) if accepted else None,
        species=("W", "W") if accepted else (),
        pbc=(True, True, True) if accepted else (),
    )


def test_build_training_dataset_preserves_authoritative_members_and_splits(tmp_path: Path) -> None:
    train = _result("train-1")
    rejected = _result("train-rejected", accepted=False)
    test = _result("test-1")
    state_store = FakeStateStore([train, test])
    result = build_training_dataset(
        tmp_path / "dataset_0001",
        {
            DatasetSplit.TRAIN: (train, rejected),
            DatasetSplit.TEST: (test,),
        },
        tmp_path,
        train_virial=True,
        allow_partial=True,
        state_store=state_store,
    )

    assert result.train_count == 1
    assert result.test_count == 1
    assert result.reports[DatasetSplit.TRAIN].rejected_reason_counts == {
        "completed_outcar_missing": 1
    }
    manifest = json.loads((result.dataset_path / ".dataset").read_text())
    assert [item["structure_id"] for item in manifest["accepted_members"]] == [
        "train-1",
        "test-1",
    ]
    assert manifest["requested_train_structures"] == 2
    assert manifest["rejected_train_structures"] == 1
    assert 'virial="' in (result.dataset_path / "train.xyz").read_text()
    assert [member.structure_id for _, member in state_store.members] == [
        "train-1",
        "test-1",
    ]
    assert len(state_store.datasets) == 1
    assert state_store.datasets[0].identity.dataset_id == manifest["dataset_id"]
    assert [
        {
            "split": member.split,
            "structure_id": member.structure_id,
            "calculation_id": member.calculation_id,
            "source_outcar_hash": member.source_outcar_hash,
            "ordinal": member.ordinal,
        }
        for _, member in state_store.members
    ] == [
        {
            "split": item["split"],
            "structure_id": item["structure_id"],
            "calculation_id": item["calculation_identity"]["calculation_id"],
            "source_outcar_hash": item["source_outcar_hash"],
            "ordinal": ordinal,
        }
        for ordinal, item in enumerate(manifest["accepted_members"])
    ]


def test_writer_failure_does_not_prepare_dataset_in_state_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = _result("writer-failure")
    state_store = FakeStateStore([accepted])

    def fail_writer(*_args, **_kwargs):
        raise OSError("simulated dataset writer failure")

    monkeypatch.setattr(dataset_module, "write_nep_dataset", fail_writer)
    with pytest.raises(OSError, match="simulated dataset writer failure"):
        build_training_dataset(
            tmp_path / "dataset_0001",
            {
                DatasetSplit.TRAIN: (accepted,),
                DatasetSplit.TEST: (accepted,),
            },
            tmp_path,
            state_store=state_store,
        )

    assert state_store.datasets == []
    assert state_store.members == []
    assert not (tmp_path / "dataset_0001").exists()


def test_manifest_write_failure_does_not_prepare_dataset_in_state_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = _result("manifest-failure")
    state_store = FakeStateStore([accepted])

    def fail_manifest(*_args, **_kwargs):
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(dataset_module, "write_json", fail_manifest)
    with pytest.raises(OSError, match="simulated manifest write failure"):
        build_training_dataset(
            tmp_path / "dataset_0001",
            {
                DatasetSplit.TRAIN: (accepted,),
                DatasetSplit.TEST: (accepted,),
            },
            tmp_path,
            state_store=state_store,
        )

    assert state_store.datasets == []
    assert state_store.members == []
    assert not (tmp_path / "dataset_0001").exists()


def test_rejected_exact_count_does_not_publish_dataset_files(tmp_path: Path) -> None:
    accepted = _result("train-1")
    rejected = _result("train-rejected", accepted=False)
    dataset_path = tmp_path / "dataset_0001"
    state_store = FakeStateStore([accepted])

    try:
        build_training_dataset(
            dataset_path,
            {
                DatasetSplit.TRAIN: (accepted, rejected),
                DatasetSplit.TEST: (accepted,),
            },
            tmp_path,
            allow_partial=False,
            state_store=state_store,
        )
    except RuntimeError as exc:
        assert "rejected selected structures" in str(exc)
    else:
        raise AssertionError("exact-count rejection must fail")

    assert not (dataset_path / "train.xyz").exists()
    assert not (dataset_path / "test.xyz").exists()
    assert not (dataset_path / ".dataset").exists()


def test_filesystem_only_result_is_rejected_without_authoritative_state(
    tmp_path: Path,
) -> None:
    result = _result("filesystem-only")
    dataset_path = tmp_path / "dataset_0001"

    with pytest.raises(RuntimeError, match="rejected selected structures"):
        build_training_dataset(
            dataset_path,
            {
                DatasetSplit.TRAIN: (result,),
                DatasetSplit.TEST: (result,),
            },
            tmp_path,
            state_store=FakeStateStore([]),
        )

    assert not (dataset_path / "train.xyz").exists()
    assert not (dataset_path / ".dataset").exists()


def test_member_identity_does_not_depend_on_legacy_folder_or_selection_position(
    tmp_path: Path,
) -> None:
    first = _result("selected-first")
    second = _result("selected-second")
    first = replace(first, source_outcar="/legacy/struct_0099/OUTCAR")
    second = replace(second, source_outcar="/legacy/struct_0000/OUTCAR")
    state_store = FakeStateStore([first, second])

    result = build_training_dataset(
        tmp_path / "dataset_0001",
        {
            DatasetSplit.TRAIN: (second, first),
            DatasetSplit.TEST: (first,),
        },
        tmp_path,
        allow_partial=True,
        state_store=state_store,
    )

    assert [member.structure_id for _, member in state_store.members] == [
        "selected-second",
        "selected-first",
        "selected-first",
    ]
    assert [member.calculation_id for _, member in state_store.members] == [
        "calculation-selected-second",
        "calculation-selected-first",
        "calculation-selected-first",
    ]
    assert result.metadata["accepted_members"][0]["source_outcar"] == "/legacy/struct_0000/OUTCAR"


def test_changed_authoritative_dft_content_gets_a_new_immutable_dataset(
    tmp_path: Path,
) -> None:
    first = _result("stable")
    changed = replace(first, source_outcar_hash="hash-stable-recomputed", energy_ev=-2.0)

    first_result = build_training_dataset(
        tmp_path / "datasets" / "first",
        {DatasetSplit.TRAIN: (first,), DatasetSplit.TEST: (first,)},
        tmp_path,
        state_store=FakeStateStore([first]),
    )
    changed_result = build_training_dataset(
        tmp_path / "datasets" / "changed",
        {DatasetSplit.TRAIN: (changed,), DatasetSplit.TEST: (changed,)},
        tmp_path,
        state_store=FakeStateStore([changed]),
    )

    assert changed_result.dataset_id != first_result.dataset_id
    assert first_result.dataset_path.is_dir()
    assert changed_result.dataset_path.is_dir()


def test_writer_rejects_missing_required_virial(tmp_path: Path) -> None:
    structure = _result("missing-virial").as_structure_dict()
    structure["virial"] = None
    try:
        write_nep_dataset(tmp_path / "train.xyz", [structure], include_virial=True)
    except ValueError as exc:
        assert str(exc) == "missing_required_virial"
    else:
        raise AssertionError("missing virial must not be silently omitted")
