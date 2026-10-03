from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nepflow.dft.vasp.outputs import VaspParseResult
from nepflow.stages.training.dataset import (
    DatasetSplit,
    build_training_dataset,
    write_nep_dataset,
)


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
    result = build_training_dataset(
        tmp_path / "dataset_0001",
        {
            DatasetSplit.TRAIN: (_result("train-1"), _result("train-rejected", accepted=False)),
            DatasetSplit.TEST: (_result("test-1"),),
        },
        tmp_path,
        train_virial=True,
        allow_partial=True,
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


def test_writer_rejects_missing_required_virial(tmp_path: Path) -> None:
    structure = _result("missing-virial").as_structure_dict()
    structure["virial"] = None
    try:
        write_nep_dataset(tmp_path / "train.xyz", [structure], include_virial=True)
    except ValueError as exc:
        assert str(exc) == "missing_required_virial"
    else:
        raise AssertionError("missing virial must not be silently omitted")
