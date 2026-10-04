import hashlib
import json

import numpy as np
import pytest
from ase import Atoms

from nepflow.domain import (
    ArtifactIdentity,
    DatasetIdentity,
    DescriptorCacheIdentity,
    DftCalculationIdentity,
    ModelRunIdentity,
    SelectedDatasetMember,
    StructureIdentity,
    StructureProvenance,
    ValidationRunIdentity,
    calculate_structure_id,
    canonical_structure_text,
)
from nepflow.domain.models import ModelArtifactMetadata, ModelRunRecord
from nepflow.state.store import StateStore
from nepflow.domain.units import stress_kbar_to_ev_per_angstrom3, virial_from_stress


def test_model_run_rejects_nep_input_artifact_identity_mismatch(tmp_path) -> None:
    identity = ModelRunIdentity("dataset-1", "a" * 64, "hyperparameters-1")
    model = ArtifactIdentity.from_bytes("nep_model", b"model")
    nep_in = ArtifactIdentity.from_bytes("nep_in", b"different-input")

    with StateStore(tmp_path / "state.db") as store:
        with pytest.raises(ValueError, match="nep_in_sha256|nep\\.in"):
            store.upsert_model_run(
                ModelRunRecord(
                    identity,
                    ModelArtifactMetadata(
                        model=model,
                        nep_in=nep_in,
                        status="completed",
                    ),
                ),
                status="completed",
            )

        assert store.get_model_run(identity.model_run_id) is None


def test_structure_identity_preserves_phase2_canonical_bytes() -> None:
    atoms = Atoms(
        symbols=["Si", "C", "Si"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.5, 0.75], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )
    text = canonical_structure_text(atoms)
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert calculate_structure_id(atoms) == expected
    assert StructureIdentity.from_atoms(atoms).structure_id == expected
    assert "config_type" not in text


def test_structure_identity_ignores_relocation_and_mutable_annotations() -> None:
    first = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4.0, pbc=True)
    second = first.copy()
    second.info["source_xyz"] = "/different/location/selected.xyz"
    second.info["config_type"] = "changed"
    assert calculate_structure_id(first) == calculate_structure_id(second)


def test_dft_identity_excludes_execution_resources() -> None:
    base = DftCalculationIdentity("s1", "incar", "potcar", nodes="a", ncore=2, kpar=1, gpus=1)
    changed_resources = DftCalculationIdentity(
        "s1", "incar", "potcar", nodes="b", ncore=16, kpar=8, gpus=4, walltime="01:00:00"
    )
    assert base.calculation_id == changed_resources.calculation_id
    assert base.to_dict()["resources"] != changed_resources.to_dict()["resources"]
    assert DftCalculationIdentity("s1", "changed-incar", "potcar").calculation_id != base.calculation_id


def test_model_identity_matches_phase2_payload() -> None:
    identity = ModelRunIdentity.from_inputs("dataset_x", "nep_sha", "params_sha")
    payload = {
        "schema_version": "nepflow.model_run_identity.v1",
        "dataset_id": "dataset_x",
        "nep_in_sha256": "nep_sha",
        "hyperparameters_hash": "params_sha",
    }
    expected = "model_run_" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert identity.model_run_id == expected


def test_dataset_identity_is_ordered_and_path_invariant() -> None:
    records = [
        {
            "split": "train",
            "structure_id": "s1",
            "source_outcar_hash": "o1",
            "source_outcar": "/old/OUTCAR",
            "energy": 1.0,
        },
        {
            "split": "test",
            "structure_id": "s2",
            "source_outcar_hash": "o2",
            "source_outcar": "/old/test.OUTCAR",
            "energy": 2.0,
        },
    ]
    first = DatasetIdentity.from_records(
        records,
        label_schema={"version": "v1"},
        units={"energy": "eV"},
        virial_convention=None,
    )
    second = DatasetIdentity.from_records(
        list(reversed(records)),
        label_schema={"version": "v1"},
        units={"energy": "eV"},
        virial_convention=None,
    )
    assert first.dataset_id != second.dataset_id
    relocated = DatasetIdentity.from_records(
        [
            {**record, "source_outcar": "/new/location/OUTCAR"}
            for record in records
        ],
        label_schema={"version": "v1"},
        units={"energy": "eV"},
        virial_convention=None,
    )
    assert relocated.dataset_id == first.dataset_id


def test_selected_dataset_member_promotes_calculation_id() -> None:
    member = SelectedDatasetMember.from_mapping(
        {
            "split": "train",
            "structure_id": "s1",
            "calculation_identity": {
                "structure_hash": "s1",
                "incar_hash": "incar",
                "potcar_hash": "potcar",
            },
            "source_outcar_hash": "outcar",
        },
        ordinal=0,
    )
    assert member.to_dict()["calculation_id"].startswith("calculation_")


def test_descriptor_cache_identity_preserves_order_and_manifest_fields() -> None:
    identity = DescriptorCacheIdentity(("s1", "s2"), "nep89.txt", "model-sha", True, (2, 4))
    assert identity.to_manifest() == {
        "schema_version": "descriptor-cache-v1",
        "structure_ids": ["s1", "s2"],
        "model": {"filename": "nep89.txt", "sha256": "model-sha"},
        "settings": {"mean_descriptor": True},
        "descriptor_shape": [2, 4],
    }


def test_validation_and_artifact_identity_are_content_based() -> None:
    validation = ValidationRunIdentity("model_1", "dataset_1", {"threshold": 0.1})
    assert validation.validation_run_id == ValidationRunIdentity(
        "model_1", "dataset_1", {"threshold": 0.1}
    ).validation_run_id
    assert validation.validation_run_id != ValidationRunIdentity(
        "model_1", "dataset_1", {"threshold": 0.2}
    ).validation_run_id
    artifact = ArtifactIdentity.from_bytes("report", b"report", path="/old/report.json")
    relocated_artifact = ArtifactIdentity.from_bytes("report", b"report", path="/new/report.json")
    assert artifact.artifact_id == relocated_artifact.artifact_id


def test_units_keep_explicit_positive_compression_sign() -> None:
    np.testing.assert_allclose(virial_from_stress(np.eye(3), 10.0), -np.eye(3) * 10.0)
    np.testing.assert_allclose(
        stress_kbar_to_ev_per_angstrom3(np.eye(3) * 1602.17663),
        np.eye(3),
    )


def test_provenance_record_is_immutable() -> None:
    provenance = StructureProvenance(
        parent_structure_id=None,
        generator="sqs",
        requested_composition={"Si": 2},
        realised_composition={"Si": 2},
        source_database_id=None,
        crystal_structure="diamond",
        perturbation_family=None,
        perturbation_parameters={},
        random_seed=7,
        operation_id="op-1",
        code_version="dev",
        config_fingerprint="cfg",
    )
    with pytest.raises(TypeError):
        provenance.requested_composition["Si"] = 3

