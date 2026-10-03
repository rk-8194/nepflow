"""Identity, geometry, and real GPUMD static-output regressions."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase.io import read as ase_read
from ase.io import write as ase_write

from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.domain.identities import ArtifactIdentity, ModelRunIdentity, StructureIdentity
from nepflow.domain.models import ModelArtifactMetadata, ModelRunRecord
from nepflow.mlip.gpumd import GpumdBackend
from nepflow.mlip.simulation import StaticPredictionRequest
from nepflow.errors import MlipError, ValidationError
from nepflow.mlip.nep.artifacts import create_model_run_manifest, update_model_run_status
from nepflow.stages.validation import (
    calculate_cell_replicates_for_cutoff,
    prepare_validation_cases,
    resolve_model_dataset,
)
from nepflow.stages.validation.protocols import (
    VALIDATION_CASE_SCHEMA,
    VALIDATION_PREPARATION_SCHEMA,
    ValidationCaseSpec,
    ValidationPreparation,
)
from modules.validate.analyze import generate_comparison_csv
from nepflow.state.store import StateStore


ROOT = Path(__file__).resolve().parents[2]
DFT_FIXTURE = ROOT / "fixtures" / "structures" / "dft_reference.extxyz.fixture"
ML_FIXTURE = ROOT / "fixtures" / "structures" / "gpumd_static_prediction.extxyz.fixture"


def _record() -> dict:
    atoms = ase_read(str(DFT_FIXTURE), format="extxyz")
    return {
        "split": "test",
        "structure_id": "fixture-structure-0001",
        "species": list(atoms.get_chemical_symbols()),
        "positions": np.asarray(atoms.positions).tolist(),
        "lattice": np.asarray(atoms.cell).tolist(),
        "pbc": [True, True, True],
        "energy": -10.5,
        "forces": np.asarray(atoms.arrays["force"]).tolist(),
        "virial": np.asarray(atoms.info["virial"]).tolist(),
        "calculation_identity": {"calculation_id": "calculation-fixture-0001"},
        "source_outcar_hash": "outcar-fixture-hash",
    }


def _materialize_authoritative_project(root: Path) -> tuple[str, Path]:
    dataset_path = root / "nep" / "datasets" / "dataset_fixture"
    potential_path = root / "nep" / "potentials" / "potential_fixture"
    dataset_path.mkdir(parents=True)
    potential_path.mkdir(parents=True)
    record = _record()
    identity = DatasetIdentity.from_identity_payload(
        {
            "schema_version": "nepflow.dataset.v1",
            "label_schema": {"energy": True, "forces": True, "virial": True},
            "units": {"energy": "eV", "forces": "eV/Angstrom", "virial": "eV"},
            "virial_convention": "positive_compression",
            "records": [record],
        }
    )
    dataset = TrainingDatasetManifest(identity=identity, records=(record,))
    (dataset_path / ".dataset").write_text(
        json.dumps({"dataset_id": identity.dataset_id, "records": [record]}),
        encoding="utf-8",
    )
    (potential_path / "nep.in").write_text("type 1 Si\n", encoding="utf-8")
    (potential_path / "nep.txt").write_text(
        "version 4\ntype 1 Si\ncutoff 6 5 112 60\n", encoding="utf-8"
    )
    with StateStore(root / "state.db") as store:
        store.upsert_dataset(dataset)
        manifest = create_model_run_manifest(
            potential_path=potential_path,
            dataset_path=dataset_path,
            dataset_id=identity.dataset_id,
            nep_in_path=potential_path / "nep.in",
            hyperparameters_hash="fixture-hyperparameters",
            state_store=store,
        )
        update_model_run_status(
            potential_path,
            "completed",
            state_store=store,
            model_run_id=manifest["model_run_id"],
        )
    return manifest["model_run_id"], dataset_path


def test_resolution_and_case_schema_keep_exact_model_dataset_association(tmp_path: Path) -> None:
    model_run_id, dataset_path = _materialize_authoritative_project(tmp_path)

    resolved = resolve_model_dataset(tmp_path, model_run_id)
    assert resolved.model_run_id == model_run_id
    assert resolved.dataset_path == dataset_path.resolve()
    assert resolved.dataset_id == resolved.model_run.identity.dataset_id

    preparation = prepare_validation_cases(tmp_path, model_run_id)
    assert preparation.model_run_id == model_run_id
    assert preparation.dataset_id == resolved.dataset_id
    assert len(preparation.cases) == 1
    case = preparation.cases[0]
    assert case.case_id.startswith("validation_case_")
    assert case.model_run_id == model_run_id
    assert case.dataset_id == resolved.dataset_id
    assert case.reference.energy_ev == -10.5
    assert "Properties=" in case.input_path.read_text(encoding="utf-8").splitlines()[1]
    np.testing.assert_allclose(case.reference.cell_angstrom, np.eye(3) * 3.0)
    assert len(case.atom_mapping) == case.reference.atom_count * int(np.prod(case.replicates))
    assert [item[0] for item in case.atom_mapping].count(0) == int(np.prod(case.replicates))
    restored = ValidationCaseSpec.from_mapping(case.to_dict())
    assert restored.atom_mapping == case.atom_mapping
    malformed = case.to_dict()
    malformed["atom_mapping"] = malformed["atom_mapping"][:-1]
    with pytest.raises(ValidationError, match="atom_mapping"):
        ValidationCaseSpec.from_mapping(malformed)
    duplicate = case.to_dict()
    duplicate["atom_mapping"][-1] = duplicate["atom_mapping"][0]
    with pytest.raises(ValidationError, match="duplicate|cover"):
        ValidationCaseSpec.from_mapping(duplicate)
    missing_mapping = case.to_dict()
    del missing_mapping["atom_mapping"]
    with pytest.raises(ValidationError, match="atom_mapping"):
        ValidationCaseSpec.from_mapping(missing_mapping)

    unknown_case_schema = case.to_dict()
    unknown_case_schema["schema_version"] = "nepflow.validation_case.v999"
    with pytest.raises(ValidationError, match="schema"):
        ValidationCaseSpec.from_mapping(unknown_case_schema)

    preparation = ValidationPreparation(
        model_run_id=model_run_id,
        dataset_id=resolved.dataset_id,
        model_path=resolved.model_path,
        dataset_path=resolved.dataset_path,
        cases=(case,),
    )
    unknown_preparation_schema = preparation.to_dict()
    unknown_preparation_schema["schema_version"] = "nepflow.validation_preparation.v999"
    with pytest.raises(ValidationError, match="schema"):
        ValidationPreparation.from_mapping(unknown_preparation_schema)
    assert preparation.schema_version == VALIDATION_PREPARATION_SCHEMA
    assert case.schema_version == VALIDATION_CASE_SCHEMA


def test_static_prediction_request_rejects_partial_expected_configuration(
    tmp_path: Path,
) -> None:
    model = ModelRunRecord(
        ModelRunIdentity("dataset-fixture", "nep-input", "hyperparameters"),
        ModelArtifactMetadata(
            model=ArtifactIdentity.from_bytes("nep_model", b"model"),
            status="completed",
        ),
    )
    common = dict(
        structure=StructureIdentity("fixture-structure-0001"),
        model=model,
        input_path=tmp_path / "model.xyz",
        working_directory=tmp_path,
        atom_count=2,
        expected_positions_angstrom=np.zeros((2, 3)),
        expected_cell_angstrom=np.eye(3),
        expected_pbc=(True, True, True),
    )
    with pytest.raises(ValidationError, match="expected configuration"):
        StaticPredictionRequest(**common)

    with pytest.raises(ValidationError, match="atom_mapping"):
        StaticPredictionRequest(atom_mapping=(0, 1), **{
            key: value
            for key, value in common.items()
            if not key.startswith("expected_")
        })


def test_triclinic_replicates_use_perpendicular_heights() -> None:
    cell = np.array(
        [[3.0, 0.0, 0.0], [1.0, 2.5, 0.0], [0.3, 0.4, 7.0]],
        dtype=float,
    )
    repeats = calculate_cell_replicates_for_cutoff(cell, 2.0)
    volume = abs(float(np.linalg.det(cell)))
    heights = np.array(
        [
            volume / np.linalg.norm(np.cross(cell[1], cell[2])),
            volume / np.linalg.norm(np.cross(cell[2], cell[0])),
            volume / np.linalg.norm(np.cross(cell[0], cell[1])),
        ]
    )
    assert repeats == (2, 2, 1)
    assert all(repeat * height > 4.0 for repeat, height in zip(repeats, heights))


def test_gpumd_backend_parses_actual_ml_values_and_metadata(tmp_path: Path) -> None:
    model = ModelRunRecord(
        ModelRunIdentity("dataset-fixture", "nep-input", "hyperparameters"),
        ModelArtifactMetadata(
            model=ArtifactIdentity.from_bytes("nep_model", b"model"),
            status="completed",
        ),
    )
    request = StaticPredictionRequest(
        structure=StructureIdentity("fixture-structure-0001"),
        model=model,
        input_path=ML_FIXTURE,
        working_directory=tmp_path,
        atom_count=2,
        virial_requested=True,
    )
    prediction = GpumdBackend().parse_prediction(request, ML_FIXTURE)
    assert prediction.energy_ev == -10.25
    assert prediction.energy_ev != -10.5
    np.testing.assert_allclose(prediction.forces_ev_per_angstrom, [[0.12, 0.01, 0.0], [-0.08, -0.01, 0.0]])
    assert prediction.cell_angstrom is not None
    np.testing.assert_allclose(prediction.cell_angstrom, np.eye(3) * 3.0)
    assert prediction.runtime.command == ("gpumd",)


def test_gpumd_backend_rejects_same_count_wrong_physical_configuration(tmp_path: Path) -> None:
    model = ModelRunRecord(
        ModelRunIdentity("dataset-fixture", "nep-input", "hyperparameters"),
        ModelArtifactMetadata(
            model=ArtifactIdentity.from_bytes("nep_model", b"model"),
            status="completed",
        ),
    )
    request = StaticPredictionRequest(
        structure=StructureIdentity("fixture-structure-0001"),
        model=model,
        input_path=ML_FIXTURE,
        working_directory=tmp_path,
        atom_count=2,
        virial_requested=True,
        expected_species=("Si", "Si"),
        expected_positions_angstrom=np.array([[0.0, 0.0, 0.0], [1.25, 1.5, 1.5]]),
        expected_cell_angstrom=np.eye(3) * 3.0,
        expected_pbc=(True, True, True),
        atom_mapping=(0, 1),
    )
    with pytest.raises(MlipError, match="positions|configuration"):
        GpumdBackend().parse_prediction(request, ML_FIXTURE)


def test_gpumd_backend_rejects_missing_energy_forces_and_requested_virial(tmp_path: Path) -> None:
    model = ModelRunRecord(
        ModelRunIdentity("dataset-fixture", "nep-input", "hyperparameters"),
        ModelArtifactMetadata(
            model=ArtifactIdentity.from_bytes("nep_model", b"model"),
            status="completed",
        ),
    )

    def request() -> StaticPredictionRequest:
        return StaticPredictionRequest(
            structure=StructureIdentity("fixture-structure-0001"),
            model=model,
            input_path=ML_FIXTURE,
            working_directory=tmp_path,
            atom_count=2,
            virial_requested=True,
        )

    missing_energy = tmp_path / "missing-energy.xyz"
    missing_energy.write_text(
        ML_FIXTURE.read_text(encoding="utf-8").replace("energy=-10.2500000000 ", "", 1),
        encoding="utf-8",
    )
    with pytest.raises(MlipError, match="energy"):
        GpumdBackend().parse_prediction(request(), missing_energy)

    missing_virial = tmp_path / "missing-virial.xyz"
    missing_virial.write_text(
        ML_FIXTURE.read_text(encoding="utf-8").replace(
            ' virial="1.1000000000 2.1000000000 3.1000000000 4.1000000000 '
            '5.1000000000 6.1000000000 7.1000000000 8.1000000000 9.1000000000"',
            "",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(MlipError, match="virial|stress"):
        GpumdBackend().parse_prediction(request(), missing_virial)

    missing_forces = ase_read(str(ML_FIXTURE), format="extxyz")
    del missing_forces.arrays["force"]
    missing_forces.info["energy"] = -10.25
    missing_forces.calc = None
    missing_forces_path = tmp_path / "missing-forces.xyz"
    ase_write(str(missing_forces_path), missing_forces, format="extxyz")
    with pytest.raises(MlipError, match="forces"):
        GpumdBackend().parse_prediction(request(), missing_forces_path)


def test_gpumd_backend_materializes_exact_potential_filename_and_checks_artifact_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "authoritative-nep.txt"
    source.write_text("version 4\ntype 1 Si\ncutoff 2\n", encoding="utf-8")
    artifact = ArtifactIdentity.from_file("nep_model", source)
    model = ModelRunRecord(
        ModelRunIdentity("dataset-fixture", "nep-input", "hyperparameters"),
        ModelArtifactMetadata(model=artifact, status="completed"),
    )
    request = StaticPredictionRequest(
        structure=StructureIdentity("fixture-structure-0001"),
        model=model,
        input_path=tmp_path / "not-yet-materialized.xyz",
        working_directory=tmp_path / "case",
        atom_count=1,
        species=("Si",),
        positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        cell_angstrom=np.eye(3) * 3.0,
        pbc=(True, True, True),
    )
    prepared = GpumdBackend(command=("gpumd", "--version")).prepare_inputs(
        request,
        potential_filename="custom-nep.txt",
    )
    assert prepared.input_path.is_file()
    assert (tmp_path / "case" / "custom-nep.txt").read_bytes() == source.read_bytes()
    assert "potential custom-nep.txt" in prepared.content

    source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(MlipError, match="SHA-256"):
        GpumdBackend().prepare_inputs(request, potential_filename="custom-nep.txt")

    missing_file_artifact = ArtifactIdentity(
        artifact.artifact_id,
        artifact.artifact_type,
        artifact.sha256,
        path=str(tmp_path / "missing-nep.txt"),
    )
    missing_file_model = ModelRunRecord(
        model.identity,
        ModelArtifactMetadata(model=missing_file_artifact, status="completed"),
    )
    missing_file_request = StaticPredictionRequest(
        request.structure,
        missing_file_model,
        request.input_path,
        tmp_path / "missing-file-case",
        atom_count=1,
        species=("Si",),
        positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        cell_angstrom=np.eye(3) * 3.0,
        pbc=(True, True, True),
    )
    with pytest.raises(MlipError, match="does not exist"):
        GpumdBackend().prepare_inputs(missing_file_request)

    missing_path_model = ModelRunRecord(
        model.identity,
        ModelArtifactMetadata(
            model=ArtifactIdentity(
                artifact.artifact_id,
                artifact.artifact_type,
                artifact.sha256,
                path=None,
            ),
            status="completed",
        ),
    )
    missing_path_request = StaticPredictionRequest(
        request.structure,
        missing_path_model,
        request.input_path,
        tmp_path / "missing-path-case",
        atom_count=1,
        species=("Si",),
        positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        cell_angstrom=np.eye(3) * 3.0,
        pbc=(True, True, True),
    )
    with pytest.raises(MlipError, match="source path"):
        GpumdBackend().prepare_inputs(missing_path_request)


def test_canonical_case_analysis_uses_backend_ml_values(tmp_path: Path) -> None:
    model_run_id, _ = _materialize_authoritative_project(tmp_path)
    resolved = resolve_model_dataset(tmp_path, model_run_id)
    reference = resolved.test_references()[0]
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "run.in").write_text("replicate 1 1 1\nrun 1\n", encoding="utf-8")
    shutil.copy2(ML_FIXTURE, case_dir / "out.xyz")
    case = ValidationCaseSpec.create(
        ordinal=0,
        model_run_id=model_run_id,
        dataset_id=resolved.dataset_id,
        reference=reference,
        input_path=case_dir / "model.xyz",
        working_directory=case_dir,
        output_path=case_dir / "out.xyz",
        replicates=(1, 1, 1),
        virial_requested=True,
    )
    report_path = tmp_path / "comparison.csv"
    generate_comparison_csv(
        case_dir.parent,
        None,
        report_path,
        cases=(case,),
        model=resolved.model_run,
    )
    row = next(csv.DictReader(report_path.open(newline="", encoding="utf-8")))
    assert float(row["energy_error_per_atom"]) == 0.125
    assert float(row["force_component_mae"]) > 0.0


def test_replicated_case_prediction_returns_reference_atom_mapping(tmp_path: Path) -> None:
    model_run_id, _ = _materialize_authoritative_project(tmp_path)
    resolved = resolve_model_dataset(tmp_path, model_run_id)
    case_dir = tmp_path / "replicated-case"
    case_dir.mkdir()
    case = ValidationCaseSpec.create(
        ordinal=0,
        model_run_id=model_run_id,
        dataset_id=resolved.dataset_id,
        reference=resolved.test_references()[0],
        input_path=case_dir / "model.xyz",
        working_directory=case_dir,
        output_path=case_dir / "out.xyz",
        replicates=(2, 1, 1),
        virial_requested=True,
    )
    output = ase_read(str(ML_FIXTURE), format="extxyz").repeat((2, 1, 1))
    output.info["energy"] = -20.5
    output.info["virial"] = np.asarray(output.info["virial"], dtype=float) * 2.0
    output_path = case_dir / "out.xyz"
    ase_write(str(output_path), output, format="extxyz")

    prediction = GpumdBackend().parse_prediction(
        case.static_prediction_request(resolved.model_run),
        output_path,
    )
    assert prediction.atom_mapping is not None
    assert prediction.atom_mapping.count(0) == 2
    assert prediction.atom_mapping.count(1) == 2
