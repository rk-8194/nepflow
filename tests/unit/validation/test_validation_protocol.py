"""Identity, geometry, and real GPUMD static-output regressions."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.domain.identities import ArtifactIdentity, ModelRunIdentity, StructureIdentity
from nepflow.domain.models import ModelArtifactMetadata, ModelRunRecord
from nepflow.mlip.gpumd import GpumdBackend
from nepflow.mlip.simulation import StaticPredictionRequest
from nepflow.mlip.nep.artifacts import create_model_run_manifest, update_model_run_status
from nepflow.stages.validation import (
    calculate_cell_replicates_for_cutoff,
    prepare_validation_cases,
    resolve_model_dataset,
)
from nepflow.stages.validation.protocols import ValidationCaseSpec
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
