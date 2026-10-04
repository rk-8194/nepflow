import json
import shutil
import tempfile
from configparser import ConfigParser
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from ase.io import read as ase_read  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402

from nepflow.mlip.nep.artifacts import (  # noqa: E402
    compute_model_run_id,
    create_model_run_manifest,
    update_model_run_status,
    validate_model_run_manifest,
)
from nepflow.domain.datasets import (  # noqa: E402
    DatasetIdentity,
    SelectedDatasetMember,
    TrainingDatasetManifest,
)
from nepflow.domain.identities import (  # noqa: E402
    ArtifactIdentity,
    DftCalculationIdentity,
    ModelRunIdentity,
    StructureIdentity,
    ValidationRunIdentity,
    calculate_structure_id,
)
from nepflow.domain.models import (  # noqa: E402
    ModelArtifactMetadata,
    ModelRunRecord,
    ValidationArtifactMetadata,
    ValidationRunRecord,
)
from nepflow.domain.structures import GeneratedStructureRecord, StructureProvenance  # noqa: E402
from nepflow.dft.vasp.inputs import read_identity  # noqa: E402
from nepflow.dft.vasp.outputs import parse_outcar_result  # noqa: E402
from nepflow.io.hashing import sha256_file  # noqa: E402
from nepflow.io.hashing import sha256_bytes  # noqa: E402
from nepflow.mlip.nep.inputs import NepHyperparameters, NepInputRenderer  # noqa: E402
from nepflow.stages.training.dataset import build_dataset_metadata  # noqa: E402
from nepflow.io.json import write_json  # noqa: E402
from nepflow.stages.validation.resolution import resolve_model_dataset  # noqa: E402
from nepflow.errors import StateError  # noqa: E402
from nepflow.state.store import StateStore  # noqa: E402


TESTS_ROOT = Path(__file__).resolve().parents[1]
OUTCAR_FIXTURE = TESTS_ROOT / "fixtures" / "outcar" / "valid_outcar"
DFT_FIXTURE = TESTS_ROOT / "fixtures" / "structures" / "dft_reference.extxyz.fixture"


def read_dft_fixture_as_vasp_result():
    """Adapt the NEP-facing fixture to ASE's VASP calculator contract."""
    atoms = ase_read(str(DFT_FIXTURE), format="extxyz")
    assert "force" in atoms.arrays
    assert np.asarray(atoms.arrays["force"]).shape == (len(atoms), 3)

    energy = atoms.get_potential_energy()
    forces = np.asarray(atoms.arrays["force"], dtype=float).copy()
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy,
        forces=forces,
    )
    return atoms


def audit_config() -> ConfigParser:
    config = ConfigParser()
    config["composition"] = {"elements": "Si"}
    config["train_nep"] = {"charge_mode": "0"}
    return config


def empty_report(requested_count: int, accepted_results: list) -> dict:
    accepted_count = len(accepted_results)
    return {
        "requested_count": requested_count,
        "accepted_count": accepted_count,
        "rejected_count": requested_count - accepted_count,
        "rejected_reason_counts": {},
        "accepted_results": accepted_results,
    }


def test_deterministic_dft_to_validation_identity_trace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        dataset_path = project_dir / "nep" / "datasets" / "dataset_0001"
        dataset_path.mkdir(parents=True)

        # Selected physical structure -> VASP calculation identity -> exact OUTCAR hash.
        vasp_job = project_dir / "vasp" / "jobs" / "train" / "struct_0000"
        vasp_job.mkdir(parents=True)
        outcar_path = vasp_job / "OUTCAR"
        shutil.copy2(OUTCAR_FIXTURE, outcar_path)
        selected = read_dft_fixture_as_vasp_result()
        calculation_identity = {
            "structure_hash": calculate_structure_id(selected),
            "incar_hash": "incar-hash-v1",
            "potcar_hash": "potcar-hash-v1",
        }
        (vasp_job / ".vasp_identity").write_text(
            json.dumps(calculation_identity), encoding="utf-8"
        )

        accepted = parse_outcar_result(
            outcar_path,
            selected,
            require_virial=True,
            calculation_identity=calculation_identity,
            reader=lambda _path: read_dft_fixture_as_vasp_result(),
            identity_reader=read_identity,
        )

        assert accepted.accepted, accepted.rejection_reason
        assert accepted.structure_id == calculation_identity["structure_hash"]
        assert accepted.calculation_identity == tuple(sorted(calculation_identity.items()))
        assert accepted.source_outcar_hash == sha256_file(outcar_path)

        # Accepted records -> content-derived dataset manifest.
        train_report = {
            "requested_count": 2,
            "accepted_count": 1,
            "rejected_count": 1,
            "rejected_reason_counts": {"completed_outcar_missing": 1},
            "accepted_results": [accepted],
        }
        test_report = empty_report(1, [accepted])
        metadata = build_dataset_metadata(
            dataset_path,
            train_report,
            test_report,
            train_virial=True,
            allow_partial=True,
        )
        dataset_path.mkdir(parents=True, exist_ok=True)
        write_json(dataset_path / ".dataset", metadata)
        shutil.copy2(DFT_FIXTURE, dataset_path / "train.xyz")
        shutil.copy2(DFT_FIXTURE, dataset_path / "test.xyz")

        persisted_dataset = json.loads((dataset_path / ".dataset").read_text(encoding="utf-8"))
        assert persisted_dataset["dataset_id"] == metadata["dataset_id"]
        assert persisted_dataset["requested_train_structures"] == 2
        assert persisted_dataset["accepted_train_structures"] == 1
        assert persisted_dataset["rejected_train_structures"] == 1
        assert persisted_dataset["rejection_reason_counts"] == {"completed_outcar_missing": 1}
        assert persisted_dataset["accepted_calculation_identities"][0]["structure_id"] == accepted.structure_id
        assert persisted_dataset["source_output_hashes"] == [accepted.source_outcar_hash] * 2
        assert persisted_dataset["label_schema"]["virial"] is True
        assert persisted_dataset["units"]["energy"] == "eV"
        assert persisted_dataset["virial_convention"] == "positive_compression"

        # Rendered NEP input -> immutable hyperparameter identity.
        config = audit_config()
        hyperparameters = NepHyperparameters.from_legacy_config(config)
        (dataset_path / "nep.in").write_text(
            NepInputRenderer().render_content(hyperparameters), encoding="utf-8"
        )
        nep_in = dataset_path / "nep.in"
        nep_in_hash = sha256_file(nep_in)
        hyperparameters_hash = hyperparameters.identity_hash()
        variant_config = audit_config()
        variant_config["train_nep"]["charge_mode"] = "1"
        assert (
            NepHyperparameters.from_legacy_config(variant_config).identity_hash()
            != hyperparameters_hash
        )

        # Model manifest binds the exact copied input and completed artifact.
        potential_path = project_dir / "nep" / "potentials" / "potential_0001"
        potential_path.mkdir(parents=True)
        shutil.copy2(nep_in, potential_path / "nep.in")
        (potential_path / "nep.txt").write_text(
            "version 4\ntype 1 Si\ncutoff 6 5 112 60\n", encoding="utf-8"
        )
        with StateStore(project_dir / "state.db") as state_store:
            dataset_identity = DatasetIdentity.from_identity_payload(
                {
                    "schema_version": "nepflow.dataset.v1",
                    "label_schema": metadata["label_schema"],
                    "units": metadata["units"],
                    "virial_convention": metadata["virial_convention"],
                    "virial_tensor_convention": metadata["virial_tensor_convention"],
                    "records": metadata["records"],
                }
            )
            state_store.upsert_dataset(
                TrainingDatasetManifest(
                    identity=dataset_identity,
                    records=tuple(metadata["records"]),
                )
            )
            model_manifest = create_model_run_manifest(
                potential_path=potential_path,
                dataset_path=dataset_path,
                dataset_id=metadata["dataset_id"],
                nep_in_path=potential_path / "nep.in",
                hyperparameters_hash=hyperparameters_hash,
                state_store=state_store,
            )
            completed_manifest = update_model_run_status(
                potential_path,
                "completed",
                state_store=state_store,
                model_run_id=model_manifest["model_run_id"],
            )
        validated_manifest = validate_model_run_manifest(
            potential_path / "model_run_manifest.json",
            expected_model_run_id=model_manifest["model_run_id"],
        )
        assert completed_manifest["model_run_id"] == validated_manifest["model_run_id"]
        assert validated_manifest["dataset_id"] == metadata["dataset_id"]
        assert validated_manifest["nep_in_sha256"] == sha256_file(potential_path / "nep.in")
        assert validated_manifest["potential_artifact_sha256"] == sha256_file(potential_path / "nep.txt")
        assert validated_manifest["nep_in_sha256"] == nep_in_hash

        resolved = resolve_model_dataset(project_dir, validated_manifest["model_run_id"])
        assert resolved.model_path.parent == potential_path.resolve()
        assert resolved.dataset_path == dataset_path.resolve()
        with pytest.raises(StateError, match="explicit model_run_id"):
            resolve_model_dataset(project_dir, "")

        # Storage relocation does not change the scientific dataset/model identities.
        relocated_result = replace(accepted, source_outcar=str(project_dir / "moved" / "OUTCAR"))
        relocated_metadata = build_dataset_metadata(
            project_dir / "moved" / "dataset_0001",
            {**train_report, "accepted_results": [relocated_result]},
            {**test_report, "accepted_results": [relocated_result]},
            train_virial=True,
            allow_partial=True,
        )
        assert relocated_metadata["dataset_id"] == metadata["dataset_id"]
        relocated_nep = project_dir / "moved" / "nep.in"
        relocated_nep.parent.mkdir(parents=True)
        shutil.copy2(potential_path / "nep.in", relocated_nep)
        assert compute_model_run_id(
            dataset_id=metadata["dataset_id"],
            nep_in_sha256=sha256_file(relocated_nep),
            hyperparameters_hash=hyperparameters_hash,
        ) == validated_manifest["model_run_id"]


def test_deterministic_identity_state_trace_reopens_as_one_chain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        selected = read_dft_fixture_as_vasp_result()
        structure = StructureIdentity.from_atoms(selected)
        provenance = StructureProvenance(
            parent_structure_id=None,
            generator="fixture",
            requested_composition={"Si": 1.0},
            realised_composition={"Si": 1.0},
            source_database_id="fixture-db-1",
            crystal_structure="diamond",
            perturbation_family="reference",
            perturbation_parameters={},
            random_seed=42,
            operation_id="trace:generation:0001",
            code_version="trace-v1",
            config_fingerprint="config-v1",
        )
        incar = project_dir / "INCAR"
        incar.write_text("ENCUT = 520\nISMEAR = 0\n", encoding="utf-8")
        potcar = project_dir / "POTCAR"
        potcar.write_text("Si potential fixture\n", encoding="utf-8")
        calculation = DftCalculationIdentity(
            structure_id=structure.structure_id,
            incar_hash=sha256_file(incar),
            potcar_hash=sha256_file(potcar),
        )
        outcar_content = b"accepted fixture output\n"
        outcar_hash = sha256_bytes(outcar_content)
        dataset = DatasetIdentity.from_records(
            [
                {
                    "split": "train",
                    "structure_id": structure.structure_id,
                    "calculation_id": calculation.calculation_id,
                    "source_outcar_hash": outcar_hash,
                }
            ],
            label_schema={"energy": True, "forces": True, "virial": True},
            units={"energy": "eV", "forces": "eV/Angstrom", "virial": "eV"},
            virial_convention="positive_compression",
        )
        outcar = project_dir / "OUTCAR"
        outcar.write_bytes(outcar_content)
        nep_in = project_dir / "nep.in"
        effective_hyperparameters = NepHyperparameters.from_legacy_config(audit_config())
        nep_in.write_text(
            NepInputRenderer().render_content(effective_hyperparameters),
            encoding="utf-8",
        )
        model_file = project_dir / "nep.txt"
        model_file.write_text("version 4\n", encoding="utf-8")
        report = project_dir / "validation.json"
        report.write_text("{\"passed\": true}\n", encoding="utf-8")

        outcar_artifact = ArtifactIdentity.from_file("vasp_outcar", outcar)
        nep_in_artifact = ArtifactIdentity.from_file("nep_in", nep_in)
        model_artifact = ArtifactIdentity.from_file("nep_model", model_file)
        report_artifact = ArtifactIdentity.from_file("validation_report", report)
        model = ModelRunIdentity(
            dataset.dataset_id,
            nep_in_artifact.sha256,
            effective_hyperparameters.identity_hash(),
        )
        assert model.nep_in_sha256 == nep_in_artifact.sha256
        assert model.hyperparameters_hash == effective_hyperparameters.identity_hash()
        validation = ValidationRunIdentity(
            model.model_run_id,
            dataset.dataset_id,
            {"protocol": "static-v1"},
        )
        dataset_manifest = TrainingDatasetManifest(
            identity=dataset,
            records=(
                {
                    "split": "train",
                    "structure_id": structure.structure_id,
                    "calculation_id": calculation.calculation_id,
                    "source_outcar_hash": outcar_artifact.sha256,
                },
            ),
        )
        member = SelectedDatasetMember(
            split="train",
            structure_id=structure.structure_id,
            calculation_id=calculation.calculation_id,
            source_outcar_hash=outcar_artifact.sha256,
            ordinal=0,
            calculation_identity=calculation.to_dict(),
        )
        model_record = ModelRunRecord(
            model,
            ModelArtifactMetadata(
                model=model_artifact,
                nep_in=nep_in_artifact,
                status="completed",
            ),
        )
        validation_record = ValidationRunRecord(
            validation,
            ValidationArtifactMetadata(
                report=report_artifact,
                metrics={"energy_rmse_ev": 0.0},
                thresholds={"energy_rmse_ev": 0.1},
                passed=True,
            ),
        )

        with StateStore(project_dir / "state.db") as store:
            store.upsert_project(
                "trace",
                name="trace",
                root_path=str(project_dir),
                config_fingerprint="config-v1",
                metadata={"schema": "trace-v1"},
            )
            for stage in ("init", "generate", "select", "run_vasp", "train_nep", "validate", "completed"):
                store.upsert_stage_run(
                    f"trace:{stage}",
                    "trace",
                    stage,
                    status="completed",
                )
            store.upsert_structure(
                GeneratedStructureRecord(
                    identity=structure,
                    provenance=provenance,
                    metadata={"fixture": True},
                )
            )
            store.upsert_selection_run(
                "selection-trace-v1",
                "trace",
                status="completed",
                method="fps",
                parameters={"selected_structure_ids": [structure.structure_id]},
            )
            store.upsert_dft_calculation(
                calculation,
                status="running",
                selected=True,
            )
            attempt_id = f"{calculation.calculation_id}:attempt:1"
            store.create_dft_attempt(
                calculation.calculation_id,
                attempt_id,
                attempt_number=1,
                status="running",
                scheduler={"job_id": "trace-job-1"},
            )
            store.register_completed_result(
                calculation.calculation_id,
                attempt_id,
                (outcar_artifact,),
            )
            store.upsert_dataset(dataset_manifest, project_id="trace", status="completed")
            store.record_dataset_member(dataset.dataset_id, member)
            store.upsert_model_run(model_record, status="completed")
            store.upsert_validation_run(validation_record, status="completed")
            store.record_validation_result(
                validation.validation_run_id,
                "energy",
                structure_id=structure.structure_id,
                metric_name="energy_rmse_ev",
                observed_value=0.0,
                threshold=0.1,
                passed=True,
            )
            store.append_event(
                "trace:training:completed",
                "training_campaign",
                model.model_run_id,
                "candidate_completed",
                {"dataset_id": dataset.dataset_id, "model_run_id": model.model_run_id},
            )
            store.append_event(
                "trace:validation:completed",
                "validation_run",
                validation.validation_run_id,
                "metrics_completed",
                {"model_run_id": model.model_run_id, "dataset_id": dataset.dataset_id},
            )

            assert store.get_structure(structure.structure_id)["provenance"]["operation_id"] == provenance.operation_id
            assert store.get_dft_calculation(calculation.calculation_id)["accepted_attempt_id"] == attempt_id
            assert store.get_dft_attempt(attempt_id)["status"] == "completed"
            assert store.get_dataset(dataset.dataset_id)["dataset_id"] == dataset.dataset_id
            assert store.get_model_run(model.model_run_id)["dataset_id"] == dataset.dataset_id
            assert store.get_model_run(model.model_run_id)["nep_in_sha256"] == nep_in_artifact.sha256
            assert store.get_validation_run(validation.validation_run_id)["model_run_id"] == model.model_run_id
            assert store.list_validation_results(validation.validation_run_id)[0]["structure_id"] == structure.structure_id

        with StateStore(project_dir / "state.db") as reopened:
            assert reopened.get_project("trace") is not None
            assert reopened.list_stage_runs("trace")
            assert reopened.list_dft_attempts(calculation.calculation_id)[0]["attempt_id"] == attempt_id
            assert reopened.list_artifacts(originating_attempt_id=attempt_id)[0]["sha256"] == outcar_artifact.sha256
            assert reopened.list_model_artifacts(model.model_run_id)
            assert reopened.list_events(entity_id=validation.validation_run_id)[0]["event_type"] == "metrics_completed"
