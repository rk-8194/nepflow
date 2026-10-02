from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from nepflow.domain.datasets import DatasetIdentity, SelectedDatasetMember
from nepflow.domain.identities import (
    ArtifactIdentity,
    DftCalculationIdentity,
    ModelRunIdentity,
    StructureIdentity,
    ValidationRunIdentity,
)
from nepflow.domain.models import (
    ModelArtifactMetadata,
    ModelRunRecord,
    ValidationArtifactMetadata,
    ValidationRunRecord,
)
from nepflow.domain.structures import GeneratedStructureRecord, StructureProvenance
from nepflow.errors import StateError
from nepflow.state.store import StateStore


def test_full_phase3_record_chain_serializes_and_reopens(tmp_path) -> None:
    """Exercise every Phase 3 ledger boundary in one transaction."""
    path = tmp_path / "state.db"
    structure = StructureIdentity("structure-fixture")
    provenance = StructureProvenance(
        parent_structure_id=None,
        generator="fixture",
        requested_composition={"Si": 1.0},
        realised_composition={"Si": 1.0},
        source_database_id="fixture-db",
        crystal_structure="bcc",
        perturbation_family=None,
        perturbation_parameters={},
        random_seed=42,
        operation_id="operation-fixture",
        code_version="test",
        config_fingerprint="config-fixture",
    )
    calculation = DftCalculationIdentity(
        structure_id=structure.structure_id,
        incar_hash="incar-fixture",
        potcar_hash="potcar-fixture",
    )
    outcar = ArtifactIdentity.from_bytes("vasp_outcar", b"completed-outcar")
    dataset = DatasetIdentity.from_identity_payload(
        {
            "schema_version": "nepflow.dataset.v1",
            "records": [{"structure_id": structure.structure_id}],
        }
    )
    member = SelectedDatasetMember(
        split="train",
        structure_id=structure.structure_id,
        calculation_id=calculation.calculation_id,
        source_outcar_hash=outcar.sha256,
        ordinal=0,
        calculation_identity=calculation.to_dict(),
    )
    model = ModelRunIdentity.from_inputs(
        dataset.dataset_id,
        "nep-in-fixture",
        "hyperparameters-fixture",
    )
    model_artifact = ArtifactIdentity.from_bytes("nep_model", b"nep-model")
    validation = ValidationRunIdentity(
        model.model_run_id,
        dataset.dataset_id,
        {"temperature_k": 300},
    )
    report = ArtifactIdentity.from_bytes("validation_report", b"report")

    with StateStore(path) as store:
        with store.transaction():
            store.record_project("project-fixture", name="fixture")
            store.record_stage_run(
                "project-fixture:init",
                "project-fixture",
                "init",
                status="running",
            )
            store.record_structure(
                GeneratedStructureRecord(identity=structure, provenance=provenance)
            )
            store.record_dft_calculation(calculation, selected=True)
            store.create_dft_attempt(
                calculation.calculation_id,
                "attempt-fixture",
                attempt_number=1,
                resources={"nodes": 1, "gpus_per_node": 1},
            )
            store.register_completed_result(
                calculation.calculation_id,
                "attempt-fixture",
                [outcar],
            )
            store.record_dataset(dataset, project_id="project-fixture")
            store.record_dataset_member(dataset.dataset_id, member)
            store.record_model_run(
                ModelRunRecord(
                    identity=model,
                    artifact=ModelArtifactMetadata(model=model_artifact),
                ),
                status="completed",
            )
            store.record_validation_run(
                ValidationRunRecord(
                    identity=validation,
                    artifact=ValidationArtifactMetadata(
                        report=report,
                        passed=True,
                    ),
                ),
                status="completed",
            )
            store.record_validation_result(
                validation.validation_run_id,
                "result-fixture",
                structure_id=structure.structure_id,
                metric_name="mae",
                observed_value=0.1,
                threshold=0.2,
                passed=True,
            )
            store.append_event(
                "event-fixture",
                "validation_run",
                validation.validation_run_id,
                "completed",
                {"source": "test"},
            )

        assert store.schema_version == 1
        assert store.get_project("project-fixture") is not None
        assert store.get_stage_run("project-fixture:init")["stage"] == "init"
        assert store.get_structure(structure.structure_id)["provenance"]["operation_id"] == "operation-fixture"
        assert store.get_dft_calculation(calculation.calculation_id)["status"] == "completed"
        assert store.list_dft_attempts(calculation.calculation_id)[0]["status"] == "completed"
        assert store.list_dataset_members(dataset.dataset_id)[0]["source_outcar_hash"] == outcar.sha256
        assert store.get_model_run(model.model_run_id)["status"] == "completed"
        assert store.get_validation_run(validation.validation_run_id)["status"] == "completed"
        assert store.list_validation_results(validation.validation_run_id)[0]["passed"] == 1
        assert store.list_events(entity_id=validation.validation_run_id, entity_type="validation_run")

    with StateStore(path) as reopened:
        assert reopened.get_project("project-fixture") is not None
        assert reopened.get_validation_run(validation.validation_run_id) is not None
        assert len(reopened.list_artifacts()) == 3


def test_transaction_rolls_back_all_ledger_changes(tmp_path) -> None:
    with StateStore(tmp_path / "state.db") as store:
        with pytest.raises(RuntimeError, match="abort"):
            with store.transaction():
                store.upsert_project("project-1", name="demo")
                raise RuntimeError("abort")

        assert store.get_project("project-1") is None


def test_project_and_structure_writes_are_idempotent(tmp_path) -> None:
    structure = StructureIdentity("structure-1")

    with StateStore(tmp_path / "state.db") as store:
        first = store.upsert_project("project-1", name="demo", root_path="/tmp/demo")
        second = store.upsert_project("project-1", name="demo", root_path="/tmp/demo")
        assert first["project_id"] == second["project_id"] == "project-1"

        store.upsert_structure(structure, metadata={"source": "fixture"})
        store.upsert_structure(structure, metadata={"source": "fixture"})
        assert store.get_structure("structure-1")["structure_id"] == "structure-1"


def test_attempt_history_is_preserved_and_lookup_is_idempotent(tmp_path) -> None:
    calculation = DftCalculationIdentity(
        structure_id="structure-1",
        incar_hash="incar-hash",
        potcar_hash="potcar-hash",
    )

    with StateStore(tmp_path / "state.db") as store:
        store.upsert_structure(StructureIdentity("structure-1"))
        store.upsert_dft_calculation(calculation)
        store.create_dft_attempt(calculation.calculation_id, "attempt-1", attempt_number=1)
        store.create_dft_attempt(calculation.calculation_id, "attempt-1", attempt_number=1)
        store.create_dft_attempt(calculation.calculation_id, "attempt-2", attempt_number=2)

        attempts = store.list_dft_attempts(calculation.calculation_id)
        assert [attempt["attempt_id"] for attempt in attempts] == ["attempt-1", "attempt-2"]


def test_completed_result_registration_is_concurrency_safe(tmp_path) -> None:
    path = tmp_path / "state.db"
    calculation = DftCalculationIdentity(
        structure_id="structure-1",
        incar_hash="incar-hash",
        potcar_hash="potcar-hash",
    )
    artifact = ArtifactIdentity.from_bytes("outcar", b"completed")

    with StateStore(path) as store:
        store.upsert_structure(StructureIdentity("structure-1"))
        store.upsert_dft_calculation(calculation)
        store.create_dft_attempt(calculation.calculation_id, "attempt-1", attempt_number=1)

    def register() -> None:
        with StateStore(path) as store:
            store.register_completed_result(
                calculation.calculation_id,
                "attempt-1",
                [artifact],
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: register(), range(2)))

    with StateStore(path) as store:
        calculation_row = store.get_dft_calculation(calculation.calculation_id)
        assert calculation_row["status"] == "completed"
        assert calculation_row["accepted_attempt_id"] == "attempt-1"
        artifacts = store.list_artifacts(originating_attempt_id="attempt-1")
        assert len(artifacts) == 1
        assert artifacts[0]["artifact_id"] == artifact.artifact_id


def test_conflicting_completed_attempt_is_rejected(tmp_path) -> None:
    calculation = DftCalculationIdentity(
        structure_id="structure-1",
        incar_hash="incar-hash",
        potcar_hash="potcar-hash",
    )
    first_artifact = ArtifactIdentity.from_bytes("outcar", b"first")
    second_artifact = ArtifactIdentity.from_bytes("outcar", b"second")

    with StateStore(tmp_path / "state.db") as store:
        store.upsert_structure(StructureIdentity("structure-1"))
        store.upsert_dft_calculation(calculation)
        store.create_dft_attempt(calculation.calculation_id, "attempt-1", attempt_number=1)
        store.create_dft_attempt(calculation.calculation_id, "attempt-2", attempt_number=2)
        store.register_completed_result(calculation.calculation_id, "attempt-1", [first_artifact])

        with pytest.raises(StateError, match="already accepted"):
            store.register_completed_result(
                calculation.calculation_id,
                "attempt-2",
                [second_artifact],
            )
