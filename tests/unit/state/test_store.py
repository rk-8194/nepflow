from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from nepflow.domain.identities import ArtifactIdentity, DftCalculationIdentity, StructureIdentity
from nepflow.errors import StateError
from nepflow.state.store import StateStore


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
