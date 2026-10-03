from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from nepflow.domain.datasets import DatasetIdentity
from nepflow.domain.identities import ArtifactIdentity, ModelRunIdentity, StructureIdentity, ValidationRunIdentity
from nepflow.domain.models import ModelArtifactMetadata, ModelRunRecord
from nepflow.hpc.jobs import (
    QueueQueryResult,
    ReconciledJobResult,
    SchedulerJobState,
    SlurmJobRecord,
    SubmissionResult,
)
from nepflow.mlip.simulation import StaticPrediction
from nepflow.state.store import StateStore
from nepflow.stages.validation.protocols import ValidationCaseSpec, ValidationReference
from nepflow.stages.validation.reconciliation import ValidationReconciliationOrchestrator


@dataclass
class FakeScheduler:
    states: dict[str, SchedulerJobState]
    query_error: Exception | None = None

    def __init__(self) -> None:
        self.states = {}
        self.submission_count = 0
        self.query_error = None

    def submit(self, command, *, cwd=None, timeout=None, resources=None):  # noqa: ANN001
        self.submission_count += 1
        job_id = f"fake-job-{self.submission_count}"
        self.states[job_id] = SchedulerJobState.PENDING
        return SubmissionResult(job_id, "", "", tuple(str(value) for value in command))

    def list_active_jobs(self, *, user=None, name_prefix=None, timeout=None):  # noqa: ANN001
        if self.query_error is not None:
            raise self.query_error
        return QueueQueryResult(
            tuple(
                SlurmJobRecord(job_id, None, state)
                for job_id, state in self.states.items()
                if state.is_active
            )
        )

    def reconcile(self, job_id, *, timeout=None):  # noqa: ANN001
        state = self.states[job_id]
        job = SlurmJobRecord(job_id, None, state)
        return ReconciledJobResult(job_id, job, "fake")


class FakeGpumdBackend:
    command = ("fake-gpumd",)

    def __init__(self, prediction: StaticPrediction | None = None) -> None:
        self.prediction = prediction
        self.parse_count = 0

    def parse_prediction(self, request, output_path=None):  # noqa: ANN001
        self.parse_count += 1
        assert output_path == request.working_directory / "out.xyz"
        if self.prediction is None:
            raise FileNotFoundError("fixture output is absent")
        return self.prediction


def _fixture(tmp_path: Path, case_count: int = 1):
    dataset = DatasetIdentity.from_identity_payload(
        {"schema_version": "nepflow.dataset.v1", "records": [{"source": "fixture"}]}
    )
    model_identity = ModelRunIdentity(dataset.dataset_id, "nep-input", "hyperparameters")
    model = ModelRunRecord(
        model_identity,
        ModelArtifactMetadata(
            model=ArtifactIdentity.from_bytes("nep_model", b"fixture-model"),
            status="completed",
        ),
    )
    validation_run = ValidationRunIdentity(model.model_run_id, dataset.dataset_id, {"fixture": "reconcile"})
    cases = []
    for ordinal in range(case_count):
        reference = ValidationReference(
            structure=StructureIdentity(f"reconcile-structure-{ordinal}"),
            species=("Si",),
            positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
            cell_angstrom=np.eye(3) * 3.0,
            pbc=(True, True, True),
            energy_ev=-2.0 - ordinal,
            forces_ev_per_angstrom=np.array([[1.0, 0.0, 0.0]]),
        )
        case_dir = tmp_path / f"case-{ordinal}"
        cases.append(
            ValidationCaseSpec.create(
                ordinal=ordinal,
                model_run_id=model.model_run_id,
                dataset_id=dataset.dataset_id,
                reference=reference,
                input_path=case_dir / "model.xyz",
                working_directory=case_dir,
                output_path=case_dir / "out.xyz",
            )
        )
    return dataset, model, validation_run, tuple(cases)


def _store(path: Path, dataset: DatasetIdentity, model: ModelRunRecord) -> StateStore:
    store = StateStore(path)
    store.upsert_dataset(dataset)
    store.upsert_model_run(model, status="completed")
    return store


def _prediction(case: ValidationCaseSpec, model: ModelRunRecord) -> StaticPrediction:
    return StaticPrediction(
        structure=case.reference.structure,
        model_run=model.identity,
        atom_count=case.atom_count,
        energy_ev=-1.5,
        forces_ev_per_angstrom=np.array([[1.25, 0.0, 0.0]]),
        atom_mapping=(0,),
    )


def test_reconciliation_submits_once_then_restarts_from_persisted_job(tmp_path: Path) -> None:
    dataset, model, validation_run, cases = _fixture(tmp_path)
    scheduler = FakeScheduler()
    backend = FakeGpumdBackend(_prediction(cases[0], model))
    store = _store(tmp_path / "state.db", dataset, model)
    try:
        first = ValidationReconciliationOrchestrator(
            validation_run=validation_run,
            model=model,
            cases=cases,
            state_store=store,
            scheduler=scheduler,
            backend=backend,
        )
        submitted = first.reconcile_once()
        assert submitted.status == "running"
        assert submitted.cases[0].status == "submitted"
        assert scheduler.submission_count == 1
        job_id = submitted.cases[0].job_id

        store.close()
        store = StateStore(tmp_path / "state.db")
        restarted = ValidationReconciliationOrchestrator(
            validation_run=validation_run,
            model=model,
            cases=cases,
            state_store=store,
            scheduler=scheduler,
            backend=backend,
        )
        scheduler.states[job_id] = SchedulerJobState.RUNNING
        running = restarted.reconcile_once()
        assert running.cases[0].status == "running"
        assert scheduler.submission_count == 1

        scheduler.states[job_id] = SchedulerJobState.COMPLETED
        completed = restarted.reconcile_once()
        assert completed.status == "completed"
        assert completed.cases[0].status == "completed"
        assert completed.cases[0].prediction is not None
        assert backend.parse_count == 1
        assert store.get_validation_run(validation_run.validation_run_id)["status"] == "completed"
        assert len(store.list_validation_events("validation_attempt", validation_run.validation_run_id)) >= 4
    finally:
        store.close()


def test_reconciliation_does_not_mark_missing_output_successful(tmp_path: Path) -> None:
    dataset, model, validation_run, cases = _fixture(tmp_path)
    scheduler = FakeScheduler()
    backend = FakeGpumdBackend()
    store = _store(tmp_path / "state.db", dataset, model)
    try:
        reconciler = ValidationReconciliationOrchestrator(
            validation_run=validation_run,
            model=model,
            cases=cases,
            state_store=store,
            scheduler=scheduler,
            backend=backend,
        )
        submitted = reconciler.reconcile_once()
        job_id = submitted.cases[0].job_id
        scheduler.states[job_id] = SchedulerJobState.NOT_FOUND
        failed = reconciler.reconcile_once()
        assert failed.status == "failed"
        assert failed.cases[0].failure_reason == "scheduler state not_found"
        assert backend.parse_count == 0
        assert store.get_validation_run(validation_run.validation_run_id)["status"] == "failed"
    finally:
        store.close()


def test_reconciliation_query_failure_leaves_pending_state_untouched(tmp_path: Path) -> None:
    dataset, model, validation_run, cases = _fixture(tmp_path)
    scheduler = FakeScheduler()
    scheduler.query_error = RuntimeError("scheduler unavailable")
    store = _store(tmp_path / "state.db", dataset, model)
    try:
        reconciler = ValidationReconciliationOrchestrator(
            validation_run=validation_run,
            model=model,
            cases=cases,
            state_store=store,
            scheduler=scheduler,
            backend=FakeGpumdBackend(),
        )
        with pytest.raises(RuntimeError, match="unavailable"):
            reconciler.reconcile_once()
        assert reconciler.cases()[0].status == "pending"
        assert store.get_validation_run(validation_run.validation_run_id)["status"] == "pending"
    finally:
        store.close()


def test_reconciliation_enforces_concurrency_before_submitting_cases(tmp_path: Path) -> None:
    dataset, model, validation_run, cases = _fixture(tmp_path, case_count=2)
    scheduler = FakeScheduler()
    store = _store(tmp_path / "state.db", dataset, model)
    try:
        reconciler = ValidationReconciliationOrchestrator(
            validation_run=validation_run,
            model=model,
            cases=cases,
            state_store=store,
            scheduler=scheduler,
            backend=FakeGpumdBackend(),
            max_concurrent=1,
        )
        result = reconciler.reconcile_once()
        assert scheduler.submission_count == 1
        assert [case.status for case in result.cases] == ["submitted", "pending"]
    finally:
        store.close()


def test_reconciliation_retries_unusable_completed_output_without_false_success(
    tmp_path: Path,
) -> None:
    dataset, model, validation_run, cases = _fixture(tmp_path)
    scheduler = FakeScheduler()
    store = _store(tmp_path / "state.db", dataset, model)
    try:
        reconciler = ValidationReconciliationOrchestrator(
            validation_run=validation_run,
            model=model,
            cases=cases,
            state_store=store,
            scheduler=scheduler,
            backend=FakeGpumdBackend(),
            max_attempts=2,
        )
        first = reconciler.reconcile_once()
        scheduler.states[first.cases[0].job_id] = SchedulerJobState.COMPLETED
        retry = reconciler.reconcile_once()
        assert retry.status == "running"
        assert retry.cases[0].status == "pending"
        assert [attempt.status for attempt in reconciler.attempts()] == ["failed", "pending"]

        second = reconciler.reconcile_once()
        scheduler.states[second.cases[0].job_id] = SchedulerJobState.COMPLETED
        failed = reconciler.reconcile_once()
        assert failed.status == "failed"
        assert [attempt.status for attempt in reconciler.attempts()] == ["failed", "failed"]
    finally:
        store.close()
