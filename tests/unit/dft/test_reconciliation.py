"""State-table tests for the canonical DFT execution reconciler."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nepflow.dft.backend import (
    DftCompletionEvidence,
    DftFailure,
    DftFailureEvidence,
    DftInputArtifacts,
    DftResult,
)
from nepflow.domain.calculations import DftResultArtifact
from nepflow.domain.identities import (
    ArtifactIdentity,
    DftCalculationIdentity,
    StructureIdentity,
)
from nepflow.hpc.jobs import (
    QueueQueryResult,
    ReconciledJobResult,
    SchedulerJobState,
    SlurmJobRecord,
    SubmissionResult,
)
from nepflow.stages.dft.orchestrator import DftPreparationResult
from nepflow.stages.dft.reconciliation import (
    DftExecutionRecord,
    DftReconciliationOrchestrator,
    DftReconciliationResult,
    DftRecoveryDecision,
)
from nepflow.stages.dft.stage import DftStageResult
from nepflow.workflow.resubmission import SelfResubmitExit
from nepflow.workflow.stages import StageRunState, WorkflowStage


@dataclass
class FakeStateStore:
    saved: list[tuple[DftExecutionRecord, Any, str | None]] = field(default_factory=list)

    def save_execution(
        self,
        record: DftExecutionRecord,
        *,
        artifact: Any = None,
        reason: str | None = None,
    ) -> None:
        self.saved.append((record, artifact, reason))


class FakeScheduler:
    def __init__(
        self,
        jobs: dict[str, ReconciledJobResult] | None = None,
        *,
        query_error: Exception | None = None,
    ) -> None:
        self.jobs = {} if jobs is None else jobs
        self.query_error = query_error
        self.active: tuple[SlurmJobRecord, ...] = ()
        self.submissions: list[Path] = []
        self.submission_kwargs: list[dict[str, Any]] = []

    def list_active_jobs(self, **_kwargs: Any) -> QueueQueryResult:
        if self.query_error is not None:
            raise self.query_error
        return QueueQueryResult(self.active)

    def reconcile(self, job_id: str, **_kwargs: Any) -> ReconciledJobResult:
        return self.jobs[job_id]

    def submit_script(self, script_path: str | Path, **kwargs: Any) -> SubmissionResult:
        self.submissions.append(Path(script_path))
        self.submission_kwargs.append(kwargs)
        return SubmissionResult(
            job_id=f"job-{len(self.submissions)}",
            stdout="",
            stderr="",
            command=("sbatch", str(script_path)),
        )


class LifecycleScheduler(FakeScheduler):
    """Fake scheduler whose submitted jobs occupy slots until reconciled."""

    def __init__(self) -> None:
        super().__init__()
        self.max_observed_active = 0
        self.active_ids: set[str] = set()

    def list_active_jobs(self, **kwargs: Any) -> QueueQueryResult:
        result = super().list_active_jobs(**kwargs)
        self.max_observed_active = max(self.max_observed_active, len(result.active_jobs))
        return result

    def submit_script(self, script_path: str | Path, **kwargs: Any) -> SubmissionResult:
        submission = super().submit_script(script_path, **kwargs)
        self.jobs[submission.job_id] = _job(submission.job_id, SchedulerJobState.NOT_FOUND)
        self.active = tuple(
            SlurmJobRecord(job_id=job_id, name="dft", state=SchedulerJobState.PENDING)
            for job_id in sorted(self.active_ids | {submission.job_id})
        )
        self.active_ids.add(submission.job_id)
        return submission

    def reconcile(self, job_id: str, **kwargs: Any) -> ReconciledJobResult:
        result = super().reconcile(job_id, **kwargs)
        self.active_ids.discard(job_id)
        self.active = tuple(item for item in self.active if item.job_id != job_id)
        return result


class FakeBackend:
    def __init__(
        self,
        *,
        completed: bool = False,
        failure: DftFailure | None = None,
    ) -> None:
        self.completed = completed
        self.failure = failure or DftFailure(
            kind="incomplete_output",
            recoverable=True,
            reason="output is incomplete",
        )
        self.completion_calls = 0
        self.result_calls = 0
        self.failure_calls = 0
        self.failure_evidence: list[DftFailureEvidence] = []

    def parse_completion(self, inputs: DftInputArtifacts, process=None) -> DftCompletionEvidence:
        del inputs, process
        self.completion_calls += 1
        return DftCompletionEvidence(completed=self.completed)

    def parse_result(self, inputs: DftInputArtifacts) -> DftResult:
        self.result_calls += 1
        artifact = DftResultArtifact(
            calculation=inputs.calculation,
            outcar=ArtifactIdentity.from_bytes("vasp_outcar", b"completed"),
            status="completed",
        )
        return DftResult(
            structure=StructureIdentity(inputs.calculation.structure_id),
            calculation=inputs.calculation,
            energy_ev=-1.0,
            forces_ev_per_angstrom=np.zeros((1, 3)),
            artifact=artifact,
            requirements=inputs.requirements,
        )

    def classify_failure(self, evidence: DftFailureEvidence) -> DftFailure:
        self.failure_evidence.append(evidence)
        self.failure_calls += 1
        return self.failure


def _job(job_id: str, state: SchedulerJobState) -> ReconciledJobResult:
    return ReconciledJobResult(
        job_id=job_id,
        job=SlurmJobRecord(job_id=job_id, name="dft", state=state),
        source="fake",
    )


@pytest.fixture
def inputs(tmp_path: Path) -> DftInputArtifacts:
    calculation = DftCalculationIdentity(
        structure_id="structure-1",
        incar_hash="incar-1",
        potcar_hash="potcar-1",
    )
    return DftInputArtifacts(
        calculation=calculation,
        working_directory=tmp_path / "structure-1",
        files=(),
    )


@pytest.fixture
def record(inputs: DftInputArtifacts) -> DftExecutionRecord:
    return DftExecutionRecord(
        inputs=inputs,
        attempt_id="attempt-1",
        status="submitted",
        job_id="job-1",
    )


@pytest.mark.parametrize(
    ("initial_status", "job_state", "completed", "failure", "expected_status"),
    (
        ("completed", None, False, None, "completed"),
        ("reused", None, False, None, "reused"),
        ("failed", None, False, None, "failed"),
        ("submitted", SchedulerJobState.PENDING, False, None, "submitted"),
        ("submitted", SchedulerJobState.RUNNING, False, None, "running"),
        ("submitted", SchedulerJobState.NOT_FOUND, True, None, "completed"),
        (
            "submitted",
            SchedulerJobState.NOT_FOUND,
            False,
            DftFailure("out_of_memory", True, "retry with more memory"),
            "pending",
        ),
        (
            "submitted",
            SchedulerJobState.FAILED,
            False,
            DftFailure("execution_failed", False, "VASP failed"),
            "failed",
        ),
        ("submitted", SchedulerJobState.NOT_FOUND, False, None, "pending"),
    ),
)
def test_reconciliation_state_table(
    inputs: DftInputArtifacts,
    record: DftExecutionRecord,
    initial_status: str,
    job_state: SchedulerJobState | None,
    completed: bool,
    failure: DftFailure | None,
    expected_status: str,
) -> None:
    store = FakeStateStore()
    scheduler = FakeScheduler({"job-1": _job("job-1", job_state)} if job_state is not None else {})
    backend = FakeBackend(completed=completed, failure=failure)
    reconciler = DftReconciliationOrchestrator(
        scheduler=scheduler,
        backend=backend,
        state_store=store,
        retry_limit=2,
        script_path=Path("run_vasp.sh"),
    )

    current = record if initial_status == record.status else record.with_status(initial_status)
    result = reconciler.reconcile_once((current,))

    assert result.records[0].status == expected_status
    if initial_status in {"completed", "reused", "failed"}:
        assert store.saved == []
    elif expected_status == "completed":
        assert len(store.saved) == 1
        assert store.saved[0][1] is not None
        assert backend.result_calls == 1
    else:
        assert len(store.saved) == (2 if expected_status == "pending" else 1)
        if expected_status == "pending":
            assert store.saved[0][0].status == "failed"
            assert store.saved[1][0].status == "pending"
            assert store.saved[0][0].attempt_id != store.saved[1][0].attempt_id


def test_scheduler_query_failure_does_not_advance_state(
    inputs: DftInputArtifacts,
    record: DftExecutionRecord,
) -> None:
    store = FakeStateStore()
    scheduler = FakeScheduler(query_error=RuntimeError("scheduler unavailable"))
    reconciler = DftReconciliationOrchestrator(
        scheduler=scheduler,
        backend=FakeBackend(),
        state_store=store,
        script_path=Path("run_vasp.sh"),
    )

    with pytest.raises(RuntimeError, match="scheduler unavailable"):
        reconciler.reconcile_once((record,))

    assert store.saved == []


def test_recoverable_failure_uses_typed_recovery_decision(
    record: DftExecutionRecord,
) -> None:
    store = FakeStateStore()
    scheduler = FakeScheduler({"job-1": _job("job-1", SchedulerJobState.OOM)})
    decisions: list[tuple[str, int]] = []

    def recovery_policy(current, failure) -> DftRecoveryDecision:
        decisions.append((failure.kind, current.retry_level))
        return DftRecoveryDecision(
            retry=True,
            retry_level=current.retry_level + 1,
            reason="typed-recovery",
        )

    reconciler = DftReconciliationOrchestrator(
        scheduler=scheduler,
        backend=FakeBackend(failure=DftFailure("out_of_memory", True, "OOM")),
        state_store=store,
        script_path=Path("run_vasp.sh"),
        recovery_policy=recovery_policy,
    )

    result = reconciler.reconcile_once((record,))

    assert result.records[0].status == "pending"
    assert decisions == [("out_of_memory", 0)]
    assert store.saved[1][2] == "typed-recovery"
    assert store.saved[0][0].failure_evidence == {
        "kind": "out_of_memory",
        "reason": "OOM",
        "scheduler_state": "oom",
        "scheduler_reason": None,
        "scheduler_exit_code": None,
        "runner_exit_code": None,
    }


def test_terminal_scheduler_evidence_is_forwarded_to_backend(
    inputs: DftInputArtifacts,
) -> None:
    store = FakeStateStore()
    scheduler = FakeScheduler(
        {
            "job-1": ReconciledJobResult(
                job_id="job-1",
                job=SlurmJobRecord(
                    job_id="job-1",
                    name="dft",
                    state=SchedulerJobState.FAILED,
                    reason="missing executable",
                    exit_code=127,
                ),
                source="fake",
            )
        }
    )
    backend = FakeBackend(failure=DftFailure("vasp_execution_failed", False, "missing executable"))
    record = DftExecutionRecord(
        inputs=inputs,
        attempt_id="attempt-1",
        status="submitted",
        job_id="job-1",
    )
    reconciler = DftReconciliationOrchestrator(
        scheduler=scheduler,
        backend=backend,
        state_store=store,
        script_path=Path("run_vasp.sh"),
    )

    result = reconciler.reconcile_once((record,))

    assert result.records[0].status == "failed"
    evidence = backend.failure_evidence[0]
    assert evidence.working_directory == inputs.working_directory
    assert evidence.scheduler_state == "failed"
    assert evidence.scheduler_reason == "missing executable"
    assert evidence.scheduler_exit_code == 127
    assert store.saved[0][0].failure_evidence == {
        "kind": "vasp_execution_failed",
        "reason": "missing executable",
        "scheduler_state": "failed",
        "scheduler_reason": "missing executable",
        "scheduler_exit_code": 127,
        "runner_exit_code": None,
    }


def test_pending_submits_when_capacity_is_available(inputs: DftInputArtifacts) -> None:
    store = FakeStateStore()
    scheduler = FakeScheduler()
    backend = FakeBackend()
    reconciler = DftReconciliationOrchestrator(
        scheduler=scheduler,
        backend=backend,
        state_store=store,
        max_concurrent=1,
        script_path=Path("run_vasp.sh"),
    )
    pending = DftExecutionRecord(inputs=inputs, attempt_id="attempt-1", status="pending")

    result = reconciler.reconcile_once((pending,))

    assert result.records[0].status == "submitted"
    assert result.records[0].job_id == "job-1"
    assert scheduler.submissions == [Path("run_vasp.sh")]
    assert scheduler.submission_kwargs == [
        {
            "job_name": "structure-1",
            "resources": None,
            "stdout_path": inputs.working_directory / "vasp_%j.out",
            "stderr_path": inputs.working_directory / "vasp_%j.err",
            "cwd": inputs.working_directory,
        }
    ]


def test_pending_waits_when_capacity_is_full(inputs: DftInputArtifacts) -> None:
    store = FakeStateStore()
    scheduler = FakeScheduler()
    scheduler.active = (
        SlurmJobRecord(job_id="other", name="dft", state=SchedulerJobState.RUNNING),
    )
    reconciler = DftReconciliationOrchestrator(
        scheduler=scheduler,
        backend=FakeBackend(),
        state_store=store,
        max_concurrent=1,
        script_path=Path("run_vasp.sh"),
    )
    pending = DftExecutionRecord(inputs=inputs, attempt_id="attempt-1", status="pending")

    result = reconciler.reconcile_once((pending,))

    assert result.records == (pending,)
    assert scheduler.submissions == []
    assert store.saved == []


def test_reconcile_until_terminal_refills_capacity_until_all_five_complete(
    inputs: DftInputArtifacts,
) -> None:
    scheduler = LifecycleScheduler()
    store = FakeStateStore()
    sleeps: list[float] = []
    reconciler = DftReconciliationOrchestrator(
        scheduler=scheduler,
        backend=FakeBackend(completed=True),
        state_store=store,
        max_concurrent=2,
        script_path=Path("run_vasp.sh"),
        sleeper=sleeps.append,
    )
    records = tuple(
        DftExecutionRecord(
            inputs=replace(
                inputs,
                calculation=DftCalculationIdentity(
                    structure_id=f"structure-{index}",
                    incar_hash="incar-1",
                    potcar_hash="potcar-1",
                ),
            ),
            attempt_id=f"attempt-{index}",
            status="pending",
        )
        for index in range(5)
    )

    result = reconciler.reconcile_until_terminal(
        records,
        poll_interval_seconds=1,
        slurm_deadline=None,
    )

    assert result.all_terminal is True
    assert result.all_successful is True
    assert len(scheduler.submissions) == 5
    assert scheduler.max_observed_active <= 2
    assert len(sleeps) > 0


def test_reconcile_until_terminal_does_not_stop_for_one_terminal_failure(
    inputs: DftInputArtifacts,
) -> None:
    class MixedBackend(FakeBackend):
        def parse_completion(self, current_inputs, process=None):
            self.completion_calls += 1
            return DftCompletionEvidence(
                completed=current_inputs.calculation.structure_id != "structure-0"
            )

        def classify_failure(self, evidence):
            self.failure_calls += 1
            return DftFailure("execution_failed", False, "terminal failure")

    scheduler = LifecycleScheduler()
    store = FakeStateStore()
    reconciler = DftReconciliationOrchestrator(
        scheduler=scheduler,
        backend=MixedBackend(completed=True),
        state_store=store,
        max_concurrent=2,
        script_path=Path("run_vasp.sh"),
        sleeper=lambda _seconds: None,
    )
    records = tuple(
        DftExecutionRecord(
            inputs=replace(
                inputs,
                calculation=DftCalculationIdentity(
                    structure_id=f"structure-{index}",
                    incar_hash="incar-1",
                    potcar_hash="potcar-1",
                ),
            ),
            attempt_id=f"attempt-{index}",
            status="pending",
        )
        for index in range(3)
    )

    result = reconciler.reconcile_until_terminal(
        records,
        poll_interval_seconds=0,
        slurm_deadline=None,
    )

    assert result.all_terminal is True
    assert result.any_failed is True
    assert result.records[0].status == "failed"
    assert result.records[1].status == "completed"
    assert result.records[2].status == "completed"
    assert len(scheduler.submissions) == 3


def test_reconcile_until_terminal_raises_resubmit_after_persisting_pass(
    inputs: DftInputArtifacts,
) -> None:
    scheduler = FakeScheduler()
    store = FakeStateStore()
    clock_values = iter((0.0, 10.0))
    reconciler = DftReconciliationOrchestrator(
        scheduler=scheduler,
        backend=FakeBackend(),
        state_store=store,
        script_path=Path("run_vasp.sh"),
        clock=lambda: next(clock_values),
    )
    record = DftExecutionRecord(inputs=inputs, attempt_id="attempt-1", status="pending")

    with pytest.raises(SelfResubmitExit):
        reconciler.reconcile_until_terminal(
            (record,),
            poll_interval_seconds=20,
            slurm_deadline=5.0,
        )

    assert scheduler.submissions == [Path("run_vasp.sh")]
    assert store.saved[0][0].status == "submitted"


def test_restart_does_not_duplicate_submission(inputs: DftInputArtifacts) -> None:
    scheduler = FakeScheduler({"job-1": _job("job-1", SchedulerJobState.PENDING)})
    store = FakeStateStore()
    reconciler = DftReconciliationOrchestrator(
        scheduler=scheduler,
        backend=FakeBackend(),
        state_store=store,
        script_path=Path("run_vasp.sh"),
    )
    submitted = DftExecutionRecord(
        inputs=inputs,
        attempt_id="attempt-1",
        status="submitted",
        job_id="job-1",
    )

    result = reconciler.reconcile_once((submitted,))

    assert result.records == (submitted,)
    assert scheduler.submissions == []


def test_failed_dft_stage_does_not_advance_to_training(
    record: DftExecutionRecord,
) -> None:
    failed = record.with_status("failed")
    stage_result = DftStageResult(
        preparation=DftPreparationResult(()),
        execution=DftReconciliationResult((failed,)),
    )

    workflow_result = stage_result.as_workflow_result()

    assert workflow_result.stage is WorkflowStage.RUN_VASP
    assert workflow_result.status is StageRunState.FAILED
    assert workflow_result.completed is False
    assert workflow_result.advanced_to is None


def test_nonterminal_dft_snapshot_with_one_failure_stays_running(
    record: DftExecutionRecord,
) -> None:
    snapshot = DftReconciliationResult(
        (record.with_status("failed"), record.with_status("pending", job_id=None))
    )

    workflow_result = DftStageResult(
        preparation=DftPreparationResult(()),
        execution=snapshot,
    ).as_workflow_result()

    assert workflow_result.status is StageRunState.RUNNING
    assert workflow_result.completed is False
    assert workflow_result.advanced_to is None


def test_empty_dft_reconciliation_does_not_advance_to_training() -> None:
    result = DftReconciliationResult(())

    assert result.all_terminal is True
    assert result.all_successful is False
    assert (
        DftStageResult(
            preparation=DftPreparationResult(()),
            execution=result,
        )
        .as_workflow_result()
        .advanced_to
        is None
    )
