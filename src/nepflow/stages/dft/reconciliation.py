"""Canonical reconciliation of prepared DFT executions.

The reconciler is deliberately small and scheduler-neutral.  It consumes
prepared backend inputs, asks the scheduler for one normalized job state, asks
the backend for scientific completion/failure evidence, and persists only the
resulting transition.  It does not inspect SLURM output or write legacy
launcher markers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from nepflow.dft.backend import (
    DftBackend,
    DftFailureEvidence,
    DftInputArtifacts,
)
from nepflow.domain.calculations import DftResultArtifact
from nepflow.errors import StateError
from nepflow.hpc.jobs import SchedulerJobState
from nepflow.hpc.resources import JobResources
from nepflow.hpc.scheduler import Scheduler


class ExecutionStateStore(Protocol):
    """Persistence seam used by the execution state machine."""

    def save_execution(
        self,
        record: "DftExecutionRecord",
        *,
        artifact: DftResultArtifact | None = None,
        reason: str | None = None,
    ) -> None:
        """Persist the latest attempt status and optional artifact evidence."""
        ...


@dataclass(frozen=True, slots=True)
class DftRecoveryDecision:
    """Typed recovery choice returned by an execution policy."""

    retry: bool
    retry_level: int
    reason: str
    resources: JobResources | None = None


class DftRecoveryPolicy(Protocol):
    """Policy that converts typed DFT failure evidence into retry/terminal choice."""
    def __call__(
        self,
        record: "DftExecutionRecord",
        failure: Any,
    ) -> DftRecoveryDecision:
        """Return a bounded retry decision for typed failure evidence."""
        ...


@dataclass(frozen=True, slots=True)
class DftExecutionRecord:
    """Persistable state for one prepared DFT attempt.

    ``status`` distinguishes preparation, scheduler activity, scientific
    completion, reuse, and terminal failure.  ``retry_level`` is persisted so
    restart cannot silently repeat a resource tier.
    """

    inputs: DftInputArtifacts
    attempt_id: str
    status: str = "pending"
    job_id: str | None = None
    job_name: str | None = None
    retry_level: int = 0
    resources: JobResources | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_status(
        self,
        status: str,
        *,
        job_id: str | None = None,
        retry_level: int | None = None,
        resources: JobResources | None = None,
    ) -> "DftExecutionRecord":
        """Return an immutable status update without changing input state."""
        return replace(
            self,
            status=status,
            job_id=job_id,
            retry_level=self.retry_level if retry_level is None else retry_level,
            resources=self.resources if resources is None else resources,
        )


@dataclass(frozen=True, slots=True)
class DftReconciliationResult:
    """Records after one idempotent reconciliation pass.

    Terminal means no further scheduler transition is expected; successful
    means only ``completed`` or verified ``reused`` records, never merely a
    terminal scheduler state.
    """

    records: tuple[DftExecutionRecord, ...]

    @property
    def all_terminal(self) -> bool:
        """Return whether every record is completed, reused, or failed."""
        return all(record.status in {"completed", "reused", "failed"} for record in self.records)

    @property
    def all_successful(self) -> bool:
        """Return whether every calculation completed successfully or reused."""
        return bool(self.records) and all(
            record.status in {"completed", "reused"} for record in self.records
        )

    @property
    def any_failed(self) -> bool:
        """Return whether at least one calculation reached irrecoverable failure."""
        return any(record.status == "failed" for record in self.records)

    @property
    def counts(self) -> dict[str, int]:
        """Return a count of records by persisted status."""
        result: dict[str, int] = {}
        for record in self.records:
            result[record.status] = result.get(record.status, 0) + 1
        return result


class DftReconciliationOrchestrator:
    """Advance prepared DFT attempts through scheduler/backend evidence.

    One pass is restart-safe: persisted submitted/running attempts are
    reconciled before new submissions, active jobs consume concurrency slots,
    and a backend-completed output is required before success is recorded.
    Failed attempts are retained before a new retry record is created.
    """

    _TERMINAL_STATUSES = frozenset({"completed", "reused", "failed"})

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        backend: DftBackend,
        state_store: ExecutionStateStore,
        max_concurrent: int = 1,
        retry_limit: int = 0,
        script_path: str | Path | None = None,
        recovery_policy: DftRecoveryPolicy | None = None,
        performance_recorder: Any | None = None,
        job_name_prefix: str | None = None,
        persist_result_artifact: bool = True,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        if retry_limit < 0:
            raise ValueError("retry_limit must not be negative")
        self.scheduler = scheduler
        self.backend = backend
        self.state_store = state_store
        self.max_concurrent = max_concurrent
        self.retry_limit = retry_limit
        self.script_path = None if script_path is None else Path(script_path)
        self.recovery_policy = recovery_policy
        self.performance_recorder = performance_recorder
        self.job_name_prefix = job_name_prefix
        self.persist_result_artifact = persist_result_artifact

    def reconcile_once(
        self,
        records: Sequence[DftExecutionRecord],
    ) -> DftReconciliationResult:
        """Apply one idempotent state pass while preserving input order.

        Queue/accounting results control operational state, but only backend
        completion and parsed labels establish scientific success.  Unknown
        states raise rather than being interpreted as an empty queue.
        """
        current = tuple(records)
        active_jobs = self.scheduler.list_active_jobs(
            name_prefix=self.job_name_prefix,
        ).active_jobs
        active_count = len(active_jobs)
        submitted_this_pass = 0
        reconciled: list[DftExecutionRecord] = []

        for record in current:
            if record.status in self._TERMINAL_STATUSES:
                reconciled.append(record)
                continue

            if record.status in {"submitted", "running"}:
                updated = self._reconcile_submitted(record)
                reconciled.append(updated)
                continue

            if record.status in {"pending", "prepared"}:
                if active_count + submitted_this_pass >= self.max_concurrent:
                    reconciled.append(record)
                    continue
                updated = self._submit(record)
                reconciled.append(updated)
                submitted_this_pass += 1
                continue

            raise ValueError(f"Unknown DFT execution status: {record.status}")

        return DftReconciliationResult(tuple(reconciled))

    def _reconcile_submitted(self, record: DftExecutionRecord) -> DftExecutionRecord:
        """Resolve one submitted attempt and persist completion or retry evidence."""
        if not record.job_id:
            raise ValueError(f"Submitted DFT attempt {record.attempt_id} has no scheduler job ID")
        scheduler_result = self.scheduler.reconcile(record.job_id)
        job_state = scheduler_result.job.state
        if job_state == SchedulerJobState.UNKNOWN:
            raise StateError(f"Scheduler returned an unknown state for DFT job {record.job_id}")
        if job_state == SchedulerJobState.PENDING:
            return self._save(record.with_status("submitted", job_id=record.job_id))
        if job_state == SchedulerJobState.RUNNING:
            return self._save(record.with_status("running", job_id=record.job_id))

        completion = self.backend.parse_completion(record.inputs)
        if completion.completed:
            result = self.backend.parse_result(record.inputs)
            artifact = result.artifact
            if artifact is None:
                raise ValueError(
                    f"Completed DFT attempt {record.attempt_id} has no result artifact"
                )
            if self.performance_recorder is not None:
                self.performance_recorder.record_completion(record, result)
            completed = record.with_status("completed")
            return self._save(
                completed,
                artifact=artifact if self.persist_result_artifact else None,
            )

        markers = (job_state.value,) if job_state == SchedulerJobState.OOM else ()
        failure = self.backend.classify_failure(
            DftFailureEvidence(
                calculation=record.inputs.calculation,
                completion=completion,
                markers=markers,
            )
        )
        if self.performance_recorder is not None:
            self.performance_recorder.record_failure(record, failure)
        decision = self._recovery_decision(record, failure)
        if decision.retry:
            # Retain the failed attempt before creating a new identity so a
            # restart cannot submit the same failed attempt twice.
            failed = record.with_status("failed")
            self._save(failed, reason=failure.reason)
            retry = DftExecutionRecord(
                inputs=record.inputs,
                attempt_id=(
                    f"{record.inputs.calculation.calculation_id}:attempt:{record.retry_level + 2}"
                ),
                status="pending",
                retry_level=decision.retry_level,
                resources=decision.resources or record.resources,
                job_name=record.job_name,
            )
            return self._save(retry, reason=decision.reason)
        return self._save(
            record.with_status("failed"),
            reason=failure.reason,
        )

    def _submit(self, record: DftExecutionRecord) -> DftExecutionRecord:
        if self.script_path is None:
            raise ValueError("A runner script is required to submit a DFT attempt")
        submission = self.scheduler.submit_script(
            self.script_path,
            job_name=record.job_name or record.inputs.calculation.calculation_id,
            resources=record.resources,
            cwd=record.inputs.working_directory,
        )
        return self._save(
            record.with_status("submitted", job_id=submission.job_id),
        )

    def _save(
        self,
        record: DftExecutionRecord,
        *,
        artifact: DftResultArtifact | None = None,
        reason: str | None = None,
    ) -> DftExecutionRecord:
        self.state_store.save_execution(record, artifact=artifact, reason=reason)
        return record

    def _recovery_decision(self, record: DftExecutionRecord, failure: Any) -> DftRecoveryDecision:
        if self.recovery_policy is not None:
            decision = self.recovery_policy(record, failure)
            if not isinstance(decision, DftRecoveryDecision):
                raise TypeError("DFT recovery policy must return DftRecoveryDecision")
            return decision
        return DftRecoveryDecision(
            retry=(failure.recoverable and record.retry_level < self.retry_limit),
            retry_level=record.retry_level + 1,
            reason=failure.reason,
        )


__all__ = [
    "DftExecutionRecord",
    "DftRecoveryDecision",
    "DftRecoveryPolicy",
    "DftReconciliationOrchestrator",
    "DftReconciliationResult",
    "ExecutionStateStore",
]
