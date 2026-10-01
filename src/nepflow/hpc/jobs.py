"""Typed scheduler job records and operation results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SchedulerJobState(str, Enum):
    """Scheduler-neutral state classification for one job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    OOM = "oom"
    OUT_OF_MEMORY = "oom"
    CANCELLED = "cancelled"
    CANCELED = "cancelled"
    TIMEOUT = "timeout"
    TIMED_OUT = "timeout"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"

    @property
    def is_active(self) -> bool:
        return self in {self.PENDING, self.RUNNING}

    @property
    def is_terminal(self) -> bool:
        return self in {
            self.COMPLETED,
            self.FAILED,
            self.OOM,
            self.CANCELLED,
            self.TIMEOUT,
            self.NOT_FOUND,
        }


@dataclass(frozen=True, slots=True)
class SlurmJobRecord:
    """Normalized queue or accounting information for one SLURM job."""

    job_id: str
    name: str | None
    state: SchedulerJobState
    raw_state: str | None = None
    reason: str | None = None
    exit_code: int | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    user: str | None = None
    partition: str | None = None

    @property
    def job_name(self) -> str | None:
        """Compatibility-friendly name for job-name lookups."""

        return self.name

    @property
    def jobid(self) -> str:
        return self.job_id

    @property
    def stdout(self) -> Path | None:
        return self.stdout_path

    @property
    def stderr(self) -> Path | None:
        return self.stderr_path

    @property
    def stdout_location(self) -> Path | None:
        return self.stdout_path

    @property
    def stderr_location(self) -> Path | None:
        return self.stderr_path

    @property
    def active(self) -> bool:
        return self.state.is_active

    @property
    def exit_reason(self) -> str | None:
        return self.reason


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    """Result of a successful scheduler submission."""

    job_id: str
    stdout: str
    stderr: str
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CancellationResult:
    """Result of a successful cancellation request."""

    job_id: str
    stdout: str
    stderr: str
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueueQueryResult:
    """Successful queue query, including the valid empty-queue case."""

    jobs: tuple[SlurmJobRecord, ...]

    @property
    def empty(self) -> bool:
        return not self.jobs

    @property
    def active_jobs(self) -> tuple[SlurmJobRecord, ...]:
        return tuple(job for job in self.jobs if job.active)


@dataclass(frozen=True, slots=True)
class QueueStatusResult:
    """Successful status query for one job.

    ``job is None`` means the job was absent from a successful queue query;
    it does not represent a failed scheduler query.
    """

    job_id: str
    job: SlurmJobRecord | None

    @property
    def present(self) -> bool:
        return self.job is not None

    @property
    def active(self) -> bool:
        return self.job is not None and self.job.active


@dataclass(frozen=True, slots=True)
class AccountingResult:
    """Successful accounting lookup, which may have no record yet."""

    job_id: str
    job: SlurmJobRecord | None

    @property
    def found(self) -> bool:
        return self.job is not None


@dataclass(frozen=True, slots=True)
class ReconciledJobResult:
    """Queue/accounting reconciliation for one submitted job."""

    job_id: str
    job: SlurmJobRecord
    source: str


# Names used by callers that prefer the operation-oriented terminology.
JobSubmissionResult = SubmissionResult
JobCancellationResult = CancellationResult
SchedulerQueryResult = QueueQueryResult


__all__ = [
    "AccountingResult",
    "CancellationResult",
    "JobCancellationResult",
    "JobSubmissionResult",
    "QueueQueryResult",
    "QueueStatusResult",
    "ReconciledJobResult",
    "SchedulerJobState",
    "SchedulerQueryResult",
    "SlurmJobRecord",
    "SubmissionResult",
]
