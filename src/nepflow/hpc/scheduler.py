"""Stable scheduler boundary used by workflow callers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .jobs import (
    AccountingResult,
    CancellationResult,
    QueueQueryResult,
    QueueStatusResult,
    ReconciledJobResult,
    SlurmJobRecord,
    SubmissionResult,
)
from .resources import JobResources


@runtime_checkable
class Scheduler(Protocol):
    """Operations required by stages without exposing SLURM syntax."""

    def submit(
        self,
        command: Sequence[str | Path],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        resources: JobResources | None = None,
    ) -> SubmissionResult:
        """Submit an already-rendered backend command."""
        ...

    def submit_script(
        self,
        script_path: str | Path,
        *,
        resources: JobResources | None = None,
        job_name: str | None = None,
        stdout_path: str | Path | None = None,
        stderr_path: str | Path | None = None,
        extra_args: Sequence[str | Path] = (),
        cwd: str | Path | None = None,
        timeout: float | None = None,
    ) -> SubmissionResult:
        """Build and submit one sbatch script command."""
        ...

    def cancel(self, job_id: str, *, timeout: float | None = None) -> CancellationResult:
        """Cancel a job or raise a scheduler communication error."""
        ...

    def queue_status(self, job_id: str, *, timeout: float | None = None) -> QueueStatusResult:
        """Return active queue status; absent is distinct from query failure."""
        ...

    def accounting_status(self, job_id: str, *, timeout: float | None = None) -> AccountingResult:
        """Return accounting status; absent accounting is distinct from failure."""
        ...

    def list_active_jobs(
        self,
        *,
        user: str | None = None,
        name_prefix: str | None = None,
        timeout: float | None = None,
    ) -> QueueQueryResult:
        """List active jobs from a successful queue query."""
        ...

    def find_job_by_name(
        self,
        job_name: str,
        *,
        user: str | None = None,
        timeout: float | None = None,
    ) -> SlurmJobRecord | None:
        """Find one active job by exact name."""
        ...

    def reconcile(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
    ) -> ReconciledJobResult:
        """Reconcile live queue state with accounting state."""
        ...


__all__ = ["Scheduler"]
