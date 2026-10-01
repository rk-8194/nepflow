"""SLURM implementation of the shared scheduler boundary."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import re

from nepflow.errors import ProcessError, SchedulerError

from .jobs import (
    AccountingResult,
    CancellationResult,
    QueueQueryResult,
    QueueStatusResult,
    ReconciledJobResult,
    SchedulerJobState,
    SlurmJobRecord,
    SubmissionResult,
)
from .process import ProcessRunner
from .resources import JobResources, render_sbatch_directives


SQUEUE_FORMAT = "%i|%j|%T|%R"
SACCT_FORMAT = "JobIDRaw|JobName|State|Reason|ExitCode|StdOut|StdErr"

_STATE_ALIASES = {
    "PD": SchedulerJobState.PENDING,
    "PENDING": SchedulerJobState.PENDING,
    "CF": SchedulerJobState.RUNNING,
    "CONFIGURING": SchedulerJobState.RUNNING,
    "R": SchedulerJobState.RUNNING,
    "RUNNING": SchedulerJobState.RUNNING,
    "S": SchedulerJobState.RUNNING,
    "SUSPENDED": SchedulerJobState.RUNNING,
    "CG": SchedulerJobState.RUNNING,
    "COMPLETING": SchedulerJobState.RUNNING,
    "CD": SchedulerJobState.COMPLETED,
    "COMPLETED": SchedulerJobState.COMPLETED,
    "CA": SchedulerJobState.CANCELLED,
    "CANCELLED": SchedulerJobState.CANCELLED,
    "CANCELED": SchedulerJobState.CANCELLED,
    "F": SchedulerJobState.FAILED,
    "FAILED": SchedulerJobState.FAILED,
    "NF": SchedulerJobState.FAILED,
    "NODE_FAIL": SchedulerJobState.FAILED,
    "BOOT_FAIL": SchedulerJobState.FAILED,
    "DEADLINE": SchedulerJobState.FAILED,
    "PREEMPTED": SchedulerJobState.FAILED,
    "SPECIAL_EXIT": SchedulerJobState.FAILED,
    "OOM": SchedulerJobState.OOM,
    "OUT_OF_MEMORY": SchedulerJobState.OOM,
    "TO": SchedulerJobState.TIMEOUT,
    "TIMEOUT": SchedulerJobState.TIMEOUT,
}


def map_slurm_state(raw_state: str) -> SchedulerJobState:
    """Map a SLURM state or abbreviation to a typed scheduler state."""

    normalized = raw_state.strip().upper().split()[0] if raw_state.strip() else ""
    normalized = normalized.rstrip("+")
    return _STATE_ALIASES.get(normalized, SchedulerJobState.UNKNOWN)


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return None if not stripped or stripped in {"-", "(null)", "N/A"} else stripped


def _optional_path(value: str | None) -> Path | None:
    text = _optional_text(value)
    return Path(text) if text is not None else None


def _parse_exit_code(value: str | None) -> int | None:
    text = _optional_text(value)
    if text is None:
        return None
    match = re.match(r"^(?P<code>-?\d+)(?::\d+)?$", text)
    if match is None:
        return None
    return int(match.group("code"))


def _record(
    *,
    job_id: str,
    name: str | None,
    raw_state: str,
    reason: str | None = None,
    exit_code: int | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
    user: str | None = None,
    partition: str | None = None,
) -> SlurmJobRecord:
    return SlurmJobRecord(
        job_id=job_id,
        name=_optional_text(name),
        state=map_slurm_state(raw_state),
        raw_state=_optional_text(raw_state),
        reason=_optional_text(reason),
        exit_code=exit_code,
        stdout_path=_optional_path(stdout_path),
        stderr_path=_optional_path(stderr_path),
        user=_optional_text(user),
        partition=_optional_text(partition),
    )


def parse_sbatch_job_id(output: str) -> str:
    """Extract a job id from canonical ``sbatch`` output."""

    match = re.search(r"\bSubmitted\s+batch\s+job\s+(\S+)", output or "")
    if match is None:
        raise SchedulerError(
            f"Malformed sbatch output; expected 'Submitted batch job <id>': {output!r}",
            kind="malformed_submission",
        )
    return match.group(1)


def parse_sbatch_output(output: str) -> str:
    """Compatibility name for :func:`parse_sbatch_job_id`."""

    return parse_sbatch_job_id(output)


def parse_squeue_output(output: str) -> tuple[SlurmJobRecord, ...]:
    """Parse headerless pipe-delimited squeue output.

    The canonical format is ``job_id|job_name|state|reason``.  A compact
    whitespace form is accepted for isolated parser use and legacy command
    fixtures, but malformed single-token rows are rejected.
    """

    records: list[SlurmJobRecord] = []
    for line_number, line in enumerate((output or "").splitlines(), start=1):
        if not line.strip():
            continue
        if "|" in line:
            fields = line.split("|")
            if len(fields) < 3:
                raise SchedulerError(
                    f"Malformed squeue row {line_number}: {line!r}",
                    kind="malformed_query",
                )
            job_id, name, raw_state = (field.strip() for field in fields[:3])
            reason = fields[3] if len(fields) >= 4 else None
        else:
            fields = line.split(None, 3)
            if len(fields) < 2:
                raise SchedulerError(
                    f"Malformed squeue row {line_number}: {line!r}",
                    kind="malformed_query",
                )
            if len(fields) == 2:
                job_id, raw_state = fields
                name, reason = None, None
            else:
                job_id, name, raw_state = fields[:3]
                reason = fields[3] if len(fields) == 4 else None
        if not job_id or not raw_state:
            raise SchedulerError(
                f"Malformed squeue row {line_number}: {line!r}",
                kind="malformed_query",
            )
        records.append(_record(job_id=job_id, name=name, raw_state=raw_state, reason=reason))
    return tuple(records)


def _sacct_fields(fields: list[str]) -> tuple[str, str | None, str, str | None, str | None, str | None, str | None]:
    if len(fields) >= 7:
        return tuple(fields[:7])  # type: ignore[return-value]
    if len(fields) >= 2:
        # Compact fixtures may omit JobName and use id|state|reason|exit|out|err.
        values = fields + [""] * (6 - len(fields))
        return values[0], None, values[1], values[2], values[3], values[4], values[5]
    raise ValueError("not enough fields")


def parse_sacct_output(output: str) -> tuple[SlurmJobRecord, ...]:
    """Parse headerless ``sacct -P`` output into normalized records."""

    records: list[SlurmJobRecord] = []
    for line_number, line in enumerate((output or "").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("|") if "|" in line else line.split()
        try:
            job_id, name, raw_state, reason, exit_code, stdout_path, stderr_path = _sacct_fields(
                [field.strip() for field in fields]
            )
        except ValueError as exc:
            raise SchedulerError(
                f"Malformed sacct row {line_number}: {line!r}",
                kind="malformed_accounting",
            ) from exc
        if not job_id or not raw_state:
            raise SchedulerError(
                f"Malformed sacct row {line_number}: {line!r}",
                kind="malformed_accounting",
            )
        records.append(
            _record(
                job_id=job_id,
                name=name,
                raw_state=raw_state,
                reason=reason,
                exit_code=_parse_exit_code(exit_code),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        )
    return tuple(records)


class SlurmScheduler:
    """Run and parse SLURM commands through :class:`ProcessRunner`."""

    def __init__(
        self,
        *,
        process_runner: ProcessRunner | None = None,
        runner: ProcessRunner | None = None,
        user: str | None = None,
        default_timeout: float = 30.0,
    ) -> None:
        if process_runner is not None and runner is not None:
            raise ValueError("pass only one of process_runner or runner")
        self.process_runner = process_runner or runner or ProcessRunner()
        self.user = user if user is not None else os.environ.get("USER") or os.environ.get("USERNAME")
        self.default_timeout = default_timeout

    def _run(
        self,
        command: Sequence[str | Path],
        *,
        cwd: str | Path | None,
        timeout: float | None,
    ):
        command_values = tuple(os.fspath(part) for part in command)
        try:
            result = self.process_runner.run(
                command_values,
                cwd=cwd,
                timeout=self.default_timeout if timeout is None else timeout,
                check=False,
                capture_output=True,
            )
        except ProcessError as exc:
            raise SchedulerError(
                f"Scheduler command could not execute: {exc}",
                command=exc.command,
                cwd=exc.cwd,
                returncode=exc.returncode,
                stdout=exc.stdout,
                stderr=exc.stderr,
                kind=f"process_{exc.kind}",
            ) from exc
        if result.returncode != 0:
            raise SchedulerError(
                f"Scheduler command failed with exit code {result.returncode}: {' '.join(command_values)}",
                command=result.command,
                cwd=result.cwd,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                kind="command_failed",
            )
        return result

    @staticmethod
    def _directive_arguments(directives: Sequence[str]) -> list[str]:
        return [directive.removeprefix("#SBATCH ") for directive in directives]

    def submit(
        self,
        command: Sequence[str | Path],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        resources: JobResources | None = None,
    ) -> SubmissionResult:
        """Submit an already-built sbatch command without changing its body."""

        command_values = [os.fspath(part) for part in command]
        if not command_values or command_values[0] != "sbatch":
            raise ValueError("SlurmScheduler.submit expects a command beginning with sbatch")
        if resources is not None:
            command_values[1:1] = self._directive_arguments(render_sbatch_directives(resources))
        result = self._run(command_values, cwd=cwd, timeout=timeout)
        job_id = parse_sbatch_job_id(result.stdout or "")
        return SubmissionResult(
            job_id=job_id,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            command=result.command,
        )

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
        directives = render_sbatch_directives(
            resources or JobResources(),
            job_name=job_name,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        command = [
            "sbatch",
            *self._directive_arguments(directives),
            *(os.fspath(part) for part in extra_args),
            os.fspath(script_path),
        ]
        return self.submit(command, cwd=cwd, timeout=timeout)

    def cancel(self, job_id: str, *, timeout: float | None = None) -> CancellationResult:
        if not job_id.strip():
            raise ValueError("job_id must not be blank")
        command = ("scancel", job_id)
        result = self._run(command, cwd=None, timeout=timeout)
        return CancellationResult(
            job_id=job_id,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            command=result.command,
        )

    def show_job(self, job_id: str, *, timeout: float | None = None) -> str:
        """Return raw ``scontrol show job`` output for scheduler-owned fallback logic."""

        if not job_id.strip():
            raise ValueError("job_id must not be blank")
        result = self._run(
            ["scontrol", "show", "job", job_id],
            cwd=None,
            timeout=timeout,
        )
        return result.stdout or ""

    def _queue_query(
        self,
        *,
        job_id: str | None = None,
        user: str | None = None,
        active_only: bool,
        timeout: float | None,
    ) -> QueueQueryResult:
        command = ["squeue", "--noheader", "--format", SQUEUE_FORMAT]
        if job_id is not None:
            command.extend(["--jobs", job_id])
        selected_user = user if user is not None else self.user
        if selected_user:
            command.extend(["--user", selected_user])
        if active_only:
            command.extend(["--states", "PENDING,RUNNING,CONFIGURING,COMPLETING"])
        result = self._run(command, cwd=None, timeout=timeout)
        return QueueQueryResult(parse_squeue_output(result.stdout or ""))

    def queue_status(self, job_id: str, *, timeout: float | None = None) -> QueueStatusResult:
        query = self._queue_query(job_id=job_id, active_only=False, timeout=timeout)
        job = next((candidate for candidate in query.jobs if candidate.job_id == job_id), None)
        return QueueStatusResult(job_id=job_id, job=job)

    def list_active_jobs(
        self,
        *,
        user: str | None = None,
        name_prefix: str | None = None,
        timeout: float | None = None,
    ) -> QueueQueryResult:
        query = self._queue_query(user=user, active_only=True, timeout=timeout)
        if name_prefix is None:
            return query
        return QueueQueryResult(
            tuple(job for job in query.jobs if job.name and job.name.startswith(name_prefix))
        )

    def find_job_by_name(
        self,
        job_name: str,
        *,
        user: str | None = None,
        timeout: float | None = None,
    ) -> SlurmJobRecord | None:
        query = self.list_active_jobs(user=user, timeout=timeout)
        return next((job for job in query.jobs if job.name == job_name), None)

    def accounting_status(self, job_id: str, *, timeout: float | None = None) -> AccountingResult:
        command = [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--jobs",
            job_id,
            "--format",
            SACCT_FORMAT,
        ]
        result = self._run(command, cwd=None, timeout=timeout)
        records = parse_sacct_output(result.stdout or "")
        job = next((record for record in records if record.job_id == job_id), None)
        if job is None and records:
            job = records[0]
        return AccountingResult(job_id=job_id, job=job)

    def reconcile(self, job_id: str, *, timeout: float | None = None) -> ReconciledJobResult:
        queue = self.queue_status(job_id, timeout=timeout)
        if queue.job is not None:
            return ReconciledJobResult(job_id=job_id, job=queue.job, source="squeue")
        accounting = self.accounting_status(job_id, timeout=timeout)
        if accounting.job is not None:
            return ReconciledJobResult(job_id=job_id, job=accounting.job, source="sacct")
        return ReconciledJobResult(
            job_id=job_id,
            job=SlurmJobRecord(
                job_id=job_id,
                name=None,
                state=SchedulerJobState.NOT_FOUND,
                reason="job absent from squeue and sacct",
            ),
            source="sacct",
        )

    # Explicit aliases keep the boundary easy to adopt without adding parser
    # logic to legacy caller modules.
    status = reconcile
    active_jobs = list_active_jobs
    job_name_lookup = find_job_by_name


__all__ = [
    "SACCT_FORMAT",
    "SQUEUE_FORMAT",
    "SlurmScheduler",
    "map_slurm_state",
    "parse_sacct_output",
    "parse_sbatch_job_id",
    "parse_sbatch_output",
    "parse_squeue_output",
]
