"""Build application-level external services and handle resubmission."""

from __future__ import annotations

import logging
from pathlib import Path

from nepflow.errors import SchedulerError
from nepflow.hpc.process import ProcessRunner
from nepflow.hpc.slurm import SlurmScheduler
from nepflow.workflow import resolve_resubmit_command as _resolve_workflow_command


def create_process_runner(logger: logging.Logger) -> ProcessRunner:
    """Create the process service used by the CLI and stage composition."""

    return ProcessRunner(logger=logger)


def create_scheduler(process_runner: ProcessRunner) -> SlurmScheduler:
    """Create the scheduler service used by the composed workflow."""

    return SlurmScheduler(process_runner=process_runner)


def resolve_resubmit_command(
    *,
    workdir: Path,
    scheduler: SlurmScheduler,
) -> tuple[list[str], Path, str]:
    """Resolve a self-resubmission command through the workflow boundary."""

    return _resolve_workflow_command(workdir=workdir, scheduler=scheduler)


def resubmit_slurm_job(
    command: list[str],
    submit_cwd: Path,
    submit_source: str,
    *,
    debug: bool,
    logger: logging.Logger,
    scheduler: SlurmScheduler,
) -> None:
    """Submit a previously resolved command, or log it in debug mode."""

    logger.info("SLURM walltime deadline approaching - resubmitting nepflow")
    if debug:
        logger.info(
            "[DEBUG] Would resubmit with: %s (cwd=%s, source=%s)",
            " ".join(command),
            submit_cwd,
            submit_source,
        )
        return

    try:
        result = scheduler.submit(command, cwd=submit_cwd)
        logger.info("Resubmission via sbatch: %s", result.stdout.strip())
    except SchedulerError as exc:
        stderr = exc.stderr or ""
        stdout = exc.stdout or ""
        logger.error(
            "Could not resubmit job via %s: %s%s%s",
            submit_source,
            exc,
            f" | stdout: {stdout.strip()}" if stdout.strip() else "",
            f" | stderr: {stderr.strip()}" if stderr.strip() else "",
        )
        raise


__all__ = [
    "create_process_runner",
    "create_scheduler",
    "resolve_resubmit_command",
    "resubmit_slurm_job",
]
