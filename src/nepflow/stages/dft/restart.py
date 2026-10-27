"""Operator-requested restart of one project's DFT execution campaign."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nepflow.errors import StateError
from nepflow.hpc.jobs import SchedulerJobState
from nepflow.hpc.scheduler import Scheduler
from nepflow.state import StateStore


@dataclass(frozen=True, slots=True)
class DftRestartResult:
    """Evidence returned after a DFT restart boundary is committed."""

    cancelled_attempt_ids: tuple[str, ...]
    cancelled_job_ids: tuple[str, ...]
    execution_generations: dict[str, int]


class DftRestartService:
    """Cancel authoritative child jobs, invalidate DFT state, and reset jobs."""

    def __init__(self, scheduler: Scheduler, *, sleeper: Any = time.sleep) -> None:
        self.scheduler = scheduler
        self._sleep = sleeper

    def restart(
        self,
        *,
        project_dir: Path,
        project_name: str,
        state_store: StateStore,
        scheduler: Scheduler | None = None,
        poll_interval_seconds: int,
        datasets: tuple[str, ...] = ("train", "test"),
    ) -> DftRestartResult:
        """Perform the destructive DFT restart in its required order."""

        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must not be negative")
        project_dir = Path(project_dir).resolve()
        active_scheduler = scheduler or self.scheduler
        self._validate_selected_inputs(project_dir, datasets)
        attempts = self._authoritative_live_attempts(state_store)
        self._cancel_and_confirm(attempts, active_scheduler, poll_interval_seconds)

        attempt_ids = tuple(str(item["attempt_id"]) for item in attempts)
        job_ids = tuple(str(item["job_id"]) for item in attempts)
        if attempt_ids:
            state_store.cancel_dft_attempts_for_restart(attempt_ids)
        generations = state_store.begin_dft_execution_generation(
            project_name,
            source="explicit_run_vasp_restart",
            selected_only=True,
        )
        self._clear_jobs_directory(project_dir)
        return DftRestartResult(
            cancelled_attempt_ids=attempt_ids,
            cancelled_job_ids=job_ids,
            execution_generations=generations,
        )

    @staticmethod
    def _validate_selected_inputs(project_dir: Path, datasets: tuple[str, ...]) -> None:
        selected_dir = project_dir / "structures" / "selected"
        missing: list[Path] = []
        for dataset in datasets:
            path = selected_dir / f"{dataset}.xyz"
            try:
                ready = path.is_file() and path.stat().st_size > 0
            except OSError as exc:
                raise StateError(
                    "Cannot restart DFT; selected input cannot be inspected: "
                    f"{path}"
                ) from exc
            if not ready:
                missing.append(path)
        if missing:
            raise StateError(
                "Cannot restart DFT; selected input(s) are missing or empty: "
                + ", ".join(str(path) for path in missing)
            )

    @staticmethod
    def _authoritative_live_attempts(state_store: StateStore) -> list[dict[str, Any]]:
        live: list[dict[str, Any]] = []
        for calculation in state_store.list_dft_calculations():
            calculation_id = calculation.get("calculation_id")
            if not isinstance(calculation_id, str):
                continue
            for attempt in state_store.list_dft_attempts(calculation_id):
                metadata = attempt.get("metadata")
                if (
                    isinstance(metadata, dict)
                    and metadata.get("execution_kind") == "vasp_benchmark"
                ):
                    continue
                if attempt.get("status") not in {"submitted", "running"}:
                    continue
                job_id = attempt.get("job_id")
                if not isinstance(job_id, str) or not job_id.strip():
                    raise StateError(
                        "Cannot restart DFT; authoritative live attempt has no persisted job ID: "
                        f"{attempt.get('attempt_id')}"
                    )
                live.append(attempt)
        return live

    def _cancel_and_confirm(
        self,
        attempts: list[dict[str, Any]],
        scheduler: Scheduler,
        poll_interval_seconds: int,
    ) -> None:
        for attempt in attempts:
            job_id = str(attempt["job_id"])
            status = scheduler.reconcile(job_id)
            if status.job.state is SchedulerJobState.UNKNOWN:
                raise StateError(f"Cannot determine scheduler state for DFT job {job_id}")
            if status.job.state.is_active:
                scheduler.cancel(job_id)
                while True:
                    status = scheduler.reconcile(job_id)
                    if status.job.state is SchedulerJobState.UNKNOWN:
                        raise StateError(
                            f"Cannot confirm cancellation of DFT job {job_id}: scheduler state unknown"
                        )
                    if not status.job.state.is_active:
                        break
                    if poll_interval_seconds:
                        self._sleep(poll_interval_seconds)

    @staticmethod
    def _clear_jobs_directory(project_dir: Path) -> None:
        jobs_dir = (project_dir / "vasp" / "jobs").resolve()
        if project_dir not in jobs_dir.parents:
            raise StateError(f"Refusing to clear jobs directory outside project: {jobs_dir}")
        try:
            if jobs_dir.exists():
                shutil.rmtree(jobs_dir)
            jobs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StateError(f"Could not reset DFT jobs directory {jobs_dir}") from exc


__all__ = ["DftRestartResult", "DftRestartService"]
