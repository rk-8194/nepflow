"""Workflow resubmission and legacy-state reconciliation result types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from nepflow.errors import StateError

from .stages import WorkflowStage


class ReconciliationSource(str, Enum):
    """Finite sources used when selecting authoritative workflow state."""

    STATE_STORE = "state_store"
    LEGACY_MARKER = "legacy_marker"
    INITIAL_STATE = "initial_state"

    @classmethod
    def from_legacy(cls, value: "ReconciliationSource | str") -> "ReconciliationSource":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as exc:
            raise StateError(f"Unknown reconciliation source: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Describe how the authoritative stage and a legacy marker were reconciled."""

    stage: WorkflowStage
    source: ReconciliationSource
    marker_stage: WorkflowStage | None = None
    authoritative_stage: WorkflowStage | None = None
    changed: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", WorkflowStage.from_legacy(self.stage))
        object.__setattr__(self, "source", ReconciliationSource.from_legacy(self.source))
        if self.marker_stage is not None:
            object.__setattr__(
                self, "marker_stage", WorkflowStage.from_legacy(self.marker_stage)
            )
        if self.authoritative_stage is not None:
            object.__setattr__(
                self,
                "authoritative_stage",
                WorkflowStage.from_legacy(self.authoritative_stage),
            )


@dataclass(frozen=True, slots=True)
class ResubmissionResult:
    """Describe a requested or completed workflow self-resubmission."""

    requested: bool
    reason: str
    stage: WorkflowStage | None = None
    command: tuple[str, ...] | None = None
    metadata: Any = None

    def __post_init__(self) -> None:
        if self.stage is not None:
            object.__setattr__(self, "stage", WorkflowStage.from_legacy(self.stage))
        if self.command is not None:
            object.__setattr__(self, "command", tuple(self.command))


class SelfResubmitExit(Exception):
    """Request process-level SLURM self-resubmission without advancing state."""

    def __init__(
        self,
        message: str = "Workflow self-resubmission requested",
        *,
        result: ResubmissionResult | None = None,
    ) -> None:
        self.result = result or ResubmissionResult(requested=True, reason=message)
        super().__init__(message)


def resolve_resubmit_command(
    workdir: Path,
    submit_script: Path | None = None,
    *,
    scheduler: Any | None = None,
    environment: dict[str, str] | None = None,
) -> tuple[list[str], Path, str]:
    """Resolve the current SLURM submission through the scheduler boundary."""

    workdir = Path(workdir).resolve()
    if submit_script is not None:
        resolved = Path(submit_script)
        if not resolved.is_absolute():
            resolved = workdir / resolved
        resolved = resolved.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Submit script not found: {resolved}")
        return ["sbatch", str(resolved)], resolved.parent, f"explicit submit script {resolved}"

    env = os.environ if environment is None else environment
    job_id = env.get("SLURM_JOB_ID")
    if job_id and scheduler is not None and hasattr(scheduler, "show_job"):
        try:
            output = scheduler.show_job(job_id)
        except Exception:
            output = ""
        command_match = re.search(r"\bCommand=(\S+)", output)
        if command_match:
            command_path = Path(command_match.group(1)).expanduser()
            if not command_path.is_absolute():
                workdir_match = re.search(r"\bWorkDir=(\S+)", output)
                if workdir_match:
                    command_path = Path(workdir_match.group(1)) / command_path
            if command_path.exists():
                return ["sbatch", str(command_path)], command_path.parent, f"scontrol job {job_id}"

    candidate_dirs: list[Path] = []
    for directory in (env.get("SLURM_SUBMIT_DIR"), str(workdir), os.getcwd()):
        if directory:
            path = Path(directory).resolve()
            if path not in candidate_dirs:
                candidate_dirs.append(path)
    tried = [directory / "submit.slurm" for directory in candidate_dirs]
    for candidate in tried:
        if candidate.exists():
            return ["sbatch", str(candidate)], candidate.parent, f"submit.slurm in {candidate.parent}"
    tried_text = "\n".join(f"  - {candidate}" for candidate in tried)
    raise FileNotFoundError(f"Could not find a submit script to resubmit. Tried:\n{tried_text}")


def submit_resubmission(
    command: list[str] | tuple[str, ...],
    cwd: Path,
    *,
    scheduler: Any | None = None,
    dry_run: bool = False,
) -> str:
    """Submit a resolved workflow command through the scheduler boundary."""

    if dry_run:
        return ""
    if scheduler is not None:
        result = scheduler.submit(command, cwd=cwd)
        return str(getattr(result, "stdout", "") or "").strip()
    result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=str(cwd))
    return result.stdout.strip()


__all__ = [
    "ReconciliationResult",
    "ReconciliationSource",
    "ResubmissionResult",
    "SelfResubmitExit",
    "resolve_resubmit_command",
    "submit_resubmission",
]
