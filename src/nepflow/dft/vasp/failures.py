"""VASP-specific failure evidence and classification.

Scheduler command failures remain scheduler failures.  This module only
interprets evidence produced by the VASP job itself or by its runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nepflow.dft.backend import DftFailure

from .outputs import outcar_is_complete


OOM_MARKER = ".vasp_oom_detected"
LEGACY_OOM_MARKER = ".vasp_oom_marker"


@dataclass(frozen=True, slots=True)
class VaspFailureEvidence:
    """Filesystem evidence collected after a VASP job leaves the queue."""

    job_directory: Path
    completed: bool
    oom_marker: bool
    output_log: str = ""
    returncode: int | None = None

    @classmethod
    def from_job_directory(cls, job_directory: Path) -> "VaspFailureEvidence":
        job_directory = Path(job_directory)
        log_path = job_directory / "vasp_output.log"
        try:
            output_log = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            output_log = ""
        return cls(
            job_directory=job_directory,
            completed=outcar_is_complete(job_directory / "OUTCAR"),
            oom_marker=(
                (job_directory / OOM_MARKER).exists()
                or (job_directory / LEGACY_OOM_MARKER).exists()
            )
            or "oom_kill" in output_log,
            output_log=output_log,
        )


def classify_failure(evidence: VaspFailureEvidence) -> DftFailure:
    """Classify only VASP-visible evidence, preserving launcher semantics."""
    if evidence.completed:
        return DftFailure(kind="completed", recoverable=False, reason="completed")
    if evidence.oom_marker:
        return DftFailure(
            kind="out_of_memory",
            recoverable=True,
            reason="VASP runner reported an out-of-memory termination",
        )
    if evidence.output_log:
        return DftFailure(
            kind="vasp_execution_failed",
            recoverable=False,
            reason="VASP runner produced failure output without an OOM marker",
        )
    return DftFailure(
        kind="incomplete_output",
        recoverable=True,
        reason="VASP left no completion or failure evidence",
    )


def is_oom_failure(job_directory: Path) -> bool:
    """Compatibility predicate used by launcher and memory stages."""
    return VaspFailureEvidence.from_job_directory(job_directory).oom_marker


def has_oom_marker(job_directory: Path) -> bool:
    """Return the launcher's exact Phase 2 marker predicate."""
    return (Path(job_directory) / OOM_MARKER).exists()


__all__ = [
    "OOM_MARKER",
    "LEGACY_OOM_MARKER",
    "VaspFailureEvidence",
    "classify_failure",
    "has_oom_marker",
    "is_oom_failure",
]
