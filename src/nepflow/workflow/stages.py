"""Typed workflow-stage identities and transition primitives.

The legacy controller still writes ``.project`` for interoperability, but the
workflow package owns the vocabulary and transition policy.  Keeping the
conversion here prevents each caller from inventing its own string handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from nepflow.errors import StateError


class WorkflowStage(str, Enum):
    """The ordered stages of a NEPFlow project."""

    INIT = "init"
    GENERATE = "generate"
    SELECT = "select"
    RUN_VASP = "run_vasp"
    TRAIN_NEP = "train_nep"
    VALIDATE = "validate"
    COMPLETED = "completed"

    @classmethod
    def from_legacy(cls, value: "WorkflowStage | str") -> "WorkflowStage":
        """Parse a legacy marker or database value without a fallback.

        Unknown, blank, and non-string values are state corruption.  In
        particular, they must not be silently treated as ``INIT``.
        """

        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise StateError(f"Workflow stage must be a string, got {type(value).__name__}")
        candidate = value.strip()
        if not candidate:
            raise StateError("Workflow stage is blank")
        try:
            return cls(candidate)
        except ValueError as exc:
            raise StateError(f"Unknown workflow stage: {value!r}") from exc

    @classmethod
    def from_string(cls, value: str) -> "WorkflowStage":
        """Compatibility spelling for callers migrating from string stages."""

        return cls.from_legacy(value)

    @property
    def legacy_value(self) -> str:
        """Return the exact value used by the legacy ``.project`` marker."""

        return self.value

    @property
    def is_terminal(self) -> bool:
        return self is type(self).COMPLETED


_ORDERED_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage.INIT,
    WorkflowStage.GENERATE,
    WorkflowStage.SELECT,
    WorkflowStage.RUN_VASP,
    WorkflowStage.TRAIN_NEP,
    WorkflowStage.VALIDATE,
    WorkflowStage.COMPLETED,
)


def parse_legacy_stage(value: WorkflowStage | str) -> WorkflowStage:
    """Canonical string-to-enum conversion for legacy stage values."""

    return WorkflowStage.from_legacy(value)


def stage_to_legacy(value: WorkflowStage | str) -> str:
    """Canonical enum-to-string conversion for legacy marker writes."""

    return WorkflowStage.from_legacy(value).legacy_value


def validate_transition(
    previous: WorkflowStage | str | None,
    target: WorkflowStage | str,
) -> WorkflowStage:
    """Validate one workflow transition and return its typed target.

    A stage may be resumed in place, or advance by exactly one adjacent stage.
    ``COMPLETED`` is terminal.  A missing previous stage may only bootstrap
    ``INIT``; legacy-marker reconciliation is handled explicitly by the
    controller rather than being an implicit transition exception.
    """

    next_stage = WorkflowStage.from_legacy(target)
    if previous is None:
        if next_stage is not WorkflowStage.INIT:
            raise StateError(
                f"Cannot start workflow at {next_stage.value!r}; initial stage is 'init'"
            )
        return next_stage

    current_stage = WorkflowStage.from_legacy(previous)
    if current_stage is next_stage:
        return next_stage
    if current_stage.is_terminal:
        raise StateError("Completed workflow is terminal and cannot transition")

    current_index = _ORDERED_STAGES.index(current_stage)
    target_index = _ORDERED_STAGES.index(next_stage)
    if target_index != current_index + 1:
        direction = "backward" if target_index < current_index else "non-adjacent"
        raise StateError(
            f"Invalid {direction} workflow transition: "
            f"{current_stage.value!r} -> {next_stage.value!r}"
        )
    return next_stage


def is_valid_transition(
    previous: WorkflowStage | str | None,
    target: WorkflowStage | str,
) -> bool:
    """Return whether a transition is valid without hiding conversion errors."""

    try:
        validate_transition(previous, target)
    except StateError:
        return False
    return True


class StageRunState(str, Enum):
    """Operational states used by a stage-run ledger record."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def from_legacy(cls, value: "StageRunState | str") -> "StageRunState":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise StateError(f"Stage-run status must be a string, got {type(value).__name__}")
        try:
            return cls(value.strip())
        except ValueError as exc:
            raise StateError(f"Unknown stage-run status: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class StageRunStatus:
    """Typed view of one persisted workflow stage-run row."""

    stage_run_id: str
    project_id: str
    stage: WorkflowStage
    status: StageRunState
    input_fingerprint: str | None = None
    output_fingerprint: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    metadata: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", WorkflowStage.from_legacy(self.stage))
        object.__setattr__(self, "status", StageRunState.from_legacy(self.status))

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "StageRunStatus":
        """Build a typed record from a StateStore row or equivalent mapping."""

        try:
            return cls(
                stage_run_id=str(row["stage_run_id"]),
                project_id=str(row["project_id"]),
                stage=row["stage"],
                status=row["status"],
                input_fingerprint=row.get("input_fingerprint"),
                output_fingerprint=row.get("output_fingerprint"),
                started_at=row.get("started_at"),
                completed_at=row.get("completed_at"),
                metadata=row.get("metadata"),
            )
        except KeyError as exc:
            raise StateError(f"Stage-run record is missing {exc.args[0]!r}") from exc

    @property
    def is_terminal(self) -> bool:
        return self.stage is WorkflowStage.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_run_id": self.stage_run_id,
            "project_id": self.project_id,
            "stage": self.stage.value,
            "status": self.status.value,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


# The plan and the existing StateStore use both spellings.  Keep one record
# implementation so a caller cannot accidentally create two state models.
StageRun = StageRunStatus


@dataclass(frozen=True, slots=True)
class StageRunResult:
    """Typed result returned by a stage boundary or controller adapter."""

    stage: WorkflowStage
    status: StageRunState
    advanced_to: WorkflowStage | None = None
    completed: bool = False
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", WorkflowStage.from_legacy(self.stage))
        object.__setattr__(self, "status", StageRunState.from_legacy(self.status))
        if self.advanced_to is not None:
            object.__setattr__(
                self,
                "advanced_to",
                WorkflowStage.from_legacy(self.advanced_to),
            )


StageResult = StageRunResult


__all__ = [
    "StageResult",
    "StageRun",
    "StageRunResult",
    "StageRunState",
    "StageRunStatus",
    "WorkflowStage",
    "is_valid_transition",
    "parse_legacy_stage",
    "stage_to_legacy",
    "validate_transition",
]
