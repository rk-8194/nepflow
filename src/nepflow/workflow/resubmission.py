"""Workflow resubmission and legacy-state reconciliation result types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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


__all__ = [
    "ReconciliationResult",
    "ReconciliationSource",
    "ResubmissionResult",
    "SelfResubmitExit",
]
