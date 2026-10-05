"""Authoritative workflow-state reconciliation and transition service.

The workflow controller coordinates execution, while this module owns the
state boundary: ``StateStore`` is authoritative and ``.project`` is only a
legacy compatibility marker.  Keeping that policy in one service makes the
controller usable with both the current legacy stage adapters and the future
canonical stages.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from nepflow.errors import StateError
from nepflow.io.atomic import atomic_write_text
from nepflow.state import StateStore

from .resubmission import ReconciliationResult, ReconciliationSource
from .stages import (
    StageRunState,
    StageRunStatus,
    WorkflowStage,
    parse_legacy_stage,
    stage_to_legacy,
    validate_transition,
)


class WorkflowState:
    """Read, reconcile, and transition one project's workflow state."""

    def __init__(
        self,
        project_id: str,
        store: StateStore,
        marker_path: str | Path,
    ) -> None:
        self.project_id = project_id
        self.store = store
        self.marker_path = Path(marker_path)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read_marker(self) -> WorkflowStage | None:
        if not self.marker_path.exists():
            return None
        try:
            marker = self.marker_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StateError(f"Failed to read workflow marker {self.marker_path}") from exc
        try:
            return parse_legacy_stage(marker)
        except StateError as exc:
            raise StateError(f"Invalid workflow stage marker in {self.marker_path}: {exc}") from exc

    def _write_marker(self, stage: WorkflowStage) -> None:
        try:
            self.marker_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                self.marker_path,
                stage_to_legacy(stage),
                encoding="utf-8",
            )
        except OSError as exc:
            raise StateError(f"Failed to write workflow marker {self.marker_path}") from exc

    def _latest_run(self) -> StageRunStatus | None:
        row = self.store.get_latest_stage_run(self.project_id)
        return StageRunStatus.from_mapping(row) if row is not None else None

    def _read_authoritative_state(
        self,
    ) -> tuple[StageRunStatus | None, WorkflowStage | None, StateError | None]:
        stored_run = self._latest_run()
        marker_stage: WorkflowStage | None = None
        marker_error: StateError | None = None
        try:
            marker_stage = self._read_marker()
        except StateError as exc:
            marker_error = exc
        return stored_run, marker_stage, marker_error

    def reconcile(self) -> ReconciliationResult:
        """Reconcile the compatibility marker from authoritative ledger state."""

        stored_run, marker_stage, marker_error = self._read_authoritative_state()

        if stored_run is not None:
            marker_needs_repair = marker_error is not None or marker_stage is not stored_run.stage
            if marker_needs_repair:
                self._write_marker(stored_run.stage)
            if marker_error is not None:
                reason = "repaired invalid legacy marker from StateStore"
            elif marker_stage is None:
                reason = "repaired missing legacy marker from StateStore"
            elif marker_stage is not stored_run.stage:
                reason = "repaired stale legacy marker from StateStore"
            else:
                reason = None
            return ReconciliationResult(
                stage=stored_run.stage,
                source=ReconciliationSource.STATE_STORE,
                marker_stage=marker_stage,
                authoritative_stage=stored_run.stage,
                changed=marker_needs_repair,
                reason=reason,
            )

        if marker_error is not None:
            raise marker_error

        if marker_stage is not None:
            now = self._timestamp()
            status = StageRunState.COMPLETED if marker_stage.is_terminal else StageRunState.RUNNING
            with self.store.transaction():
                self._record_stage(
                    marker_stage,
                    status=status,
                    started_at=now,
                    completed_at=now if marker_stage.is_terminal else None,
                    metadata={"source": "legacy_project_marker"},
                )
            return ReconciliationResult(
                stage=marker_stage,
                source=ReconciliationSource.LEGACY_MARKER,
                marker_stage=marker_stage,
                authoritative_stage=marker_stage,
                changed=True,
            )

        raise StateError(
            "Authoritative workflow state is missing; restore the stage ledger "
            "or perform an explicit migration"
        )

    def current_stage(self) -> WorkflowStage:
        """Return the current stage after explicit reconciliation."""

        return self.reconcile().stage

    def _stage_run_id(self, stage: WorkflowStage) -> str:
        return f"{self.project_id}:{stage.value}"

    def _record_stage(
        self,
        stage: WorkflowStage,
        *,
        status: StageRunState,
        metadata: Mapping[str, Any] | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        self.store.upsert_stage_run(
            self._stage_run_id(stage),
            self.project_id,
            stage.value,
            status=status.value,
            started_at=started_at,
            completed_at=completed_at,
            metadata={} if metadata is None else dict(metadata),
        )

    def set_status(
        self,
        stage: WorkflowStage | str,
        status: StageRunState | str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> StageRunStatus:
        """Persist an operational status without changing workflow stage."""

        typed_stage = parse_legacy_stage(stage)
        typed_status = StageRunState.from_legacy(status)
        current = self.current_stage()
        if current is not typed_stage:
            raise StateError(
                f"Cannot update {typed_stage.value!r} while current stage is {current.value!r}"
            )
        existing = self._latest_run()
        now = self._timestamp()
        started_at = existing.started_at if existing is not None else now
        completed_at = (
            now
            if typed_status
            in {
                StageRunState.COMPLETED,
                StageRunState.FAILED,
            }
            else None
        )
        with self.store.transaction():
            self._record_stage(
                typed_stage,
                status=typed_status,
                started_at=started_at,
                completed_at=completed_at,
                metadata=metadata,
            )
        row = self.store.get_stage_run(self._stage_run_id(typed_stage))
        if row is None:  # pragma: no cover - guarded by the store write
            raise StateError(f"Stage run was not persisted: {typed_stage.value}")
        return StageRunStatus.from_mapping(row)

    def transition_to(
        self,
        target: WorkflowStage | str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> StageRunStatus:
        """Validate and atomically persist one adjacent stage transition."""

        typed_target = parse_legacy_stage(target)
        current = self.reconcile().stage
        previous_run = self._latest_run()
        if previous_run is None:
            raise StateError("Workflow reconciliation did not create authoritative state")
        validate_transition(current, typed_target)

        now = self._timestamp()
        with self.store.transaction():
            if current is not typed_target:
                self._record_stage(
                    current,
                    status=StageRunState.COMPLETED,
                    started_at=previous_run.started_at,
                    completed_at=now,
                    metadata=metadata or {"source": "workflow_controller"},
                )
                target_started_at = self._timestamp()
                if target_started_at <= now:
                    target_started_at = (
                        datetime.fromisoformat(now) + timedelta(microseconds=1)
                    ).isoformat()
            else:
                target_started_at = previous_run.started_at or now
            self._record_stage(
                typed_target,
                status=(
                    StageRunState.COMPLETED if typed_target.is_terminal else StageRunState.RUNNING
                ),
                started_at=target_started_at,
                completed_at=target_started_at if typed_target.is_terminal else None,
                metadata=metadata or {"source": "workflow_controller"},
            )

        # The marker is a cache.  If this write fails, the next reconciliation
        # repairs it from the already-committed ledger record.
        self._write_marker(typed_target)
        row = self.store.get_stage_run(self._stage_run_id(typed_target))
        if row is None:  # pragma: no cover - guarded by the store write
            raise StateError(f"Stage transition was not persisted: {typed_target.value}")
        return StageRunStatus.from_mapping(row)

    def mark_failed(self, error: BaseException) -> StageRunStatus:
        """Record failure of the current stage without advancing it."""

        current = self.current_stage()
        return self.set_status(
            current,
            StageRunState.FAILED,
            metadata={
                "source": "workflow_controller",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )


__all__ = ["WorkflowState"]
