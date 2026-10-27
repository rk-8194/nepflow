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
        """Reconcile the marker from authoritative ledger state.

        A committed StateStore stage always wins; the marker is repaired as a
        cache.  Only when no ledger row exists is a valid legacy marker
        imported into the store, making migration explicit and one-way.
        """

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

    def restart_selection(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> StageRunStatus:
        """Guardedly rewind workflow authority to ``select``.

        This is intentionally separate from ``transition_to``: routine
        processing remains adjacent-only, while an explicit operator restart
        invalidates downstream stage authority and prior selection reuse in
        one ledger transaction.  Scientific files are retained as history.
        """

        generated_path = (
            self.marker_path.parent
            / "structures"
            / "generated"
            / "generated_structures.xyz"
        )
        try:
            generated_ready = generated_path.is_file() and generated_path.stat().st_size > 0
        except OSError as exc:
            raise StateError(
                f"Cannot restart selection; generated input cannot be inspected: {generated_path}"
            ) from exc
        if not generated_ready:
            raise StateError(
                "Cannot restart selection; generated structures are missing or empty: "
                f"{generated_path}"
            )

        current = self.reconcile().stage
        restartable = {
            WorkflowStage.SELECT,
            WorkflowStage.RUN_VASP,
            WorkflowStage.TRAIN_NEP,
            WorkflowStage.VALIDATE,
            WorkflowStage.COMPLETED,
        }
        if current not in restartable:
            raise StateError(
                "Explicit selection restart requires the current workflow stage to be "
                f"select or later; current stage is {current.value!r}"
            )
        previous_runs = {
            stage.value: self.store.get_stage_run(self._stage_run_id(stage))
            for stage in (
                WorkflowStage.SELECT,
                WorkflowStage.RUN_VASP,
                WorkflowStage.TRAIN_NEP,
                WorkflowStage.VALIDATE,
                WorkflowStage.COMPLETED,
            )
        }
        now = self._timestamp()
        base_metadata = dict(metadata or {})
        base_metadata.update(
            {
                "source": "explicit_select_restart",
                "restart_target": WorkflowStage.SELECT.value,
                "invalidated_at": now,
            }
        )
        # Give the replacement select run a strictly newer ordering key than
        # the invalidated downstream rows.  The ledger uses started_at when
        # deciding which stage is authoritative, so equal timestamps would
        # make the result depend on stage-id tie breaking.
        select_started_at = (
            datetime.fromisoformat(now) + timedelta(microseconds=1)
        ).isoformat()
        with self.store.transaction():
            self.store.invalidate_selection_runs(self.project_id)
            for stage, previous in previous_runs.items():
                if previous is None and stage != WorkflowStage.SELECT.value:
                    continue
                previous_record = None
                if previous is not None:
                    previous_record = {
                        "stage_run_id": previous.get("stage_run_id"),
                        "status": previous.get("status"),
                        "started_at": previous.get("started_at"),
                        "completed_at": previous.get("completed_at"),
                        "metadata": previous.get("metadata"),
                    }
                stage_value = WorkflowStage.from_legacy(stage)
                stage_metadata = dict(base_metadata)
                stage_metadata["previous_stage_run"] = previous_record
                if stage_value is WorkflowStage.SELECT:
                    self._record_stage(
                        stage_value,
                        status=StageRunState.RUNNING,
                        started_at=select_started_at,
                        completed_at=None,
                        metadata=stage_metadata,
                    )
                else:
                    self._record_stage(
                        stage_value,
                        status=StageRunState.FAILED,
                        started_at=now,
                        completed_at=now,
                        metadata={**stage_metadata, "invalidated": True},
                    )

        self._write_marker(WorkflowStage.SELECT)
        row = self.store.get_stage_run(self._stage_run_id(WorkflowStage.SELECT))
        if row is None:  # pragma: no cover - guarded by the store write
            raise StateError("Selection restart was not persisted")
        return StageRunStatus.from_mapping(row)

    def restart_dft(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> StageRunStatus:
        """Rewind workflow authority to ``run_vasp`` for an explicit restart."""

        current = self.reconcile().stage
        restartable = {
            WorkflowStage.RUN_VASP,
            WorkflowStage.TRAIN_NEP,
            WorkflowStage.VALIDATE,
            WorkflowStage.COMPLETED,
        }
        if current not in restartable:
            raise StateError(
                "Explicit DFT restart requires the current workflow stage to be "
                f"run_vasp or later; current stage is {current.value!r}"
            )

        previous_runs = {
            stage.value: self.store.get_stage_run(self._stage_run_id(stage))
            for stage in restartable
        }
        now = self._timestamp()
        base_metadata = dict(metadata or {})
        base_metadata.update(
            {
                "source": "explicit_run_vasp_restart",
                "restart_target": WorkflowStage.RUN_VASP.value,
                "invalidated_at": now,
            }
        )
        run_vasp_started_at = (
            datetime.fromisoformat(now) + timedelta(microseconds=1)
        ).isoformat()
        with self.store.transaction():
            for stage, previous in previous_runs.items():
                if previous is None:
                    continue
                previous_record = {
                    "stage_run_id": previous.get("stage_run_id"),
                    "status": previous.get("status"),
                    "started_at": previous.get("started_at"),
                    "completed_at": previous.get("completed_at"),
                    "metadata": previous.get("metadata"),
                }
                stage_value = WorkflowStage.from_legacy(stage)
                stage_metadata = dict(base_metadata)
                stage_metadata["previous_stage_run"] = previous_record
                if stage_value is WorkflowStage.RUN_VASP:
                    self._record_stage(
                        stage_value,
                        status=StageRunState.RUNNING,
                        started_at=run_vasp_started_at,
                        completed_at=None,
                        metadata=stage_metadata,
                    )
                else:
                    self._record_stage(
                        stage_value,
                        status=StageRunState.FAILED,
                        started_at=now,
                        completed_at=now,
                        metadata={**stage_metadata, "invalidated": True},
                    )

        self._write_marker(WorkflowStage.RUN_VASP)
        row = self.store.get_stage_run(self._stage_run_id(WorkflowStage.RUN_VASP))
        if row is None:  # pragma: no cover - guarded by the store write
            raise StateError("DFT restart was not persisted")
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
