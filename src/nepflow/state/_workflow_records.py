"""Persistence operations for projects, stage runs, and selection runs."""

from __future__ import annotations

from typing import Any

from nepflow.errors import StateError

from ._record_codec import canonical_timestamp, decode_row, encode_json, now
from ._typing import StateStoreMixinSupport, require_state_row


class WorkflowRecordsMixin(StateStoreMixinSupport):
    """Provide StateStore persistence for workflow ledger records."""

    def upsert_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        root_path: str | None = None,
        config_fingerprint: str | None = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        """Insert or update one project ledger record by stable project ID."""

        project_name = name or project_id
        timestamp = now()
        return self._write(
            lambda: self._write_project(
                project_id, project_name, root_path, config_fingerprint, metadata, timestamp
            )
        )

    def _write_project(
        self,
        project_id: str,
        project_name: str,
        root_path: str | None,
        config_fingerprint: str | None,
        metadata: Any,
        timestamp: str,
    ) -> dict[str, Any]:
        self._connection.execute(
            """
            INSERT INTO project
                (project_id, name, root_path, config_fingerprint, metadata_json,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                name = excluded.name,
                root_path = excluded.root_path,
                config_fingerprint = excluded.config_fingerprint,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                project_id,
                project_name,
                root_path,
                config_fingerprint,
                encode_json({} if metadata is None else metadata),
                timestamp,
                timestamp,
            ),
        )
        return require_state_row(self.get_project(project_id), "project")

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM project WHERE project_id = ?", (project_id,))
        return None if row is None else decode_row(row, ("metadata_json",))

    def record_project(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        """Compatibility-shaped name for recording a project ledger row."""

        return self.upsert_project(project_id, **kwargs)

    def upsert_stage_run(
        self,
        stage_run_id: str,
        project_id: str,
        stage: str,
        *,
        status: str = "pending",
        input_fingerprint: str | None = None,
        output_fingerprint: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        """Insert or update one workflow stage-run record."""

        timestamp = now()
        started_at = canonical_timestamp(started_at)
        completed_at = canonical_timestamp(completed_at)
        return self._write(
            lambda: self._write_stage_run(
                stage_run_id,
                project_id,
                stage,
                status,
                input_fingerprint,
                output_fingerprint,
                started_at,
                completed_at,
                metadata,
                timestamp,
            )
        )

    def _write_stage_run(
        self,
        stage_run_id: str,
        project_id: str,
        stage: str,
        status: str,
        input_fingerprint: str | None,
        output_fingerprint: str | None,
        started_at: str | None,
        completed_at: str | None,
        metadata: Any,
        timestamp: str,
    ) -> dict[str, Any]:
        existing = self._connection.execute(
            "SELECT project_id, stage FROM stage_runs WHERE stage_run_id = ?",
            (stage_run_id,),
        ).fetchone()
        if existing is not None and (
            existing["project_id"] != project_id or existing["stage"] != stage
        ):
            raise StateError(f"Stage-run identity conflict: {stage_run_id}")
        self._connection.execute(
            """
            INSERT INTO stage_runs
                (stage_run_id, project_id, stage, status, input_fingerprint,
                 output_fingerprint, started_at, completed_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stage_run_id) DO UPDATE SET
                status = excluded.status,
                input_fingerprint = excluded.input_fingerprint,
                output_fingerprint = excluded.output_fingerprint,
                started_at = excluded.started_at,
                completed_at = excluded.completed_at,
                metadata_json = excluded.metadata_json
            """,
            (
                stage_run_id,
                project_id,
                stage,
                status,
                input_fingerprint,
                output_fingerprint,
                started_at,
                completed_at,
                encode_json({} if metadata is None else metadata),
            ),
        )
        return require_state_row(self.get_stage_run(stage_run_id), "stage run")

    def get_stage_run(self, stage_run_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM stage_runs WHERE stage_run_id = ?", (stage_run_id,))
        return None if row is None else decode_row(row, ("metadata_json",))

    def list_stage_runs(self, project_id: str) -> list[dict[str, Any]]:
        """Return stage-run records for a project in most-recent-first order."""

        rows = self._fetchall(
            "SELECT * FROM stage_runs "
            "WHERE project_id = ? "
            "ORDER BY COALESCE(started_at, completed_at, '') DESC, stage_run_id DESC",
            (project_id,),
        )
        return [decode_row(row, ("metadata_json",)) for row in rows]

    def get_latest_stage_run(self, project_id: str) -> dict[str, Any] | None:
        """Return the current-most stage-run record for a project."""

        rows = self.list_stage_runs(project_id)
        return rows[0] if rows else None

    def record_stage_run(
        self,
        stage_run_id: str,
        project_id: str,
        stage: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Record one stage run by its stable run ID."""

        return self.upsert_stage_run(stage_run_id, project_id, stage, **kwargs)

    def upsert_selection_run(
        self,
        selection_run_id: str,
        project_id: str,
        *,
        status: str = "pending",
        method: str | None = None,
        parameters: Any = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update one selection-run record."""

        timestamp = now()
        started_at = canonical_timestamp(started_at)
        completed_at = canonical_timestamp(completed_at)
        return self._write(
            lambda: self._write_selection_run(
                selection_run_id,
                project_id,
                status,
                method,
                parameters,
                started_at,
                completed_at,
                timestamp,
            )
        )

    def _write_selection_run(
        self,
        selection_run_id: str,
        project_id: str,
        status: str,
        method: str | None,
        parameters: Any,
        started_at: str | None,
        completed_at: str | None,
        timestamp: str,
    ) -> dict[str, Any]:
        existing = self._connection.execute(
            "SELECT project_id FROM selection_runs WHERE selection_run_id = ?",
            (selection_run_id,),
        ).fetchone()
        if existing is not None and existing["project_id"] != project_id:
            raise StateError(f"Selection-run identity conflict: {selection_run_id}")
        self._connection.execute(
            """
            INSERT INTO selection_runs
                (selection_run_id, project_id, status, method, parameters_json,
                 started_at, completed_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(selection_run_id) DO UPDATE SET
                status = excluded.status,
                method = excluded.method,
                parameters_json = excluded.parameters_json,
                started_at = excluded.started_at,
                completed_at = excluded.completed_at
            """,
            (
                selection_run_id,
                project_id,
                status,
                method,
                encode_json({} if parameters is None else parameters),
                started_at,
                completed_at,
                timestamp,
            ),
        )
        return require_state_row(self.get_selection_run(selection_run_id), "selection run")

    def get_selection_run(self, selection_run_id: str) -> dict[str, Any] | None:
        row = self._fetchone(
            "SELECT * FROM selection_runs WHERE selection_run_id = ?",
            (selection_run_id,),
        )
        return None if row is None else decode_row(row, ("parameters_json",))
