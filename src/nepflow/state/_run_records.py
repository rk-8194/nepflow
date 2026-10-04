"""Persistence operations for model and validation run records."""

from __future__ import annotations

from typing import Any

from nepflow.domain.models import ModelRunRecord, ValidationRunRecord
from nepflow.errors import StateError

from ._record_codec import decode_json, decode_row, encode_json, now


class RunRecordsMixin:
    """Provide StateStore persistence for model and validation runs."""

    def upsert_model_run(
        self,
        record: ModelRunRecord,
        *,
        status: str = "pending",
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update one model-run identity and its artifact links."""

        return self._write(
            lambda: self._write_model_run(
                record, status, started_at, completed_at, now()
            )
        )

    def _write_model_run(
        self,
        record: ModelRunRecord,
        status: str,
        started_at: str | None,
        completed_at: str | None,
        timestamp: str,
    ) -> dict[str, Any]:
        identity = record.identity
        existing = self._connection.execute(
            "SELECT identity_json FROM model_runs WHERE model_run_id = ?",
            (identity.model_run_id,),
        ).fetchone()
        if existing is not None and encode_json(
            decode_json(existing["identity_json"], "identity_json")
        ) != encode_json(identity.to_dict()):
            raise StateError(f"Model-run identity conflict: {identity.model_run_id}")
        self._connection.execute(
            """
            INSERT INTO model_runs
                (model_run_id, dataset_id, identity_json, status,
                 execution_metadata_json, started_at, completed_at,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_run_id) DO UPDATE SET
                status = excluded.status,
                execution_metadata_json = excluded.execution_metadata_json,
                started_at = excluded.started_at,
                completed_at = excluded.completed_at,
                updated_at = excluded.updated_at
            """,
            (
                identity.model_run_id,
                identity.dataset_id,
                encode_json(identity.to_dict()),
                status,
                encode_json(
                    {} if record.execution_metadata is None else record.execution_metadata
                ),
                started_at,
                completed_at,
                timestamp,
                timestamp,
            ),
        )
        self._link_model_artifacts(record)
        return self.get_model_run(identity.model_run_id)  # type: ignore[return-value]

    def _link_model_artifacts(self, record: ModelRunRecord) -> None:
        if record.artifact is None:
            return
        for role, artifact in (
            ("model", record.artifact.model),
            ("nep_in", record.artifact.nep_in),
        ):
            if artifact is None:
                continue
            self._record_artifact(
                artifact,
                originating_attempt_id=None,
                retention_status="active",
                metadata={
                    "status": record.artifact.status,
                    "metrics": record.artifact.metrics,
                },
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO model_artifacts "
                "(model_run_id, artifact_id, role, metadata_json) VALUES (?, ?, ?, ?)",
                (record.identity.model_run_id, artifact.artifact_id, role, encode_json({})),
            )

    def upsert_validation_run(
        self,
        record: ValidationRunRecord,
        *,
        status: str = "pending",
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update one validation-run identity and artifact links."""

        return self._write(
            lambda: self._write_validation_run(
                record, status, started_at, completed_at, now()
            )
        )

    def _write_validation_run(
        self,
        record: ValidationRunRecord,
        status: str,
        started_at: str | None,
        completed_at: str | None,
        timestamp: str,
    ) -> dict[str, Any]:
        identity = record.identity
        existing = self._connection.execute(
            "SELECT identity_json FROM validation_runs WHERE validation_run_id = ?",
            (identity.validation_run_id,),
        ).fetchone()
        if existing is not None and encode_json(
            decode_json(existing["identity_json"], "identity_json")
        ) != encode_json(identity.to_dict()):
            raise StateError(
                f"Validation-run identity conflict: {identity.validation_run_id}"
            )
        passed = None if record.artifact is None else record.artifact.passed
        self._connection.execute(
            """
            INSERT INTO validation_runs
                (validation_run_id, model_run_id, dataset_id, identity_json,
                 status, passed, metadata_json, started_at, completed_at,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(validation_run_id) DO UPDATE SET
                status = excluded.status,
                passed = excluded.passed,
                metadata_json = excluded.metadata_json,
                started_at = excluded.started_at,
                completed_at = excluded.completed_at,
                updated_at = excluded.updated_at
            """,
            (
                identity.validation_run_id,
                identity.model_run_id,
                identity.dataset_id,
                encode_json(identity.to_dict()),
                status,
                None if passed is None else int(passed),
                encode_json({} if record.metadata is None else record.metadata),
                started_at,
                completed_at,
                timestamp,
                timestamp,
            ),
        )
        self._link_validation_artifacts(record)
        return self.get_validation_run(identity.validation_run_id)  # type: ignore[return-value]

    def _link_validation_artifacts(self, record: ValidationRunRecord) -> None:
        if record.artifact is None:
            return
        for role, artifact in (
            ("report", record.artifact.report),
            ("trajectory", record.artifact.trajectory),
        ):
            if artifact is None:
                continue
            self._record_artifact(
                artifact,
                originating_attempt_id=None,
                retention_status="active",
                metadata={
                    "metrics": record.artifact.metrics,
                    "thresholds": record.artifact.thresholds,
                },
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO validation_artifacts "
                "(validation_run_id, artifact_id, role, metadata_json) VALUES (?, ?, ?, ?)",
                (
                    record.identity.validation_run_id,
                    artifact.artifact_id,
                    role,
                    encode_json({}),
                ),
            )

    def get_model_run(self, model_run_id: str) -> dict[str, Any] | None:
        """Return one authoritative model-run record."""

        row = self._fetchone("SELECT * FROM model_runs WHERE model_run_id = ?", (model_run_id,))
        return (
            None
            if row is None
            else decode_row(row, ("identity_json", "execution_metadata_json"))
        )

    def list_model_artifacts(self, model_run_id: str) -> list[dict[str, Any]]:
        """Return artifacts linked to one authoritative model run."""

        rows = self._fetchall(
            "SELECT a.*, ma.role, ma.metadata_json AS link_metadata_json "
            "FROM model_artifacts AS ma "
            "JOIN artifacts AS a ON a.artifact_id = ma.artifact_id "
            "WHERE ma.model_run_id = ? ORDER BY ma.role, a.artifact_id",
            (model_run_id,),
        )
        return [
            decode_row(row, ("metadata_json", "link_metadata_json"))
            for row in rows
        ]

    def record_model_run(self, record: ModelRunRecord, **kwargs: Any) -> dict[str, Any]:
        """Record one model-run identity and its artifact links."""

        return self.upsert_model_run(record, **kwargs)

    def get_validation_run(self, validation_run_id: str) -> dict[str, Any] | None:
        """Return one authoritative validation-run record."""

        row = self._fetchone(
            "SELECT * FROM validation_runs WHERE validation_run_id = ?",
            (validation_run_id,),
        )
        return None if row is None else decode_row(row, ("identity_json", "metadata_json"))

    def record_validation_run(
        self,
        record: ValidationRunRecord,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Record one validation-run identity and its artifact links."""

        return self.upsert_validation_run(record, **kwargs)
