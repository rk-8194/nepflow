"""Persistence operations for content-addressed artifact records."""

from __future__ import annotations

import sqlite3
from typing import Any

from nepflow.domain.identities import ArtifactIdentity
from nepflow.errors import StateError

from ._record_codec import decode_row, encode_json, now


class ArtifactRecordsMixin:
    """Provide StateStore persistence for shared artifact rows."""

    def record_artifact(
        self,
        artifact: ArtifactIdentity,
        *,
        originating_attempt_id: str | None = None,
        retention_status: str = "active",
        metadata: Any = None,
    ) -> dict[str, Any]:
        """Record one content-addressed artifact identity."""

        return self._write(
            lambda: self._record_artifact(
                artifact,
                originating_attempt_id=originating_attempt_id,
                retention_status=retention_status,
                metadata=metadata,
            )
        )

    def _record_artifact(
        self,
        artifact: ArtifactIdentity,
        *,
        originating_attempt_id: str | None,
        retention_status: str,
        metadata: Any,
    ) -> dict[str, Any]:
        existing = self._connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact.artifact_id,)
        ).fetchone()
        if existing is not None:
            if (
                existing["artifact_type"] != artifact.artifact_type
                or existing["sha256"] != artifact.sha256
            ):
                raise StateError(f"Artifact identity conflict: {artifact.artifact_id}")
            return decode_row(existing, ("metadata_json",))
        try:
            self._connection.execute(
                """
                INSERT INTO artifacts
                    (artifact_id, artifact_type, sha256, path, originating_attempt_id,
                     retention_status, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.artifact_type,
                    artifact.sha256,
                    artifact.path,
                    originating_attempt_id,
                    retention_status,
                    encode_json({} if metadata is None else metadata),
                    now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise StateError(
                "Artifact content is already registered under another identity: "
                f"{artifact.sha256}"
            ) from exc
        row = self._connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact.artifact_id,)
        ).fetchone()
        return decode_row(row, ("metadata_json",))

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        return None if row is None else decode_row(row, ("metadata_json",))

    def list_artifacts(
        self,
        *,
        originating_attempt_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List artifacts, optionally restricted to one DFT attempt."""

        if originating_attempt_id is None:
            rows = self._fetchall("SELECT * FROM artifacts ORDER BY created_at")
        else:
            rows = self._fetchall(
                "SELECT * FROM artifacts WHERE originating_attempt_id = ? ORDER BY created_at",
                (originating_attempt_id,),
            )
        return [decode_row(row, ("metadata_json",)) for row in rows]
