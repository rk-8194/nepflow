"""Persistence operations for content-addressed artifact records."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from nepflow.domain.identities import ArtifactIdentity
from nepflow.errors import StateError
from nepflow.io.paths import project_relative_path

from ._record_codec import decode_row, encode_json, now
from ._typing import StateStoreMixinSupport


class ArtifactRecordsMixin(StateStoreMixinSupport):
    """Provide StateStore persistence for shared artifact rows."""

    def record_artifact(
        self,
        artifact: ArtifactIdentity,
        *,
        originating_attempt_id: str | None = None,
        retention_status: str = "active",
        metadata: Any = None,
        project_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Record one content-addressed artifact identity.

        When ``project_root`` is supplied, only the operational path metadata
        is normalised to a project-relative value; the content identity is
        untouched.
        """

        persisted_artifact = artifact
        if project_root is not None and artifact.path is not None:
            persisted_artifact = artifact.with_path(
                project_relative_path(
                    artifact.path,
                    project_root,
                    field="artifact path",
                    error_type=StateError,
                )
            )

        return self._write(
            lambda: self._record_artifact(
                persisted_artifact,
                originating_attempt_id=originating_attempt_id,
                retention_status=retention_status,
                metadata=metadata,
                project_root=project_root,
            )
        )

    def _record_artifact(
        self,
        artifact: ArtifactIdentity,
        *,
        originating_attempt_id: str | None,
        retention_status: str,
        metadata: Any,
        project_root: str | Path | None = None,
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
            if project_root is not None and artifact.path is not None:
                existing_path = existing["path"]
                if existing_path != artifact.path:
                    self._connection.execute(
                        "UPDATE artifacts SET path = ? WHERE artifact_id = ?",
                        (artifact.path, artifact.artifact_id),
                    )
                    existing = self._connection.execute(
                        "SELECT * FROM artifacts WHERE artifact_id = ?",
                        (artifact.artifact_id,),
                    ).fetchone()
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
                f"Artifact content is already registered under another identity: {artifact.sha256}"
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
