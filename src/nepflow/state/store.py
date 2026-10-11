"""Public transactional facade for the authoritative project ledger."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from nepflow.errors import StateError
from nepflow.io.paths import resolve_project_path

from ._artifact_records import ArtifactRecordsMixin
from ._dataset_records import DatasetRecordsMixin
from ._dft_records import DftRecordsMixin
from ._event_records import EventRecordsMixin
from ._run_records import RunRecordsMixin
from ._structure_records import StructureRecordsMixin
from ._workflow_records import WorkflowRecordsMixin
from .migrations import CURRENT_SCHEMA_VERSION, migrate

_T = TypeVar("_T")


class StateStore(
    WorkflowRecordsMixin,
    StructureRecordsMixin,
    DftRecordsMixin,
    ArtifactRecordsMixin,
    DatasetRecordsMixin,
    RunRecordsMixin,
    EventRecordsMixin,
):
    """Own one SQLite ledger connection and its transaction boundary.

    Record-specific persistence lives in the responsibility-focused mixins;
    this class remains the public facade and owns connection lifecycle,
    locking, transaction boundaries, and write-error translation.
    """

    _connection: sqlite3.Connection

    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self._database = str(path)
        self.path = None if self._database == ":memory:" else Path(path)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                self._database,
                timeout=timeout,
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            migrate(self._connection)
        except StateError:
            self.close()
            raise
        except sqlite3.DatabaseError as exc:
            self.close()
            raise StateError(f"Could not open state database {path}") from exc

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the connection for read-only diagnostics and migrations."""

        return self._connection

    @property
    def schema_version(self) -> int:
        from .migrations import schema_version

        return schema_version(self._connection)

    def close(self) -> None:
        """Close the owned SQLite connection."""

        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            # Keep the closed-handle sentinel used by the existing lifecycle;
            # record mixins only access the connection while the store is open.
            self._connection = None  # type: ignore[assignment]

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @contextmanager
    def transaction(self):
        """Run a group of writes atomically using an immediate SQLite lock."""

        with self._lock:
            nested = self._connection.in_transaction
            if not nested:
                self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self
            except BaseException:
                # Roll back on every abort, including interruption, so a
                # partially completed state transaction is never committed.
                if not nested:
                    self._connection.rollback()
                raise
            else:
                if not nested:
                    self._connection.commit()

    def _write(self, operation: Callable[[], _T]) -> _T:
        try:
            with self.transaction():
                return operation()
        except StateError:
            raise
        except sqlite3.IntegrityError as exc:
            raise StateError("State write violates the ledger schema") from exc

    def _fetchone(self, query: str, parameters: Sequence[Any]) -> sqlite3.Row | None:
        with self._lock:
            return self._connection.execute(query, parameters).fetchone()

    def _fetchall(
        self,
        query: str,
        parameters: Sequence[Any] = (),
    ) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._connection.execute(query, parameters).fetchall())

    def resolve_artifact_path(
        self,
        artifact: Mapping[str, Any],
        project_root: str | Path,
        *,
        allow_legacy_absolute: bool = False,
    ) -> Path:
        """Resolve and validate an artifact row's project-scoped path."""

        value = artifact.get("path")
        if not isinstance(value, str) or not value.strip():
            raise StateError("Authoritative artifact has no persisted path")
        return resolve_project_path(
            value,
            project_root,
            field="authoritative artifact path",
            allow_legacy_absolute=allow_legacy_absolute,
            error_type=StateError,
        )

    def relocate_project_root(
        self,
        project_id: str,
        project_root: str | Path,
        *,
        expected_previous_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Migrate only mutable project-location metadata after relocation.

        Project identity, stage history, artifact IDs, and provenance remain
        untouched.  The optional previous-root check makes the compatibility
        operation deterministic instead of silently adopting an unrelated DB.
        """

        existing = self.get_project(project_id)
        if existing is None:
            raise StateError(f"Cannot relocate unknown project {project_id!r}")
        current_root = Path(project_root).resolve()
        stored_root = existing.get("root_path")
        if expected_previous_root is not None:
            expected = Path(expected_previous_root).resolve()
            if stored_root and Path(str(stored_root)).resolve() != expected:
                raise StateError(
                    f"Project {project_id!r} root changed unexpectedly: {stored_root}"
                )
        return self.upsert_project(
            project_id,
            name=str(existing.get("name") or project_id),
            root_path=str(current_root),
            config_fingerprint=existing.get("config_fingerprint"),
            metadata=existing.get("metadata", {}),
        )


__all__ = ["StateStore", "CURRENT_SCHEMA_VERSION"]
