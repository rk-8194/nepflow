"""Public transactional facade for the authoritative project ledger."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from nepflow.errors import StateError

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


__all__ = ["StateStore", "CURRENT_SCHEMA_VERSION"]
