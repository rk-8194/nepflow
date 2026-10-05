"""Shared implementation surface for the StateStore record mixins."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any, TypeVar

from nepflow.domain.identities import ArtifactIdentity
from nepflow.errors import StateError

_T = TypeVar("_T")


def require_state_row(row: dict[str, Any] | None, label: str) -> dict[str, Any]:
    """Assert that a row written in the current transaction is readable."""

    if row is None:
        raise StateError(f"StateStore could not read {label} after writing it")
    return row


class StateStoreMixinSupport:
    """Internal members supplied by the concrete :class:`StateStore`."""

    _connection: sqlite3.Connection
    _lock: threading.RLock

    def transaction(self) -> AbstractContextManager[StateStoreMixinSupport]:
        raise NotImplementedError

    def _write(self, operation: Callable[[], _T]) -> _T:
        raise NotImplementedError

    def _fetchone(self, query: str, parameters: Sequence[Any]) -> sqlite3.Row | None:
        raise NotImplementedError

    def _fetchall(
        self,
        query: str,
        parameters: Sequence[Any] = (),
    ) -> list[sqlite3.Row]:
        raise NotImplementedError

    def _record_artifact(
        self,
        artifact: ArtifactIdentity,
        *,
        originating_attempt_id: str | None,
        retention_status: str,
        metadata: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError


__all__ = ["StateStoreMixinSupport", "require_state_row"]
