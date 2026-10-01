from __future__ import annotations

import sqlite3

import pytest

from nepflow.errors import StateError
from nepflow.state.migrations import CURRENT_SCHEMA_VERSION, migrate, schema_version
from nepflow.state.schema import REQUIRED_TABLES
from nepflow.state.store import StateStore


def test_new_store_applies_complete_ledger_schema(tmp_path) -> None:
    path = tmp_path / "state.db"

    with StateStore(path) as store:
        assert store.schema_version == CURRENT_SCHEMA_VERSION
        tables = {
            row[0]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert REQUIRED_TABLES <= tables


def test_migration_is_idempotent(tmp_path) -> None:
    path = tmp_path / "state.db"

    with StateStore(path) as store:
        first_version = store.schema_version
    with StateStore(path) as store:
        assert store.schema_version == first_version == CURRENT_SCHEMA_VERSION

    connection = sqlite3.connect(path)
    try:
        assert schema_version(connection) == CURRENT_SCHEMA_VERSION
        migrate(connection)
        assert schema_version(connection) == CURRENT_SCHEMA_VERSION
    finally:
        connection.close()


def test_future_schema_version_is_rejected(tmp_path) -> None:
    path = tmp_path / "state.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA user_version = 999")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StateError, match="newer schema"):
        StateStore(path)


def test_versioned_but_incomplete_schema_is_rejected(tmp_path) -> None:
    path = tmp_path / "state.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE project (project_id TEXT PRIMARY KEY)")
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StateError, match="missing ledger tables"):
        StateStore(path)


def test_failed_initial_migration_rolls_back(tmp_path) -> None:
    path = tmp_path / "state.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE project (project_id TEXT PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StateError):
        StateStore(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert tables == {"project"}
    finally:
        connection.close()
