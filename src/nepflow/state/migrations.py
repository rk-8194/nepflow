"""Versioned SQLite schema migrations for the project ledger."""

from __future__ import annotations

import sqlite3

from nepflow.errors import StateError

from .schema import MIGRATION_STATEMENTS, REQUIRED_COLUMNS, REQUIRED_TABLES, SCHEMA_VERSION


CURRENT_SCHEMA_VERSION = SCHEMA_VERSION


def schema_version(connection: sqlite3.Connection) -> int:
    """Return SQLite's application schema version."""

    try:
        row = connection.execute("PRAGMA user_version").fetchone()
    except sqlite3.DatabaseError as exc:
        raise StateError("Could not read the state database schema version") from exc
    if row is None or not isinstance(row[0], int):
        raise StateError("State database returned an invalid schema version")
    return int(row[0])


def _set_schema_version(connection: sqlite3.Connection, version: int) -> None:
    # SQLite does not allow a bound parameter in PRAGMA assignment.  The value
    # comes only from the module's versioned migration table.
    connection.execute(f"PRAGMA user_version = {version}")


def validate_schema(connection: sqlite3.Connection, version: int | None = None) -> None:
    """Reject a versioned database whose required ledger objects are missing."""

    actual_version = schema_version(connection) if version is None else version
    if actual_version != CURRENT_SCHEMA_VERSION:
        raise StateError(
            "State database schema validation requires the current schema version: "
            f"{actual_version} != {CURRENT_SCHEMA_VERSION}"
        )

    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise StateError("Could not inspect the state database schema") from exc
    tables = {str(row[0]) for row in rows}
    missing_tables = REQUIRED_TABLES - tables
    if missing_tables:
        missing = ", ".join(sorted(missing_tables))
        raise StateError(f"State database is missing ledger tables: {missing}")

    for table, required_columns in REQUIRED_COLUMNS.items():
        try:
            columns = {
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
        except sqlite3.DatabaseError as exc:
            raise StateError(f"Could not inspect state table {table}") from exc
        missing_columns = required_columns - columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise StateError(f"State table {table} is missing columns: {missing}")


def migrate(connection: sqlite3.Connection) -> int:
    """Apply all known migrations and return the resulting schema version.

    Migrations are executed in one transaction.  A failed migration is rolled
    back so a partially-created ledger is never presented as current.
    """

    connection.execute("PRAGMA foreign_keys = ON")
    version = schema_version(connection)
    if version > CURRENT_SCHEMA_VERSION:
        raise StateError(
            "State database uses a newer schema than this NEPFlow version: "
            f"{version} > {CURRENT_SCHEMA_VERSION}"
        )

    started_transaction = connection.in_transaction
    if not started_transaction:
        connection.execute("BEGIN IMMEDIATE")
    try:
        for target_version in range(version + 1, CURRENT_SCHEMA_VERSION + 1):
            statements = MIGRATION_STATEMENTS.get(target_version)
            if statements is None:
                raise StateError(f"No migration is registered for schema {target_version}")
            for statement in statements:
                connection.execute(statement)
            _set_schema_version(connection, target_version)
        validate_schema(connection)
        if not started_transaction:
            connection.commit()
    except StateError:
        if not started_transaction:
            connection.rollback()
        raise
    except sqlite3.DatabaseError as exc:
        if not started_transaction:
            connection.rollback()
        raise StateError("Could not migrate the state database") from exc
    except Exception as exc:
        if not started_transaction:
            connection.rollback()
        raise StateError("Unexpected failure while migrating the state database") from exc
    return CURRENT_SCHEMA_VERSION


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "migrate",
    "schema_version",
    "validate_schema",
]
