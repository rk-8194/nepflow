"""Typed transactional access to the authoritative project ledger."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Any, TypeVar

from nepflow.domain.datasets import DatasetIdentity, SelectedDatasetMember, TrainingDatasetManifest
from nepflow.domain.identities import (
    ArtifactIdentity,
    DftCalculationIdentity,
    StructureIdentity,
)
from nepflow.domain.models import ModelRunRecord, ValidationRunRecord
from nepflow.domain.structures import GeneratedStructureRecord, StructureProvenance
from nepflow.errors import StateError
from nepflow.io.json import dumps, loads, to_jsonable

from .migrations import CURRENT_SCHEMA_VERSION, migrate


_T = TypeVar("_T")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return dumps(to_jsonable(value), canonical=True, trailing_newline=False)


def _decode_json(value: str | None, field: str) -> Any:
    if value is None:
        return None
    try:
        return loads(value)
    except (TypeError, ValueError) as exc:
        raise StateError(f"Malformed {field} in state database") from exc


def _decode_row(row: sqlite3.Row, json_columns: Iterable[str] = ()) -> dict[str, Any]:
    result = dict(row)
    for column in json_columns:
        result[column.removesuffix("_json")] = _decode_json(
            result.pop(column),
            column,
        )
    return result


def _identity_payload(identity: Any) -> dict[str, Any]:
    if isinstance(identity, DftCalculationIdentity):
        payload = identity.scientific_payload()
        payload["calculation_id"] = identity.calculation_id
        return payload
    if hasattr(identity, "to_dict"):
        return dict(identity.to_dict())
    if isinstance(identity, Mapping):
        return dict(identity)
    raise TypeError("identity must be a domain identity record or mapping")


class StateStore:
    """Own one SQLite ledger connection and its transaction boundary.

    The store is deliberately independent of legacy marker files.  New callers
    can write authoritative records here while existing stage implementations
    continue to reconcile their files during the later migration.
    """

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
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
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

    def _fetchall(self, query: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._connection.execute(query, parameters).fetchall())

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
        now = _now()

        def write() -> dict[str, Any]:
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
                    _json({} if metadata is None else metadata),
                    now,
                    now,
                ),
            )
            return self.get_project(project_id)  # type: ignore[return-value]

        return self._write(write)

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM project WHERE project_id = ?", (project_id,))
        return None if row is None else _decode_row(row, ("metadata_json",))

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
        now = _now()

        def write() -> dict[str, Any]:
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
                    _json({} if metadata is None else metadata),
                ),
            )
            return self.get_stage_run(stage_run_id)  # type: ignore[return-value]

        return self._write(write)

    def get_stage_run(self, stage_run_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM stage_runs WHERE stage_run_id = ?", (stage_run_id,))
        return None if row is None else _decode_row(row, ("metadata_json",))

    def list_stage_runs(self, project_id: str) -> list[dict[str, Any]]:
        """Return stage-run records for a project in most-recent-first order.

        Ordering belongs in the state boundary because ``started_at`` and the
        stage-run identity are ledger fields.  Workflow policy still decides
        whether the returned stage is a valid transition.
        """

        rows = self._fetchall(
            "SELECT * FROM stage_runs "
            "WHERE project_id = ? "
            "ORDER BY COALESCE(started_at, completed_at, '') DESC, stage_run_id DESC",
            (project_id,),
        )
        return [_decode_row(row, ("metadata_json",)) for row in rows]

    def get_latest_stage_run(self, project_id: str) -> dict[str, Any] | None:
        """Return the current-most stage-run record for a project."""

        rows = self.list_stage_runs(project_id)
        return rows[0] if rows else None

    def record_stage_run(self, stage_run_id: str, project_id: str, stage: str, **kwargs: Any) -> dict[str, Any]:
        """Record one stage run by its stable run ID."""

        return self.upsert_stage_run(stage_run_id, project_id, stage, **kwargs)

    def upsert_structure(
        self,
        identity_or_record: StructureIdentity | GeneratedStructureRecord,
        *,
        provenance: StructureProvenance | None = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        if isinstance(identity_or_record, GeneratedStructureRecord):
            identity = identity_or_record.identity
            provenance = identity_or_record.provenance
            metadata = identity_or_record.metadata
        else:
            identity = identity_or_record
        now = _now()

        def write() -> dict[str, Any]:
            existing = self._connection.execute(
                "SELECT identity_schema FROM structures WHERE structure_id = ?",
                (identity.structure_id,),
            ).fetchone()
            if existing is not None and existing["identity_schema"] != identity.schema_version:
                raise StateError(f"Structure identity schema conflict: {identity.structure_id}")
            self._connection.execute(
                """
                INSERT INTO structures (structure_id, identity_schema, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(structure_id) DO UPDATE SET
                    identity_schema = excluded.identity_schema,
                    metadata_json = excluded.metadata_json
                """,
                (
                    identity.structure_id,
                    identity.schema_version,
                    _json(metadata),
                    now,
                ),
            )
            if provenance is not None:
                existing_provenance = self._connection.execute(
                    "SELECT structure_id FROM structure_provenance WHERE operation_id = ?",
                    (provenance.operation_id,),
                ).fetchone()
                if (
                    existing_provenance is not None
                    and existing_provenance["structure_id"] != identity.structure_id
                ):
                    raise StateError(
                        f"Structure provenance identity conflict: {provenance.operation_id}"
                    )
                self._connection.execute(
                    """
                    INSERT INTO structure_provenance
                        (operation_id, structure_id, parent_structure_id, generator,
                         requested_composition_json, realised_composition_json,
                         source_database_id, crystal_structure, perturbation_family,
                         perturbation_parameters_json, random_seed, code_version,
                         config_fingerprint, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(operation_id) DO UPDATE SET
                        structure_id = excluded.structure_id,
                        parent_structure_id = excluded.parent_structure_id,
                        generator = excluded.generator,
                        requested_composition_json = excluded.requested_composition_json,
                        realised_composition_json = excluded.realised_composition_json,
                        source_database_id = excluded.source_database_id,
                        crystal_structure = excluded.crystal_structure,
                        perturbation_family = excluded.perturbation_family,
                        perturbation_parameters_json = excluded.perturbation_parameters_json,
                        random_seed = excluded.random_seed,
                        code_version = excluded.code_version,
                        config_fingerprint = excluded.config_fingerprint
                    """,
                    (
                        provenance.operation_id,
                        identity.structure_id,
                        provenance.parent_structure_id,
                        provenance.generator,
                        _json(provenance.requested_composition),
                        _json(provenance.realised_composition),
                        provenance.source_database_id,
                        provenance.crystal_structure,
                        provenance.perturbation_family,
                        _json(provenance.perturbation_parameters),
                        provenance.random_seed,
                        provenance.code_version,
                        provenance.config_fingerprint,
                        now,
                    ),
                )
            return self.get_structure(identity.structure_id)  # type: ignore[return-value]

        return self._write(write)

    def get_structure(self, structure_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM structures WHERE structure_id = ?", (structure_id,))
        if row is None:
            return None
        result = _decode_row(row, ("metadata_json",))
        provenance_row = self._fetchone(
            "SELECT * FROM structure_provenance WHERE structure_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (structure_id,),
        )
        if provenance_row is not None:
            result["provenance"] = _decode_row(
                provenance_row,
                (
                    "requested_composition_json",
                    "realised_composition_json",
                    "perturbation_parameters_json",
                ),
            )
        return result

    def record_structure(self, identity_or_record: StructureIdentity | GeneratedStructureRecord, **kwargs: Any) -> dict[str, Any]:
        """Record a structure identity and optional provenance."""

        return self.upsert_structure(identity_or_record, **kwargs)

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
        now = _now()

        def write() -> dict[str, Any]:
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
                    _json({} if parameters is None else parameters),
                    started_at,
                    completed_at,
                    now,
                ),
            )
            return self.get_selection_run(selection_run_id)  # type: ignore[return-value]

        return self._write(write)

    def get_selection_run(self, selection_run_id: str) -> dict[str, Any] | None:
        row = self._fetchone(
            "SELECT * FROM selection_runs WHERE selection_run_id = ?",
            (selection_run_id,),
        )
        return None if row is None else _decode_row(row, ("parameters_json",))

    def upsert_dft_calculation(
        self,
        identity: DftCalculationIdentity | Mapping[str, Any],
        *,
        status: str = "pending",
        selected: bool = False,
        priority: int = 0,
        accepted_attempt_id: str | None = None,
        reused_from_calculation_id: str | None = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        payload = _identity_payload(identity)
        calculation_id = payload.get("calculation_id")
        structure_id = payload.get("structure_id")
        if not isinstance(calculation_id, str) or not calculation_id:
            raise StateError("DFT calculation identity is missing calculation_id")
        if not isinstance(structure_id, str) or not structure_id:
            raise StateError("DFT calculation identity is missing structure_id")
        now = _now()

        def write() -> dict[str, Any]:
            existing = self._connection.execute(
                "SELECT identity_json FROM dft_calculations WHERE calculation_id = ?",
                (calculation_id,),
            ).fetchone()
            if existing is not None and _json(_decode_json(existing["identity_json"], "identity_json")) != _json(payload):
                raise StateError(f"DFT calculation identity conflict: {calculation_id}")
            self._connection.execute(
                """
                INSERT INTO dft_calculations
                    (calculation_id, structure_id, identity_json, status, selected,
                     priority, accepted_attempt_id, reused_from_calculation_id,
                     metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(calculation_id) DO UPDATE SET
                    status = CASE
                        WHEN dft_calculations.accepted_attempt_id IS NOT NULL
                        THEN dft_calculations.status
                        ELSE excluded.status
                    END,
                    selected = excluded.selected,
                    priority = excluded.priority,
                    accepted_attempt_id = COALESCE(
                        dft_calculations.accepted_attempt_id,
                        excluded.accepted_attempt_id
                    ),
                    reused_from_calculation_id = excluded.reused_from_calculation_id,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    calculation_id,
                    structure_id,
                    _json(payload),
                    status,
                    int(selected),
                    priority,
                    accepted_attempt_id,
                    reused_from_calculation_id,
                    _json({} if metadata is None else metadata),
                    now,
                    now,
                ),
            )
            return self.get_dft_calculation(calculation_id)  # type: ignore[return-value]

        return self._write(write)

    def get_dft_calculation(self, calculation_id: str) -> dict[str, Any] | None:
        row = self._fetchone(
            "SELECT * FROM dft_calculations WHERE calculation_id = ?",
            (calculation_id,),
        )
        return (
            None
            if row is None
            else _decode_row(row, ("identity_json", "metadata_json"))
        )

    def list_dft_calculations(
        self,
        *,
        statuses: Sequence[str] | None = None,
        selected_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List authoritative DFT calculations for reconciliation loaders."""
        query = "SELECT * FROM dft_calculations"
        parameters: list[Any] = []
        clauses: list[str] = []
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(statuses)
        if selected_only:
            clauses.append("selected = 1")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, calculation_id"
        return [
            _decode_row(row, ("identity_json", "metadata_json"))
            for row in self._fetchall(query, parameters)
        ]

    def record_dft_calculation(
        self,
        identity: DftCalculationIdentity | Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Record a scientific DFT calculation identity."""

        return self.upsert_dft_calculation(identity, **kwargs)

    def create_dft_attempt(
        self,
        calculation_id: str,
        attempt_id: str,
        *,
        attempt_number: int,
        status: str = "pending",
        resources: Any = None,
        recovery: Any = None,
        scheduler: Any = None,
        failure_evidence: Any = None,
        job_id: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        if attempt_number < 1:
            raise StateError("DFT attempt_number must be positive")
        now = _now()

        def write() -> dict[str, Any]:
            existing = self._connection.execute(
                "SELECT * FROM dft_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["calculation_id"] != calculation_id
                    or existing["attempt_number"] != attempt_number
                ):
                    raise StateError(f"DFT attempt identity conflict: {attempt_id}")
                return _decode_row(
                    existing,
                    (
                        "resources_json",
                        "recovery_json",
                        "scheduler_json",
                        "failure_evidence_json",
                        "metadata_json",
                    ),
                )
            duplicate_number = self._connection.execute(
                "SELECT attempt_id FROM dft_attempts "
                "WHERE calculation_id = ? AND attempt_number = ?",
                (calculation_id, attempt_number),
            ).fetchone()
            if duplicate_number is not None:
                raise StateError(
                    "DFT attempt number already exists for calculation: "
                    f"{calculation_id}/{attempt_number}"
                )
            self._connection.execute(
                """
                INSERT INTO dft_attempts
                    (attempt_id, calculation_id, attempt_number, status, resources_json,
                     recovery_json, scheduler_json, failure_evidence_json, job_id,
                     started_at, completed_at, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    calculation_id,
                    attempt_number,
                    status,
                    _json({} if resources is None else resources),
                    _json({} if recovery is None else recovery),
                    _json({} if scheduler is None else scheduler),
                    _json({} if failure_evidence is None else failure_evidence),
                    job_id,
                    started_at,
                    completed_at,
                    _json({} if metadata is None else metadata),
                    now,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM dft_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            return _decode_row(
                row,
                (
                    "resources_json",
                    "recovery_json",
                    "scheduler_json",
                    "failure_evidence_json",
                    "metadata_json",
                ),
            )

        return self._write(write)

    def get_dft_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM dft_attempts WHERE attempt_id = ?", (attempt_id,))
        if row is None:
            return None
        return _decode_row(
            row,
            (
                "resources_json",
                "recovery_json",
                "scheduler_json",
                "failure_evidence_json",
                "metadata_json",
            ),
        )

    def list_dft_attempts(self, calculation_id: str) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT * FROM dft_attempts WHERE calculation_id = ? ORDER BY attempt_number",
            (calculation_id,),
        )
        return [
            _decode_row(
                row,
                (
                    "resources_json",
                    "recovery_json",
                    "scheduler_json",
                    "failure_evidence_json",
                    "metadata_json",
                ),
            )
            for row in rows
        ]

    def update_dft_attempt(
        self,
        attempt_id: str,
        *,
        status: str | None = None,
        resources: Any = None,
        scheduler: Any = None,
        failure_evidence: Any = None,
        job_id: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        """Update mutable execution fields without replacing attempt history."""

        def write() -> dict[str, Any]:
            existing = self._connection.execute(
                "SELECT * FROM dft_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if existing is None:
                raise StateError(f"Unknown DFT attempt: {attempt_id}")
            self._connection.execute(
                """
                UPDATE dft_attempts SET
                    status = COALESCE(?, status),
                    resources_json = COALESCE(?, resources_json),
                    scheduler_json = COALESCE(?, scheduler_json),
                    failure_evidence_json = COALESCE(?, failure_evidence_json),
                    job_id = COALESCE(?, job_id),
                    started_at = COALESCE(?, started_at),
                    completed_at = COALESCE(?, completed_at),
                    metadata_json = COALESCE(?, metadata_json)
                WHERE attempt_id = ?
                """,
                (
                    status,
                    None if resources is None else _json(resources),
                    None if scheduler is None else _json(scheduler),
                    None if failure_evidence is None else _json(failure_evidence),
                    job_id,
                    started_at,
                    completed_at,
                    None if metadata is None else _json(metadata),
                    attempt_id,
                ),
            )
            return self.get_dft_attempt(attempt_id)  # type: ignore[return-value]

        return self._write(write)

    def record_artifact(
        self,
        artifact: ArtifactIdentity,
        *,
        originating_attempt_id: str | None = None,
        retention_status: str = "active",
        metadata: Any = None,
    ) -> dict[str, Any]:
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
            return _decode_row(existing, ("metadata_json",))
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
                    _json({} if metadata is None else metadata),
                    _now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise StateError(
                f"Artifact content is already registered under another identity: {artifact.sha256}"
            ) from exc
        row = self._connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact.artifact_id,)
        ).fetchone()
        return _decode_row(row, ("metadata_json",))

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        return None if row is None else _decode_row(row, ("metadata_json",))

    def list_artifacts(self, *, originating_attempt_id: str | None = None) -> list[dict[str, Any]]:
        if originating_attempt_id is None:
            rows = self._fetchall("SELECT * FROM artifacts ORDER BY created_at")
        else:
            rows = self._fetchall(
                "SELECT * FROM artifacts WHERE originating_attempt_id = ? ORDER BY created_at",
                (originating_attempt_id,),
            )
        return [_decode_row(row, ("metadata_json",)) for row in rows]

    def register_completed_result(
        self,
        calculation_id: str,
        attempt_id: str,
        artifacts: Iterable[ArtifactIdentity],
    ) -> dict[str, Any]:
        """Atomically accept one attempt and register all result artifacts."""

        artifact_list = tuple(artifacts)

        def write() -> dict[str, Any]:
            calculation = self._connection.execute(
                "SELECT * FROM dft_calculations WHERE calculation_id = ?",
                (calculation_id,),
            ).fetchone()
            if calculation is None:
                raise StateError(f"Unknown DFT calculation: {calculation_id}")
            attempt = self._connection.execute(
                "SELECT * FROM dft_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if attempt is None or attempt["calculation_id"] != calculation_id:
                raise StateError(f"Unknown DFT attempt for calculation: {attempt_id}")
            accepted = calculation["accepted_attempt_id"]
            if accepted is not None and accepted != attempt_id:
                raise StateError(
                    "A different DFT attempt is already accepted for calculation "
                    f"{calculation_id}: {accepted}"
                )
            if accepted == attempt_id:
                return _decode_row(calculation, ("identity_json", "metadata_json"))
            for artifact in artifact_list:
                self._record_artifact(
                    artifact,
                    originating_attempt_id=attempt_id,
                    retention_status="active",
                    metadata=None,
                )
            completed_at = _now()
            self._connection.execute(
                "UPDATE dft_attempts SET status = 'completed', completed_at = ? "
                "WHERE attempt_id = ?",
                (completed_at, attempt_id),
            )
            self._connection.execute(
                "UPDATE dft_calculations SET status = 'completed', accepted_attempt_id = ?, "
                "updated_at = ? WHERE calculation_id = ?",
                (attempt_id, completed_at, calculation_id),
            )
            row = self._connection.execute(
                "SELECT * FROM dft_calculations WHERE calculation_id = ?",
                (calculation_id,),
            ).fetchone()
            return _decode_row(row, ("identity_json", "metadata_json"))

        return self._write(write)

    def save_execution(
        self,
        record: Any,
        *,
        artifact: Any = None,
        reason: str | None = None,
    ) -> None:
        """Persist one canonical DFT execution transition.

        The execution reconciler supplies an immutable record.  This adapter
        keeps attempt history append-only for retries and uses the existing
        result-registration transaction when a backend artifact is accepted.
        It intentionally does not write legacy marker or registry files.
        """
        calculation = record.inputs.calculation
        existing_attempt = self.get_dft_attempt(record.attempt_id)
        existing_calculation = self.get_dft_calculation(calculation.calculation_id)
        metadata = (
            dict(existing_calculation.get("metadata", {}))
            if existing_calculation is not None
            and isinstance(existing_calculation.get("metadata"), Mapping)
            else {}
        )
        metadata["working_directory"] = str(record.inputs.working_directory)
        metadata["execution_status"] = record.status
        if reason is not None:
            metadata["last_failure_reason"] = reason

        with self.transaction():
            self.upsert_dft_calculation(
                calculation,
                status=record.status,
                selected=(
                    bool(existing_calculation.get("selected", False))
                    if existing_calculation is not None
                    else True
                ),
                priority=(
                    int(existing_calculation.get("priority", 0))
                    if existing_calculation is not None
                    else 0
                ),
                metadata=metadata,
            )
            if existing_attempt is None:
                attempts = self.list_dft_attempts(calculation.calculation_id)
                attempt_number = (
                    max((int(item["attempt_number"]) for item in attempts), default=0)
                    + 1
                )
                self.create_dft_attempt(
                    calculation.calculation_id,
                    record.attempt_id,
                    attempt_number=attempt_number,
                    status=record.status,
                    resources=(
                        None
                        if record.resources is None
                        else asdict(record.resources)
                    ),
                    job_id=record.job_id,
                    failure_evidence=(
                        {"reason": reason} if reason is not None else None
                    ),
                    metadata=metadata,
                )
            else:
                self.update_dft_attempt(
                    record.attempt_id,
                    status=record.status,
                    resources=(
                        None
                        if record.resources is None
                        else asdict(record.resources)
                    ),
                    job_id=record.job_id,
                    failure_evidence=(
                        {"reason": reason} if reason is not None else None
                    ),
                    completed_at=_now() if record.status in {"completed", "failed"} else None,
                    metadata=metadata,
                )

            if artifact is not None:
                if not hasattr(artifact, "outcar"):
                    raise StateError("DFT execution artifact has no VASP artifact fields")
                identities = tuple(
                    item
                    for item in (artifact.outcar, artifact.vasprun)
                    if item is not None
                )
                self.register_completed_result(
                    calculation.calculation_id,
                    record.attempt_id,
                    identities,
                )

    def upsert_dataset(
        self,
        identity_or_manifest: DatasetIdentity | TrainingDatasetManifest,
        *,
        project_id: str | None = None,
        status: str = "prepared",
    ) -> dict[str, Any]:
        if isinstance(identity_or_manifest, TrainingDatasetManifest):
            identity = identity_or_manifest.identity
            manifest = identity_or_manifest.to_dict()
        else:
            identity = identity_or_manifest
            manifest = identity.to_dict()
        now = _now()

        def write() -> dict[str, Any]:
            existing = self._connection.execute(
                "SELECT identity_json FROM datasets WHERE dataset_id = ?",
                (identity.dataset_id,),
            ).fetchone()
            if existing is not None and _json(_decode_json(existing["identity_json"], "identity_json")) != _json(identity.to_dict()):
                raise StateError(f"Dataset identity conflict: {identity.dataset_id}")
            self._connection.execute(
                """
                INSERT INTO datasets
                    (dataset_id, project_id, identity_json, status, manifest_json,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    project_id = excluded.project_id,
                    status = excluded.status,
                    manifest_json = excluded.manifest_json,
                    updated_at = excluded.updated_at
                """,
                (
                    identity.dataset_id,
                    project_id,
                    _json(identity.to_dict()),
                    status,
                    _json(manifest),
                    now,
                    now,
                ),
            )
            return self.get_dataset(identity.dataset_id)  # type: ignore[return-value]

        return self._write(write)

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
        return None if row is None else _decode_row(row, ("identity_json", "manifest_json"))

    def record_dataset(
        self,
        identity_or_manifest: DatasetIdentity | TrainingDatasetManifest,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Record a dataset identity or complete training manifest."""

        return self.upsert_dataset(identity_or_manifest, **kwargs)

    def record_dataset_member(
        self,
        dataset_id: str,
        member: SelectedDatasetMember,
        *,
        metadata: Any = None,
    ) -> dict[str, Any]:
        def write() -> dict[str, Any]:
            existing = self._connection.execute(
                "SELECT structure_id, calculation_id FROM dataset_members "
                "WHERE dataset_id = ? AND ordinal = ?",
                (dataset_id, member.ordinal),
            ).fetchone()
            if existing is not None and (
                existing["structure_id"] != member.structure_id
                or existing["calculation_id"] != member.calculation_id
            ):
                raise StateError(
                    f"Dataset member identity conflict: {dataset_id}/{member.ordinal}"
                )
            self._connection.execute(
                """
                INSERT INTO dataset_members
                    (dataset_id, ordinal, split, structure_id, calculation_id,
                     source_outcar_hash, calculation_identity_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id, ordinal) DO UPDATE SET
                    split = excluded.split,
                    structure_id = excluded.structure_id,
                    calculation_id = excluded.calculation_id,
                    source_outcar_hash = excluded.source_outcar_hash,
                    calculation_identity_json = excluded.calculation_identity_json,
                    metadata_json = excluded.metadata_json
                """,
                (
                    dataset_id,
                    member.ordinal,
                    member.split,
                    member.structure_id,
                    member.calculation_id,
                    member.source_outcar_hash,
                    _json(dict(member.calculation_identity)),
                    _json({} if metadata is None else metadata),
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM dataset_members WHERE dataset_id = ? AND ordinal = ?",
                (dataset_id, member.ordinal),
            ).fetchone()
            return _decode_row(row, ("calculation_identity_json", "metadata_json"))

        return self._write(write)

    def list_dataset_members(self, dataset_id: str) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT * FROM dataset_members WHERE dataset_id = ? ORDER BY ordinal",
            (dataset_id,),
        )
        return [
            _decode_row(row, ("calculation_identity_json", "metadata_json"))
            for row in rows
        ]

    def upsert_model_run(
        self,
        record: ModelRunRecord,
        *,
        status: str = "pending",
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        identity = record.identity
        execution_metadata = record.execution_metadata
        now = _now()

        def write() -> dict[str, Any]:
            existing = self._connection.execute(
                "SELECT identity_json FROM model_runs WHERE model_run_id = ?",
                (identity.model_run_id,),
            ).fetchone()
            if existing is not None and _json(_decode_json(existing["identity_json"], "identity_json")) != _json(identity.to_dict()):
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
                    _json(identity.to_dict()),
                    status,
                    _json({} if execution_metadata is None else execution_metadata),
                    started_at,
                    completed_at,
                    now,
                    now,
                ),
            )
            if record.artifact is not None:
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
                        (identity.model_run_id, artifact.artifact_id, role, _json({})),
                    )
            return self.get_model_run(identity.model_run_id)  # type: ignore[return-value]

        return self._write(write)

    def get_model_run(self, model_run_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM model_runs WHERE model_run_id = ?", (model_run_id,))
        return (
            None
            if row is None
            else _decode_row(row, ("identity_json", "execution_metadata_json"))
        )

    def record_model_run(self, record: ModelRunRecord, **kwargs: Any) -> dict[str, Any]:
        """Record one model-run identity and its artifact links."""

        return self.upsert_model_run(record, **kwargs)

    def upsert_validation_run(
        self,
        record: ValidationRunRecord,
        *,
        status: str = "pending",
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        identity = record.identity
        passed = None
        if record.artifact is not None:
            passed = record.artifact.passed
        now = _now()

        def write() -> dict[str, Any]:
            existing = self._connection.execute(
                "SELECT identity_json FROM validation_runs WHERE validation_run_id = ?",
                (identity.validation_run_id,),
            ).fetchone()
            if existing is not None and _json(_decode_json(existing["identity_json"], "identity_json")) != _json(identity.to_dict()):
                raise StateError(
                    f"Validation-run identity conflict: {identity.validation_run_id}"
                )
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
                    _json(identity.to_dict()),
                    status,
                    None if passed is None else int(passed),
                    _json({} if record.metadata is None else record.metadata),
                    started_at,
                    completed_at,
                    now,
                    now,
                ),
            )
            if record.artifact is not None:
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
                        (identity.validation_run_id, artifact.artifact_id, role, _json({})),
                    )
            return self.get_validation_run(identity.validation_run_id)  # type: ignore[return-value]

        return self._write(write)

    def get_validation_run(self, validation_run_id: str) -> dict[str, Any] | None:
        row = self._fetchone(
            "SELECT * FROM validation_runs WHERE validation_run_id = ?",
            (validation_run_id,),
        )
        return None if row is None else _decode_row(row, ("identity_json", "metadata_json"))

    def record_validation_run(self, record: ValidationRunRecord, **kwargs: Any) -> dict[str, Any]:
        """Record one validation-run identity and its artifact links."""

        return self.upsert_validation_run(record, **kwargs)

    def record_validation_result(
        self,
        validation_run_id: str,
        result_id: str,
        *,
        structure_id: str | None = None,
        metric_name: str | None = None,
        observed_value: float | None = None,
        threshold: float | None = None,
        passed: bool | None = None,
        failure_reason: str | None = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        def write() -> dict[str, Any]:
            self._connection.execute(
                """
                INSERT INTO validation_results
                    (validation_run_id, result_id, structure_id, metric_name,
                     observed_value, threshold, passed, failure_reason, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(validation_run_id, result_id) DO UPDATE SET
                    structure_id = excluded.structure_id,
                    metric_name = excluded.metric_name,
                    observed_value = excluded.observed_value,
                    threshold = excluded.threshold,
                    passed = excluded.passed,
                    failure_reason = excluded.failure_reason,
                    metadata_json = excluded.metadata_json
                """,
                (
                    validation_run_id,
                    result_id,
                    structure_id,
                    metric_name,
                    observed_value,
                    threshold,
                    None if passed is None else int(passed),
                    failure_reason,
                    _json({} if metadata is None else metadata),
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM validation_results WHERE validation_run_id = ? AND result_id = ?",
                (validation_run_id, result_id),
            ).fetchone()
            return _decode_row(row, ("metadata_json",))

        return self._write(write)

    def list_validation_results(self, validation_run_id: str) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT * FROM validation_results WHERE validation_run_id = ? ORDER BY result_id",
            (validation_run_id,),
        )
        return [_decode_row(row, ("metadata_json",)) for row in rows]

    def append_event(
        self,
        event_id: str,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: Any = None,
        *,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        def write() -> dict[str, Any]:
            existing = self._connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            encoded_payload = _json({} if payload is None else payload)
            if existing is not None:
                if (
                    existing["entity_type"] != entity_type
                    or existing["entity_id"] != entity_id
                    or existing["event_type"] != event_type
                    or existing["payload_json"] != encoded_payload
                ):
                    raise StateError(f"Event identity conflict: {event_id}")
                return _decode_row(existing, ("payload_json",))
            self._connection.execute(
                """
                INSERT INTO events
                    (event_id, entity_type, entity_id, event_type, payload_json, occurred_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    entity_type,
                    entity_id,
                    event_type,
                    encoded_payload,
                    occurred_at or _now(),
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            return _decode_row(row, ("payload_json",))

        return self._write(write)

    def list_events(self, *, entity_type: str | None = None, entity_id: str | None = None) -> list[dict[str, Any]]:
        if entity_type is None and entity_id is None:
            rows = self._fetchall("SELECT * FROM events ORDER BY occurred_at, event_id")
        elif entity_type is not None and entity_id is not None:
            rows = self._fetchall(
                "SELECT * FROM events WHERE entity_type = ? AND entity_id = ? "
                "ORDER BY occurred_at, event_id",
                (entity_type, entity_id),
            )
        else:
            raise ValueError("entity_type and entity_id must be provided together")
        return [_decode_row(row, ("payload_json",)) for row in rows]


__all__ = ["StateStore", "CURRENT_SCHEMA_VERSION"]
