"""Persistence operations for DFT identities, attempts, and completion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from nepflow.domain.identities import ArtifactIdentity, DftCalculationIdentity
from nepflow.errors import StateError

from ._record_codec import decode_json, decode_row, encode_json, identity_payload, now
from ._typing import StateStoreMixinSupport, require_state_row


class DftRecordsMixin(StateStoreMixinSupport):
    """Provide StateStore persistence for DFT execution records."""

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
        """Insert or update one scientific DFT calculation identity."""

        payload = identity_payload(identity)
        calculation_id, structure_id = _calculation_keys(payload)
        return self._write(
            lambda: self._write_dft_calculation(
                payload,
                calculation_id,
                structure_id,
                status,
                selected,
                priority,
                accepted_attempt_id,
                reused_from_calculation_id,
                metadata,
                now(),
            )
        )

    def _write_dft_calculation(
        self,
        payload: dict[str, Any],
        calculation_id: str,
        structure_id: str,
        status: str,
        selected: bool,
        priority: int,
        accepted_attempt_id: str | None,
        reused_from_calculation_id: str | None,
        metadata: Any,
        timestamp: str,
    ) -> dict[str, Any]:
        existing = self._connection.execute(
            "SELECT identity_json FROM dft_calculations WHERE calculation_id = ?",
            (calculation_id,),
        ).fetchone()
        if existing is not None and encode_json(
            decode_json(existing["identity_json"], "identity_json")
        ) != encode_json(payload):
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
                encode_json(payload),
                status,
                int(selected),
                priority,
                accepted_attempt_id,
                reused_from_calculation_id,
                encode_json({} if metadata is None else metadata),
                timestamp,
                timestamp,
            ),
        )
        return require_state_row(self.get_dft_calculation(calculation_id), "DFT calculation")

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
        """Create one immutable DFT attempt, rejecting identity collisions."""

        if attempt_number < 1:
            raise StateError("DFT attempt_number must be positive")
        return self._write(
            lambda: self._write_dft_attempt(
                calculation_id,
                attempt_id,
                attempt_number,
                status,
                resources,
                recovery,
                scheduler,
                failure_evidence,
                job_id,
                started_at,
                completed_at,
                metadata,
                now(),
            )
        )

    def _write_dft_attempt(
        self,
        calculation_id: str,
        attempt_id: str,
        attempt_number: int,
        status: str,
        resources: Any,
        recovery: Any,
        scheduler: Any,
        failure_evidence: Any,
        job_id: str | None,
        started_at: str | None,
        completed_at: str | None,
        metadata: Any,
        timestamp: str,
    ) -> dict[str, Any]:
        existing = self._connection.execute(
            "SELECT * FROM dft_attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        if existing is not None:
            if (
                existing["calculation_id"] != calculation_id
                or existing["attempt_number"] != attempt_number
            ):
                raise StateError(f"DFT attempt identity conflict: {attempt_id}")
            return _decode_attempt(existing)
        duplicate_number = self._connection.execute(
            "SELECT attempt_id FROM dft_attempts WHERE calculation_id = ? AND attempt_number = ?",
            (calculation_id, attempt_number),
        ).fetchone()
        if duplicate_number is not None:
            raise StateError(
                "DFT attempt number already exists for calculation: "
                f"{calculation_id}/{attempt_number}"
            )
        self._insert_dft_attempt(
            calculation_id,
            attempt_id,
            attempt_number,
            status,
            resources,
            recovery,
            scheduler,
            failure_evidence,
            job_id,
            started_at,
            completed_at,
            metadata,
            timestamp,
        )
        row = self._connection.execute(
            "SELECT * FROM dft_attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        return _decode_attempt(row)

    def _insert_dft_attempt(
        self,
        calculation_id: str,
        attempt_id: str,
        attempt_number: int,
        status: str,
        resources: Any,
        recovery: Any,
        scheduler: Any,
        failure_evidence: Any,
        job_id: str | None,
        started_at: str | None,
        completed_at: str | None,
        metadata: Any,
        timestamp: str,
    ) -> None:
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
                encode_json({} if resources is None else resources),
                encode_json({} if recovery is None else recovery),
                encode_json({} if scheduler is None else scheduler),
                encode_json({} if failure_evidence is None else failure_evidence),
                job_id,
                started_at,
                completed_at,
                encode_json({} if metadata is None else metadata),
                timestamp,
            ),
        )

    def register_completed_result(
        self,
        calculation_id: str,
        attempt_id: str,
        artifacts: Iterable[ArtifactIdentity],
    ) -> dict[str, Any]:
        """Atomically accept one attempt and register all result artifacts."""

        return self._write(
            lambda: self._complete_dft_result(calculation_id, attempt_id, tuple(artifacts))
        )

    def _complete_dft_result(
        self,
        calculation_id: str,
        attempt_id: str,
        artifacts: tuple[Any, ...],
    ) -> dict[str, Any]:
        calculation = self._connection.execute(
            "SELECT * FROM dft_calculations WHERE calculation_id = ?",
            (calculation_id,),
        ).fetchone()
        if calculation is None:
            raise StateError(f"Unknown DFT calculation: {calculation_id}")
        attempt = self._connection.execute(
            "SELECT * FROM dft_attempts WHERE attempt_id = ?", (attempt_id,)
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
            return decode_row(calculation, ("identity_json", "metadata_json"))
        for artifact in artifacts:
            self._record_artifact(
                artifact,
                originating_attempt_id=attempt_id,
                retention_status="active",
                metadata=None,
            )
        completed_at = now()
        self._connection.execute(
            "UPDATE dft_attempts SET status = 'completed', completed_at = ? WHERE attempt_id = ?",
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
        return decode_row(row, ("identity_json", "metadata_json"))

    def save_execution(
        self,
        record: Any,
        *,
        artifact: Any = None,
        reason: str | None = None,
    ) -> None:
        """Persist one canonical DFT execution transition."""

        calculation = record.inputs.calculation
        existing_attempt = self.get_dft_attempt(record.attempt_id)
        existing_calculation = self.get_dft_calculation(calculation.calculation_id)
        metadata = _execution_metadata(record, existing_calculation, reason)
        with self.transaction():
            self.upsert_dft_calculation(
                calculation,
                status=record.status,
                selected=_selected_value(existing_calculation),
                priority=_priority_value(existing_calculation),
                metadata=metadata,
            )
            self._save_execution_attempt(
                record, existing_attempt, metadata, reason, calculation.calculation_id
            )
            if artifact is not None:
                self._save_execution_artifact(
                    artifact, calculation.calculation_id, record.attempt_id
                )

    def _save_execution_attempt(
        self,
        record: Any,
        existing_attempt: dict[str, Any] | None,
        metadata: dict[str, Any],
        reason: str | None,
        calculation_id: str,
    ) -> None:
        resources = None if record.resources is None else asdict(record.resources)
        failure_evidence = {"reason": reason} if reason is not None else None
        if existing_attempt is None:
            attempts = self.list_dft_attempts(calculation_id)
            attempt_number = max((int(item["attempt_number"]) for item in attempts), default=0) + 1
            self.create_dft_attempt(
                calculation_id,
                record.attempt_id,
                attempt_number=attempt_number,
                status=record.status,
                resources=resources,
                job_id=record.job_id,
                failure_evidence=failure_evidence,
                metadata=metadata,
            )
            return
        self.update_dft_attempt(
            record.attempt_id,
            status=record.status,
            resources=resources,
            job_id=record.job_id,
            failure_evidence=failure_evidence,
            completed_at=now() if record.status in {"completed", "failed"} else None,
            metadata=metadata,
        )

    def _save_execution_artifact(
        self,
        artifact: Any,
        calculation_id: str,
        attempt_id: str,
    ) -> None:
        if not hasattr(artifact, "outcar"):
            raise StateError("DFT execution artifact has no VASP artifact fields")
        identities = tuple(item for item in (artifact.outcar, artifact.vasprun) if item is not None)
        self.register_completed_result(calculation_id, attempt_id, identities)

    def get_dft_calculation(self, calculation_id: str) -> dict[str, Any] | None:
        """Return one authoritative DFT calculation record."""

        row = self._fetchone(
            "SELECT * FROM dft_calculations WHERE calculation_id = ?",
            (calculation_id,),
        )
        return None if row is None else decode_row(row, ("identity_json", "metadata_json"))

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
            decode_row(row, ("identity_json", "metadata_json"))
            for row in self._fetchall(query, parameters)
        ]

    def record_dft_calculation(
        self,
        identity: DftCalculationIdentity | Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Record a scientific DFT calculation identity."""

        return self.upsert_dft_calculation(identity, **kwargs)

    def get_dft_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        """Return one DFT attempt with decoded JSON fields."""

        row = self._fetchone("SELECT * FROM dft_attempts WHERE attempt_id = ?", (attempt_id,))
        return None if row is None else _decode_attempt(row)

    def list_dft_attempts(self, calculation_id: str) -> list[dict[str, Any]]:
        """Return attempts for a calculation in attempt-number order."""

        rows = self._fetchall(
            "SELECT * FROM dft_attempts WHERE calculation_id = ? ORDER BY attempt_number",
            (calculation_id,),
        )
        return [_decode_attempt(row) for row in rows]

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

        return self._write(
            lambda: self._update_dft_attempt(
                attempt_id,
                status,
                resources,
                scheduler,
                failure_evidence,
                job_id,
                started_at,
                completed_at,
                metadata,
            )
        )

    def _update_dft_attempt(
        self,
        attempt_id: str,
        status: str | None,
        resources: Any,
        scheduler: Any,
        failure_evidence: Any,
        job_id: str | None,
        started_at: str | None,
        completed_at: str | None,
        metadata: Any,
    ) -> dict[str, Any]:
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
                None if resources is None else encode_json(resources),
                None if scheduler is None else encode_json(scheduler),
                None if failure_evidence is None else encode_json(failure_evidence),
                job_id,
                started_at,
                completed_at,
                None if metadata is None else encode_json(metadata),
                attempt_id,
            ),
        )
        return require_state_row(self.get_dft_attempt(attempt_id), "DFT attempt")


def _calculation_keys(payload: Mapping[str, Any]) -> tuple[str, str]:
    calculation_id = payload.get("calculation_id")
    structure_id = payload.get("structure_id")
    if not isinstance(calculation_id, str) or not calculation_id:
        raise StateError("DFT calculation identity is missing calculation_id")
    if not isinstance(structure_id, str) or not structure_id:
        raise StateError("DFT calculation identity is missing structure_id")
    return calculation_id, structure_id


def _decode_attempt(row: Any) -> dict[str, Any]:
    return decode_row(
        row,
        (
            "resources_json",
            "recovery_json",
            "scheduler_json",
            "failure_evidence_json",
            "metadata_json",
        ),
    )


def _execution_metadata(
    record: Any,
    existing_calculation: dict[str, Any] | None,
    reason: str | None,
) -> dict[str, Any]:
    metadata = (
        dict(existing_calculation.get("metadata", {}))
        if existing_calculation is not None
        and isinstance(existing_calculation.get("metadata"), Mapping)
        else {}
    )
    if isinstance(getattr(record, "metadata", None), Mapping):
        metadata.update(dict(record.metadata))
    metadata["working_directory"] = str(record.inputs.working_directory)
    metadata["execution_status"] = record.status
    if reason is not None:
        metadata["last_failure_reason"] = reason
    return metadata


def _selected_value(existing_calculation: dict[str, Any] | None) -> bool:
    return bool(existing_calculation.get("selected", False)) if existing_calculation else True


def _priority_value(existing_calculation: dict[str, Any] | None) -> int:
    return int(existing_calculation.get("priority", 0)) if existing_calculation else 0
