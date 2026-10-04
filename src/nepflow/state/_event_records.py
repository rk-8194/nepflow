"""Persistence operations for validation results and append-only events."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from nepflow.errors import StateError

from ._record_codec import decode_row, encode_json, now


class EventRecordsMixin:
    """Provide StateStore persistence for result and event ledger rows."""

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
        """Insert or update one validation result."""

        return self._write(
            lambda: self._write_validation_result(
                validation_run_id,
                result_id,
                structure_id,
                metric_name,
                observed_value,
                threshold,
                passed,
                failure_reason,
                metadata,
            )
        )

    def _write_validation_result(
        self,
        validation_run_id: str,
        result_id: str,
        structure_id: str | None,
        metric_name: str | None,
        observed_value: float | None,
        threshold: float | None,
        passed: bool | None,
        failure_reason: str | None,
        metadata: Any,
    ) -> dict[str, Any]:
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
                encode_json({} if metadata is None else metadata),
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM validation_results WHERE validation_run_id = ? AND result_id = ?",
            (validation_run_id, result_id),
        ).fetchone()
        return decode_row(row, ("metadata_json",))

    def list_validation_results(self, validation_run_id: str) -> list[dict[str, Any]]:
        """Return validation results in deterministic result-id order."""

        rows = self._fetchall(
            "SELECT * FROM validation_results WHERE validation_run_id = ? ORDER BY result_id",
            (validation_run_id,),
        )
        return [decode_row(row, ("metadata_json",)) for row in rows]

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
        """Append one immutable ledger event by deterministic event identity."""

        return self._write(
            lambda: self._write_event(
                event_id,
                entity_type,
                entity_id,
                event_type,
                payload,
                occurred_at,
            )
        )

    def _write_event(
        self,
        event_id: str,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: Any,
        occurred_at: str | None,
    ) -> dict[str, Any]:
        existing = self._connection.execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        encoded_payload = encode_json({} if payload is None else payload)
        if existing is not None:
            if (
                existing["entity_type"] != entity_type
                or existing["entity_id"] != entity_id
                or existing["event_type"] != event_type
                or existing["payload_json"] != encoded_payload
            ):
                raise StateError(f"Event identity conflict: {event_id}")
            return decode_row(existing, ("payload_json",))
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
                occurred_at or now(),
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return decode_row(row, ("payload_json",))

    def list_events(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List events using the supported entity filters and ledger ordering."""

        query, parameters = _event_query(entity_type, entity_id)
        return [
            decode_row(row, ("payload_json",))
            for row in self._fetchall(query, parameters)
        ]

    def record_training_event(
        self,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: Any = None,
        *,
        event_id: str | None = None,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        """Append one training-campaign transition to the ledger."""

        if not entity_type.startswith("training_"):
            raise ValueError("training events must use a training_ entity type")
        return self.append_event(
            event_id or f"training:{entity_type}:{entity_id}:{uuid4().hex}",
            entity_type,
            entity_id,
            event_type,
            payload,
            occurred_at=occurred_at,
        )

    def list_training_events(
        self,
        entity_type: str,
        entity_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return append-only training transitions in ledger order."""

        if not entity_type.startswith("training_"):
            raise ValueError("training events must use a training_ entity type")
        return self.list_events(entity_type=entity_type, entity_id=entity_id)

    def record_validation_event(
        self,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: Any = None,
        *,
        event_id: str | None = None,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        """Append one validation reconciliation transition to the ledger."""

        if not entity_type.startswith("validation_"):
            raise ValueError("validation events must use a validation_ entity type")
        return self.append_event(
            event_id or f"validation:{entity_type}:{entity_id}:{uuid4().hex}",
            entity_type,
            entity_id,
            event_type,
            payload,
            occurred_at=occurred_at,
        )

    def list_validation_events(
        self,
        entity_type: str,
        entity_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return append-only validation transitions in ledger order."""

        if not entity_type.startswith("validation_"):
            raise ValueError("validation events must use a validation_ entity type")
        return self.list_events(entity_type=entity_type, entity_id=entity_id)


def _event_query(
    entity_type: str | None,
    entity_id: str | None,
) -> tuple[str, tuple[str, ...]]:
    order = " ORDER BY occurred_at, event_id"
    if entity_type is None and entity_id is None:
        return "SELECT * FROM events" + order, ()
    if entity_type is not None and entity_id is not None:
        return (
            "SELECT * FROM events WHERE entity_type = ? AND entity_id = ?" + order,
            (entity_type, entity_id),
        )
    if entity_type is not None:
        return (
            "SELECT * FROM events WHERE entity_type = ?" + order,
            (entity_type,),
        )
    return (
        "SELECT * FROM events WHERE entity_id = ?" + order,
        (entity_id,),
    )
