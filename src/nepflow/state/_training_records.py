"""Durable authority and projections for append-only training events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nepflow.errors import StateError

from ._record_codec import decode_json, decode_row, encode_json, now
from ._typing import StateStoreMixinSupport


class TrainingRecordsMixin(StateStoreMixinSupport):
    """Persist training execution generations without deleting history."""

    def ensure_training_execution_authority(
        self,
        campaign_id: str,
        dataset_id: str,
    ) -> dict[str, Any]:
        return self._write(
            lambda: self._ensure_training_execution_authority(campaign_id, dataset_id)
        )

    def _ensure_training_execution_authority(
        self,
        campaign_id: str,
        dataset_id: str,
    ) -> dict[str, Any]:
        current = self._connection.execute(
            "SELECT * FROM training_execution_authority "
            "WHERE campaign_id = ? AND status = 'current' "
            "ORDER BY execution_generation DESC LIMIT 1",
            (campaign_id,),
        ).fetchone()
        if current is not None:
            if current["dataset_id"] != dataset_id:
                raise StateError(
                    f"Training campaign {campaign_id} is owned by a different dataset"
                )
            return decode_row(current, ())
        dataset = self._connection.execute(
            "SELECT project_id FROM datasets WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        if dataset is None:
            raise StateError(f"Cannot create training authority for unknown dataset: {dataset_id}")
        latest = self._connection.execute(
            "SELECT COALESCE(MAX(execution_generation), 0) AS generation "
            "FROM training_execution_authority WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        generation = int(latest["generation"]) + 1
        timestamp = now()
        self._connection.execute(
            "INSERT INTO training_execution_authority "
            "(campaign_id, dataset_id, project_id, execution_generation, status, source, "
            "created_at, updated_at, invalidated_at) VALUES (?, ?, ?, ?, 'current', ?, ?, ?, NULL)",
            (
                campaign_id,
                dataset_id,
                dataset["project_id"],
                generation,
                "initial_training_execution" if generation == 1 else "training_rollback",
                timestamp,
                timestamp,
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM training_execution_authority "
            "WHERE campaign_id = ? AND execution_generation = ?",
            (campaign_id, generation),
        ).fetchone()
        if row is None:  # pragma: no cover - guarded by the insert
            raise StateError("Training execution authority was not persisted")
        return decode_row(row, ())

    def get_training_execution_authority(
        self,
        campaign_id: str,
        dataset_id: str | None = None,
    ) -> dict[str, Any] | None:
        query = (
            "SELECT * FROM training_execution_authority "
            "WHERE campaign_id = ? AND status = 'current'"
        )
        parameters: list[Any] = [campaign_id]
        if dataset_id is not None:
            query += " AND dataset_id = ?"
            parameters.append(dataset_id)
        query += " ORDER BY execution_generation DESC LIMIT 1"
        row = self._fetchone(query, tuple(parameters))
        return None if row is None else decode_row(row, ())

    def list_training_execution_authority_history(
        self,
        campaign_id: str,
    ) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT * FROM training_execution_authority "
            "WHERE campaign_id = ? ORDER BY execution_generation",
            (campaign_id,),
        )
        return [decode_row(row, ()) for row in rows]

    def list_authoritative_training_campaigns(self, project_id: str) -> list[dict[str, Any]]:
        """Return current training authorities discovered from ledger data."""

        return self._write(lambda: self._list_authoritative_training_campaigns(project_id))

    def _list_authoritative_training_campaigns(self, project_id: str) -> list[dict[str, Any]]:
        dataset_rows = self._connection.execute(
            "SELECT dataset_id FROM datasets WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        dataset_ids = {str(row["dataset_id"]) for row in dataset_rows}
        campaigns: dict[str, str] = {}
        rows = self._connection.execute(
            "SELECT entity_id, payload_json FROM events "
            "WHERE entity_type = 'training_campaign' ORDER BY occurred_at, event_id"
        ).fetchall()
        for row in rows:
            payload = decode_json(row["payload_json"], "payload_json")
            if not isinstance(payload, Mapping):
                continue
            dataset_id = payload.get("dataset_id")
            if isinstance(dataset_id, str) and dataset_id in dataset_ids:
                campaigns[str(row["entity_id"])] = dataset_id
        for campaign_id, dataset_id in campaigns.items():
            self._ensure_training_execution_authority(campaign_id, dataset_id)
        current = self._connection.execute(
            "SELECT * FROM training_execution_authority "
            "WHERE project_id = ? AND status = 'current' "
            "ORDER BY campaign_id",
            (project_id,),
        ).fetchall()
        return [decode_row(row, ()) for row in current]

    @staticmethod
    def _event_generation(payload: Mapping[str, Any]) -> int:
        value = payload.get("execution_generation", 1)
        try:
            generation = int(value)
        except (TypeError, ValueError) as exc:
            raise StateError("Training event has an invalid execution generation") from exc
        if generation < 1:
            raise StateError("Training event execution generation must be positive")
        return generation

    def list_training_candidate_projections(
        self,
        campaign_id: str,
        execution_generation: int,
    ) -> list[dict[str, Any]]:
        return self._list_training_projections(
            "training_candidate", campaign_id, execution_generation, "model_run_id"
        )

    def list_training_attempt_projections(
        self,
        campaign_id: str,
        execution_generation: int,
    ) -> list[dict[str, Any]]:
        return self._list_training_projections(
            "training_attempt", campaign_id, execution_generation, "attempt_id"
        )

    def _list_training_projections(
        self,
        entity_type: str,
        campaign_id: str,
        execution_generation: int,
        key: str,
    ) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        rows = self._fetchall(
            "SELECT payload_json FROM events WHERE entity_type = ? AND entity_id = ? "
            "ORDER BY occurred_at, event_id",
            (entity_type, campaign_id),
        )
        for row in rows:
            payload = decode_json(row["payload_json"], "payload_json")
            if not isinstance(payload, Mapping):
                continue
            if self._event_generation(payload) != execution_generation:
                continue
            value = payload.get(key)
            if value:
                latest[str(value)] = dict(payload)
        return [latest[value] for value in sorted(latest)]

    def invalidate_training_execution_authority(
        self,
        project_id: str,
        *,
        source: str,
    ) -> dict[str, int]:
        """Close current training generations and open fresh authorities."""

        return self._write(
            lambda: self._invalidate_training_execution_authority(project_id, source)
        )

    def _invalidate_training_execution_authority(
        self,
        project_id: str,
        source: str,
    ) -> dict[str, int]:
        authorities = self._list_authoritative_training_campaigns(project_id)
        timestamp = now()
        generations: dict[str, int] = {}
        for authority in authorities:
            campaign_id = str(authority["campaign_id"])
            previous = int(authority["execution_generation"])
            next_generation = previous + 1
            self._connection.execute(
                "UPDATE training_execution_authority SET status = 'invalidated', source = ?, "
                "updated_at = ?, invalidated_at = ? WHERE campaign_id = ? AND status = 'current'",
                (source, timestamp, timestamp, campaign_id),
            )
            self._connection.execute(
                "INSERT INTO training_execution_authority "
                "(campaign_id, dataset_id, project_id, execution_generation, status, source, "
                "created_at, updated_at, invalidated_at) VALUES "
                "(?, ?, ?, ?, 'current', ?, ?, ?, NULL)",
                (
                    campaign_id,
                    authority["dataset_id"],
                    authority["project_id"],
                    next_generation,
                    source,
                    timestamp,
                    timestamp,
                ),
            )
            generations[campaign_id] = next_generation

        model_rows = self._connection.execute(
            "SELECT m.model_run_id, m.execution_metadata_json FROM model_runs AS m "
            "JOIN datasets AS d ON d.dataset_id = m.dataset_id WHERE d.project_id = ?",
            (project_id,),
        ).fetchall()
        for row in model_rows:
            metadata = decode_json(row["execution_metadata_json"], "execution_metadata_json")
            if not isinstance(metadata, Mapping):
                metadata = {}
            metadata = dict(metadata)
            metadata.update(
                {
                    "invalidated_at": timestamp,
                    "invalidation_source": source,
                    "training_authority_generations": generations,
                }
            )
            self._connection.execute(
                "UPDATE model_runs SET status = 'invalidated', execution_metadata_json = ?, "
                "updated_at = ? WHERE model_run_id = ?",
                (encode_json(metadata), timestamp, row["model_run_id"]),
            )
        invalidate_validation = getattr(self, "_invalidate_validation_for_project", None)
        if callable(invalidate_validation):
            invalidate_validation(project_id, timestamp, source)
        return generations

    def begin_model_run_execution(
        self,
        model_run_id: str,
        *,
        campaign_id: str,
        execution_generation: int,
    ) -> dict[str, Any]:
        """Make a fresh execution current without changing scientific identity."""

        def operation() -> dict[str, Any]:
            row = self._connection.execute(
                "SELECT execution_metadata_json FROM model_runs WHERE model_run_id = ?",
                (model_run_id,),
            ).fetchone()
            if row is None:
                raise StateError(f"Unknown model run: {model_run_id}")
            metadata = decode_json(row["execution_metadata_json"], "execution_metadata_json")
            if not isinstance(metadata, Mapping):
                metadata = {}
            metadata = dict(metadata)
            metadata.update(
                {
                    "campaign_id": campaign_id,
                    "training_execution_generation": execution_generation,
                    "execution_status": "prepared",
                }
            )
            self._connection.execute(
                "UPDATE model_runs SET status = 'prepared', completed_at = NULL, "
                "execution_metadata_json = ?, updated_at = ? WHERE model_run_id = ?",
                (encode_json(metadata), now(), model_run_id),
            )
            result = self._connection.execute(
                "SELECT * FROM model_runs WHERE model_run_id = ?", (model_run_id,)
            ).fetchone()
            return decode_row(result, ("identity_json", "execution_metadata_json"))

        return self._write(operation)


__all__ = ["TrainingRecordsMixin"]
