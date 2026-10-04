"""Persistence operations for training datasets and their members."""

from __future__ import annotations

from typing import Any

from nepflow.domain.datasets import DatasetIdentity, SelectedDatasetMember, TrainingDatasetManifest
from nepflow.errors import StateError

from ._record_codec import decode_json, decode_row, encode_json, now


class DatasetRecordsMixin:
    """Provide StateStore persistence for dataset ledger records."""

    def upsert_dataset(
        self,
        identity_or_manifest: DatasetIdentity | TrainingDatasetManifest,
        *,
        project_id: str | None = None,
        status: str = "prepared",
    ) -> dict[str, Any]:
        """Insert or update a dataset identity and manifest."""

        if isinstance(identity_or_manifest, TrainingDatasetManifest):
            identity = identity_or_manifest.identity
            manifest = identity_or_manifest.to_dict()
        else:
            identity = identity_or_manifest
            manifest = identity.to_dict()
        return self._write(
            lambda: self._write_dataset(identity, manifest, project_id, status, now())
        )

    def _write_dataset(
        self,
        identity: DatasetIdentity,
        manifest: dict[str, Any],
        project_id: str | None,
        status: str,
        timestamp: str,
    ) -> dict[str, Any]:
        existing = self._connection.execute(
            "SELECT identity_json FROM datasets WHERE dataset_id = ?",
            (identity.dataset_id,),
        ).fetchone()
        if existing is not None and encode_json(
            decode_json(existing["identity_json"], "identity_json")
        ) != encode_json(identity.to_dict()):
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
                encode_json(identity.to_dict()),
                status,
                encode_json(manifest),
                timestamp,
                timestamp,
            ),
        )
        return self.get_dataset(identity.dataset_id)  # type: ignore[return-value]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
        return None if row is None else decode_row(row, ("identity_json", "manifest_json"))

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
        """Insert or update one ordered member of a training dataset."""

        return self._write(lambda: self._write_dataset_member(dataset_id, member, metadata))

    def _write_dataset_member(
        self,
        dataset_id: str,
        member: SelectedDatasetMember,
        metadata: Any,
    ) -> dict[str, Any]:
        existing = self._connection.execute(
            "SELECT structure_id, calculation_id FROM dataset_members "
            "WHERE dataset_id = ? AND ordinal = ?",
            (dataset_id, member.ordinal),
        ).fetchone()
        if existing is not None and (
            existing["structure_id"] != member.structure_id
            or existing["calculation_id"] != member.calculation_id
        ):
            raise StateError(f"Dataset member identity conflict: {dataset_id}/{member.ordinal}")
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
                encode_json(dict(member.calculation_identity)),
                encode_json({} if metadata is None else metadata),
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM dataset_members WHERE dataset_id = ? AND ordinal = ?",
            (dataset_id, member.ordinal),
        ).fetchone()
        return decode_row(row, ("calculation_identity_json", "metadata_json"))

    def list_dataset_members(self, dataset_id: str) -> list[dict[str, Any]]:
        """Return ordered members for a training dataset."""

        rows = self._fetchall(
            "SELECT * FROM dataset_members WHERE dataset_id = ? ORDER BY ordinal",
            (dataset_id,),
        )
        return [decode_row(row, ("calculation_identity_json", "metadata_json")) for row in rows]
