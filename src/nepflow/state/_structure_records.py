"""Persistence operations for generated structures and provenance."""

from __future__ import annotations

from typing import Any

from nepflow.domain.structures import (
    GeneratedStructureRecord,
    StructureIdentity,
    StructureProvenance,
)
from nepflow.errors import StateError

from ._record_codec import decode_row, encode_json, now
from ._typing import StateStoreMixinSupport, require_state_row


class StructureRecordsMixin(StateStoreMixinSupport):
    """Provide the StateStore structure-record persistence boundary."""

    def upsert_structure(
        self,
        identity_or_record: StructureIdentity | GeneratedStructureRecord,
        *,
        provenance: StructureProvenance | None = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        """Insert or update a structure identity and optional provenance."""

        identity, provenance, metadata = _structure_values(identity_or_record, provenance, metadata)
        timestamp = now()
        return self._write(lambda: self._write_structure(identity, provenance, metadata, timestamp))

    def _write_structure(
        self,
        identity: StructureIdentity,
        provenance: StructureProvenance | None,
        metadata: Any,
        timestamp: str,
    ) -> dict[str, Any]:
        existing = self._connection.execute(
            "SELECT identity_schema FROM structures WHERE structure_id = ?",
            (identity.structure_id,),
        ).fetchone()
        if existing is not None and existing["identity_schema"] != identity.schema_version:
            raise StateError(f"Structure identity schema conflict: {identity.structure_id}")
        if existing is None:
            self._connection.execute(
                """
                INSERT INTO structures (structure_id, identity_schema, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    identity.structure_id,
                    identity.schema_version,
                    encode_json(metadata),
                    timestamp,
                ),
            )
        elif metadata is not None:
            self._connection.execute(
                """
                UPDATE structures
                SET identity_schema = ?, metadata_json = ?
                WHERE structure_id = ?
                """,
                (identity.schema_version, encode_json(metadata), identity.structure_id),
            )
        if provenance is not None:
            self._write_provenance(identity.structure_id, provenance, timestamp)
        return require_state_row(self.get_structure(identity.structure_id), "structure")

    def append_structure_provenance(
        self,
        structure_id: str,
        provenance: StructureProvenance,
    ) -> dict[str, Any]:
        """Append one provenance operation without replacing structure metadata."""

        timestamp = now()

        def write_provenance() -> dict[str, Any]:
            structure = self._connection.execute(
                "SELECT structure_id FROM structures WHERE structure_id = ?",
                (structure_id,),
            ).fetchone()
            if structure is None:
                raise StateError(f"Unknown structure identity: {structure_id}")
            self._write_provenance(structure_id, provenance, timestamp)
            return require_state_row(self.get_structure(structure_id), "structure")

        return self._write(write_provenance)

    def record_provenance(
        self,
        structure_id: str,
        provenance: StructureProvenance,
    ) -> dict[str, Any]:
        """Compatibility alias for appending one structure provenance operation."""

        return self.append_structure_provenance(structure_id, provenance)

    def record_structure_provenance(
        self,
        structure_id: str,
        provenance: StructureProvenance,
    ) -> dict[str, Any]:
        """Explicitly named alias for appending one structure provenance operation."""

        return self.append_structure_provenance(structure_id, provenance)

    def append_provenance(
        self,
        structure_id: str,
        provenance: StructureProvenance,
    ) -> dict[str, Any]:
        """Short alias for appending one structure provenance operation."""

        return self.append_structure_provenance(structure_id, provenance)

    def _write_provenance(
        self,
        structure_id: str,
        provenance: StructureProvenance,
        timestamp: str,
    ) -> None:
        existing = self._connection.execute(
            "SELECT structure_id FROM structure_provenance WHERE operation_id = ?",
            (provenance.operation_id,),
        ).fetchone()
        if existing is not None and existing["structure_id"] != structure_id:
            raise StateError(f"Structure provenance identity conflict: {provenance.operation_id}")
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
                structure_id,
                provenance.parent_structure_id,
                provenance.generator,
                encode_json(provenance.requested_composition),
                encode_json(provenance.realised_composition),
                provenance.source_database_id,
                provenance.crystal_structure,
                provenance.perturbation_family,
                encode_json(provenance.perturbation_parameters),
                provenance.random_seed,
                provenance.code_version,
                provenance.config_fingerprint,
                timestamp,
            ),
        )

    def get_structure(self, structure_id: str) -> dict[str, Any] | None:
        """Return one structure and its latest provenance record."""

        row = self._fetchone("SELECT * FROM structures WHERE structure_id = ?", (structure_id,))
        if row is None:
            return None
        result = decode_row(row, ("metadata_json",))
        provenance_row = self._fetchone(
            "SELECT * FROM structure_provenance WHERE structure_id = ? "
            "ORDER BY created_at DESC, operation_id DESC LIMIT 1",
            (structure_id,),
        )
        if provenance_row is not None:
            result["provenance"] = decode_row(
                provenance_row,
                (
                    "requested_composition_json",
                    "realised_composition_json",
                    "perturbation_parameters_json",
                ),
            )
        return result

    def get_structure_provenance(self, structure_id: str) -> list[dict[str, Any]]:
        """Return all provenance operations for a structure in stable order."""

        rows = self._fetchall(
            "SELECT * FROM structure_provenance WHERE structure_id = ? "
            "ORDER BY created_at ASC, operation_id ASC",
            (structure_id,),
        )
        return [
            decode_row(
                row,
                (
                    "requested_composition_json",
                    "realised_composition_json",
                    "perturbation_parameters_json",
                ),
            )
            for row in rows
        ]

    def list_structure_provenance(self, structure_id: str) -> list[dict[str, Any]]:
        """Compatibility alias for retrieving all structure provenance operations."""

        return self.get_structure_provenance(structure_id)

    def record_structure(
        self,
        identity_or_record: StructureIdentity | GeneratedStructureRecord,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Record a structure identity and optional provenance."""

        return self.upsert_structure(identity_or_record, **kwargs)


def _structure_values(
    identity_or_record: StructureIdentity | GeneratedStructureRecord,
    provenance: StructureProvenance | None,
    metadata: Any,
) -> tuple[StructureIdentity, StructureProvenance | None, Any]:
    if isinstance(identity_or_record, GeneratedStructureRecord):
        return (
            identity_or_record.identity,
            identity_or_record.provenance,
            identity_or_record.metadata,
        )
    return identity_or_record, provenance, metadata
