"""Immutable dataset membership and manifest identity records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from nepflow.io.hashing import sha256_canonical_json
from nepflow.io.json import to_jsonable

from .identities import (
    _freeze,
    normalise_dft_calculation_identity,
)

DATASET_IDENTITY_SCHEMA = "nepflow.dataset.v1"
DATASET_MANIFEST_SCHEMA = "nepflow.dataset_manifest.v1"


@dataclass(frozen=True)
class SelectedDatasetMember:
    """One selected structure and its accepted, hash-bound DFT result."""
    split: str
    structure_id: str
    calculation_id: str
    source_outcar_hash: str
    ordinal: int
    calculation_identity: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        identity = self.calculation_identity
        if isinstance(identity, Mapping):
            identity = identity.items()
        object.__setattr__(
            self,
            "calculation_identity",
            tuple(sorted((str(key), str(value)) for key, value in identity)),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], ordinal: int) -> "SelectedDatasetMember":
        """Parse one persisted member and require a usable calculation ID."""
        identity = value.get("calculation_identity", value.get("calculation_id", {}))
        if isinstance(identity, str):
            identity = {"calculation_id": identity}
        normalized_identity = normalise_dft_calculation_identity(dict(identity))
        calculation_id = normalized_identity.get("calculation_id")
        if not isinstance(calculation_id, str) or not calculation_id.strip():
            raise ValueError("selected dataset member is missing calculation_id")
        return cls(
            split=str(value["split"]),
            structure_id=str(value["structure_id"]),
            calculation_id=calculation_id,
            source_outcar_hash=str(value["source_outcar_hash"]),
            ordinal=ordinal,
            calculation_identity=tuple(sorted((str(k), str(v)) for k, v in dict(identity).items())),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize membership with normalized calculation identity fields."""
        return {
            "split": self.split,
            "structure_id": self.structure_id,
            "calculation_id": self.calculation_id,
            "calculation_identity": dict(self.calculation_identity),
            "source_outcar_hash": self.source_outcar_hash,
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True)
class DatasetIdentity:
    """Immutable content identity for a complete train/test dataset."""
    dataset_id: str
    identity_payload: Mapping[str, Any]
    schema_version: str = DATASET_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_payload", _freeze(self.identity_payload))

    @classmethod
    def from_identity_payload(cls, payload: Mapping[str, Any]) -> "DatasetIdentity":
        """Hash a JSON-shaped payload without filesystem paths."""
        frozen_payload = to_jsonable(dict(payload))
        return cls("dataset_" + sha256_canonical_json(frozen_payload), frozen_payload)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        label_schema: Any,
        units: Mapping[str, str],
        virial_convention: str | None,
        virial_tensor_convention: str | None = None,
    ) -> "DatasetIdentity":
        """Derive identity from label schema, units, and ordered records.

        Source paths are excluded so relocating unchanged data does not change
        its scientific identity.
        """
        payload: dict[str, Any] = {
            "schema_version": DATASET_IDENTITY_SCHEMA,
            "label_schema": label_schema,
            "units": dict(units),
            "virial_convention": virial_convention,
            "records": [
                to_jsonable(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {"path", "source_path", "source_outcar", "source_xyz"}
                    }
                )
                for record in records
            ],
        }
        payload["virial_tensor_convention"] = virial_tensor_convention
        return cls.from_identity_payload(payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataset ID together with its identity payload."""
        return {"dataset_id": self.dataset_id, **to_jsonable(self.identity_payload)}


@dataclass(frozen=True)
class TrainingDatasetManifest:
    """Persisted dataset membership, identity, and selection provenance."""
    identity: DatasetIdentity
    records: tuple[Mapping[str, Any], ...]
    selection_method: str | None = None
    selection_parameters: Mapping[str, Any] | None = None
    descriptor_model_fingerprint: Mapping[str, Any] | None = None
    created_at: str | None = None
    code_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(_freeze(record) for record in self.records))
        for name in (
            "selection_parameters",
            "descriptor_model_fingerprint",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze(value))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest using the canonical dataset schema."""
        result: dict[str, Any] = {
            "schema_version": DATASET_MANIFEST_SCHEMA,
            "dataset_id": self.identity.dataset_id,
            "identity": self.identity.to_dict(),
            "records": [to_jsonable(record) for record in self.records],
        }
        for key, value in (
            ("selection_method", self.selection_method),
            ("selection_parameters", self.selection_parameters),
            ("descriptor_model_fingerprint", self.descriptor_model_fingerprint),
            ("created_at", self.created_at),
            ("code_version", self.code_version),
        ):
            if value is not None:
                result[key] = to_jsonable(value)
        return result
