"""Immutable generated-structure provenance records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nepflow.io.json import to_jsonable

from .identities import STRUCTURE_IDENTITY_SCHEMA, StructureIdentity, _freeze


@dataclass(frozen=True)
class StructureProvenance:
    parent_structure_id: str | None
    generator: str
    requested_composition: Any
    realised_composition: Any
    source_database_id: str | None
    crystal_structure: str | None
    perturbation_family: str | None
    perturbation_parameters: Any
    random_seed: int | None
    operation_id: str
    code_version: str | None
    config_fingerprint: str | None

    def __post_init__(self) -> None:
        for name in (
            "requested_composition",
            "realised_composition",
            "perturbation_parameters",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        """Serialize through public values so process workers can return records.

        ``_freeze`` intentionally uses ``MappingProxyType`` for immutable
        provenance mappings, but the standard multiprocessing pickler cannot
        serialize mapping proxies.  The public JSON-shaped representation is
        the explicit transport boundary; reconstruction calls the normal
        constructor and therefore reapplies ``_freeze``.
        """

        return (_restore_structure_provenance, (self.to_dict(),))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_structure_id": self.parent_structure_id,
            "generator": self.generator,
            "requested_composition": to_jsonable(self.requested_composition),
            "realised_composition": to_jsonable(self.realised_composition),
            "source_database_id": self.source_database_id,
            "crystal_structure": self.crystal_structure,
            "perturbation_family": self.perturbation_family,
            "perturbation_parameters": to_jsonable(self.perturbation_parameters),
            "random_seed": self.random_seed,
            "operation_id": self.operation_id,
            "code_version": self.code_version,
            "config_fingerprint": self.config_fingerprint,
        }


@dataclass(frozen=True)
class GeneratedStructureRecord:
    identity: StructureIdentity
    provenance: StructureProvenance
    metadata: Any = None

    def __post_init__(self) -> None:
        if self.metadata is not None:
            object.__setattr__(self, "metadata", _freeze(self.metadata))

    @property
    def structure_id(self) -> str:
        return self.identity.structure_id

    def to_dict(self) -> dict[str, Any]:
        result = self.identity.to_dict()
        result["provenance"] = self.provenance.to_dict()
        if self.metadata is not None:
            result["metadata"] = to_jsonable(self.metadata)
        return result

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        """Serialize the canonical record through its public representation."""

        return (_restore_generated_structure_record, (self.to_dict(),))


def _restore_structure_provenance(payload: dict[str, Any]) -> StructureProvenance:
    """Rebuild an immutable provenance record from its JSON-shaped values."""

    return StructureProvenance(
        parent_structure_id=payload["parent_structure_id"],
        generator=payload["generator"],
        requested_composition=payload["requested_composition"],
        realised_composition=payload["realised_composition"],
        source_database_id=payload["source_database_id"],
        crystal_structure=payload["crystal_structure"],
        perturbation_family=payload["perturbation_family"],
        perturbation_parameters=payload["perturbation_parameters"],
        random_seed=payload["random_seed"],
        operation_id=payload["operation_id"],
        code_version=payload["code_version"],
        config_fingerprint=payload["config_fingerprint"],
    )


def _restore_generated_structure_record(
    payload: dict[str, Any],
) -> GeneratedStructureRecord:
    """Rebuild a canonical generated record through its normal constructors."""

    identity = StructureIdentity(
        structure_id=payload["structure_id"],
        schema_version=payload.get("structure_id_version", STRUCTURE_IDENTITY_SCHEMA),
    )
    provenance = _restore_structure_provenance(payload["provenance"])
    return GeneratedStructureRecord(
        identity=identity,
        provenance=provenance,
        metadata=payload.get("metadata"),
    )
