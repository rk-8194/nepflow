"""Immutable generated-structure provenance records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .identities import StructureIdentity, _freeze, _jsonable


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_structure_id": self.parent_structure_id,
            "generator": self.generator,
            "requested_composition": _jsonable(self.requested_composition),
            "realised_composition": _jsonable(self.realised_composition),
            "source_database_id": self.source_database_id,
            "crystal_structure": self.crystal_structure,
            "perturbation_family": self.perturbation_family,
            "perturbation_parameters": _jsonable(self.perturbation_parameters),
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
            result["metadata"] = _jsonable(self.metadata)
        return result

