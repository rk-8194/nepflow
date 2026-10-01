"""Immutable DFT calculation and result-artifact records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nepflow.io.json import to_jsonable

from .identities import ArtifactIdentity, DftCalculationIdentity, _freeze


@dataclass(frozen=True)
class DftExecutionResources:
    nodes: str | None = None
    ncore: int | None = None
    kpar: int | None = None
    gpus: int | None = None
    walltime: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("nodes", self.nodes),
                ("ncore", self.ncore),
                ("kpar", self.kpar),
                ("gpus", self.gpus),
                ("walltime", self.walltime),
            )
            if value is not None
        }


@dataclass(frozen=True)
class DftResultArtifact:
    calculation: DftCalculationIdentity
    outcar: ArtifactIdentity | None = None
    vasprun: ArtifactIdentity | None = None
    status: str = "unknown"
    metadata: Any = None
    resources: DftExecutionResources = field(default_factory=DftExecutionResources)

    def __post_init__(self) -> None:
        if self.metadata is not None:
            object.__setattr__(self, "metadata", _freeze(self.metadata))

    @property
    def calculation_id(self) -> str:
        return self.calculation.calculation_id

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "calculation_id": self.calculation_id,
            "calculation": self.calculation.to_dict(),
            "status": self.status,
            "resources": self.resources.to_dict(),
        }
        if self.outcar is not None:
            result["outcar"] = self.outcar.to_dict()
        if self.vasprun is not None:
            result["vasprun"] = self.vasprun.to_dict()
        if self.metadata is not None:
            result["metadata"] = to_jsonable(self.metadata)
        return result

