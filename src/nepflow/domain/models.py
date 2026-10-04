"""Immutable model and validation artifact metadata records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from nepflow.io.json import to_jsonable

from .identities import ArtifactIdentity, ModelRunIdentity, ValidationRunIdentity, _freeze


@dataclass(frozen=True)
class ModelArtifactMetadata:
    model: ArtifactIdentity
    nep_in: ArtifactIdentity | None = None
    metrics: Mapping[str, Any] | None = None
    status: str = "unknown"
    started_at: str | None = None
    completed_at: str | None = None

    def __post_init__(self) -> None:
        if self.metrics is not None:
            object.__setattr__(self, "metrics", _freeze(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "model": self.model.to_dict(),
            "status": self.status,
        }
        if self.nep_in is not None:
            result["nep_in"] = self.nep_in.to_dict()
        if self.metrics is not None:
            result["metrics"] = to_jsonable(self.metrics)
        if self.started_at is not None:
            result["started_at"] = self.started_at
        if self.completed_at is not None:
            result["completed_at"] = self.completed_at
        return result


@dataclass(frozen=True)
class ModelRunRecord:
    identity: ModelRunIdentity
    artifact: ModelArtifactMetadata | None = None
    execution_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            self.artifact is not None
            and self.artifact.nep_in is not None
            and self.artifact.nep_in.sha256 != self.identity.nep_in_sha256
        ):
            raise ValueError(
                "Model-run identity nep_in_sha256 does not match the attached "
                "nep.in artifact SHA-256"
            )
        if self.execution_metadata is not None:
            object.__setattr__(self, "execution_metadata", _freeze(self.execution_metadata))

    @property
    def model_run_id(self) -> str:
        return self.identity.model_run_id

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = self.identity.to_dict()
        if self.artifact is not None:
            result["artifact"] = self.artifact.to_dict()
        if self.execution_metadata is not None:
            result["execution_metadata"] = to_jsonable(self.execution_metadata)
        return result


@dataclass(frozen=True)
class ValidationArtifactMetadata:
    report: ArtifactIdentity | None = None
    trajectory: ArtifactIdentity | None = None
    metrics: Mapping[str, Any] | None = None
    thresholds: Mapping[str, Any] | None = None
    passed: bool | None = None

    def __post_init__(self) -> None:
        for name in ("metrics", "thresholds"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze(value))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.report is not None:
            result["report"] = self.report.to_dict()
        if self.trajectory is not None:
            result["trajectory"] = self.trajectory.to_dict()
        if self.metrics is not None:
            result["metrics"] = to_jsonable(self.metrics)
        if self.thresholds is not None:
            result["thresholds"] = to_jsonable(self.thresholds)
        if self.passed is not None:
            result["passed"] = self.passed
        return result


@dataclass(frozen=True)
class ValidationRunRecord:
    identity: ValidationRunIdentity
    artifact: ValidationArtifactMetadata | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is not None:
            object.__setattr__(self, "metadata", _freeze(self.metadata))

    @property
    def validation_run_id(self) -> str:
        return self.identity.validation_run_id

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = self.identity.to_dict()
        if self.artifact is not None:
            result["artifact"] = self.artifact.to_dict()
        if self.metadata is not None:
            result["metadata"] = to_jsonable(self.metadata)
        return result

