"""Typed request and result records for generation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from nepflow.config.models import CompositionConfig, GenerationConfig
from nepflow.domain.identities import ArtifactIdentity


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """All inputs needed by the generation coordinator.

    The scientific generator implementations deliberately do not appear in
    this record.  They are supplied as stage dependencies by the application
    composition root, keeping configuration discovery out of the stage.
    """

    project_name: str
    project_dir: Path
    composition: CompositionConfig
    generation: GenerationConfig
    random_seed: int
    structures_path: Path = Path("structures")
    state_store: Any = None
    seeds_only: bool = False
    debug: bool = False

    @property
    def seeds_file(self) -> Path:
        return self.project_dir / self.structures_path / "seeds" / "base_structures.xyz"


@dataclass(frozen=True, slots=True)
class GenerationManifest:
    """The durable summary of one base-structure preparation step."""

    artifact: ArtifactIdentity | None
    path: Path
    structure_ids: tuple[str, ...]
    count: int
    resumed: bool = False
    candidate_artifact: ArtifactIdentity | None = None
    candidate_path: Path | None = None

    @property
    def seed_artifact(self) -> ArtifactIdentity | None:
        """Explicit spelling for callers that distinguish seed/candidate output."""

        return self.artifact

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "path": str(self.path),
            "structure_ids": list(self.structure_ids),
            "count": self.count,
            "resumed": self.resumed,
            "candidate_artifact": (
                None if self.candidate_artifact is None else self.candidate_artifact.to_dict()
            ),
            "candidate_path": None if self.candidate_path is None else str(self.candidate_path),
        }


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Typed output of :class:`GenerationStage`."""

    status: str
    base_structures: tuple[Any, ...]
    manifest: GenerationManifest
    summary: Mapping[str, Any] = field(default_factory=dict)
    seeds_only: bool = False

    @property
    def completed(self) -> bool:
        return self.status in {"completed", "empty"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "manifest": self.manifest.to_dict(),
            "summary": dict(self.summary),
            "seeds_only": self.seeds_only,
        }
