"""Typed request and result records for generation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from nepflow.config.models import CompositionConfig, GenerationConfig, MagnetismConfig
from nepflow.domain.identities import ArtifactIdentity

GENERATION_MANIFEST_SCHEMA = "nepflow.generation_manifest.v2"


class GenerationExecutionMode(str, Enum):
    """Execution intent supplied to the generation stage."""

    RESTART = "restart"
    RESUME = "resume"

    @classmethod
    def from_context_mode(cls, value: object) -> "GenerationExecutionMode":
        """Translate controller context modes into generation intent."""

        if isinstance(value, cls):
            return value
        mode = "resume" if value is None else str(value).strip().lower()
        if mode in {"restart", "local", "debug"}:
            return cls.RESTART
        if mode in {"resume", "normal"}:
            return cls.RESUME
        raise ValueError(f"Unsupported generation execution mode: {value!r}")


# Keep the domain terminology available to callers that refer to this as an
# execution intent rather than an execution mode.
GenerationExecutionIntent = GenerationExecutionMode


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
    execution_mode: GenerationExecutionMode = GenerationExecutionMode.RESUME
    magnetism: MagnetismConfig = field(default_factory=MagnetismConfig)

    @property
    def execution_intent(self) -> GenerationExecutionMode:
        """Compatibility spelling for the generation execution mode."""

        return self.execution_mode

    @property
    def seeds_file(self) -> Path:
        return self.project_dir / self.structures_path / "seeds" / "base_structures.xyz"


@dataclass(frozen=True, slots=True)
class GenerationManifest:
    """The durable, authoritative summary of one generation run.

    ``structure_ids`` remains the ordered identity list for the seed/base
    artifact.  Final candidates deliberately have their own identity lists:
    magnetic candidates can share a physical ``structure_id`` while retaining
    distinct ``candidate_id`` values.
    """

    artifact: ArtifactIdentity | None
    path: Path
    structure_ids: tuple[str, ...]
    count: int
    resumed: bool = False
    candidate_artifact: ArtifactIdentity | None = None
    candidate_path: Path | None = None
    candidate_ids: tuple[str, ...] = ()
    candidate_structure_ids: tuple[str, ...] = ()
    accepted_candidate_count: int = 0
    counts_by_configurational_type: Mapping[str, int] = field(default_factory=dict)
    counts_by_perturbation_family: Mapping[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    rejections_by_family: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    rejected_count: int = 0
    requested_family_counts: Mapping[str, int] = field(default_factory=dict)
    realised_family_counts: Mapping[str, int] = field(default_factory=dict)
    coverage: Mapping[str, Any] = field(default_factory=dict)
    config_fingerprint: str | None = None
    manifest_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "structure_ids", tuple(str(value) for value in self.structure_ids))
        object.__setattr__(self, "candidate_ids", tuple(str(value) for value in self.candidate_ids))
        object.__setattr__(
            self,
            "candidate_structure_ids",
            tuple(str(value) for value in self.candidate_structure_ids),
        )
        if self.accepted_candidate_count == 0 and self.candidate_ids:
            object.__setattr__(self, "accepted_candidate_count", len(self.candidate_ids))
        if self.candidate_ids and self.accepted_candidate_count != len(self.candidate_ids):
            raise ValueError("accepted candidate count must match candidate IDs")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("final generation candidate IDs must be unique")
        if (
            self.candidate_ids
            and self.candidate_structure_ids
            and len(self.candidate_structure_ids) != len(self.candidate_ids)
        ):
            raise ValueError("candidate and structure identity counts must match")
        object.__setattr__(
            self,
            "counts_by_configurational_type",
            {str(key): int(value) for key, value in self.counts_by_configurational_type.items()},
        )
        object.__setattr__(
            self,
            "counts_by_perturbation_family",
            {str(key): int(value) for key, value in self.counts_by_perturbation_family.items()},
        )
        object.__setattr__(
            self,
            "rejections_by_family",
            {
                str(family): {str(reason): int(count) for reason, count in reasons.items()}
                for family, reasons in self.rejections_by_family.items()
            },
        )
        if self.rejected_count == 0:
            object.__setattr__(
                self,
                "rejected_count",
                sum(sum(reasons.values()) for reasons in self.rejections_by_family.values()),
            )
        object.__setattr__(
            self,
            "requested_family_counts",
            {str(key): int(value) for key, value in self.requested_family_counts.items()},
        )
        object.__setattr__(
            self,
            "realised_family_counts",
            {str(key): int(value) for key, value in self.realised_family_counts.items()},
        )

    @property
    def seed_artifact(self) -> ArtifactIdentity | None:
        """Explicit spelling for callers that distinguish seed/candidate output."""

        return self.artifact

    @property
    def candidate_count(self) -> int:
        """Return the accepted final candidate count."""

        return self.accepted_candidate_count

    @property
    def base_structure_ids(self) -> tuple[str, ...]:
        """Return seed/base structure IDs in their persisted order."""

        return self.structure_ids

    @property
    def underlying_structure_ids(self) -> tuple[str, ...]:
        """Return physical structure IDs in final candidate order."""

        return self.candidate_structure_ids

    @property
    def counts_by_family(self) -> Mapping[str, int]:
        """Compatibility spelling for perturbation-family counts."""

        return self.counts_by_perturbation_family

    @property
    def counts_by_derived_family(self) -> Mapping[str, int]:
        """Return final counts grouped by derived generation family."""

        return self.counts_by_perturbation_family

    @property
    def duplicate_count(self) -> int:
        """Compatibility spelling for exact final duplicates removed."""

        return self.duplicates_removed

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GENERATION_MANIFEST_SCHEMA,
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "path": str(self.path),
            "structure_ids": list(self.structure_ids),
            "count": self.count,
            "resumed": self.resumed,
            "candidate_artifact": (
                None if self.candidate_artifact is None else self.candidate_artifact.to_dict()
            ),
            "candidate_path": None if self.candidate_path is None else str(self.candidate_path),
            "candidate_ids": list(self.candidate_ids),
            "candidate_structure_ids": list(self.candidate_structure_ids),
            "accepted_candidate_count": self.accepted_candidate_count,
            "counts_by_configurational_type": dict(self.counts_by_configurational_type),
            "counts_by_perturbation_family": dict(self.counts_by_perturbation_family),
            "duplicates_removed": self.duplicates_removed,
            "rejections_by_family": {
                family: dict(reasons) for family, reasons in self.rejections_by_family.items()
            },
            "rejected_count": self.rejected_count,
            "requested_family_counts": dict(self.requested_family_counts),
            "realised_family_counts": dict(self.realised_family_counts),
            "coverage": dict(self.coverage),
            "config_fingerprint": self.config_fingerprint,
            "manifest_path": (None if self.manifest_path is None else str(self.manifest_path)),
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
