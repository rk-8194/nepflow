"""Typed boundary for machine-learning interatomic-potential backends.

This module covers the current single-run NEP training contract.  Search,
candidate ranking, promotion, and scheduler orchestration remain outside the
backend boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from nepflow.config.models import CompositionConfig, NepTrainingConfig
from nepflow.domain.datasets import TrainingDatasetManifest
from nepflow.domain.identities import ArtifactIdentity, ModelRunIdentity
from nepflow.domain.models import ModelArtifactMetadata
from nepflow.errors import MlipError


@dataclass(frozen=True, slots=True)
class TrainingInputRequest:
    """Typed dataset/settings request for rendering one NEP training input."""

    dataset: TrainingDatasetManifest
    hyperparameters: NepTrainingConfig
    composition: CompositionConfig
    working_directory: Path
    hyperparameters_hash: str
    template_path: Path | None = None


@dataclass(frozen=True, slots=True)
class TrainingInput:
    """Rendered ``nep.in`` content and the identity of that exact input."""

    dataset: TrainingDatasetManifest
    working_directory: Path
    content: str
    nep_in: ArtifactIdentity
    hyperparameters_hash: str
    # Zero is retained as a compatibility value for lightweight test/fake
    # backends that predate the typed generation contract.  The canonical NEP
    # renderer always supplies a positive value.
    target_generations: int = 0

    def __post_init__(self) -> None:
        if self.target_generations < 0:
            raise MlipError("training target generations cannot be negative")

    @property
    def model_run_identity(self) -> ModelRunIdentity:
        """Return the canonical run identity for the rendered input."""

        return ModelRunIdentity.from_inputs(
            self.dataset.identity.dataset_id,
            self.nep_in.sha256,
            self.hyperparameters_hash,
        )


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """The progress values currently exposed by NEP ``loss.out``."""

    generation: int
    loss: float

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise MlipError("NEP training generation cannot be negative")
        if not math.isfinite(float(self.loss)):
            raise MlipError("NEP training loss must be finite")


@dataclass(frozen=True, slots=True)
class TrainingCompletion:
    """Typed backend evidence before the required model artifact is collected.

    ``completed`` is a conclusion from the complete evidence set, not an
    indication that a model-shaped file happens to exist.  The optional fields
    keep the interface backend-neutral while allowing callers to audit the
    process marker and the exact generation evidence used for that conclusion.
    """

    completed: bool
    artifact_paths: tuple[Path, ...] = ()
    segment_target_generations: int | None = None
    segment_observed_generations: int | None = None
    process_exit_code: int | None = None
    success_marker: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.completed and len(self.artifact_paths) != 1:
            raise MlipError("completed training requires exactly one model artifact")
        for name in ("segment_target_generations", "segment_observed_generations"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise MlipError(f"training {name} cannot be negative")

    @property
    def target_generation(self) -> int | None:
        """Backward-compatible alias for the segment-local target."""

        return self.segment_target_generations

    @property
    def observed_generation(self) -> int | None:
        """Backward-compatible alias for the segment-local observation."""

        return self.segment_observed_generations


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    """Verified backend checkpoint evidence for one scheduler attempt."""

    local_generation: int
    restart_path: Path
    restart_sha256: str
    model_path: Path
    model_sha256: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.local_generation <= 0:
            raise MlipError("training checkpoint generation must be positive")
        if not self.restart_sha256 or not self.model_sha256:
            raise MlipError("training checkpoint artifacts require hashes")


@dataclass(frozen=True, slots=True)
class CollectedModelArtifacts:
    """Successful model collection with a required, identity-bound artifact."""

    training_input: TrainingInput
    model_run: ModelRunIdentity
    artifact: ModelArtifactMetadata

    def __post_init__(self) -> None:
        if self.artifact is None or not isinstance(self.artifact.model, ArtifactIdentity):
            raise MlipError("collected model artifacts require model metadata")
        if self.model_run != self.training_input.model_run_identity:
            raise MlipError("collected model artifact belongs to a different model run")


@runtime_checkable
class MlipBackend(Protocol):
    """Stable operations required by current NEP training."""

    def render_training_input(self, request: TrainingInputRequest) -> TrainingInput:
        """Render ``nep.in`` from typed dataset and scientific settings."""
        ...

    def training_command(self, inputs: TrainingInput) -> tuple[str, ...]:
        """Return the training executable and arguments without shell syntax."""
        ...

    def parse_progress(self, run_directory: Path) -> TrainingProgress | None:
        """Read optional in-progress generation/loss evidence."""
        ...

    def parse_completion(self, run_directory: Path) -> TrainingCompletion:
        """Determine completion from authoritative backend output evidence."""
        ...

    def parse_checkpoint(self, run_directory: Path) -> TrainingCheckpoint | None:
        """Return verified resumable checkpoint evidence, when present."""
        ...

    def collect_model_artifacts(
        self,
        run_directory: Path,
        inputs: TrainingInput,
    ) -> CollectedModelArtifacts:
        """Collect required artifacts bound to the rendered model-run identity."""
        ...

    def model_run_identity(self, inputs: TrainingInput) -> ModelRunIdentity:
        """Expose the exact dataset/input/settings identity of the run."""
        ...


__all__ = [
    "CollectedModelArtifacts",
    "MlipBackend",
    "TrainingCheckpoint",
    "TrainingCompletion",
    "TrainingInput",
    "TrainingInputRequest",
    "TrainingProgress",
]
