"""Typed boundary for machine-learning interatomic-potential backends.

This module covers the current single-run NEP training contract.  Search,
candidate ranking, promotion, and scheduler orchestration remain outside the
backend boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

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
    """Completion evidence before the required model artifact is collected."""

    completed: bool
    artifact_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.completed and not self.artifact_paths:
            raise MlipError(
                "completed NEP training requires at least one model artifact"
            )


@dataclass(frozen=True, slots=True)
class CollectedModelArtifacts:
    """Successful model collection with a required, identity-bound artifact."""

    training_input: TrainingInput
    model_run: ModelRunIdentity
    artifact: ModelArtifactMetadata

    def __post_init__(self) -> None:
        if self.artifact is None or self.artifact.model is None:  # type: ignore[comparison-overlap]
            raise MlipError("collected model artifacts require model metadata")
        if self.model_run != self.training_input.model_run_identity:
            raise MlipError(
                "collected model artifact belongs to a different model run"
            )


@runtime_checkable
class MlipBackend(Protocol):
    """Stable operations required by current NEP training."""

    def render_training_input(self, request: TrainingInputRequest) -> TrainingInput:
        """Render ``nep.in`` from typed dataset and scientific settings."""

    def training_command(self, inputs: TrainingInput) -> tuple[str, ...]:
        """Return the training executable and arguments without shell syntax."""

    def parse_progress(self, run_directory: Path) -> TrainingProgress | None:
        """Read optional in-progress generation/loss evidence."""

    def parse_completion(self, run_directory: Path) -> TrainingCompletion:
        """Determine completion from backend output evidence."""

    def collect_model_artifacts(
        self,
        run_directory: Path,
        inputs: TrainingInput,
    ) -> CollectedModelArtifacts:
        """Collect required artifacts bound to the rendered model-run identity."""

    def model_run_identity(self, inputs: TrainingInput) -> ModelRunIdentity:
        """Expose the exact dataset/input/settings identity of the run."""


__all__ = [
    "CollectedModelArtifacts",
    "MlipBackend",
    "TrainingCompletion",
    "TrainingInput",
    "TrainingInputRequest",
    "TrainingProgress",
]
