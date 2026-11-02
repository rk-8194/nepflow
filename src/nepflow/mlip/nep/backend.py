"""Concrete NEP backend with no scheduler or campaign policy."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Sequence

from nepflow.domain.identities import ArtifactIdentity, ModelRunIdentity
from nepflow.mlip.backend import (
    CollectedModelArtifacts,
    TrainingCheckpoint,
    TrainingCompletion,
    TrainingInput,
    TrainingInputRequest,
    TrainingProgress,
)

from .inputs import NepInputRenderer
from .metrics import classify_training_error, parse_progress
from .outputs import collect_model_artifacts, parse_checkpoint, parse_completion


class NepBackend:
    """Implement one prepared NEP training run without scheduler policy.

    Rendering, progress/completion parsing, and artifact collection remain
    bound to the exact dataset/input/model identity; campaign retries and
    promotion are owned by :class:`TrainingCampaign`.
    """

    def __init__(
        self,
        command: str | Sequence[str] = ("nep",),
        *,
        renderer: NepInputRenderer | None = None,
    ) -> None:
        if isinstance(command, str):
            command = tuple(shlex.split(command))
        self._command = tuple(str(part) for part in command)
        if not self._command or any(not part for part in self._command):
            raise ValueError("NEP training command must contain executable arguments")
        self.renderer = renderer or NepInputRenderer()

    def render_training_input(self, request: TrainingInputRequest) -> TrainingInput:
        """Render and identity-bind one ``nep.in`` file."""
        return self.renderer.render(request)

    def training_command(self, inputs: TrainingInput) -> tuple[str, ...]:
        """Return only backend arguments; scheduler submission is a caller concern."""

        return self._command

    def materialize_attempt_input(
        self,
        inputs: TrainingInput,
        destination: Path,
        segment_target_generations: int,
    ) -> ArtifactIdentity:
        """Write a segment-local input without changing the canonical input."""

        return self.renderer.materialize_attempt_input(
            inputs,
            destination,
            segment_target_generations,
        )

    @property
    def command(self) -> tuple[str, ...]:
        """Expose configured backend arguments to a scheduler-script caller."""

        return self._command

    def parse_progress(self, run_directory: Path) -> TrainingProgress | None:
        """Read optional generation/loss progress from a run directory."""
        return parse_progress(run_directory)

    def parse_completion(self, run_directory: Path) -> TrainingCompletion:
        """Return completion evidence only when the full attempt contract passes."""
        return parse_completion(run_directory)

    def parse_checkpoint(self, run_directory: Path) -> TrainingCheckpoint | None:
        """Return verified attempt-local restart evidence, when present."""
        return parse_checkpoint(run_directory)

    def collect_model_artifacts(
        self,
        run_directory: Path,
        inputs: TrainingInput,
    ) -> CollectedModelArtifacts:
        """Collect model artifacts and verify their run identity and hashes."""
        return collect_model_artifacts(run_directory, inputs)

    def model_run_identity(self, inputs: TrainingInput) -> ModelRunIdentity:
        """Return the identity derived from the exact rendered training input."""
        return inputs.model_run_identity

    def classify_error(self, run_directory: Path) -> str | None:
        """Classify observable NEP failure evidence without marking success."""
        classified = classify_training_error(run_directory)
        if classified is not None:
            return classified
        return None if self.parse_completion(run_directory).completed else "missing_output"


__all__ = ["NepBackend"]
