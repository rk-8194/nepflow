"""Concrete NEP backend with no scheduler or campaign policy."""

from __future__ import annotations

from pathlib import Path
import shlex
from typing import Sequence

from nepflow.domain.identities import ModelRunIdentity
from nepflow.mlip.backend import (
    CollectedModelArtifacts,
    TrainingCompletion,
    TrainingInput,
    TrainingInputRequest,
    TrainingProgress,
)

from .inputs import NepInputRenderer
from .metrics import classify_training_error, parse_progress
from .outputs import collect_model_artifacts, parse_completion


class NepBackend:
    """Implement the backend contract for one prepared NEP training run."""

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
        return self.renderer.render(request)

    def training_command(self, inputs: TrainingInput) -> tuple[str, ...]:
        """Return only backend arguments; scheduler submission is a caller concern."""

        return self._command

    @property
    def command(self) -> tuple[str, ...]:
        """Expose configured backend arguments to a scheduler-script caller."""

        return self._command

    def parse_progress(self, run_directory: Path) -> TrainingProgress | None:
        return parse_progress(run_directory)

    def parse_completion(self, run_directory: Path) -> TrainingCompletion:
        return parse_completion(run_directory)

    def collect_model_artifacts(
        self,
        run_directory: Path,
        inputs: TrainingInput,
    ) -> CollectedModelArtifacts:
        return collect_model_artifacts(run_directory, inputs)

    def model_run_identity(self, inputs: TrainingInput) -> ModelRunIdentity:
        return inputs.model_run_identity

    def classify_error(self, run_directory: Path) -> str | None:
        classified = classify_training_error(run_directory)
        if classified is not None:
            return classified
        return None if self.parse_completion(run_directory).completed else "missing_output"


__all__ = ["NepBackend"]
