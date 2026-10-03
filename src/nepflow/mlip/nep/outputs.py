"""NEP completion evidence and model artifact collection."""

from __future__ import annotations

from pathlib import Path

from nepflow.domain.identities import ArtifactIdentity
from nepflow.domain.models import ModelArtifactMetadata
from nepflow.errors import MlipError
from nepflow.mlip.backend import (
    CollectedModelArtifacts,
    TrainingCompletion,
    TrainingInput,
)


def model_artifact_paths(run_directory: Path) -> tuple[Path, ...]:
    """Resolve the exact completed NEP artifact without latest-file discovery."""

    directory = Path(run_directory)
    primary = directory / "nep.txt"
    if primary.is_file() and primary.stat().st_size > 0:
        return (primary,)
    alternates = tuple(
        path
        for path in sorted(directory.glob("nep*.txt"), key=lambda item: item.name)
        if path.is_file() and path.stat().st_size > 0 and path.name != "nep.txt"
    )
    if len(alternates) > 1:
        raise MlipError(
            "NEP training produced multiple alternate model artifacts: "
            + ", ".join(path.name for path in alternates)
        )
    return alternates


def parse_completion(run_directory: Path) -> TrainingCompletion:
    artifacts = model_artifact_paths(run_directory)
    return TrainingCompletion(bool(artifacts), artifacts)


def collect_model_artifacts(
    run_directory: Path,
    inputs: TrainingInput,
) -> CollectedModelArtifacts:
    artifacts = model_artifact_paths(run_directory)
    if len(artifacts) != 1:
        raise MlipError("Completed NEP training requires exactly one model artifact")
    model = ArtifactIdentity.from_file("nep_model", artifacts[0])
    metadata = ModelArtifactMetadata(
        model=model,
        nep_in=inputs.nep_in,
        status="completed",
    )
    return CollectedModelArtifacts(
        training_input=inputs,
        model_run=inputs.model_run_identity,
        artifact=metadata,
    )


__all__ = [
    "collect_model_artifacts",
    "model_artifact_paths",
    "parse_completion",
]
