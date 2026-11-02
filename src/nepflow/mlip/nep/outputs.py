"""NEP completion evidence and model artifact collection."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from nepflow.domain.identities import ArtifactIdentity
from nepflow.domain.models import ModelArtifactMetadata
from nepflow.errors import MlipError
from nepflow.io.hashing import sha256_file
from nepflow.mlip.backend import (
    CollectedModelArtifacts,
    TrainingCheckpoint,
    TrainingCompletion,
    TrainingInput,
)

from .metrics import parse_progress_history

SUCCESS_MARKER_NAME = ".nepflow_training_success.json"
SUCCESS_MARKER_SCHEMA = "nepflow.training_success.v2"
_GENERATION_LINE = re.compile(r"^\s*generation\s+([0-9]+)\s*(?:#.*)?$")


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


def _read_success_marker(run_directory: Path) -> dict[str, Any] | None:
    marker_path = Path(run_directory) / SUCCESS_MARKER_NAME
    if not marker_path.is_file():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MlipError(f"Malformed NEP training success marker: {marker_path}") from exc
    if not isinstance(payload, dict):
        raise MlipError("NEP training success marker must contain a JSON object")
    required = {
        "schema_version",
        "model_run_id",
        "attempt_id",
        "attempt_number",
        "execution_generation",
        "generation_offset",
        "segment_target_generations",
        "process_exit_code",
        "completion_timestamp",
    }
    if not required.issubset(payload):
        missing = ", ".join(sorted(required - payload.keys()))
        raise MlipError(f"NEP training success marker is missing: {missing}")
    if payload["schema_version"] != SUCCESS_MARKER_SCHEMA:
        raise MlipError("Unsupported NEP training success marker schema")
    if not isinstance(payload["model_run_id"], str) or not payload["model_run_id"].strip():
        raise MlipError("NEP training success marker has no model_run_id")
    if not isinstance(payload["attempt_id"], str) or not payload["attempt_id"].strip():
        raise MlipError("NEP training success marker has no attempt_id")
    if (
        isinstance(payload["attempt_number"], bool)
        or not isinstance(payload["attempt_number"], int)
        or payload["attempt_number"] < 1
    ):
        raise MlipError("NEP training success marker has an invalid attempt_number")
    if (
        isinstance(payload["execution_generation"], bool)
        or not isinstance(payload["execution_generation"], int)
        or payload["execution_generation"] < 1
    ):
        raise MlipError("NEP training success marker has an invalid execution_generation")
    if (
        isinstance(payload["generation_offset"], bool)
        or not isinstance(payload["generation_offset"], int)
        or payload["generation_offset"] < 0
    ):
        raise MlipError("NEP training success marker has an invalid generation_offset")
    if (
        isinstance(payload["segment_target_generations"], bool)
        or not isinstance(payload["segment_target_generations"], int)
        or payload["segment_target_generations"] <= 0
    ):
        raise MlipError("NEP training success marker has an invalid segment target")
    if (
        isinstance(payload["process_exit_code"], bool)
        or not isinstance(payload["process_exit_code"], int)
        or payload["process_exit_code"] != 0
    ):
        raise MlipError("NEP training success marker does not record a successful process")
    timestamp = payload["completion_timestamp"]
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise MlipError("NEP training success marker has no completion timestamp")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MlipError(
            "NEP training success marker has an invalid completion timestamp"
        ) from exc
    return payload


def _target_generation(run_directory: Path) -> int:
    input_path = Path(run_directory) / "nep.in"
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MlipError(f"Could not read the attempt-local NEP input: {input_path}") from exc
    matches = [
        match
        for line in lines
        if (match := _GENERATION_LINE.fullmatch(line)) is not None
    ]
    if len(matches) != 1:
        raise MlipError(
            f"Attempt-local NEP input must contain exactly one generation target: {input_path}"
        )
    return int(matches[0].group(1))


def _observed_generation(run_directory: Path) -> int | None:
    history = parse_progress_history(run_directory)
    return history[-1].generation if history else None


def parse_completion(run_directory: Path) -> TrainingCompletion:
    marker = _read_success_marker(run_directory)
    artifacts = model_artifact_paths(run_directory)
    target = _target_generation(run_directory)
    observed = _observed_generation(run_directory)
    marker_matches_segment = marker is not None and (
        marker["segment_target_generations"] == target
        and marker["generation_offset"] >= 0
    )
    completed = (
        marker is not None
        and marker_matches_segment
        and marker["process_exit_code"] == 0
        and len(artifacts) == 1
        and observed is not None
        and observed >= target
    )
    return TrainingCompletion(
        completed=completed,
        artifact_paths=artifacts,
        segment_target_generations=target,
        segment_observed_generations=observed,
        process_exit_code=None if marker is None else int(marker["process_exit_code"]),
        success_marker=marker,
    )


def parse_checkpoint(run_directory: Path) -> TrainingCheckpoint | None:
    """Validate one attempt-local NEP restart checkpoint."""

    directory = Path(run_directory)
    restart_path = directory / "nep.restart"
    model_path = directory / "nep.txt"
    restart_exists = restart_path.exists()
    model_exists = model_path.exists()
    if not restart_exists and not model_exists:
        return None
    if not restart_exists or not model_exists:
        raise MlipError("NEP checkpoint requires both nep.restart and nep.txt")
    if restart_path.stat().st_size <= 0 or model_path.stat().st_size <= 0:
        raise MlipError("NEP checkpoint artifacts must be non-empty")
    target = _target_generation(directory)
    history = parse_progress_history(directory)
    if not history:
        raise MlipError("NEP checkpoint requires local loss.out progress")
    local_generation = history[-1].generation
    if local_generation <= 0 or local_generation > target:
        raise MlipError(
            "NEP checkpoint local generation is outside the attempt segment budget"
        )
    return TrainingCheckpoint(
        local_generation=local_generation,
        restart_path=restart_path.resolve(),
        restart_sha256=sha256_file(restart_path),
        model_path=model_path.resolve(),
        model_sha256=sha256_file(model_path),
        metadata={
            "segment_target_generations": target,
            "segment_observed_generations": local_generation,
        },
    )


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
    "SUCCESS_MARKER_NAME",
    "SUCCESS_MARKER_SCHEMA",
    "collect_model_artifacts",
    "model_artifact_paths",
    "parse_checkpoint",
    "parse_completion",
]
