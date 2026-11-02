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
from nepflow.mlip.backend import (
    CollectedModelArtifacts,
    TrainingCompletion,
    TrainingInput,
)

from .metrics import parse_progress_line

SUCCESS_MARKER_NAME = ".nepflow_training_success.json"
SUCCESS_MARKER_SCHEMA = "nepflow.training_success.v1"
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
    progress_path = Path(run_directory) / "loss.out"
    if not progress_path.is_file():
        return None
    try:
        lines = progress_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MlipError(f"Could not read NEP progress evidence: {progress_path}") from exc
    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty:
        return None
    parsed = [parse_progress_line(line) for line in nonempty]
    if any(progress is None for progress in parsed):
        raise MlipError(f"Malformed NEP progress evidence: {progress_path}")
    valid = [progress for progress in parsed if progress is not None]
    return valid[-1].generation


def parse_completion(run_directory: Path) -> TrainingCompletion:
    marker = _read_success_marker(run_directory)
    artifacts = model_artifact_paths(run_directory)
    target = _target_generation(run_directory)
    observed = _observed_generation(run_directory)
    completed = (
        marker is not None
        and marker["process_exit_code"] == 0
        and len(artifacts) == 1
        and observed is not None
        and observed >= target
    )
    return TrainingCompletion(
        completed=completed,
        artifact_paths=artifacts,
        target_generation=target,
        observed_generation=observed,
        process_exit_code=None if marker is None else int(marker["process_exit_code"]),
        success_marker=marker,
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
    "parse_completion",
]
