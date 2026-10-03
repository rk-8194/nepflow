"""Finalize authoritative validation artifacts for the legacy GPUMD stage."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Tuple

from nepflow.io.hashing import sha256_file
from nepflow.mlip.nep.artifacts import NepArtifactError
from nepflow.stages.validation.resolution import resolve_model_dataset

logger = logging.getLogger("nepflow.validate")


def finalize_nep_potential(
    project_dir: Path,
    model_run_id: str | None = None,
) -> Tuple[Path, str]:
    """Finalize the selected model artifact for the legacy GPUMD stage.

    Model/dataset resolution remains authoritative and explicit. Scientific
    case preparation is owned by ``prepare_validation_cases`` in the canonical
    validation stage.
    """

    if not model_run_id:
        raise ValueError("model_run_id is required to finalize a potential")
    resolved = resolve_model_dataset(project_dir, model_run_id)
    source_artifact = resolved.model_path.resolve()
    potential_src = source_artifact.parent
    dataset_path = resolved.dataset_path.resolve()

    gpumd_potential_dir = (
        project_dir / "gpumd" / dataset_path.name / potential_src.name
    )
    gpumd_potential_dir.mkdir(parents=True, exist_ok=True)

    destination = gpumd_potential_dir / "nep.txt"
    if not source_artifact.exists():
        raise FileNotFoundError(f"nep.txt not found at {source_artifact}")
    if sha256_file(
        source_artifact,
        required=True,
        error_type=NepArtifactError,
    ) != resolved.model_artifact.sha256:
        raise NepArtifactError(
            f"Manifest-bound artifact changed after validation: {source_artifact}"
        )

    if destination.exists():
        if sha256_file(
            destination,
            required=True,
            error_type=NepArtifactError,
        ) != resolved.model_artifact.sha256:
            raise NepArtifactError(
                f"Existing finalized artifact does not match manifest: {destination}"
            )
        logger.info("nep.txt already exists at %s", destination)
    else:
        shutil.copy2(source_artifact, destination)
        logger.info("Copied nep.txt from %s to %s", source_artifact, destination)

    return gpumd_potential_dir, dataset_path.name
