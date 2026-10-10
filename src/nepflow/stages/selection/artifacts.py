"""Persistent selection artifacts."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Any, Mapping

from ase.io import write as ase_write

from nepflow.domain.identities import CANDIDATE_IDENTITY_SCHEMA, STRUCTURE_IDENTITY_SCHEMA
from nepflow.errors import ArtifactError
from nepflow.io.atomic import atomic_write_bytes, atomic_write_text
from nepflow.io.hashing import sha256_file
from nepflow.io.json import read_json_object, write_json

from .persistence import candidate_ids as ordered_candidate_ids
from .persistence import structure_ids as ordered_structure_ids

logger = logging.getLogger(__name__)
SELECTION_ARTIFACT_SCHEMA = "nepflow.selection_artifact.v2"
LEGACY_SELECTION_ARTIFACT_SCHEMA = "nepflow.selection_artifact.v1"
SELECTION_MANIFEST_FILENAME = "selection_manifest.json"


def write_selected_structures(
    project_dir: Path,
    ase_structures: list,
    train_indices: list[int],
    test_indices: list[int],
    candidate_ids: list[str] | None = None,
    structure_ids: list[str] | None = None,
    *,
    algorithm_id: str = "fps",
    train_anchor_indices: list[int] | None = None,
    train_acquisition_order: list[str] | None = None,
    entropy_diagnostics: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write selected train/test structures using the established paths."""

    project_dir = Path(project_dir)
    all_candidate_ids = (
        ordered_candidate_ids(ase_structures) if candidate_ids is None else list(candidate_ids)
    )
    all_structure_ids = (
        ordered_structure_ids(ase_structures) if structure_ids is None else list(structure_ids)
    )
    if len(all_candidate_ids) != len(ase_structures) or len(all_structure_ids) != len(
        ase_structures
    ):
        raise ArtifactError("Selection artifact identity counts do not match structures")
    if len(set(all_candidate_ids)) != len(all_candidate_ids):
        raise ArtifactError("Selection artifact candidate IDs are not unique")
    if all(
        candidate_id == structure_id
        for candidate_id, structure_id in zip(all_candidate_ids, all_structure_ids)
    ):
        artifact_schema = LEGACY_SELECTION_ARTIFACT_SCHEMA
    else:
        artifact_schema = SELECTION_ARTIFACT_SCHEMA
    selected_dir = project_dir / "structures" / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)

    def write_extxyz(path: Path, structures: list) -> None:
        # ASE accepts a text file object for extxyz.  Rendering in memory lets
        # the repository-owned atomic writer publish the complete artifact in
        # one replace operation instead of exposing a partially written file.
        rendered = StringIO()
        ase_write(rendered, structures, format="extxyz")
        atomic_write_text(path, rendered.getvalue(), encoding="utf-8")

    def selected_structures(indices: list[int]) -> list:
        result = []
        for index in indices:
            if index < 0 or index >= len(ase_structures):
                raise ArtifactError(f"Selection artifact index is out of range: {index}")
            structure = ase_structures[index].copy()
            structure.info["structure_id"] = all_structure_ids[index]
            structure.info["structure_id_version"] = STRUCTURE_IDENTITY_SCHEMA
            structure.info["candidate_id"] = all_candidate_ids[index]
            structure.info["candidate_id_version"] = CANDIDATE_IDENTITY_SCHEMA
            result.append(structure)
        return result

    train_path = selected_dir / "train.xyz"
    test_path = selected_dir / "test.xyz"
    manifest_path = selected_dir / SELECTION_MANIFEST_FILENAME
    previous = {
        path: path.read_bytes() if path.is_file() else None
        for path in (train_path, test_path, manifest_path)
    }
    try:
        write_extxyz(train_path, selected_structures(train_indices))
        logger.info("  Training set saved to %s", train_path)

        write_extxyz(test_path, selected_structures(test_indices))
        logger.info("  Test set saved to %s", test_path)

        write_json(
            manifest_path,
            {
                "schema_version": artifact_schema,
                "candidate_identity_schema": CANDIDATE_IDENTITY_SCHEMA,
                "train_candidate_ids": [all_candidate_ids[index] for index in train_indices],
                "test_candidate_ids": [all_candidate_ids[index] for index in test_indices],
                "train_structure_ids": [all_structure_ids[index] for index in train_indices],
                "test_structure_ids": [all_structure_ids[index] for index in test_indices],
                "algorithm_id": algorithm_id,
                "train_anchor_count": len(set(train_anchor_indices or [])),
                "train_anchor_candidate_ids": [
                    all_candidate_ids[index] for index in sorted(set(train_anchor_indices or []))
                ],
                "train_acquisition_order": list(train_acquisition_order or []),
                "entropy_diagnostics": (
                    None if entropy_diagnostics is None else dict(entropy_diagnostics)
                ),
                "artifacts": {
                    "train.xyz": {
                        "path": "train.xyz",
                        "sha256": sha256_file(train_path, required=True),
                    },
                    "test.xyz": {
                        "path": "test.xyz",
                        "sha256": sha256_file(test_path, required=True),
                    },
                },
            },
        )
    except BaseException:
        for path, content in previous.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write_bytes(path, content)
        raise
    return train_path, test_path


def read_selection_manifest(project_dir: Path) -> dict:
    """Read and verify the identity manifest for selected structure files."""

    selected_dir = Path(project_dir) / "structures" / "selected"
    path = selected_dir / SELECTION_MANIFEST_FILENAME
    manifest = read_json_object(path, error_type=ArtifactError)
    if manifest.get("schema_version") not in {
        SELECTION_ARTIFACT_SCHEMA,
        LEGACY_SELECTION_ARTIFACT_SCHEMA,
    }:
        raise ArtifactError(
            f"Unsupported selection artifact schema: {manifest.get('schema_version')!r}"
        )
    identity_schema = manifest.get("candidate_identity_schema")
    if identity_schema is not None and identity_schema != CANDIDATE_IDENTITY_SCHEMA:
        raise ArtifactError(f"Unsupported candidate identity schema: {identity_schema!r}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ArtifactError(f"Selection artifact manifest is malformed: {path}")
    for key in ("train_structure_ids", "test_structure_ids"):
        values = manifest.get(key)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ArtifactError(f"Selection artifact manifest is malformed: {path}")
    for key in ("train_candidate_ids", "test_candidate_ids"):
        values = manifest.get(key)
        if values is None and manifest.get("schema_version") == LEGACY_SELECTION_ARTIFACT_SCHEMA:
            continue
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ArtifactError(f"Selection artifact manifest is malformed: {path}")
    train_candidate_ids = manifest.get("train_candidate_ids")
    test_candidate_ids = manifest.get("test_candidate_ids")
    if train_candidate_ids is None:
        train_candidate_ids = manifest["train_structure_ids"]
        manifest["train_candidate_ids"] = list(train_candidate_ids)
    if test_candidate_ids is None:
        test_candidate_ids = manifest["test_structure_ids"]
        manifest["test_candidate_ids"] = list(test_candidate_ids)
    if set(train_candidate_ids) & set(test_candidate_ids):
        raise ArtifactError(f"Selection artifact train/test candidate IDs overlap: {path}")
    diagnostics = manifest.get("entropy_diagnostics")
    if manifest.get("algorithm_id") == "information_entropy" and diagnostics is None:
        raise ArtifactError(f"Selection entropy diagnostics are missing: {path}")
    if diagnostics is not None:
        try:
            from .algorithms.information_entropy.diagnostics import EntropyScientificDiagnostics

            record = EntropyScientificDiagnostics.from_manifest(diagnostics)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"Selection entropy diagnostics are malformed: {path}") from exc
        if tuple(record.selected_candidate_ids) != tuple(train_candidate_ids):
            raise ArtifactError(
                f"Selection entropy diagnostics training IDs disagree with manifest: {path}"
            )
        if tuple(record.test_candidate_ids) != tuple(test_candidate_ids):
            raise ArtifactError(
                f"Selection entropy diagnostics test IDs disagree with manifest: {path}"
            )
    for filename in ("train.xyz", "test.xyz"):
        record = artifacts.get(filename)
        artifact_path = selected_dir / filename
        if (
            not isinstance(record, dict)
            or record.get("path") != filename
            or not artifact_path.is_file()
            or sha256_file(artifact_path, required=True) != record.get("sha256")
        ):
            raise ArtifactError(f"Selection artifact changed or is missing: {artifact_path}")
    return manifest


__all__ = [
    "SELECTION_ARTIFACT_SCHEMA",
    "LEGACY_SELECTION_ARTIFACT_SCHEMA",
    "SELECTION_MANIFEST_FILENAME",
    "read_selection_manifest",
    "write_selected_structures",
]
