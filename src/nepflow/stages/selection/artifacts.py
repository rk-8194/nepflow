"""Persistent selection artifacts."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

from ase.io import write as ase_write

from nepflow.domain.identities import calculate_structure_id
from nepflow.errors import ArtifactError
from nepflow.io.atomic import atomic_write_bytes, atomic_write_text
from nepflow.io.hashing import sha256_file
from nepflow.io.json import read_json_object, write_json

logger = logging.getLogger(__name__)
SELECTION_ARTIFACT_SCHEMA = "nepflow.selection_artifact.v1"
SELECTION_MANIFEST_FILENAME = "selection_manifest.json"


def _structure_id(structure: object) -> str:
    return calculate_structure_id(structure)


def write_selected_structures(
    project_dir: Path,
    ase_structures: list,
    train_indices: list[int],
    test_indices: list[int],
) -> tuple[Path, Path]:
    """Write selected train/test structures using the established paths."""

    project_dir = Path(project_dir)
    selected_dir = project_dir / "structures" / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)

    def write_extxyz(path: Path, structures: list) -> None:
        # ASE accepts a text file object for extxyz.  Rendering in memory lets
        # the repository-owned atomic writer publish the complete artifact in
        # one replace operation instead of exposing a partially written file.
        rendered = StringIO()
        ase_write(rendered, structures, format="extxyz")
        atomic_write_text(path, rendered.getvalue(), encoding="utf-8")

    train_path = selected_dir / "train.xyz"
    test_path = selected_dir / "test.xyz"
    manifest_path = selected_dir / SELECTION_MANIFEST_FILENAME
    previous = {
        path: path.read_bytes() if path.is_file() else None
        for path in (train_path, test_path, manifest_path)
    }
    try:
        write_extxyz(train_path, [ase_structures[index] for index in train_indices])
        logger.info("  Training set saved to %s", train_path)

        write_extxyz(test_path, [ase_structures[index] for index in test_indices])
        logger.info("  Test set saved to %s", test_path)

        write_json(
            manifest_path,
            {
                "schema_version": SELECTION_ARTIFACT_SCHEMA,
                "train_structure_ids": [
                    _structure_id(ase_structures[index]) for index in train_indices
                ],
                "test_structure_ids": [
                    _structure_id(ase_structures[index]) for index in test_indices
                ],
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
    if manifest.get("schema_version") != SELECTION_ARTIFACT_SCHEMA:
        raise ArtifactError(
            f"Unsupported selection artifact schema: {manifest.get('schema_version')!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ArtifactError(f"Selection artifact manifest is malformed: {path}")
    for key in ("train_structure_ids", "test_structure_ids"):
        values = manifest.get(key)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ArtifactError(f"Selection artifact manifest is malformed: {path}")
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
    "SELECTION_MANIFEST_FILENAME",
    "read_selection_manifest",
    "write_selected_structures",
]
