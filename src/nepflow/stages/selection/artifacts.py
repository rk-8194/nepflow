"""Persistent selection artifacts."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Any, Mapping

from ase.io import write as ase_write

from nepflow.config.models import TEST_SELECTION_POLICIES
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
    test_selection_policy: str = "candidate_mean_fps_legacy",
    test_selection_version: str = "candidate-mean-fps-v1",
    test_selection_provenance: Mapping[str, Any] | None = None,
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
    if set(train_indices) & set(test_indices):
        raise ArtifactError("Selection artifact train/test candidate indices overlap")
    try:
        train_structure_set = {all_structure_ids[index] for index in train_indices}
        test_structure_set = {all_structure_ids[index] for index in test_indices}
    except (IndexError, TypeError) as exc:
        raise ArtifactError("Selection artifact index is out of range") from exc
    if train_structure_set & test_structure_set:
        raise ArtifactError("Selection artifact train/test physical IDs overlap")
    anchor_indices = set(train_anchor_indices or [])
    if anchor_indices & set(test_indices):
        raise ArtifactError("Selection artifact test indices overlap mandatory anchors")
    if {all_structure_ids[index] for index in anchor_indices} & test_structure_set:
        raise ArtifactError("Selection artifact test physical IDs overlap mandatory anchors")
    if (
        not isinstance(test_selection_policy, str)
        or test_selection_policy not in TEST_SELECTION_POLICIES
    ):
        raise ArtifactError("Selection artifact test-selection policy is unsupported")
    if not isinstance(test_selection_version, str) or not test_selection_version.strip():
        raise ArtifactError("Selection artifact test-selection version is blank")
    if test_selection_provenance is not None and not isinstance(
        test_selection_provenance, Mapping
    ):
        raise ArtifactError("Selection artifact test-selection provenance is malformed")
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
    if isinstance(entropy_diagnostics, Mapping):
        objective = entropy_diagnostics.get("objective")
        numeric = objective.get("numeric_arrays") if isinstance(objective, Mapping) else None
        numeric_name = numeric.get("path") if isinstance(numeric, Mapping) else None
        if isinstance(numeric_name, str) and numeric_name.strip():
            numeric_path = Path(numeric_name)
            if numeric_path.is_absolute():
                numeric_path = selected_dir / numeric_path.name
            else:
                numeric_path = selected_dir / numeric_path
            try:
                numeric_path.resolve().relative_to(selected_dir.resolve())
            except ValueError as exc:
                raise ArtifactError(
                    "Selection entropy diagnostics artifact path escapes the artifact "
                    "directory"
                ) from exc
            previous[numeric_path] = (
                numeric_path.read_bytes() if numeric_path.is_file() else None
            )
    try:
        write_extxyz(train_path, selected_structures(train_indices))
        logger.info("  Training set saved to %s", train_path)

        write_extxyz(test_path, selected_structures(test_indices))
        logger.info("  Test set saved to %s", test_path)

        diagnostics_payload = None if entropy_diagnostics is None else dict(entropy_diagnostics)
        if isinstance(diagnostics_payload, dict):
            objective_payload = diagnostics_payload.get("objective")
            if isinstance(objective_payload, Mapping):
                objective_payload = dict(objective_payload)
                numeric_payload = objective_payload.get("numeric_arrays")
                if isinstance(numeric_payload, Mapping):
                    numeric_payload = dict(numeric_payload)
                    numeric_path = numeric_payload.get("path")
                    if isinstance(numeric_path, str) and Path(numeric_path).is_absolute():
                        numeric_payload["path"] = Path(numeric_path).name
                    objective_payload["numeric_arrays"] = numeric_payload
                diagnostics_payload["objective"] = objective_payload
        artifact_records: dict[str, Any] = {
            "train.xyz": {
                "path": "train.xyz",
                "sha256": sha256_file(train_path, required=True),
            },
            "test.xyz": {
                "path": "test.xyz",
                "sha256": sha256_file(test_path, required=True),
            },
        }
        if diagnostics_payload is not None:
            objective = diagnostics_payload.get("objective")
            numeric = objective.get("numeric_arrays") if isinstance(objective, Mapping) else None
            numeric_path = numeric.get("path") if isinstance(numeric, Mapping) else None
            if (
                isinstance(numeric, Mapping)
                and numeric.get("storage") == "file"
                and isinstance(numeric_path, str)
            ):
                diagnostic_path = selected_dir / numeric_path
                if not diagnostic_path.is_file():
                    raise ArtifactError(
                        "Selection entropy diagnostics numeric artifact is missing: "
                        f"{diagnostic_path}"
                    )
                artifact_records[Path(numeric_path).name] = {
                    "path": Path(numeric_path).name,
                    "sha256": sha256_file(diagnostic_path, required=True),
                }
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
                "test_selection_policy": test_selection_policy,
                "test_selection_version": test_selection_version,
                "test_selection_provenance": dict(test_selection_provenance or {}),
                "train_anchor_count": len(set(train_anchor_indices or [])),
                "train_anchor_candidate_ids": [
                    all_candidate_ids[index] for index in sorted(set(train_anchor_indices or []))
                ],
                "train_acquisition_order": list(train_acquisition_order or []),
                "entropy_diagnostics": (
                    diagnostics_payload
                ),
                "artifacts": artifact_records,
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
    train_structure_ids = manifest.get("train_structure_ids", [])
    test_structure_ids = manifest.get("test_structure_ids", [])
    if set(train_structure_ids) & set(test_structure_ids):
        raise ArtifactError(f"Selection artifact train/test physical IDs overlap: {path}")
    policy = manifest.get("test_selection_policy", "candidate_mean_fps_legacy")
    version = manifest.get("test_selection_version", "candidate-mean-fps-v1")
    provenance = manifest.get("test_selection_provenance", {})
    if not isinstance(policy, str) or policy not in TEST_SELECTION_POLICIES:
        raise ArtifactError(f"Selection artifact test-selection policy is malformed: {path}")
    if not isinstance(version, str) or not version.strip():
        raise ArtifactError(f"Selection artifact test-selection version is malformed: {path}")
    if not isinstance(provenance, dict):
        raise ArtifactError(f"Selection artifact test-selection provenance is malformed: {path}")
    diagnostics = manifest.get("entropy_diagnostics")
    if manifest.get("algorithm_id") == "information_entropy" and diagnostics is None:
        raise ArtifactError(f"Selection entropy diagnostics are missing: {path}")
    if diagnostics is not None:
        try:
            from .algorithms.information_entropy.diagnostics import EntropyScientificDiagnostics

            record = EntropyScientificDiagnostics.from_manifest(
                diagnostics,
                artifact_root=selected_dir,
            )
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
    diagnostics = manifest.get("entropy_diagnostics")
    if isinstance(diagnostics, Mapping):
        objective = diagnostics.get("objective")
        numeric = objective.get("numeric_arrays") if isinstance(objective, Mapping) else None
        if isinstance(numeric, Mapping) and numeric.get("storage") == "file":
            name = numeric.get("path")
            if not isinstance(name, str) or Path(name).is_absolute():
                raise ArtifactError(
                    f"Selection entropy diagnostics artifact path is malformed: {path}"
                )
            diagnostic_path = selected_dir / name
            try:
                diagnostic_path.resolve().relative_to(selected_dir.resolve())
            except ValueError as exc:
                raise ArtifactError(
                    "Selection entropy diagnostics artifact path escapes the artifact "
                    f"directory: {path}"
                ) from exc
            record = artifacts.get(Path(name).name)
            if (
                not diagnostic_path.is_file()
                or not isinstance(record, Mapping)
                or record.get("sha256") != sha256_file(diagnostic_path, required=True)
            ):
                raise ArtifactError(
                    "Selection entropy diagnostics artifact changed or is missing: "
                    f"{diagnostic_path}"
                )
    return manifest


def verify_selection_artifacts(
    project_dir: Path,
    *,
    algorithm_id: str = "fps",
    require_reports: bool = True,
) -> dict:
    """Verify published selection artifacts and required presentation files."""

    project_dir = Path(project_dir)
    manifest = read_selection_manifest(project_dir)
    selected_dir = project_dir / "structures" / "selected"
    reports_dir = project_dir / "reports"
    if require_reports:
        descriptor_report = reports_dir / "descriptor_space.png"
        if not descriptor_report.is_file() or descriptor_report.stat().st_size == 0:
            raise ArtifactError(
                f"Selection descriptor report is missing or empty: {descriptor_report}"
            )
        if algorithm_id == "information_entropy":
            diagnostics_report = reports_dir / "entropy_diagnostics.md"
            if not diagnostics_report.is_file() or diagnostics_report.stat().st_size == 0:
                raise ArtifactError(
                    "Selection entropy diagnostics report is missing or empty: "
                    f"{diagnostics_report}"
                )
    for filename in ("train.xyz", "test.xyz", SELECTION_MANIFEST_FILENAME):
        path = selected_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ArtifactError(f"Selection artifact is missing or empty: {path}")
    return manifest


__all__ = [
    "SELECTION_ARTIFACT_SCHEMA",
    "LEGACY_SELECTION_ARTIFACT_SCHEMA",
    "SELECTION_MANIFEST_FILENAME",
    "read_selection_manifest",
    "verify_selection_artifacts",
    "write_selected_structures",
]
