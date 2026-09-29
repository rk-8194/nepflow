"""Persisted identity and integrity records for NEP model runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_RUN_MANIFEST_FILENAME = "model_run_manifest.json"
MODEL_RUN_MANIFEST_SCHEMA = "nepflow.model_run_manifest.v1"


class ModelManifestError(ValueError):
    """Raised when a model-run manifest is missing or inconsistent."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ModelManifestError(f"Cannot hash manifest artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_model_run_manifest(path: Path) -> dict[str, Any]:
    """Read one manifest and reject malformed JSON or non-object content."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Model-run manifest not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelManifestError(f"Could not read model-run manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelManifestError(f"Model-run manifest must be an object: {path}")
    return value


def write_model_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write one complete manifest atomically enough for local workflow use."""
    _write_json(path, manifest)


def create_model_run_manifest(
    *,
    potential_path: Path,
    dataset_path: Path,
    dataset_id: str,
    nep_in_path: Path,
    hyperparameters_hash: str,
) -> dict[str, Any]:
    """Create the prepared manifest for a training run."""
    nep_in_path = nep_in_path.resolve()
    potential_path = potential_path.resolve()
    if not nep_in_path.is_file():
        raise ModelManifestError(f"Training input does not exist: {nep_in_path}")
    if not dataset_id:
        raise ModelManifestError("Cannot create a model run without dataset_id")

    nep_in_hash = sha256_file(nep_in_path)
    created_at = _now()
    identity_payload = {
        "schema_version": MODEL_RUN_MANIFEST_SCHEMA,
        "dataset_id": dataset_id,
        "nep_in_sha256": nep_in_hash,
        "hyperparameters_hash": hyperparameters_hash,
        "created_at": created_at,
    }
    model_run_id = "model_run_" + hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact_path = (potential_path / "nep.txt").resolve()
    manifest = {
        "schema_version": MODEL_RUN_MANIFEST_SCHEMA,
        "model_run_id": model_run_id,
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path.resolve()),
        "potential_artifact_path": str(artifact_path),
        "potential_artifact_sha256": None,
        "nep_in_path": str(nep_in_path),
        "nep_in_sha256": nep_in_hash,
        "status": "prepared",
        "completion_evidence": {
            "status": "pending",
            "artifact_exists": False,
            "artifact_sha256": None,
        },
        "created_at": created_at,
        "updated_at": _now(),
    }
    write_model_run_manifest(potential_path / MODEL_RUN_MANIFEST_FILENAME, manifest)
    return manifest


def update_model_run_status(
    potential_path: Path,
    status: str,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    """Record training completion/failure and the final model artifact hash."""
    manifest_path = potential_path / MODEL_RUN_MANIFEST_FILENAME
    manifest = read_model_run_manifest(manifest_path)
    artifact_path = Path(str(manifest.get("potential_artifact_path", "")))
    if status == "completed":
        if not artifact_path.is_file():
            alternate = sorted(
                path
                for path in potential_path.glob("nep*.txt")
                if path.is_file() and path.stat().st_size > 0
            )
            if alternate:
                artifact_path = alternate[0].resolve()
                manifest["potential_artifact_path"] = str(artifact_path)
        if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
            raise ModelManifestError(
                f"Completed model run has no non-empty potential artifact: {artifact_path}"
            )
        artifact_hash = sha256_file(artifact_path)
        manifest["potential_artifact_sha256"] = artifact_hash
        evidence = {
            "status": "completed",
            "artifact_exists": True,
            "artifact_sha256": artifact_hash,
            "completed_at": _now(),
        }
    else:
        evidence = {
            "status": status,
            "artifact_exists": artifact_path.is_file(),
            "artifact_sha256": sha256_file(artifact_path) if artifact_path.is_file() else None,
        }
        if error:
            evidence["error"] = error
    manifest["status"] = status
    manifest["completion_evidence"] = evidence
    manifest["updated_at"] = _now()
    write_model_run_manifest(manifest_path, manifest)
    return manifest


def validate_model_run_manifest(
    manifest_path: Path,
    *,
    expected_model_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate identity, completion status, paths, and content hashes."""
    manifest = read_model_run_manifest(manifest_path)
    required = (
        "model_run_id",
        "dataset_id",
        "dataset_path",
        "potential_artifact_path",
        "potential_artifact_sha256",
        "nep_in_path",
        "nep_in_sha256",
        "status",
        "completion_evidence",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ModelManifestError(
            f"Model-run manifest {manifest_path} is missing: {', '.join(missing)}"
        )
    if expected_model_run_id and manifest["model_run_id"] != expected_model_run_id:
        raise ModelManifestError(
            f"Manifest identity mismatch: expected {expected_model_run_id}, "
            f"found {manifest['model_run_id']}"
        )
    if manifest["status"] != "completed":
        raise ModelManifestError(
            f"Model run {manifest['model_run_id']} is not completed "
            f"(status={manifest['status']!r})"
        )

    dataset_path = Path(str(manifest["dataset_path"]))
    metadata_path = dataset_path / ".dataset"
    if not dataset_path.is_dir() or not metadata_path.is_file():
        raise ModelManifestError(f"Manifest dataset artifact is missing: {dataset_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelManifestError(f"Could not read dataset manifest {metadata_path}: {exc}") from exc
    if metadata.get("dataset_id") != manifest["dataset_id"]:
        raise ModelManifestError(
            f"Dataset identity mismatch for {dataset_path}: "
            f"expected {manifest['dataset_id']}, found {metadata.get('dataset_id')}"
        )

    for path_key, hash_key in (
        ("potential_artifact_path", "potential_artifact_sha256"),
        ("nep_in_path", "nep_in_sha256"),
    ):
        artifact = Path(str(manifest[path_key]))
        expected_hash = manifest[hash_key]
        if not artifact.is_file():
            raise ModelManifestError(f"Manifest artifact is missing: {artifact}")
        if not expected_hash or sha256_file(artifact) != expected_hash:
            raise ModelManifestError(f"Manifest hash mismatch for {artifact}")

    return manifest


def find_model_run_manifest(project_dir: Path, model_run_id: str) -> Path:
    """Find exactly one manifest by explicit scientific model-run identity."""
    if not model_run_id:
        raise ValueError("model_run_id is required; latest model discovery is disabled")
    potentials_dir = project_dir / "nep" / "potentials"
    if not potentials_dir.is_dir():
        raise FileNotFoundError(f"No canonical NEP potential directory found: {potentials_dir}")

    matches = []
    for manifest_path in potentials_dir.rglob(MODEL_RUN_MANIFEST_FILENAME):
        try:
            manifest = read_model_run_manifest(manifest_path)
        except ModelManifestError:
            continue
        if manifest.get("model_run_id") == model_run_id:
            matches.append(manifest_path)
    if not matches:
        raise FileNotFoundError(f"No model-run manifest found for model_run_id={model_run_id}")
    if len(matches) != 1:
        raise ModelManifestError(
            f"Multiple manifests found for model_run_id={model_run_id}: {matches}"
        )
    return matches[0]
