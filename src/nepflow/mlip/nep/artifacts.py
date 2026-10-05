"""Content-addressed NEP model-run manifests and artifact integrity."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nepflow.domain.identities import ArtifactIdentity, ModelRunIdentity
from nepflow.domain.models import ModelArtifactMetadata, ModelRunRecord
from nepflow.errors import ArtifactError
from nepflow.io.hashing import sha256_file
from nepflow.io.json import read_json_object, write_json

MODEL_RUN_MANIFEST_FILENAME = "model_run_manifest.json"
MODEL_RUN_MANIFEST_SCHEMA = "nepflow.model_run_manifest.v1"
MODEL_RUN_IDENTITY_SCHEMA = "nepflow.model_run_identity.v1"


class NepArtifactError(ArtifactError):
    """Raised when a NEP model-run manifest or artifact is inconsistent."""


def parse_nep_cutoff_angstrom(nep_path: Path) -> float:
    """Read the positive cutoff from an authoritative NEP artifact."""

    for line in Path(nep_path).read_text(encoding="utf-8").splitlines():
        values = line.split("#", 1)[0].split()
        if values and values[0].lower() == "cutoff" and len(values) >= 2:
            try:
                cutoff = float(values[1])
            except ValueError as exc:
                raise NepArtifactError(f"Invalid NEP cutoff in {nep_path}") from exc
            if math.isfinite(cutoff) and cutoff > 0:
                return cutoff
            break
    raise NepArtifactError(f"Could not find a positive NEP cutoff in {nep_path}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_model_run_manifest(path: Path) -> dict[str, Any]:
    try:
        value = read_json_object(path, error_type=NepArtifactError)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Model-run manifest not found: {path}") from exc
    if value.get("schema_version") != MODEL_RUN_MANIFEST_SCHEMA:
        raise NepArtifactError(
            f"Unsupported model-run manifest schema: {value.get('schema_version')!r}"
        )
    if value.get("identity_schema_version") != MODEL_RUN_IDENTITY_SCHEMA:
        raise NepArtifactError(
            f"Unsupported model-run identity schema: {value.get('identity_schema_version')!r}"
        )
    return value


def write_model_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != MODEL_RUN_MANIFEST_SCHEMA:
        raise NepArtifactError(
            f"Unsupported model-run manifest schema: {manifest.get('schema_version')!r}"
        )
    write_json(path, manifest)


def compute_model_run_id(
    *,
    dataset_id: str,
    nep_in_sha256: str,
    hyperparameters_hash: str,
) -> str:
    """Compute identity from exact dataset/input content and effective settings."""

    return ModelRunIdentity.from_inputs(
        dataset_id=dataset_id,
        nep_in_sha256=nep_in_sha256,
        hyperparameters_hash=hyperparameters_hash,
    ).model_run_id


def create_model_run_manifest(
    *,
    potential_path: Path,
    dataset_path: Path,
    dataset_id: str,
    nep_in_path: Path,
    hyperparameters_hash: str,
    state_store: Any | None = None,
) -> dict[str, Any]:
    potential_path = Path(potential_path)
    dataset_path = Path(dataset_path)
    nep_in_path = Path(nep_in_path)
    nep_in_path = nep_in_path.resolve()
    potential_path = potential_path.resolve()
    if not nep_in_path.is_file():
        raise NepArtifactError(f"Training input does not exist: {nep_in_path}")
    if not dataset_id:
        raise NepArtifactError("Cannot create a model run without dataset_id")
    nep_in_hash = sha256_file(nep_in_path, required=True, error_type=NepArtifactError)
    if nep_in_hash is None:
        raise NepArtifactError(f"Could not hash NEP input: {nep_in_path}")
    model_run_id = compute_model_run_id(
        dataset_id=dataset_id,
        nep_in_sha256=nep_in_hash,
        hyperparameters_hash=hyperparameters_hash,
    )
    identity = ModelRunIdentity.from_inputs(
        dataset_id=dataset_id,
        nep_in_sha256=nep_in_hash,
        hyperparameters_hash=hyperparameters_hash,
    )
    if identity.model_run_id != model_run_id:
        raise NepArtifactError("Model-run identity construction is inconsistent")
    if state_store is not None:
        state_store.upsert_model_run(
            ModelRunRecord(
                identity=identity,
                execution_metadata={
                    "dataset_path": str(dataset_path.resolve()),
                    "potential_path": str(potential_path),
                },
            ),
            status="prepared",
            started_at=_now(),
        )
    manifest = {
        "schema_version": MODEL_RUN_MANIFEST_SCHEMA,
        "model_run_id": model_run_id,
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path.resolve()),
        "identity_schema_version": MODEL_RUN_IDENTITY_SCHEMA,
        "hyperparameters_hash": hyperparameters_hash,
        "potential_artifact_path": str((potential_path / "nep.txt").resolve()),
        "potential_artifact_sha256": None,
        "nep_in_path": str(nep_in_path),
        "nep_in_sha256": nep_in_hash,
        "status": "prepared",
        "completion_evidence": {
            "status": "pending",
            "artifact_exists": False,
            "artifact_sha256": None,
        },
        "created_at": _now(),
        "updated_at": _now(),
    }
    write_model_run_manifest(potential_path / MODEL_RUN_MANIFEST_FILENAME, manifest)
    return manifest


def update_model_run_status(
    potential_path: Path,
    status: str,
    *,
    error: str | None = None,
    state_store: Any | None = None,
    model_run_id: str | None = None,
) -> dict[str, Any]:
    potential_path = Path(potential_path)
    manifest_path = potential_path / MODEL_RUN_MANIFEST_FILENAME
    manifest = read_model_run_manifest(manifest_path)
    if manifest.get("schema_version") != MODEL_RUN_MANIFEST_SCHEMA:
        raise NepArtifactError(
            f"Unsupported model-run manifest schema: {manifest.get('schema_version')}"
        )
    if manifest.get("identity_schema_version") != MODEL_RUN_IDENTITY_SCHEMA:
        raise NepArtifactError(
            f"Unsupported model-run identity schema: {manifest.get('identity_schema_version')}"
        )
    authoritative_identity: ModelRunIdentity | None = None
    execution_metadata: dict[str, Any] = {}
    if state_store is not None:
        requested_id = model_run_id or str(manifest.get("model_run_id", ""))
        if not requested_id:
            raise NepArtifactError("StateStore model-run update requires model_run_id")
        row = state_store.get_model_run(requested_id)
        if row is None:
            raise NepArtifactError(f"Unknown authoritative model run: {requested_id}")
        identity_payload = row.get("identity_json", row.get("identity", {}))
        if isinstance(row.get("execution_metadata"), dict):
            execution_metadata.update(row["execution_metadata"])
        if not isinstance(identity_payload, dict):
            raise NepArtifactError("StateStore model-run identity is malformed")
        authoritative_identity = ModelRunIdentity.from_inputs(
            dataset_id=str(identity_payload["dataset_id"]),
            nep_in_sha256=str(identity_payload["nep_in_sha256"]),
            hyperparameters_hash=str(identity_payload["hyperparameters_hash"]),
        )
        for key in (
            "model_run_id",
            "dataset_id",
            "nep_in_sha256",
            "hyperparameters_hash",
        ):
            expected = authoritative_identity.to_dict()[key]
            if manifest.get(key) != expected:
                raise NepArtifactError(
                    f"Filesystem model-run manifest conflicts with StateStore for {key}"
                )
        if manifest.get("identity_schema_version") != authoritative_identity.schema_version:
            raise NepArtifactError(
                "Filesystem model-run manifest conflicts with StateStore for "
                "identity_schema_version"
            )
        if manifest.get("schema_version") != MODEL_RUN_MANIFEST_SCHEMA:
            raise NepArtifactError(
                f"Unsupported model-run manifest schema: {manifest.get('schema_version')}"
            )
        stored_dataset_path = execution_metadata.get("dataset_path")
        if (
            stored_dataset_path
            and Path(str(manifest.get("dataset_path", ""))).resolve()
            != Path(str(stored_dataset_path)).resolve()
        ):
            raise NepArtifactError(
                "Filesystem model-run manifest dataset path conflicts with StateStore"
            )
        if requested_id != authoritative_identity.model_run_id:
            raise NepArtifactError("Requested model-run ID does not match StateStore identity")

    artifact_path = Path(str(manifest.get("potential_artifact_path", "")))
    completed_at = _now() if status == "completed" else None
    if status == "completed":
        if not artifact_path.is_file():
            alternate = sorted(
                path
                for path in potential_path.glob("nep*.txt")
                if path.is_file() and path.stat().st_size > 0
            )
            if len(alternate) != 1:
                raise NepArtifactError(
                    "Expected exactly one supported alternate NEP artifact when "
                    f"nep.txt is absent; found {len(alternate)}"
                )
            artifact_path = alternate[0].resolve()
            manifest["potential_artifact_path"] = str(artifact_path)
        if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
            raise NepArtifactError(
                f"Completed model run has no non-empty potential artifact: {artifact_path}"
            )
        artifact_hash = sha256_file(artifact_path, required=True, error_type=NepArtifactError)
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
            "artifact_sha256": sha256_file(
                artifact_path, required=False, error_type=NepArtifactError
            ),
        }
        if error:
            evidence["error"] = error
    if state_store is not None and authoritative_identity is not None:
        nep_artifact = ArtifactIdentity.from_file(
            "nep_input",
            Path(str(manifest["nep_in_path"])),
        )
        if status == "completed":
            model_artifact = ArtifactIdentity.from_file("nep_model", artifact_path)
            artifact = ModelArtifactMetadata(
                model=model_artifact,
                nep_in=nep_artifact,
                status="completed",
                completed_at=completed_at,
            )
        else:
            artifact = None
        if error:
            execution_metadata["error"] = error
        state_store.upsert_model_run(
            ModelRunRecord(
                identity=authoritative_identity,
                artifact=artifact,
                execution_metadata=execution_metadata or None,
            ),
            status=status,
            completed_at=completed_at,
        )
    manifest["status"] = status
    manifest["completion_evidence"] = evidence
    manifest["updated_at"] = _now()
    write_model_run_manifest(manifest_path, manifest)
    return manifest


def validate_model_run_manifest(
    manifest_path: Path,
    *,
    expected_model_run_id: str | None = None,
    state_store: Any | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = read_model_run_manifest(manifest_path)
    required = (
        "model_run_id",
        "dataset_id",
        "identity_schema_version",
        "hyperparameters_hash",
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
        raise NepArtifactError(
            f"Model-run manifest {manifest_path} is missing: {', '.join(missing)}"
        )
    if manifest.get("schema_version") != MODEL_RUN_MANIFEST_SCHEMA:
        raise NepArtifactError(
            f"Unsupported model-run manifest schema: {manifest.get('schema_version')}"
        )
    if expected_model_run_id and manifest["model_run_id"] != expected_model_run_id:
        raise NepArtifactError(
            f"Manifest identity mismatch: expected {expected_model_run_id}, "
            f"found {manifest['model_run_id']}"
        )
    if state_store is not None:
        row = state_store.get_model_run(str(manifest["model_run_id"]))
        if row is None:
            raise NepArtifactError(
                f"Model run {manifest['model_run_id']} is absent from StateStore"
            )
        identity_payload = row.get("identity_json", row.get("identity", {}))
        if not isinstance(identity_payload, dict):
            raise NepArtifactError("StateStore model-run identity is malformed")
        for key in (
            "model_run_id",
            "dataset_id",
            "nep_in_sha256",
            "hyperparameters_hash",
        ):
            if identity_payload.get(key) != manifest.get(key):
                raise NepArtifactError(
                    f"Filesystem model-run manifest conflicts with StateStore for {key}"
                )
        identity_schema = identity_payload.get("schema_version")
        if manifest.get("identity_schema_version") != identity_schema:
            raise NepArtifactError(
                "Filesystem model-run manifest conflicts with StateStore for "
                "identity_schema_version"
            )
        if row.get("status") != manifest.get("status"):
            raise NepArtifactError(
                f"Model-run status conflicts with StateStore: {row.get('status')!r}"
            )
        execution_metadata = row.get("execution_metadata", {})
        stored_dataset_path = (
            execution_metadata.get("dataset_path") if isinstance(execution_metadata, dict) else None
        )
        if (
            stored_dataset_path
            and Path(str(manifest["dataset_path"])).resolve()
            != Path(str(stored_dataset_path)).resolve()
        ):
            raise NepArtifactError(
                "Filesystem model-run manifest dataset path conflicts with StateStore"
            )
        list_model_artifacts = getattr(state_store, "list_model_artifacts", None)
        if callable(list_model_artifacts):
            linked = list_model_artifacts(str(manifest["model_run_id"]))
            if manifest["status"] == "completed":
                if not isinstance(linked, (list, tuple)):
                    raise NepArtifactError("StateStore model artifacts are malformed")
                model_artifacts = [
                    item
                    for item in linked
                    if isinstance(item, Mapping) and item.get("role") == "model"
                ]
                if len(model_artifacts) != 1 or model_artifacts[0].get("sha256") != manifest.get(
                    "potential_artifact_sha256"
                ):
                    raise NepArtifactError(
                        "Filesystem model artifact hash is not the persisted StateStore artifact"
                    )
    if manifest["status"] != "completed":
        raise NepArtifactError(
            f"Model run {manifest['model_run_id']} is not completed (status={manifest['status']!r})"
        )
    if manifest["identity_schema_version"] != MODEL_RUN_IDENTITY_SCHEMA:
        raise NepArtifactError(
            f"Unsupported model-run identity schema: {manifest['identity_schema_version']}"
        )
    computed_model_run_id = compute_model_run_id(
        dataset_id=str(manifest["dataset_id"]),
        nep_in_sha256=str(manifest["nep_in_sha256"]),
        hyperparameters_hash=str(manifest["hyperparameters_hash"]),
    )
    if manifest["model_run_id"] != computed_model_run_id:
        raise NepArtifactError(
            f"Model-run identity hash mismatch: expected {computed_model_run_id}, "
            f"found {manifest['model_run_id']}"
        )

    dataset_path = Path(str(manifest["dataset_path"]))
    metadata_path = dataset_path / ".dataset"
    if not dataset_path.is_dir() or not metadata_path.is_file():
        raise NepArtifactError(f"Manifest dataset artifact is missing: {dataset_path}")
    metadata = read_json_object(metadata_path, error_type=NepArtifactError)
    if metadata.get("dataset_id") != manifest["dataset_id"]:
        raise NepArtifactError(
            f"Dataset identity mismatch for {dataset_path}: "
            f"expected {manifest['dataset_id']}, found {metadata.get('dataset_id')}"
        )
    dataset_artifacts = metadata.get("artifacts")
    if dataset_artifacts is not None:
        if not isinstance(dataset_artifacts, Mapping):
            raise NepArtifactError("Dataset artifact index is malformed")
        for filename in ("train.xyz", "test.xyz"):
            record = dataset_artifacts.get(filename)
            artifact = dataset_path / filename
            if (
                not isinstance(record, Mapping)
                or record.get("path") != filename
                or not artifact.is_file()
                or sha256_file(artifact, required=True, error_type=NepArtifactError)
                != record.get("sha256")
            ):
                raise NepArtifactError(f"Dataset artifact hash mismatch for {artifact}")
    for path_key, hash_key in (
        ("potential_artifact_path", "potential_artifact_sha256"),
        ("nep_in_path", "nep_in_sha256"),
    ):
        artifact = Path(str(manifest[path_key]))
        expected_hash = manifest[hash_key]
        if not artifact.is_file():
            raise NepArtifactError(f"Manifest artifact is missing: {artifact}")
        if (
            not expected_hash
            or sha256_file(artifact, required=True, error_type=NepArtifactError) != expected_hash
        ):
            raise NepArtifactError(f"Manifest hash mismatch for {artifact}")
    return manifest


def find_model_run_manifest(project_dir: Path, model_run_id: str) -> Path:
    """Resolve exactly one explicitly requested model-run identity."""

    if not model_run_id:
        raise ValueError("model_run_id is required; latest model discovery is disabled")
    project_dir = Path(project_dir)
    potentials_dir = project_dir / "nep" / "potentials"
    if not potentials_dir.is_dir():
        raise FileNotFoundError(f"No canonical NEP potential directory found: {potentials_dir}")
    matches = []
    for manifest_path in potentials_dir.rglob(MODEL_RUN_MANIFEST_FILENAME):
        try:
            manifest = read_model_run_manifest(manifest_path)
        except NepArtifactError as exc:
            # A corrupt manifest is authoritative-state evidence, not an
            # absent candidate.  Do not hide it by returning another model or
            # by reporting that the requested identity was never written.
            raise NepArtifactError(f"Could not read model-run manifest: {manifest_path}") from exc
        if manifest.get("model_run_id") == model_run_id:
            matches.append(manifest_path)
    if not matches:
        raise FileNotFoundError(f"No model-run manifest found for model_run_id={model_run_id}")
    if len(matches) != 1:
        raise NepArtifactError(
            f"Multiple manifests found for model_run_id={model_run_id}: {matches}"
        )
    return matches[0]


__all__ = [
    "MODEL_RUN_IDENTITY_SCHEMA",
    "MODEL_RUN_MANIFEST_FILENAME",
    "MODEL_RUN_MANIFEST_SCHEMA",
    "NepArtifactError",
    "parse_nep_cutoff_angstrom",
    "compute_model_run_id",
    "create_model_run_manifest",
    "find_model_run_manifest",
    "read_model_run_manifest",
    "update_model_run_status",
    "validate_model_run_manifest",
    "write_model_run_manifest",
]
