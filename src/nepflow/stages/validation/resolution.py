"""Explicit resolution of one completed model run and its dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.domain.identities import ArtifactIdentity, ModelRunIdentity, StructureIdentity
from nepflow.domain.models import ModelArtifactMetadata, ModelRunRecord
from nepflow.errors import StateError
from nepflow.io.hashing import sha256_file
from nepflow.io.json import read_json_object
from nepflow.mlip.nep.artifacts import (
    NepArtifactError,
    find_model_run_manifest,
    validate_model_run_manifest,
)
from nepflow.state.store import StateStore

from .protocols import ValidationReference


@dataclass(frozen=True, slots=True)
class ResolvedModelDataset:
    """Authoritative model/dataset pair used to construct validation cases."""

    model_run: ModelRunRecord
    dataset: TrainingDatasetManifest
    model_path: Path
    dataset_path: Path
    model_manifest: Mapping[str, Any]

    @property
    def model_run_id(self) -> str:
        return self.model_run.model_run_id

    @property
    def dataset_id(self) -> str:
        return self.dataset.identity.dataset_id

    def test_references(self) -> tuple[ValidationReference, ...]:
        """Return ordered test labels from the persisted dataset manifest."""

        references: list[ValidationReference] = []
        for record in self.dataset.records:
            if str(record.get("split", "test")) != "test":
                continue
            references.append(_reference_from_record(record))
        if not references:
            raise StateError(
                f"Authoritative dataset has no persisted test records: {self.dataset_id}"
            )
        return tuple(references)


def _reference_from_record(record: Mapping[str, Any]) -> ValidationReference:
    structure_id = record.get("structure_id")
    if not isinstance(structure_id, str) or not structure_id.strip():
        raise StateError("Authoritative validation record is missing structure_id")
    metadata = {
        key: record[key]
        for key in ("calculation_identity", "source_outcar_hash", "split")
        if key in record
    }
    return ValidationReference(
        structure=StructureIdentity(structure_id),
        species=tuple(str(value) for value in record.get("species", ())),
        positions_angstrom=record.get("positions"),
        cell_angstrom=record.get("lattice", record.get("cell")),
        pbc=tuple(record.get("pbc", (True, True, True))),
        energy_ev=record.get("energy"),
        forces_ev_per_angstrom=record.get("forces"),
        virial_ev=record.get("virial"),
        metadata=metadata,
    )


def _identity_from_row(row: Mapping[str, Any]) -> ModelRunIdentity:
    payload = row.get("identity_json", row.get("identity", {}))
    if not isinstance(payload, Mapping):
        raise StateError("Authoritative model-run identity is malformed")
    try:
        identity = ModelRunIdentity.from_inputs(
            dataset_id=str(payload["dataset_id"]),
            nep_in_sha256=str(payload["nep_in_sha256"]),
            hyperparameters_hash=str(payload["hyperparameters_hash"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StateError("Authoritative model-run identity is incomplete") from exc
    if payload.get("model_run_id") not in (None, identity.model_run_id):
        raise StateError("Authoritative model-run identity hash does not match its fields")
    return identity


def _dataset_from_row(row: Mapping[str, Any]) -> TrainingDatasetManifest:
    stored = row.get("manifest", row.get("manifest_json", {}))
    if not isinstance(stored, Mapping):
        raise StateError("Authoritative dataset manifest is malformed")
    identity_payload = stored.get("identity", row.get("identity_json", {}))
    if not isinstance(identity_payload, Mapping):
        raise StateError("Authoritative dataset identity is malformed")
    dataset_id = str(row.get("dataset_id", stored.get("dataset_id", "")))
    if not dataset_id:
        raise StateError("Authoritative dataset is missing dataset_id")
    payload = dict(identity_payload)
    payload.pop("dataset_id", None)
    identity = DatasetIdentity(dataset_id, payload)
    if identity.to_dict() != {
        "dataset_id": dataset_id,
        **payload,
    }:
        raise StateError("Authoritative dataset identity could not be reconstructed")
    records = stored.get("records", ())
    if not isinstance(records, (list, tuple)):
        raise StateError("Authoritative dataset records are malformed")
    return TrainingDatasetManifest(
        identity=identity,
        records=tuple(records),
        selection_method=stored.get("selection_method"),
        selection_parameters=stored.get("selection_parameters"),
        descriptor_model_fingerprint=stored.get("descriptor_model_fingerprint"),
        created_at=stored.get("created_at"),
        code_version=stored.get("code_version"),
    )


def _resolve_with_store(
    project_dir: Path,
    model_run_id: str,
    state_store: Any,
    *,
    dataset_id: str | None,
) -> ResolvedModelDataset:
    model_row = state_store.get_model_run(model_run_id)
    if model_row is None:
        raise StateError(f"Unknown authoritative model run: {model_run_id}")
    identity = _identity_from_row(model_row)
    if identity.model_run_id != model_run_id:
        raise StateError("Requested model_run_id does not match StateStore identity")
    if dataset_id is not None and identity.dataset_id != dataset_id:
        raise StateError(
            "Requested dataset_id conflicts with the model run's authoritative dataset"
        )
    if str(model_row.get("status", "")) != "completed":
        raise StateError(f"Model run is not completed: {model_run_id}")

    dataset_row = state_store.get_dataset(identity.dataset_id)
    if dataset_row is None:
        raise StateError(f"Unknown authoritative dataset: {identity.dataset_id}")
    dataset = _dataset_from_row(dataset_row)

    manifest_path = find_model_run_manifest(project_dir, model_run_id)
    try:
        model_manifest = validate_model_run_manifest(
            manifest_path,
            expected_model_run_id=model_run_id,
            state_store=state_store,
        )
    except (FileNotFoundError, NepArtifactError) as exc:
        raise StateError(f"Invalid authoritative model-run manifest: {model_run_id}") from exc
    manifest_dataset_id = model_manifest.get("dataset_id")
    if manifest_dataset_id != dataset.identity.dataset_id:
        raise StateError("Model manifest dataset_id conflicts with StateStore")

    execution = model_row.get("execution_metadata", {})
    dataset_path_value = (
        execution.get("dataset_path")
        if isinstance(execution, Mapping)
        else None
    ) or model_manifest.get("dataset_path")
    if not isinstance(dataset_path_value, str) or not dataset_path_value.strip():
        raise StateError("Authoritative model run has no dataset path")
    dataset_path = Path(dataset_path_value).resolve()
    canonical_dataset_dir = (project_dir / "nep" / "datasets").resolve()
    try:
        if dataset_path.parent != canonical_dataset_dir:
            raise StateError(
                f"Authoritative dataset is outside canonical storage: {dataset_path}"
            )
    except OSError as exc:
        raise StateError(f"Could not resolve authoritative dataset path: {dataset_path}") from exc
    metadata_path = dataset_path / ".dataset"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Materialized dataset is missing: {dataset_path}")
    materialized = read_json_object(metadata_path)
    if materialized.get("dataset_id") != dataset.identity.dataset_id:
        raise StateError("Materialized dataset conflicts with authoritative dataset identity")

    artifacts = [
        artifact
        for artifact in state_store.list_model_artifacts(model_run_id)
        if artifact.get("role") == "model"
    ]
    if len(artifacts) != 1:
        raise StateError(f"StateStore must contain exactly one model artifact: {model_run_id}")
    artifact_row = artifacts[0]
    model_path_value = artifact_row.get("path") or model_manifest.get("potential_artifact_path")
    if not isinstance(model_path_value, str) or not model_path_value.strip():
        raise StateError("Authoritative model artifact has no path")
    model_path = Path(model_path_value).resolve()
    canonical_model_dir = (project_dir / "nep" / "potentials").resolve()
    if model_path.parent.parent != canonical_model_dir:
        raise StateError(
            f"Authoritative model artifact is outside canonical storage: {model_path}"
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"Authoritative model artifact is missing: {model_path}")
    if sha256_file(model_path) != artifact_row.get("sha256"):
        raise StateError("Authoritative model artifact hash does not match StateStore")
    artifact = ArtifactIdentity(
        artifact_id=str(artifact_row["artifact_id"]),
        artifact_type=str(artifact_row["artifact_type"]),
        sha256=str(artifact_row["sha256"]),
        path=str(model_path),
    )
    record = ModelRunRecord(
        identity=identity,
        artifact=ModelArtifactMetadata(model=artifact, status="completed"),
        execution_metadata=execution if isinstance(execution, Mapping) else None,
    )
    return ResolvedModelDataset(record, dataset, model_path, dataset_path, model_manifest)


def _resolve_without_store(
    project_dir: Path,
    model_run_id: str,
    *,
    dataset_id: str | None,
) -> ResolvedModelDataset:
    """Read the pre-ledger manifest only for migration-era projects."""

    manifest_path = find_model_run_manifest(project_dir, model_run_id)
    try:
        manifest = validate_model_run_manifest(
            manifest_path,
            expected_model_run_id=model_run_id,
        )
    except (FileNotFoundError, NepArtifactError) as exc:
        raise StateError(f"Invalid model-run manifest: {model_run_id}") from exc
    resolved_dataset_id = str(manifest.get("dataset_id", ""))
    if dataset_id is not None and dataset_id != resolved_dataset_id:
        raise StateError("Requested dataset_id conflicts with model manifest")
    identity = ModelRunIdentity.from_inputs(
        dataset_id=resolved_dataset_id,
        nep_in_sha256=str(manifest["nep_in_sha256"]),
        hyperparameters_hash=str(manifest["hyperparameters_hash"]),
    )
    if identity.model_run_id != model_run_id or manifest.get("status") != "completed":
        raise StateError("Model manifest is not a completed authoritative model run")
    dataset_path = Path(str(manifest["dataset_path"])).resolve()
    if dataset_path.parent != (project_dir / "nep" / "datasets").resolve():
        raise StateError(f"Dataset is outside canonical storage: {dataset_path}")
    metadata = read_json_object(dataset_path / ".dataset")
    if metadata.get("dataset_id") != resolved_dataset_id:
        raise StateError("Materialized dataset conflicts with model manifest")
    identity_payload = dict(metadata)
    identity_payload.pop("dataset_id", None)
    dataset = TrainingDatasetManifest(
        identity=DatasetIdentity(resolved_dataset_id, identity_payload),
        records=tuple(metadata.get("records", ())),
    )
    model_path = Path(str(manifest["potential_artifact_path"])).resolve()
    canonical_model_dir = (project_dir / "nep" / "potentials").resolve()
    if model_path.parent.parent != canonical_model_dir:
        raise StateError(
            f"Model artifact is outside canonical storage: {model_path}"
        )
    artifact = ArtifactIdentity.from_file("nep_model", model_path)
    record = ModelRunRecord(
        identity=identity,
        artifact=ModelArtifactMetadata(model=artifact, status="completed"),
    )
    return ResolvedModelDataset(record, dataset, model_path, dataset_path, manifest)


def resolve_model_dataset(
    project_dir: Path,
    model_run_id: str,
    *,
    dataset_id: str | None = None,
    state_store: Any | None = None,
) -> ResolvedModelDataset:
    """Resolve an exact completed model run and its persisted dataset.

    No directory ordering, timestamp, or ``latest`` convention participates
    in this lookup.  When a StateStore is present it is authoritative for the
    model, dataset, and artifact identities; the filesystem is verified as a
    materialized projection.
    """

    project_dir = Path(project_dir).resolve()
    if not str(model_run_id).strip():
        raise StateError("Validation requires an explicit model_run_id")
    if state_store is not None:
        return _resolve_with_store(project_dir, model_run_id, state_store, dataset_id=dataset_id)
    state_path = project_dir / "state.db"
    if state_path.is_file():
        with StateStore(state_path) as store:
            return _resolve_with_store(project_dir, model_run_id, store, dataset_id=dataset_id)
    return _resolve_without_store(project_dir, model_run_id, dataset_id=dataset_id)


__all__ = ["ResolvedModelDataset", "resolve_model_dataset"]
