"""Explicit resolution of one completed model run and its dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.domain.identities import ArtifactIdentity, ModelRunIdentity, StructureIdentity
from nepflow.domain.models import ModelArtifactMetadata, ModelRunRecord
from nepflow.domain.units import (
    VIRIAL_CONVENTION_POSITIVE_COMPRESSION,
    VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3,
)
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
    def model(self) -> ModelRunRecord:
        """Canonical model record spelling for typed validation callers."""

        return self.model_run

    @property
    def dataset_id(self) -> str:
        return self.dataset.identity.dataset_id

    @property
    def model_artifact(self) -> ArtifactIdentity:
        if self.model_run.artifact is None:
            raise StateError("Resolved model run has no model artifact")
        return self.model_run.artifact.model

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
    positions = record.get("positions")
    cell = record.get("lattice", record.get("cell"))
    energy = record.get("energy")
    forces = record.get("forces")
    if positions is None or cell is None or energy is None or forces is None:
        raise StateError(f"Authoritative validation record is missing labels: {structure_id}")
    pbc = record.get("pbc", (True, True, True))
    if (
        not isinstance(pbc, (list, tuple))
        or len(pbc) != 3
        or not all(isinstance(value, bool) for value in pbc)
    ):
        raise StateError("Validation record has invalid periodic boundary conditions")
    return ValidationReference(
        structure=StructureIdentity(structure_id),
        species=tuple(str(value) for value in record.get("species", ())),
        positions_angstrom=np.asarray(positions, dtype=float),
        cell_angstrom=np.asarray(cell, dtype=float),
        pbc=(pbc[0], pbc[1], pbc[2]),
        energy_ev=float(energy),
        forces_ev_per_angstrom=np.asarray(forces, dtype=float),
        virial_ev=(
            None if record.get("virial") is None else np.asarray(record["virial"], dtype=float)
        ),
        metadata=metadata,
        virial_convention=str(
            record.get("virial_convention") or VIRIAL_CONVENTION_POSITIVE_COMPRESSION
        ),
        virial_tensor_convention=str(
            record.get("virial_tensor_convention") or VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3
        ),
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
        execution.get("dataset_path") if isinstance(execution, Mapping) else None
    ) or model_manifest.get("dataset_path")
    if not isinstance(dataset_path_value, str) or not dataset_path_value.strip():
        raise StateError("Authoritative model run has no dataset path")
    dataset_path = Path(dataset_path_value).resolve()
    canonical_dataset_dir = (project_dir / "nep" / "datasets").resolve()
    try:
        if dataset_path.parent != canonical_dataset_dir:
            raise StateError(f"Authoritative dataset is outside canonical storage: {dataset_path}")
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
        raise StateError(f"Authoritative model artifact is outside canonical storage: {model_path}")
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
    if not state_path.is_file():
        raise StateError(f"Authoritative StateStore is required for validation: {state_path}")
    with StateStore(state_path) as store:
        return _resolve_with_store(project_dir, model_run_id, store, dataset_id=dataset_id)


__all__ = ["ResolvedModelDataset", "resolve_model_dataset"]
