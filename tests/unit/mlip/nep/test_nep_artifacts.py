from __future__ import annotations

import json
from pathlib import Path

import pytest

from nepflow.domain.datasets import DatasetIdentity
from nepflow.mlip.nep.artifacts import (
    NepArtifactError,
    create_model_run_manifest,
    update_model_run_status,
    validate_model_run_manifest,
)
from nepflow.state.store import StateStore


def _state_store_with_dataset(path: Path, dataset_id: str) -> StateStore:
    store = StateStore(path / "state.db")
    store.upsert_dataset(
        DatasetIdentity(
            dataset_id,
            {"schema_version": "nepflow.dataset.v1", "records": []},
        )
    )
    return store


def test_filesystem_manifest_cannot_override_state_identity(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_0001"
    dataset_path.mkdir()
    (dataset_path / ".dataset").write_text(json.dumps({"dataset_id": "dataset-authoritative"}))
    potential_path = tmp_path / "potential_0001"
    potential_path.mkdir()
    nep_in = potential_path / "nep.in"
    nep_in.write_text("generation 10\n")
    (potential_path / "nep.txt").write_text("model\n")

    with _state_store_with_dataset(tmp_path, "dataset-authoritative") as store:
        manifest = create_model_run_manifest(
            potential_path=potential_path,
            dataset_path=dataset_path,
            dataset_id="dataset-authoritative",
            nep_in_path=nep_in,
            hyperparameters_hash="hyperparameters-authoritative",
            state_store=store,
        )
        filesystem_manifest = json.loads(
            (potential_path / "model_run_manifest.json").read_text()
        )
        filesystem_manifest["dataset_id"] = "dataset-filesystem-only"
        (potential_path / "model_run_manifest.json").write_text(
            json.dumps(filesystem_manifest)
        )

        with pytest.raises(NepArtifactError, match="conflicts with StateStore"):
            validate_model_run_manifest(
                potential_path / "model_run_manifest.json",
                expected_model_run_id=manifest["model_run_id"],
                state_store=store,
            )


def test_completed_model_artifact_is_linked_to_authoritative_run(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_0001"
    dataset_path.mkdir()
    (dataset_path / ".dataset").write_text(json.dumps({"dataset_id": "dataset-authoritative"}))
    potential_path = tmp_path / "potential_0001"
    potential_path.mkdir()
    nep_in = potential_path / "nep.in"
    nep_in.write_text("generation 10\n")
    model_path = potential_path / "nep.txt"
    model_path.write_text("model\n")

    with _state_store_with_dataset(tmp_path, "dataset-authoritative") as store:
        manifest = create_model_run_manifest(
            potential_path=potential_path,
            dataset_path=dataset_path,
            dataset_id="dataset-authoritative",
            nep_in_path=nep_in,
            hyperparameters_hash="hyperparameters-authoritative",
            state_store=store,
        )
        completed = update_model_run_status(
            potential_path,
            "completed",
            state_store=store,
            model_run_id=manifest["model_run_id"],
        )

        row = store.get_model_run(manifest["model_run_id"])
        assert row is not None
        assert row["status"] == "completed"
        linked_models = [
            artifact
            for artifact in store.list_model_artifacts(manifest["model_run_id"])
            if artifact["role"] == "model"
        ]
        assert len(linked_models) == 1
        assert linked_models[0]["sha256"] == completed["potential_artifact_sha256"]
        validate_model_run_manifest(
            potential_path / "model_run_manifest.json",
            expected_model_run_id=manifest["model_run_id"],
            state_store=store,
        )


