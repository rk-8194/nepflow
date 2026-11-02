from __future__ import annotations

import json
from pathlib import Path

import pytest

from nepflow.config.models import CompositionConfig, NepTrainingConfig
from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.errors import MlipError
from nepflow.mlip.backend import TrainingInputRequest
from nepflow.mlip.nep.backend import NepBackend
from nepflow.mlip.nep.inputs import NepHyperparameters
from nepflow.mlip.nep.outputs import SUCCESS_MARKER_NAME, SUCCESS_MARKER_SCHEMA


def _dataset() -> TrainingDatasetManifest:
    identity = DatasetIdentity.from_identity_payload(
        {"schema_version": "nepflow.dataset.v1", "records": (), "units": {}}
    )
    return TrainingDatasetManifest(identity=identity, records=())


def _request(tmp_path: Path, training: NepTrainingConfig | None = None) -> TrainingInputRequest:
    composition = CompositionConfig(elements=("W",))
    training = training or NepTrainingConfig()
    settings = NepHyperparameters.from_config(composition, training)
    return TrainingInputRequest(
        dataset=_dataset(),
        hyperparameters=training,
        composition=composition,
        working_directory=tmp_path / "run",
        hyperparameters_hash=settings.identity_hash(),
    )


def test_backend_renders_identity_bound_input_and_keeps_resources_out(tmp_path: Path) -> None:
    backend = NepBackend("mpirun --bind-to none /opt/gpumd/nep")
    baseline = backend.render_training_input(_request(tmp_path))
    changed_resources = backend.render_training_input(
        _request(tmp_path / "other", NepTrainingConfig(max_resubmit=99))
    )

    assert "type 1 W" in baseline.content
    assert "generation 250000" in baseline.content
    assert baseline.target_generations == 250000
    assert baseline.model_run_identity == changed_resources.model_run_identity
    assert backend.training_command(baseline) == (
        "mpirun",
        "--bind-to",
        "none",
        "/opt/gpumd/nep",
    )


def test_backend_progress_completion_and_exact_artifact_collection(tmp_path: Path) -> None:
    backend = NepBackend()
    inputs = backend.render_training_input(_request(tmp_path))
    (tmp_path / "run" / "loss.out").write_text("1 2.0\n4 0.25\n")
    assert backend.parse_progress(tmp_path / "run").generation == 4
    assert backend.parse_progress(tmp_path / "run").loss == 0.25

    model_path = tmp_path / "run" / "nep.txt"
    model_path.write_text("model\n")
    (tmp_path / "run" / "loss.out").write_text("249999 0.25\n250000 0.2\n")
    (tmp_path / "run" / SUCCESS_MARKER_NAME).write_text(
        json.dumps(
            {
                "schema_version": SUCCESS_MARKER_SCHEMA,
                "model_run_id": "model-run",
                "attempt_id": "model-run:attempt:1",
                "attempt_number": 1,
                "execution_generation": 1,
                "generation_offset": 0,
                "segment_target_generations": 250000,
                "process_exit_code": 0,
                "completion_timestamp": "2026-11-02T10:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    completion = backend.parse_completion(tmp_path / "run")
    assert completion.completed is True
    assert completion.target_generation == 250000
    assert completion.observed_generation == 250000
    assert completion.process_exit_code == 0
    collected = backend.collect_model_artifacts(tmp_path / "run", inputs)
    assert collected.artifact.model.sha256


def test_backend_checkpoint_without_success_marker_is_not_completion(tmp_path: Path) -> None:
    backend = NepBackend()
    inputs = backend.render_training_input(_request(tmp_path))
    (inputs.working_directory / "nep.txt").write_text("checkpoint\n")
    (inputs.working_directory / "loss.out").write_text("250000 0.2\n")

    completion = backend.parse_completion(inputs.working_directory)

    assert completion.completed is False
    assert completion.artifact_paths == (inputs.working_directory / "nep.txt",)


def test_backend_success_marker_below_target_generation_is_not_completion(tmp_path: Path) -> None:
    backend = NepBackend()
    inputs = backend.render_training_input(_request(tmp_path))
    run_directory = inputs.working_directory
    run_directory.joinpath("nep.txt").write_text("checkpoint\n")
    run_directory.joinpath("loss.out").write_text("249999 0.2\n")
    run_directory.joinpath(SUCCESS_MARKER_NAME).write_text(
        json.dumps(
            {
                "schema_version": SUCCESS_MARKER_SCHEMA,
                "model_run_id": "model-run",
                "attempt_id": "model-run:attempt:1",
                "attempt_number": 1,
                "execution_generation": 1,
                "generation_offset": 0,
                "segment_target_generations": 250000,
                "process_exit_code": 0,
                "completion_timestamp": "2026-11-02T10:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    completion = backend.parse_completion(run_directory)

    assert completion.completed is False
    assert completion.observed_generation == 249999


def test_backend_malformed_marker_or_progress_is_explicitly_rejected(tmp_path: Path) -> None:
    backend = NepBackend()
    inputs = backend.render_training_input(_request(tmp_path))
    run_directory = inputs.working_directory
    run_directory.joinpath("nep.txt").write_text("checkpoint\n")
    run_directory.joinpath(SUCCESS_MARKER_NAME).write_text("not-json", encoding="utf-8")

    with pytest.raises(MlipError, match="success marker"):
        backend.parse_completion(run_directory)

    run_directory.joinpath(SUCCESS_MARKER_NAME).unlink()
    run_directory.joinpath("loss.out").write_text("not progress\n")

    with pytest.raises(MlipError, match="progress"):
        backend.parse_completion(run_directory)


def test_gpumd_progress_headers_are_ignored_but_bad_data_is_rejected(tmp_path: Path) -> None:
    backend = NepBackend()
    inputs = backend.render_training_input(_request(tmp_path))
    run_directory = inputs.working_directory
    run_directory.joinpath("loss.out").write_text(
        "# format_version 1\n"
        "# output_interval 1000\n"
        "# columns generation loss\n"
        "1000 1.0\n"
        "# columns generation loss\n"
        "2000 0.5\n",
        encoding="utf-8",
    )

    progress = backend.parse_progress(run_directory)
    assert progress is not None
    assert progress.generation == 2000

    run_directory.joinpath("loss.out").write_text("# header\nnot data\n", encoding="utf-8")
    with pytest.raises(MlipError, match="progress"):
        backend.parse_progress(run_directory)


def test_backend_returns_verified_local_checkpoint_evidence(tmp_path: Path) -> None:
    backend = NepBackend()
    inputs = backend.render_training_input(
        _request(tmp_path, NepTrainingConfig(generations=500000))
    )
    run_directory = inputs.working_directory
    run_directory.joinpath("nep.restart").write_bytes(b"restart")
    run_directory.joinpath("nep.txt").write_text("checkpoint\n", encoding="utf-8")
    run_directory.joinpath("loss.out").write_text(
        "# columns generation loss\n180000 0.2\n", encoding="utf-8"
    )

    checkpoint = backend.parse_checkpoint(run_directory)

    assert checkpoint is not None
    assert checkpoint.local_generation == 180000
    assert checkpoint.restart_sha256
    assert checkpoint.model_sha256
