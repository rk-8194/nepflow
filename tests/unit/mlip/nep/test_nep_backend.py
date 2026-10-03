from __future__ import annotations

from pathlib import Path

from nepflow.config.models import CompositionConfig, NepTrainingConfig
from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.mlip.backend import TrainingInputRequest
from nepflow.mlip.nep.backend import NepBackend
from nepflow.mlip.nep.inputs import NepHyperparameters


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
    completion = backend.parse_completion(tmp_path / "run")
    assert completion.completed is True
    collected = backend.collect_model_artifacts(tmp_path / "run", inputs)
    assert collected.artifact.model.sha256


