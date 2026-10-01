from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nepflow.config.models import CompositionConfig, NepTrainingConfig
from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.domain.identities import ArtifactIdentity, ModelRunIdentity, StructureIdentity
from nepflow.domain.models import ModelArtifactMetadata, ModelRunRecord
from nepflow.errors import MlipError, ValidationError
from nepflow.mlip.backend import (
    CollectedModelArtifacts,
    MlipBackend,
    TrainingCompletion,
    TrainingInput,
    TrainingInputRequest,
    TrainingProgress,
)
from nepflow.mlip.simulation import (
    PredictionRuntimeMetadata,
    StaticPrediction,
    StaticPredictionBackend,
    StaticPredictionRequest,
)


def _dataset() -> TrainingDatasetManifest:
    identity = DatasetIdentity.from_identity_payload(
        {
            "schema_version": "nepflow.dataset.v1",
            "records": (),
            "units": {"energy": "eV", "forces": "eV/Angstrom"},
        }
    )
    return TrainingDatasetManifest(identity=identity, records=())


def _training_request(
    tmp_path: Path,
    *,
    hyperparameters: NepTrainingConfig | None = None,
) -> TrainingInputRequest:
    return TrainingInputRequest(
        dataset=_dataset(),
        hyperparameters=hyperparameters or NepTrainingConfig(),
        composition=CompositionConfig(elements=("W",)),
        working_directory=tmp_path / "training",
        hyperparameters_hash="canonical-nep-hyperparameters",
    )


class FakeMlipBackend:
    def render_training_input(self, request: TrainingInputRequest) -> TrainingInput:
        content = "type 1 W\ngeneration 10\n"
        artifact = ArtifactIdentity.from_bytes("nep.in", content.encode())
        return TrainingInput(
            request.dataset,
            request.working_directory,
            content,
            artifact,
            request.hyperparameters_hash,
        )

    def training_command(self, inputs: TrainingInput) -> tuple[str, ...]:
        return ("nep", "--input", "nep.in")

    def parse_progress(self, run_directory: Path) -> TrainingProgress | None:
        return TrainingProgress(4, 0.25)

    def parse_completion(self, run_directory: Path) -> TrainingCompletion:
        return TrainingCompletion(True, (run_directory / "nep.txt",))

    def collect_model_artifacts(
        self,
        run_directory: Path,
        inputs: TrainingInput,
    ) -> CollectedModelArtifacts:
        model = ArtifactIdentity.from_bytes("nep-model", b"model")
        return CollectedModelArtifacts(
            inputs,
            inputs.model_run_identity,
            ModelArtifactMetadata(model=model, nep_in=inputs.nep_in, status="completed"),
        )

    def model_run_identity(self, inputs: TrainingInput) -> ModelRunIdentity:
        return inputs.model_run_identity


def _model_run() -> ModelRunRecord:
    model = ArtifactIdentity.from_bytes("nep-model", b"model")
    identity = ModelRunIdentity("dataset-1", "nep-in", "hyperparameters")
    return ModelRunRecord(identity, ModelArtifactMetadata(model=model, status="completed"))


class FakeStaticBackend:
    def predict(self, request: StaticPredictionRequest) -> StaticPrediction:
        return StaticPrediction(
            request.structure,
            request.model.identity,
            request.atom_count,
            -2.0,
            np.zeros((request.atom_count, 3)),
            np.eye(3) if request.virial_requested else None,
            request.virial_requested,
            PredictionRuntimeMetadata(elapsed_seconds=0.5, backend_version="fake"),
        )


def test_fake_mlip_backend_conforms_and_keeps_commands_argument_oriented(
    tmp_path: Path,
) -> None:
    backend = FakeMlipBackend()
    assert isinstance(backend, MlipBackend)

    inputs = backend.render_training_input(_training_request(tmp_path))
    command = backend.training_command(inputs)
    assert command == ("nep", "--input", "nep.in")
    assert command and all(isinstance(part, str) and part for part in command)
    assert ";" not in command
    assert backend.model_run_identity(inputs) == inputs.model_run_identity
    collected = backend.collect_model_artifacts(tmp_path, inputs)
    assert collected.artifact.model.sha256


def test_runtime_training_policy_does_not_change_carried_model_identity(
    tmp_path: Path,
) -> None:
    backend = FakeMlipBackend()
    baseline = backend.render_training_input(_training_request(tmp_path))
    changed_policy = backend.render_training_input(
        _training_request(tmp_path, hyperparameters=NepTrainingConfig(max_resubmit=99))
    )

    assert baseline.hyperparameters_hash == changed_policy.hyperparameters_hash
    assert baseline.model_run_identity == changed_policy.model_run_identity


def test_collected_model_artifacts_reject_missing_or_mismatched_identity() -> None:
    training_input = FakeMlipBackend().render_training_input(
        _training_request(Path("training"))
    )
    artifact = ModelArtifactMetadata(
        model=ArtifactIdentity.from_bytes("nep-model", b"model"),
        status="completed",
    )
    with pytest.raises(MlipError, match="require model metadata"):
        CollectedModelArtifacts(
            training_input,
            training_input.model_run_identity,
            None,  # type: ignore[arg-type]
        )

    different = ModelRunIdentity("dataset-2", "nep-in", "hyperparameters")
    with pytest.raises(MlipError, match="different model run"):
        CollectedModelArtifacts(training_input, different, artifact)


def test_static_prediction_keeps_runtime_metadata_separate() -> None:
    backend = FakeStaticBackend()
    request = StaticPredictionRequest(
        StructureIdentity("structure-1"),
        _model_run(),
        Path("model.xyz"),
        Path("run"),
        atom_count=2,
        virial_requested=True,
    )
    assert isinstance(backend, StaticPredictionBackend)
    result = backend.predict(request)
    assert result.energy_unit == "eV"
    assert result.force_unit == "eV/Angstrom"
    assert result.virial_convention == "positive_compression"
    assert result.runtime.elapsed_seconds == 0.5
    assert result.runtime.backend_version == "fake"


def test_static_prediction_missing_requested_virial_is_typed_validation_error() -> None:
    with pytest.raises(ValidationError, match="virial"):
        StaticPrediction(
            StructureIdentity("structure-1"),
            ModelRunIdentity("dataset-1", "nep-in", "hyperparameters"),
            1,
            -1.0,
            np.zeros((1, 3)),
            virial_requested=True,
        )


def test_static_prediction_requires_a_model_artifact() -> None:
    with pytest.raises(ValidationError, match="model artifact"):
        StaticPredictionRequest(
            StructureIdentity("structure-1"),
            ModelRunRecord(ModelRunIdentity("dataset-1", "nep-in", "hyperparameters")),
            Path("model.xyz"),
            Path("run"),
            atom_count=1,
        )
