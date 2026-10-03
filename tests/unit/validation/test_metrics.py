from pathlib import Path

import numpy as np
import pytest

from nepflow.domain.identities import ArtifactIdentity, ModelRunIdentity, StructureIdentity
from nepflow.domain.models import ModelArtifactMetadata, ModelRunRecord
from nepflow.errors import ValidationError
from nepflow.mlip.simulation import StaticPrediction
from nepflow.stages.validation.metrics import calculate_metrics, pair_prediction
from nepflow.stages.validation.protocols import ValidationCaseSpec, ValidationReference


def _model() -> ModelRunRecord:
    identity = ModelRunIdentity("dataset-fixture", "nep-input", "hyperparameters")
    artifact = ArtifactIdentity.from_bytes("nep_model", b"model-fixture")
    return ModelRunRecord(
        identity,
        ModelArtifactMetadata(model=artifact, status="completed"),
    )


def _case(tmp_path: Path) -> ValidationCaseSpec:
    reference = ValidationReference(
        structure=StructureIdentity("metrics-structure"),
        species=("Si", "Si"),
        positions_angstrom=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        cell_angstrom=np.eye(3) * 3.0,
        pbc=(True, True, True),
        energy_ev=-10.0,
        forces_ev_per_angstrom=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        virial_ev=np.diag([1.0, 2.0, 3.0]),
    )
    return ValidationCaseSpec.create(
        ordinal=0,
        model_run_id=_model().model_run_id,
        dataset_id="dataset-fixture",
        reference=reference,
        input_path=tmp_path / "input.xyz",
        working_directory=tmp_path / "case",
        output_path=tmp_path / "output.xyz",
        virial_requested=True,
    )


def test_pairing_and_metrics_use_distinct_dft_and_ml_fixture_values(tmp_path: Path) -> None:
    model = _model()
    case = _case(tmp_path)
    prediction = StaticPrediction(
        structure=case.reference.structure,
        model_run=model.identity,
        atom_count=2,
        energy_ev=-9.0,
        forces_ev_per_angstrom=np.array([[1.2, 0.0, 0.0], [-0.8, 0.0, 0.0]]),
        virial_ev=np.diag([1.5, 1.5, 3.0]),
        virial_requested=True,
        atom_mapping=(0, 1),
    )

    paired = pair_prediction(case, prediction)
    metrics = calculate_metrics((paired,))

    assert paired.dft_energy_ev == -10.0
    assert paired.ml_energy_ev == -9.0
    assert metrics.energy_mae == pytest.approx(0.5)
    assert metrics.energy_rmse == pytest.approx(0.5)
    assert metrics.force_component_mae == pytest.approx(0.4 / 6.0)
    assert metrics.force_component_rmse == pytest.approx(np.sqrt(0.08 / 6.0))
    assert metrics.force_magnitude_mae == pytest.approx(0.2)
    assert metrics.force_magnitude_rmse == pytest.approx(0.2)
    assert metrics.virial_mae == pytest.approx(1.0 / 9.0)
    assert metrics.virial_rmse == pytest.approx(np.sqrt(0.5 / 9.0))


def test_pairing_requires_backend_atom_provenance(tmp_path: Path) -> None:
    model = _model()
    case = _case(tmp_path)
    prediction = StaticPrediction(
        structure=case.reference.structure,
        model_run=model.identity,
        atom_count=2,
        energy_ev=-9.0,
        forces_ev_per_angstrom=np.zeros((2, 3)),
    )

    with pytest.raises(ValidationError, match="provenance"):
        pair_prediction(case, prediction)


def test_pairing_rejects_wrong_model_even_when_values_look_valid(tmp_path: Path) -> None:
    case = _case(tmp_path)
    other_model = ModelRunIdentity("dataset-fixture", "other-nep", "hyperparameters")
    prediction = StaticPrediction(
        structure=case.reference.structure,
        model_run=other_model,
        atom_count=2,
        energy_ev=123.0,
        forces_ev_per_angstrom=np.ones((2, 3)),
        atom_mapping=(0, 1),
    )

    with pytest.raises(ValidationError, match="model identity"):
        pair_prediction(case, prediction)
