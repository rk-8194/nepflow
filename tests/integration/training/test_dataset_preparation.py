from pathlib import Path

import pytest

from nepflow.config.models import NepTrainingConfig
from nepflow.stages.training.dataset import build_dataset_metadata
from nepflow.stages.training.optimisation import CandidateConfiguration
from nepflow.stages.training.stage import TrainingStage


def test_dataset_metadata_identity_is_independent_of_storage_location(tmp_path: Path) -> None:
    report = {
        "requested_count": 0,
        "accepted_results": [],
        "rejected_reason_counts": {},
    }
    first = build_dataset_metadata(
        tmp_path / "first" / "dataset",
        report,
        report,
        train_virial=False,
        allow_partial=False,
    )
    second = build_dataset_metadata(
        tmp_path / "second" / "dataset",
        report,
        report,
        train_virial=False,
        allow_partial=False,
    )

    assert first["dataset_id"] == second["dataset_id"]


def test_training_stage_campaign_identity_is_deterministic() -> None:
    base = NepTrainingConfig()
    first = CandidateConfiguration(0, base, {})
    changed = CandidateConfiguration(1, NepTrainingConfig(lambda_f=2.0), {"lambda_f": 2.0})

    assert TrainingStage._campaign_id("dataset-a", (first, changed)) == TrainingStage._campaign_id(
        "dataset-a", (first, changed)
    )
    assert TrainingStage._campaign_id("dataset-a", (first,)) != TrainingStage._campaign_id(
        "dataset-a", (first, changed)
    )


def test_training_stage_uses_explicit_selected_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TrainingStage._dataset_path(tmp_path)
