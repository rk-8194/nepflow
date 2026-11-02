from __future__ import annotations

from pathlib import Path

import pytest

from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.errors import StateError
from nepflow.hpc.jobs import ReconciledJobResult, SchedulerJobState, SlurmJobRecord
from nepflow.stages.training.rollback import TrainingRollbackService
from nepflow.state import StateStore


class RollbackScheduler:
    def __init__(self, *, confirms: bool = True) -> None:
        self.state = SchedulerJobState.RUNNING
        self.confirms = confirms
        self.cancelled: list[str] = []

    def reconcile(self, job_id: str) -> ReconciledJobResult:
        return ReconciledJobResult(
            job_id=job_id,
            job=SlurmJobRecord(job_id=job_id, name="nep", state=self.state),
            source="rollback-test",
        )

    def cancel(self, job_id: str):
        self.cancelled.append(job_id)
        if self.confirms:
            self.state = SchedulerJobState.CANCELLED
        else:
            self.state = SchedulerJobState.UNKNOWN
        return None


def _dataset() -> TrainingDatasetManifest:
    return TrainingDatasetManifest(
        identity=DatasetIdentity(
            "dataset-training-rollback",
            {"schema_version": "nepflow.dataset.v1", "records": []},
        ),
        records=(),
    )


def _seed_campaign(store: StateStore, *, status: str = "running") -> None:
    store.record_training_event(
        "training_campaign",
        "campaign-training-rollback",
        "created",
        {
            "campaign_id": "campaign-training-rollback",
            "dataset_id": "dataset-training-rollback",
            "status": status,
            "specification": {},
        },
    )
    store.record_training_event(
        "training_candidate",
        "campaign-training-rollback",
        "status",
        {
            "campaign_id": "campaign-training-rollback",
            "dataset_id": "dataset-training-rollback",
            "model_run_id": "model-training-rollback",
            "ordinal": 0,
            "status": status,
            "execution_generation": 1,
        },
    )
    store.record_training_event(
        "training_attempt",
        "campaign-training-rollback",
        "status",
        {
            "campaign_id": "campaign-training-rollback",
            "model_run_id": "model-training-rollback",
            "attempt_id": "model-training-rollback:attempt:1",
            "attempt_number": 1,
            "status": status,
            "job_id": "job-training-rollback",
            "execution_generation": 1,
        },
    )


def test_training_rollback_opens_new_generation_and_preserves_history(tmp_path: Path) -> None:
    scheduler = RollbackScheduler()
    with StateStore(tmp_path / "state.db") as store:
        store.upsert_project("project-training-rollback")
        store.upsert_dataset(_dataset().identity, project_id="project-training-rollback")
        store.ensure_training_execution_authority(
            "campaign-training-rollback", "dataset-training-rollback"
        )
        _seed_campaign(store)

        result = TrainingRollbackService(scheduler).rollback(
            project_id="project-training-rollback",
            state_store=store,
        )

        assert scheduler.cancelled == ["job-training-rollback"]
        assert result.next_execution_generations == {"campaign-training-rollback": 2}
        assert store.get_training_execution_authority("campaign-training-rollback")[
            "execution_generation"
        ] == 2
        history = store.list_training_execution_authority_history("campaign-training-rollback")
        assert [row["status"] for row in history] == ["invalidated", "current"]
        old_attempt = store.list_training_attempt_projections("campaign-training-rollback", 1)
        assert old_attempt[0]["status"] == "cancelled"
        assert store.list_training_candidate_projections("campaign-training-rollback", 2) == []


def test_training_rollback_failure_does_not_rewind_authority(tmp_path: Path) -> None:
    scheduler = RollbackScheduler(confirms=False)
    with StateStore(tmp_path / "state.db") as store:
        store.upsert_project("project-training-rollback")
        store.upsert_dataset(_dataset().identity, project_id="project-training-rollback")
        store.ensure_training_execution_authority(
            "campaign-training-rollback", "dataset-training-rollback"
        )
        _seed_campaign(store)

        with pytest.raises(StateError, match="scheduler state unknown"):
            TrainingRollbackService(scheduler).rollback(
                project_id="project-training-rollback",
                state_store=store,
            )

        assert store.get_training_execution_authority("campaign-training-rollback")[
            "execution_generation"
        ] == 1
        assert store.list_training_attempt_projections("campaign-training-rollback", 1)[0][
            "status"
        ] == "running"
