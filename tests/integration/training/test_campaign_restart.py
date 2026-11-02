from pathlib import Path

import pytest

from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.domain.identities import ArtifactIdentity
from nepflow.hpc.jobs import (
    ReconciledJobResult,
    SchedulerJobState,
    SlurmJobRecord,
    SubmissionResult,
)
from nepflow.mlip.backend import TrainingCompletion, TrainingInput
from nepflow.stages.training.campaign import TrainingCampaign
from nepflow.state.store import StateStore
from nepflow.workflow.resubmission import SelfResubmitExit


class RestartScheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, SlurmJobRecord] = {}
        self.states: dict[str, SchedulerJobState] = {}
        self.submissions = 0
        self.reconciled_ids: list[str] = []
        self.launcher_job_id = "9000"

    def find_job_by_name(self, name: str, **_kwargs):
        return next((job for job in self.jobs.values() if job.name == name), None)

    def submit_script(self, script_path: Path, *, job_name: str, **_kwargs):
        self.submissions += 1
        job_id = f"900{self.submissions}"
        self.states[job_id] = SchedulerJobState.PENDING
        self.jobs[job_id] = SlurmJobRecord(job_id, job_name, SchedulerJobState.PENDING)
        return SubmissionResult(job_id, "", "", ("sbatch", str(script_path)))

    def reconcile(self, job_id: str):
        self.reconciled_ids.append(job_id)
        state = self.states[job_id]
        return ReconciledJobResult(
            job_id,
            SlurmJobRecord(job_id, self.jobs[job_id].name, state),
            "restart-test",
        )


class RestartBackend:
    def model_run_identity(self, inputs):
        return inputs.model_run_identity

    def training_command(self, _inputs):
        return ("nep",)

    def parse_progress(self, _run_directory):
        return None

    def parse_completion(self, _run_directory):
        return TrainingCompletion(False)


def test_campaign_reopens_persisted_submission_without_resubmitting(tmp_path: Path) -> None:
    dataset = TrainingDatasetManifest(
        DatasetIdentity(
            "dataset-restart",
            {"schema_version": "nepflow.dataset.v1", "records": []},
        ),
        (),
    )
    training_input = TrainingInput(
        dataset=dataset,
        working_directory=tmp_path / "candidate",
        content="type 1 Si\n",
        nep_in=ArtifactIdentity.from_bytes("nep_input", b"type 1 Si\n"),
        hyperparameters_hash="hyper-restart",
    )
    training_input.working_directory.mkdir(parents=True, exist_ok=True)
    with (training_input.working_directory / "nep.in").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        handle.write(training_input.content)
    scheduler = RestartScheduler()
    backend = RestartBackend()
    state_path = tmp_path / "state.db"

    with StateStore(state_path) as store:
        store.upsert_dataset(dataset.identity)
        campaign = TrainingCampaign.create(
            campaign_id="campaign-restart",
            dataset_id=dataset.identity.dataset_id,
            state_store=store,
            scheduler=scheduler,
            backend=backend,
            working_directory=tmp_path / "potentials",
        )
        campaign.create_run(training_input, ordinal=0)
        first = campaign.reconcile()
        assert first.candidates[0].status == "submitted"
        persisted_job_id = first.candidates[0].job_id

    with StateStore(state_path) as store:
        restarted = TrainingCampaign(
            campaign_id="campaign-restart",
            dataset_id=dataset.identity.dataset_id,
            state_store=store,
            scheduler=scheduler,
            backend=backend,
            working_directory=tmp_path / "potentials",
        )
        with pytest.raises(SelfResubmitExit):
            restarted.reconcile_until_terminal(
                poll_interval_seconds=0,
                slurm_deadline=1.0,
                clock=iter((0.0, 1.0)).__next__,
            )
        second = restarted.snapshot()

    assert second.candidates[0].status == "submitted"
    assert second.candidates[0].job_id == persisted_job_id
    assert persisted_job_id == "9001"
    assert persisted_job_id != scheduler.launcher_job_id
    assert scheduler.reconciled_ids == [persisted_job_id]
    assert scheduler.submissions == 1
