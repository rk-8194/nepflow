from pathlib import Path

import pytest

from nepflow.config.models import CompositionConfig, NepTrainingConfig
from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.domain.identities import ArtifactIdentity
from nepflow.hpc.jobs import (
    ReconciledJobResult,
    SchedulerJobState,
    SlurmJobRecord,
    SubmissionResult,
)
from nepflow.mlip.backend import TrainingCompletion, TrainingInput, TrainingInputRequest
from nepflow.mlip.nep.backend import NepBackend
from nepflow.mlip.nep.inputs import NepHyperparameters
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


def test_launcher_turnover_preserves_checkpoint_resumed_worker(tmp_path: Path) -> None:
    dataset = TrainingDatasetManifest(
        DatasetIdentity(
            "dataset-checkpoint-launcher-restart",
            {"schema_version": "nepflow.dataset.v1", "records": []},
        ),
        (),
    )
    training = NepTrainingConfig(generations=500000, max_resubmit=2)
    composition = CompositionConfig(elements=("W",))
    settings = NepHyperparameters.from_config(composition, training)
    backend = NepBackend()
    training_input = backend.render_training_input(
        TrainingInputRequest(
            dataset=dataset,
            hyperparameters=training,
            composition=composition,
            working_directory=tmp_path / "checkpoint-candidate",
            hyperparameters_hash=settings.identity_hash(),
        )
    )
    scheduler = RestartScheduler()
    state_path = tmp_path / "state-checkpoint.db"

    with StateStore(state_path) as store:
        store.upsert_dataset(dataset.identity)
        campaign = TrainingCampaign.create(
            campaign_id="campaign-checkpoint-launcher-restart",
            dataset_id=dataset.identity.dataset_id,
            state_store=store,
            scheduler=scheduler,
            backend=backend,
            working_directory=tmp_path / "potentials-checkpoint",
            max_attempts=3,
        )
        candidate = campaign.create_run(training_input, ordinal=0)
        campaign.reconcile()
        first_attempt = campaign.attempts(candidate.model_run_id)[0]
        assert first_attempt.execution_directory is not None
        first_attempt.execution_directory.joinpath("nep.restart").write_bytes(b"restart-1")
        first_attempt.execution_directory.joinpath("nep.txt").write_text(
            "checkpoint-1\n", encoding="utf-8"
        )
        first_attempt.execution_directory.joinpath("loss.out").write_text(
            "# columns generation loss\n180000 0.2\n",
            encoding="utf-8",
        )
        scheduler.states["9001"] = SchedulerJobState.TIMEOUT

        resumed = campaign.reconcile()
        worker_two = resumed.candidates[0].job_id
        assert worker_two == "9002"
        assert resumed.candidates[0].committed_generations == 180000
        scheduler.states[worker_two] = SchedulerJobState.RUNNING

        with pytest.raises(SelfResubmitExit):
            campaign.reconcile_until_terminal(
                poll_interval_seconds=0,
                slurm_deadline=1.0,
                clock=iter((0.0, 1.0)).__next__,
            )
        before_turnover = campaign.snapshot()
        assert before_turnover.candidates[0].job_id == worker_two
        assert before_turnover.candidates[0].committed_generations == 180000

    scheduler.reconciled_ids.clear()
    with StateStore(state_path) as store:
        restarted = TrainingCampaign(
            campaign_id="campaign-checkpoint-launcher-restart",
            dataset_id=dataset.identity.dataset_id,
            state_store=store,
            scheduler=scheduler,
            backend=backend,
            working_directory=tmp_path / "potentials-checkpoint",
            max_attempts=3,
        )
        with pytest.raises(SelfResubmitExit):
            restarted.reconcile_until_terminal(
                poll_interval_seconds=0,
                slurm_deadline=1.0,
                clock=iter((0.0, 1.0)).__next__,
            )
        after_turnover = restarted.snapshot()
        attempts = restarted.attempts(candidate.model_run_id)

    assert scheduler.submissions == 2
    assert scheduler.reconciled_ids == [worker_two]
    assert after_turnover.candidates[0].job_id == worker_two
    assert after_turnover.candidates[0].committed_generations == 180000
    assert attempts[1].restart_source_attempt_id == attempts[0].attempt_id
