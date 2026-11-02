from __future__ import annotations

import json
from pathlib import Path

import pytest

from nepflow.config.models import CompositionConfig, NepTrainingConfig
from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.domain.identities import ArtifactIdentity
from nepflow.domain.models import ModelArtifactMetadata
from nepflow.errors import StateError
from nepflow.hpc.jobs import (
    ReconciledJobResult,
    SchedulerJobState,
    SlurmJobRecord,
    SubmissionResult,
)
from nepflow.mlip.backend import (
    CollectedModelArtifacts,
    TrainingCompletion,
    TrainingInput,
    TrainingInputRequest,
    TrainingProgress,
)
from nepflow.mlip.nep.backend import NepBackend
from nepflow.mlip.nep.inputs import NepHyperparameters
from nepflow.mlip.nep.outputs import SUCCESS_MARKER_NAME, SUCCESS_MARKER_SCHEMA
from nepflow.stages.training.campaign import TrainingAttempt, TrainingCampaign
from nepflow.stages.training.optimisation import ControlledSweep
from nepflow.state.store import StateStore
from nepflow.workflow.resubmission import SelfResubmitExit


class FakeScheduler:
    def __init__(self) -> None:
        self.states: dict[str, SchedulerJobState] = {}
        self.jobs_by_name: dict[str, SlurmJobRecord] = {}
        self.submissions: list[str] = []
        self.reconcile_sequences: dict[str, list[SchedulerJobState]] = {}
        self.reconcile_sequence_on_submit: list[SchedulerJobState] | None = None
        self.fail_find = False
        self.fail_submit = False

    def find_job_by_name(self, job_name: str, **_kwargs):
        if self.fail_find:
            raise RuntimeError("scheduler query unavailable")
        return self.jobs_by_name.get(job_name)

    def submit_script(self, script_path, *, job_name=None, **_kwargs):
        if self.fail_submit:
            raise RuntimeError("scheduler submission unavailable")
        job_id = f"job-{len(self.submissions) + 1}"
        self.submissions.append(job_id)
        self.states[job_id] = SchedulerJobState.PENDING
        self.jobs_by_name[job_name] = SlurmJobRecord(
            job_id=job_id,
            name=job_name,
            state=SchedulerJobState.PENDING,
        )
        if self.reconcile_sequence_on_submit is not None:
            self.reconcile_sequences[job_id] = list(self.reconcile_sequence_on_submit)
        return SubmissionResult(job_id, "", "", ("sbatch", str(script_path)))

    def reconcile(self, job_id: str):
        sequence = self.reconcile_sequences.get(job_id)
        if sequence:
            state = sequence.pop(0)
            self.states[job_id] = state
        else:
            state = self.states[job_id]
        return ReconciledJobResult(
            job_id=job_id,
            job=SlurmJobRecord(job_id=job_id, name="candidate", state=state),
            source="fake",
        )


class FakeBackend:
    def __init__(self) -> None:
        self.completed = False
        self.progress = TrainingProgress(4, 0.25)

    def model_run_identity(self, inputs):
        return inputs.model_run_identity

    def training_command(self, _inputs):
        return ("nep", "--quiet")

    def parse_progress(self, _run_directory):
        return self.progress

    def parse_completion(self, run_directory):
        artifact = Path(run_directory) / "nep.txt"
        return TrainingCompletion(self.completed, (artifact,) if self.completed else ())

    def collect_model_artifacts(self, run_directory, inputs):
        artifact_path = Path(run_directory) / "nep.txt"
        artifact_path.write_text("model", encoding="utf-8")
        model = ArtifactIdentity.from_file("nep_model", artifact_path)
        return CollectedModelArtifacts(
            training_input=inputs,
            model_run=inputs.model_run_identity,
            artifact=ModelArtifactMetadata(
                model=model,
                nep_in=inputs.nep_in,
                status="completed",
            ),
        )

    def classify_error(self, _run_directory):
        return "backend_failed"


class AttemptDirectoryBackend(FakeBackend):
    """Completion evidence is present only when the active attempt has output."""

    def parse_completion(self, run_directory):
        artifact = Path(run_directory) / "nep.txt"
        return TrainingCompletion(
            artifact.is_file(),
            (artifact,) if artifact.is_file() else (),
        )


def _dataset() -> TrainingDatasetManifest:
    return TrainingDatasetManifest(
        identity=DatasetIdentity(
            "dataset-campaign",
            {"schema_version": "nepflow.dataset.v1", "records": []},
        ),
        records=(),
    )


def _input(tmp_path: Path, name: str) -> TrainingInput:
    run_directory = tmp_path / name
    run_directory.mkdir(parents=True, exist_ok=True)
    content = f"input={name}\n"
    with (run_directory / "nep.in").open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return TrainingInput(
        dataset=_dataset(),
        working_directory=run_directory,
        content=content,
        nep_in=ArtifactIdentity.from_bytes(
            "nep_input",
            content.encode(),
            path=str(run_directory / "nep.in"),
        ),
        hyperparameters_hash=f"hyper-{name}",
    )


def _campaign(
    tmp_path: Path, store: StateStore, scheduler: FakeScheduler, backend: FakeBackend, **kwargs
):
    store.upsert_dataset(_dataset().identity)
    return TrainingCampaign(
        campaign_id="campaign-1",
        dataset_id="dataset-campaign",
        state_store=store,
        scheduler=scheduler,
        backend=backend,
        working_directory=tmp_path / "potentials",
        **kwargs,
    )


def test_single_candidate_state_table_and_exact_artifact(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend)
        candidate = campaign.create_run(_input(tmp_path, "candidate"), ordinal=0)

        submitted = campaign.reconcile()
        assert submitted.candidates[0].status == "submitted"
        assert len(scheduler.submissions) == 1

        scheduler.states[candidate.job_id or "job-1"] = SchedulerJobState.RUNNING
        (candidate.run_directory / "nep.txt").write_text("checkpoint", encoding="utf-8")
        running = campaign.reconcile()
        assert running.candidates[0].status == "running"
        assert running.candidates[0].progress == {
            "segment_generation": 4,
            "segment_target_generations": 0,
            "generation_offset": 0,
            "observed_cumulative_generation": 4,
            "committed_generations": 0,
            "target_generations": 0,
            "loss": 0.25,
        }

        backend.completed = True
        scheduler.states["job-1"] = SchedulerJobState.COMPLETED
        completed = campaign.reconcile()
        assert completed.status == "completed"
        assert completed.candidates[0].status == "completed"
        assert completed.promotion_decision is None
        assert store.list_model_artifacts(candidate.model_run_id)
        assert store.get_model_run(candidate.model_run_id)["status"] == "completed"
        assert [attempt.status for attempt in campaign.attempts(candidate.model_run_id)] == [
            "completed"
        ]


def test_reconcile_until_terminal_polls_one_worker_to_completion(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend)
        campaign.create_run(_input(tmp_path, "candidate"), ordinal=0)
        backend.completed = True
        scheduler.reconcile_sequence_on_submit = [
            SchedulerJobState.PENDING,
            SchedulerJobState.RUNNING,
            SchedulerJobState.COMPLETED,
        ]
        sleeps: list[float] = []

        result = campaign.reconcile_until_terminal(
            poll_interval_seconds=3,
            slurm_deadline=None,
            sleeper=sleeps.append,
        )

    job_id = "job-1"
    assert result.status == "completed"
    assert len(scheduler.submissions) == 1
    assert scheduler.reconcile_sequences[job_id] == []
    assert sleeps == [3, 3]


def test_reconcile_until_terminal_deadline_preserves_active_worker(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend)
        candidate = campaign.create_run(_input(tmp_path, "candidate"), ordinal=0)
        submitted = campaign.reconcile()
        job_id = submitted.candidates[0].job_id or candidate.job_id or "job-1"
        times = iter((0.0, 1.0))

        with pytest.raises(SelfResubmitExit, match="NEP training reconciliation"):
            campaign.reconcile_until_terminal(
                poll_interval_seconds=0,
                slurm_deadline=1.0,
                clock=lambda: next(times),
            )

        persisted = campaign.candidates()[0]

    assert persisted.job_id == job_id
    assert persisted.status == "submitted"
    assert len(scheduler.submissions) == 1


def test_concurrency_and_restart_do_not_duplicate_submission(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(
            tmp_path,
            store,
            scheduler,
            backend,
            max_concurrent=1,
        )
        first = campaign.create_run(_input(tmp_path, "first"), ordinal=0)
        campaign.create_run(_input(tmp_path, "second"), ordinal=1)

        result = campaign.reconcile()
        assert result.counts == {"submitted": 1, "pending": 1}
        assert len(scheduler.submissions) == 1

        restarted = _campaign(
            tmp_path,
            store,
            scheduler,
            backend,
            max_concurrent=1,
        )
        scheduler.states[first.job_id or "job-1"] = SchedulerJobState.RUNNING
        restarted.reconcile()
        assert len(scheduler.submissions) == 1


def test_restart_repairs_candidate_after_persisted_submission(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend)
        candidate = campaign.create_run(_input(tmp_path, "candidate"), ordinal=0)
        scheduler.states["job-recovered"] = SchedulerJobState.PENDING
        campaign._save_attempt(
            TrainingAttempt(
                attempt_id=f"{candidate.model_run_id}:attempt:1",
                model_run_id=candidate.model_run_id,
                attempt_number=1,
                status="submitted",
                job_id="job-recovered",
                job_name="nepflow-campaign-1-recovered",
            )
        )

        result = campaign.reconcile()

        assert result.candidates[0].status == "submitted"
        assert result.candidates[0].job_id == "job-recovered"
        assert len(scheduler.submissions) == 0


def test_scheduler_query_failure_does_not_advance_pending_candidate(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend)
        campaign.create_run(_input(tmp_path, "candidate"), ordinal=0)
        scheduler.fail_find = True

        with pytest.raises(RuntimeError, match="scheduler query unavailable"):
            campaign.reconcile()

        assert campaign.candidates()[0].status == "pending"
        assert campaign.attempts()[0].status == "submitting"
        assert scheduler.submissions == []


def test_scheduler_submission_failure_remains_an_error_and_keeps_intent(
    tmp_path: Path,
) -> None:
    scheduler = FakeScheduler()
    scheduler.fail_submit = True
    backend = FakeBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend)
        campaign.create_run(_input(tmp_path, "candidate"), ordinal=0)

        with pytest.raises(RuntimeError, match="scheduler submission unavailable"):
            campaign.reconcile()

        assert campaign.candidates()[0].status == "pending"
        assert campaign.attempts()[0].status == "submitting"

        scheduler.fail_submit = False
        recovered = campaign.reconcile()

        assert recovered.candidates[0].status == "submitted"
        assert len(scheduler.submissions) == 1


def test_timeout_checkpoint_commits_local_progress_and_submits_remaining_segment(
    tmp_path: Path,
) -> None:
    scheduler = FakeScheduler()
    backend = NepBackend()
    training = NepTrainingConfig(generations=500000, max_resubmit=2)
    composition = CompositionConfig(elements=("W",))
    settings = NepHyperparameters.from_config(composition, training)
    inputs = backend.render_training_input(
        TrainingInputRequest(
            dataset=_dataset(),
            hyperparameters=training,
            composition=composition,
            working_directory=tmp_path / "checkpoint-candidate",
            hyperparameters_hash=settings.identity_hash(),
        )
    )
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend, max_attempts=3)
        candidate = campaign.create_run(inputs, ordinal=0)
        first = campaign.reconcile()
        first_attempt = campaign.attempts(candidate.model_run_id)[0]
        assert first.candidates[0].target_generations == 500000
        assert first_attempt.generation_offset == 0
        assert first_attempt.segment_target_generations == 500000
        assert first_attempt.execution_directory is not None
        assert "generation 500000" in first_attempt.execution_directory.joinpath(
            "nep.in"
        ).read_text(encoding="utf-8")
        first_attempt.execution_directory.joinpath("nep.restart").write_bytes(b"restart-1")
        first_attempt.execution_directory.joinpath("nep.txt").write_text(
            "checkpoint-1\n", encoding="utf-8"
        )
        first_attempt.execution_directory.joinpath("loss.out").write_text(
            "# columns generation loss\n180000 0.2\n", encoding="utf-8"
        )
        scheduler.states[candidate.job_id or "job-1"] = SchedulerJobState.TIMEOUT

        resumed = campaign.reconcile()

        attempts = campaign.attempts(candidate.model_run_id)
        assert attempts[0].status == "checkpointed"
        assert resumed.candidates[0].committed_generations == 180000
        assert attempts[1].generation_offset == 180000
        assert attempts[1].segment_target_generations == 320000
        assert attempts[1].restart_source_attempt_id == attempts[0].attempt_id
        assert attempts[1].execution_directory is not None
        assert "generation 320000" in attempts[1].execution_directory.joinpath(
            "nep.in"
        ).read_text(encoding="utf-8")
        assert attempts[1].execution_directory.joinpath("nep.restart").read_bytes() == b"restart-1"
        assert attempts[1].execution_directory.joinpath("nep.txt").read_text(encoding="utf-8") == (
            "checkpoint-1\n"
        )
        assert not attempts[1].execution_directory.joinpath("loss.out").exists()


def test_failed_candidate_history_is_retained_and_not_promoted(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend, max_attempts=1)
        candidate = campaign.create_run(_input(tmp_path, "candidate"), ordinal=0)
        campaign.reconcile()
        scheduler.states[candidate.job_id or "job-1"] = SchedulerJobState.FAILED

        result = campaign.reconcile()
        assert result.status == "failed"
        assert result.candidates[0].status == "failed"
        assert result.promotion_decision is None
        assert campaign.attempts(candidate.model_run_id)[0].failure_reason == "scheduler_failed"
        assert store.get_model_run(candidate.model_run_id)["status"] == "failed"


def test_cancelled_checkpoint_is_never_promoted(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    backend.completed = True
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend, max_attempts=1)
        candidate = campaign.create_run(_input(tmp_path, "cancelled"), ordinal=0)
        campaign.reconcile()
        (candidate.run_directory / "nep.txt").write_text("checkpoint", encoding="utf-8")
        scheduler.states[candidate.job_id or "job-1"] = SchedulerJobState.CANCELLED

        result = campaign.reconcile()

        assert result.candidates[0].status == "failed"
        assert result.candidates[0].failure_reason == "scheduler_cancelled"
        assert not store.list_model_artifacts(candidate.model_run_id)


def test_completed_scheduler_without_success_marker_is_not_promoted(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = NepBackend()
    training = NepTrainingConfig()
    composition = CompositionConfig(elements=("W",))
    settings = NepHyperparameters.from_config(composition, training)
    inputs = backend.render_training_input(
        TrainingInputRequest(
            dataset=_dataset(),
            hyperparameters=training,
            composition=composition,
            working_directory=tmp_path / "canonical",
            hyperparameters_hash=settings.identity_hash(),
        )
    )
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend, max_attempts=1)
        candidate = campaign.create_run(inputs, ordinal=0)
        campaign.reconcile()
        attempt = campaign.attempts(candidate.model_run_id)[0]
        assert attempt.execution_directory is not None
        attempt.execution_directory.joinpath("nep.txt").write_text("checkpoint\n")
        attempt.execution_directory.joinpath("loss.out").write_text("250000 0.2\n")
        scheduler.states[candidate.job_id or "job-1"] = SchedulerJobState.COMPLETED

        result = campaign.reconcile()

        assert result.candidates[0].status == "failed"
        assert not store.list_model_artifacts(candidate.model_run_id)


def test_completed_scheduler_with_valid_current_attempt_evidence_is_promoted(
    tmp_path: Path,
) -> None:
    scheduler = FakeScheduler()
    backend = NepBackend()
    training = NepTrainingConfig()
    composition = CompositionConfig(elements=("W",))
    settings = NepHyperparameters.from_config(composition, training)
    inputs = backend.render_training_input(
        TrainingInputRequest(
            dataset=_dataset(),
            hyperparameters=training,
            composition=composition,
            working_directory=tmp_path / "canonical-valid",
            hyperparameters_hash=settings.identity_hash(),
        )
    )
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend, max_attempts=1)
        candidate = campaign.create_run(inputs, ordinal=0)
        campaign.reconcile()
        attempt = campaign.attempts(candidate.model_run_id)[0]
        assert attempt.execution_directory is not None
        attempt.execution_directory.joinpath("nep.txt").write_text("model\n")
        attempt.execution_directory.joinpath("loss.out").write_text("250000 0.2\n")
        attempt.execution_directory.joinpath(SUCCESS_MARKER_NAME).write_text(
            json.dumps(
                {
                    "schema_version": SUCCESS_MARKER_SCHEMA,
                    "model_run_id": candidate.model_run_id,
                    "attempt_id": attempt.attempt_id,
                    "attempt_number": attempt.attempt_number,
                    "execution_generation": attempt.execution_generation,
                    "generation_offset": attempt.generation_offset,
                    "segment_target_generations": attempt.segment_target_generations,
                    "process_exit_code": 0,
                    "completion_timestamp": "2026-11-02T10:00:00Z",
                }
            ),
            encoding="utf-8",
        )
        scheduler.states[candidate.job_id or "job-1"] = SchedulerJobState.COMPLETED

        result = campaign.reconcile()

        assert result.candidates[0].status == "completed"
        row = store.get_model_run(candidate.model_run_id)
        assert row is not None
        assert row["status"] == "completed"
        assert row["execution_metadata"]["completion_evidence"]["attempt_id"] == (
            attempt.attempt_id
        )


def test_retry_keeps_model_run_nonterminal_until_attempts_are_exhausted(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend, max_attempts=2)
        candidate = campaign.create_run(_input(tmp_path, "retry"), ordinal=0)
        campaign.reconcile()
        scheduler.states[candidate.job_id or "job-1"] = SchedulerJobState.FAILED

        result = campaign.reconcile()

        assert result.candidates[0].status == "submitted"
        assert len(scheduler.submissions) == 2
        assert store.get_model_run(candidate.model_run_id)["status"] == "prepared"


def test_retry_restart_uses_a_distinct_discoverable_scheduler_name(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend, max_attempts=2)
        candidate = campaign.create_run(_input(tmp_path, "restart-retry"), ordinal=0)
        campaign.reconcile()
        first_job = campaign.candidates()[0].job_id or "job-1"
        scheduler.states[first_job] = SchedulerJobState.FAILED

        campaign.reconcile()
        attempts = campaign.attempts(candidate.model_run_id)
        assert len(attempts) == 2
        assert attempts[0].job_name != attempts[1].job_name
        assert attempts[0].job_name.endswith("-a0001")
        assert attempts[1].job_name.endswith("-a0002")

        restarted = _campaign(tmp_path, store, scheduler, backend, max_attempts=2)
        restarted.reconcile()
        assert len(scheduler.submissions) == 2


def test_stale_output_from_failed_attempt_cannot_complete_retry(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = AttemptDirectoryBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend, max_attempts=2)
        candidate = campaign.create_run(_input(tmp_path, "stale-output"), ordinal=0)
        campaign.reconcile()
        first_job = campaign.candidates()[0].job_id or "job-1"
        (candidate.run_directory / "nep.txt").write_text("stale", encoding="utf-8")
        scheduler.states[first_job] = SchedulerJobState.FAILED
        campaign.reconcile()
        second_job = campaign.candidates()[0].job_id or "job-2"
        scheduler.states[second_job] = SchedulerJobState.FAILED

        result = campaign.reconcile()
        assert result.candidates[0].status == "failed"
        assert campaign.attempts(candidate.model_run_id)[1].status == "failed"


def test_campaign_persists_execution_policy_and_rejects_matrix_changes(
    tmp_path: Path,
) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    specification = {
        "candidate_keys": ["candidate-a"],
        "candidate_matrix": [{"ordinal": 0, "candidate_key": "candidate-a"}],
        "max_attempts": 2,
        "max_concurrent": 1,
        "backend": {"kind": "fake", "command_fingerprint": "command-a"},
        "resource_policy": {"walltime": "01:00:00"},
    }
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend)
        campaign.ensure(specification)
        campaign.ensure({**specification, "max_concurrent": 2})

        events = store.list_training_events("training_campaign", "campaign-1")
        assert any(event["event_type"] == "policy_updated" for event in events)

        with pytest.raises(StateError, match="conflicting immutable"):
            campaign.ensure(
                {
                    **specification,
                    "candidate_keys": ["candidate-b"],
                    "candidate_matrix": [{"ordinal": 0, "candidate_key": "candidate-b"}],
                }
            )


def test_scheduler_disappearance_is_not_completion_evidence(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    backend.completed = True
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend, max_attempts=1)
        candidate = campaign.create_run(_input(tmp_path, "disappeared"), ordinal=0)
        campaign.reconcile()
        scheduler.states[candidate.job_id or "job-1"] = SchedulerJobState.NOT_FOUND

        result = campaign.reconcile()

        assert result.status == "failed"
        assert result.candidates[0].status == "failed"
        assert not store.list_model_artifacts(candidate.model_run_id)


def test_controlled_sweep_is_deterministic_and_identity_bearing() -> None:
    base = NepTrainingConfig()
    sweep = ControlledSweep.from_mapping(
        base,
        {
            "cutoff": [("6", "5"), ("7", "5")],
            "lambda_f": [1.0, 2.0],
        },
    )
    first = sweep.configurations()
    second = sweep.configurations()
    assert [candidate.candidate_key for candidate in first] == [
        candidate.candidate_key for candidate in second
    ]
    assert len({candidate.candidate_key for candidate in first}) == 4
    assert first[0].hyperparameters != first[-1].hyperparameters


def test_model_run_identity_changes_only_with_effective_input_identity(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    backend = FakeBackend()
    with StateStore(tmp_path / "state.db") as store:
        campaign = _campaign(tmp_path, store, scheduler, backend)
        first = campaign.create_run(_input(tmp_path, "same"), ordinal=0)
        same = campaign.create_run(_input(tmp_path, "same"), ordinal=1)
        changed = campaign.create_run(_input(tmp_path, "changed"), ordinal=2)

        assert same.model_run_id == first.model_run_id
        assert changed.model_run_id != first.model_run_id
