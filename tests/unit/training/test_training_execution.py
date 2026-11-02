from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nepflow.hpc.jobs import SubmissionResult
from nepflow.hpc.resources import JobResources
from nepflow.stages.training.execution import TrainingExecution


def test_training_script_records_success_only_after_backend_command(tmp_path: Path) -> None:
    execution = TrainingExecution(
        backend=object(),
        resources=None,
        input_for=lambda _candidate: None,
    )

    script = execution._script_content(
        tmp_path,
        ("nep", "--quiet"),
        model_run_id="model-run-1",
        attempt_id="model-run-1:attempt:1",
        attempt_number=1,
        execution_generation=2,
        generation_offset=180000,
        segment_target_generations=320000,
    )

    command_position = script.index("nep --quiet")
    marker_position = script.index("mv -f --")
    assert "set -euo pipefail" in script
    assert "rm -f -- \"$success_marker\"" in script
    assert ".nepflow_training_success.json" in script
    assert "model-run-1" in script
    assert '"generation_offset":180000' in script
    assert '"segment_target_generations":320000' in script
    assert command_position < marker_position


def test_later_execution_generation_uses_a_distinct_attempt_directory(tmp_path: Path) -> None:
    candidate = type("Candidate", (), {"run_directory": tmp_path})()

    assert TrainingExecution.attempt_directory(candidate, 1) == tmp_path / "attempts" / "a0001"
    assert TrainingExecution.attempt_directory(candidate, 1, 2) == (
        tmp_path / "attempts" / "g0002" / "a0001"
    )


def test_worker_submission_receives_typed_training_resources(tmp_path: Path) -> None:
    resources = JobResources(
        nodes=1,
        gpus_per_node=1,
        mpi_ranks=1,
        walltime="24:00:00",
        account="project",
        partition="training-gpu",
    )

    class Scheduler:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def submit_script(self, _script_path, **kwargs):
            self.kwargs = kwargs
            return SubmissionResult("job-1", "", "", ("sbatch", "worker.sh"))

    scheduler = Scheduler()
    execution = TrainingExecution(
        backend=object(),
        resources=resources,
        input_for=lambda _candidate: None,
    )
    attempt = SimpleNamespace(
        script_path=tmp_path / "train_nep.sh",
        execution_directory=tmp_path,
    )

    assert execution.submit(scheduler, attempt, "nepflow-worker") == "job-1"
    assert scheduler.kwargs is not None
    assert scheduler.kwargs["resources"] == resources
    submitted_resources = scheduler.kwargs["resources"]
    assert isinstance(submitted_resources, JobResources)
    assert submitted_resources.walltime == "24:00:00"
    assert scheduler.kwargs["job_name"] == "nepflow-worker"
