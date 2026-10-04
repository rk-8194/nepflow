from nepflow.workflow.resubmission import (
    ReconciliationResult,
    ResubmissionResult,
    SelfResubmitExit,
    submit_resubmission,
)
from nepflow.workflow.stages import WorkflowStage


def test_resubmission_exception_has_typed_result() -> None:
    error = SelfResubmitExit("walltime reached")

    assert isinstance(error.result, ResubmissionResult)
    assert error.result.requested is True
    assert error.result.reason == "walltime reached"


def test_reconciliation_result_has_typed_stages() -> None:
    result = ReconciliationResult(
        stage="completed",
        source="state_store",
        marker_stage="completed",
        authoritative_stage="completed",
    )

    assert result.stage is WorkflowStage.COMPLETED
    assert result.authoritative_stage is WorkflowStage.COMPLETED


def test_resubmission_without_scheduler_uses_process_runner_boundary(tmp_path) -> None:
    class FakeProcessRunner:
        def __init__(self) -> None:
            self.calls = []

        def run(self, command, **kwargs):
            self.calls.append((tuple(command), kwargs))
            return type("Result", (), {"stdout": "Submitted batch job 123\n"})()

    runner = FakeProcessRunner()
    output = submit_resubmission(
        ["sbatch", "submit.slurm"],
        tmp_path,
        process_runner=runner,  # type: ignore[arg-type]
    )

    assert output == "Submitted batch job 123"
    assert runner.calls == [
        (
            ("sbatch", "submit.slurm"),
            {"cwd": tmp_path, "check": True, "capture_output": True},
        )
    ]
