from nepflow.workflow.resubmission import (
    ReconciliationResult,
    ResubmissionResult,
    SelfResubmitExit,
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
