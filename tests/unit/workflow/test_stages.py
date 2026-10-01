import pytest

from nepflow.errors import StateError
from nepflow.workflow.stages import (
    StageRunState,
    StageRunStatus,
    WorkflowStage,
    is_valid_transition,
    parse_legacy_stage,
    stage_to_legacy,
    validate_transition,
)


def test_legacy_stage_conversion_is_canonical() -> None:
    for stage in WorkflowStage:
        assert parse_legacy_stage(stage.value) is stage
        assert stage_to_legacy(stage) == stage.value


def test_every_adjacent_transition_is_valid() -> None:
    ordered = list(WorkflowStage)
    for previous, target in zip(ordered, ordered[1:]):
        assert validate_transition(previous, target) is target
        assert is_valid_transition(previous, target)


def test_same_stage_resume_is_valid() -> None:
    assert validate_transition(WorkflowStage.VALIDATE, "validate") is WorkflowStage.VALIDATE


@pytest.mark.parametrize(
    ("previous", "target"),
    [
        (WorkflowStage.SELECT, WorkflowStage.GENERATE),
        (WorkflowStage.GENERATE, WorkflowStage.RUN_VASP),
        (WorkflowStage.VALIDATE, WorkflowStage.COMPLETED),
        (WorkflowStage.COMPLETED, WorkflowStage.VALIDATE),
    ],
)
def test_backward_skip_and_terminal_transitions_are_rejected(
    previous: WorkflowStage,
    target: WorkflowStage,
) -> None:
    with pytest.raises(StateError):
        validate_transition(previous, target)


def test_unknown_legacy_stage_never_defaults_to_init() -> None:
    with pytest.raises(StateError):
        parse_legacy_stage("not-a-stage")


def test_stage_run_status_round_trips_typed_values() -> None:
    record = StageRunStatus.from_mapping(
        {
            "stage_run_id": "demo:validate",
            "project_id": "demo",
            "stage": "validate",
            "status": "running",
            "metadata": {"source": "test"},
        }
    )

    assert record.stage is WorkflowStage.VALIDATE
    assert record.status is StageRunState.RUNNING
    assert record.to_dict()["stage"] == "validate"
