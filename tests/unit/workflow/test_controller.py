from pathlib import Path
from unittest.mock import patch

import pytest

from nepflow.errors import StateError
from nepflow.state import StateStore
from nepflow.workflow.stages import WorkflowStage
from workflow import WorkflowController


def make_controller(tmp_path: Path, marker: str | None = None) -> WorkflowController:
    project_dir = tmp_path / "outputs" / "project_demo"
    (project_dir / "config").mkdir(parents=True)
    if marker is not None:
        (project_dir / ".project").write_text(marker, encoding="utf-8")
    return WorkflowController("demo", tmp_path / "outputs")


def test_controller_persists_reconciled_stage_run(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "validate")

    assert controller._determine_current_stage() is WorkflowStage.VALIDATE
    with StateStore(controller.state_file) as store:
        row = store.get_latest_stage_run("demo")

    assert row is not None
    assert row["stage"] == "validate"


def test_completed_controller_rerun_stays_terminal(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "completed")

    assert controller._determine_current_stage() is WorkflowStage.COMPLETED
    with pytest.raises(StateError):
        controller._set_current_stage(WorkflowStage.VALIDATE)


def test_marker_cannot_override_authoritative_store(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "generate")
    assert controller._determine_current_stage() is WorkflowStage.GENERATE
    controller.project_file.write_text("validate", encoding="utf-8")

    with pytest.raises(StateError, match="contradicts authoritative"):
        controller._determine_current_stage()


def test_invalid_marker_does_not_default_to_init(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "not-a-stage")

    with pytest.raises(ValueError, match="Invalid workflow stage"):
        controller._determine_current_stage()


def test_failed_transition_validation_leaves_previous_stage(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "generate")
    assert controller._determine_current_stage() is WorkflowStage.GENERATE

    with pytest.raises(StateError):
        controller._set_current_stage(WorkflowStage.VALIDATE)

    assert controller._determine_current_stage() is WorkflowStage.GENERATE
    assert controller.project_file.read_text(encoding="utf-8") == "generate"


def test_state_write_failure_rolls_back_previous_authority(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "generate")
    assert controller._determine_current_stage() is WorkflowStage.GENERATE

    with patch.object(
        controller._state_store,
        "upsert_stage_run",
        side_effect=RuntimeError("ledger write failed"),
    ):
        with pytest.raises(RuntimeError, match="ledger write failed"):
            controller._set_current_stage(WorkflowStage.SELECT)

    assert controller._determine_current_stage() is WorkflowStage.GENERATE
    assert controller.project_file.read_text(encoding="utf-8") == "generate"
