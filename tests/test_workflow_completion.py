from pathlib import Path
import pytest

from modules.validate.launcher import read_validation_status, write_validation_status
from nepflow.errors import StateError
from nepflow.reporting import WorkflowStatusPresenter
from nepflow.state import StateStore
from nepflow.workflow import (
    StageRegistry,
    StageRunResult,
    StageRunState,
    WorkflowController,
    WorkflowStage,
)


VALID_PROJECT_CONFIG = """
[project]
name=demo
schema_version=1

[composition]
elements=W
composition_step=0.125

[generation]
crystal_structures=bcc
target_n_atoms=64

[hpc]
vasp_command=vasp_std
""".strip() + "\n"


def make_controller(tmp_path: Path, stage: str = "validate", *, debug: bool = False) -> WorkflowController:
    output_dir = tmp_path / "outputs"
    project_dir = output_dir / "project_demo"
    config_path = project_dir / "config" / "project.config"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(VALID_PROJECT_CONFIG, encoding="utf-8")
    (project_dir / ".project").write_text(stage, encoding="utf-8")
    with StateStore(project_dir / "state.db") as store:
        store.upsert_project(
            "demo",
            name="demo",
            root_path=str(project_dir),
        )
    registry = StageRegistry()
    handlers = {
        workflow_stage: (lambda _context: None)
        for workflow_stage in WorkflowStage
        if not workflow_stage.is_terminal
    }

    for workflow_stage in handlers:
        registry.register(
            workflow_stage,
            lambda context, workflow_stage=workflow_stage: handlers[workflow_stage](context),
        )

    def validate_handler(context):
        handlers[WorkflowStage.VALIDATE](context)
        status = read_validation_status(context.project_dir)
        complete = (
            status.get("validation_complete") is True
            and status.get("analysis_complete") is True
        )
        return StageRunResult(
            stage=WorkflowStage.VALIDATE,
            status=StageRunState.COMPLETED if complete else StageRunState.RUNNING,
            advanced_to=WorkflowStage.COMPLETED if complete else None,
            completed=complete,
        )

    registry.register(WorkflowStage.VALIDATE, validate_handler, replace=True)
    controller = WorkflowController(
        project_name="demo",
        output_dir=output_dir,
        debug=debug,
        stage_registry=registry,
    )
    controller.test_handlers = handlers
    return controller


def test_completed_stage_is_read_from_project_file(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="completed")

    assert controller.current_stage() is WorkflowStage.COMPLETED


def test_completed_rerun_invokes_no_stage(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="completed")

    controller.run()

    assert controller.current_stage() is WorkflowStage.COMPLETED


def test_complete_validation_advances_to_completed(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    write_validation_status(
        controller.project_dir,
        validation_complete=True,
        analysis_complete=True,
    )

    controller.run()

    assert controller.test_handlers[WorkflowStage.VALIDATE] is not None
    assert controller.current_stage() is WorkflowStage.COMPLETED


def test_incomplete_validation_remains_at_validate(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    write_validation_status(
        controller.project_dir,
        validation_complete=False,
        analysis_complete=True,
    )

    controller.run()

    assert controller.current_stage() is WorkflowStage.VALIDATE


def test_incomplete_analysis_remains_at_validate(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    write_validation_status(
        controller.project_dir,
        validation_complete=True,
        analysis_complete=False,
    )

    controller.run()

    assert controller.current_stage() is WorkflowStage.VALIDATE


def test_validation_error_does_not_mark_completed(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    error = RuntimeError("validation failed")

    controller.test_handlers[WorkflowStage.VALIDATE] = lambda _context: (_ for _ in ()).throw(error)
    with pytest.raises(RuntimeError, match="validation failed"):
        controller.run()

    assert controller.current_stage() is WorkflowStage.VALIDATE


def test_corrupt_validation_status_propagates_without_completion(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    status_file = controller.project_dir / "gpumd" / ".validation_status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text("{malformed", encoding="utf-8")

    with pytest.raises(StateError):
        controller.run()

    assert controller.current_stage() is WorkflowStage.VALIDATE


def test_earlier_stage_progression_is_unchanged(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="generate")

    controller.run()

    assert controller.current_stage() is WorkflowStage.SELECT


def test_debug_completion_uses_persisted_validation_evidence(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, debug=True)
    write_validation_status(
        controller.project_dir,
        validation_complete=True,
        analysis_complete=True,
    )

    controller.run()

    assert controller.current_stage() is WorkflowStage.COMPLETED


def test_corrupt_workflow_stage_is_not_reset_to_init(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="not-a-stage")

    with pytest.raises(StateError, match="Invalid workflow stage"):
        controller.current_stage()


def test_corrupt_vasp_status_is_not_reported_as_pending(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="run_vasp")
    status_file = controller.project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text("{malformed", encoding="utf-8")

    status = WorkflowStatusPresenter().snapshot(controller.workflow_state)
    assert status.stage is WorkflowStage.RUN_VASP
