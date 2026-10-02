from pathlib import Path
from unittest.mock import patch

import pytest

from modules.validate.launcher import write_validation_status
from nepflow.errors import StateError
from nepflow.state import StateStore
from workflow import WorkflowController


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
    return WorkflowController(
        project_name="demo",
        output_dir=output_dir,
        debug=debug,
    )


def test_completed_stage_is_read_from_project_file(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="completed")

    assert controller._determine_current_stage() == "completed"


def test_completed_rerun_invokes_no_stage(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="completed")

    with patch.object(controller, "_validate") as validate_mock:
        controller.run()

    validate_mock.assert_not_called()


def test_complete_validation_advances_to_completed(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    write_validation_status(
        controller.project_dir,
        validation_complete=True,
        analysis_complete=True,
    )

    with patch.object(controller, "_validate") as validate_mock:
        controller.run()

    validate_mock.assert_called_once_with()
    assert controller._determine_current_stage() == "completed"


def test_incomplete_validation_remains_at_validate(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    write_validation_status(
        controller.project_dir,
        validation_complete=False,
        analysis_complete=True,
    )

    with patch.object(controller, "_validate") as validate_mock:
        controller.run()

    validate_mock.assert_called_once_with()
    assert controller._determine_current_stage() == "validate"


def test_incomplete_analysis_remains_at_validate(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    write_validation_status(
        controller.project_dir,
        validation_complete=True,
        analysis_complete=False,
    )

    with patch.object(controller, "_validate") as validate_mock:
        controller.run()

    validate_mock.assert_called_once_with()
    assert controller._determine_current_stage() == "validate"


def test_validation_error_does_not_mark_completed(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    error = RuntimeError("validation failed")

    with patch.object(controller, "_validate", side_effect=error):
        with pytest.raises(RuntimeError, match="validation failed"):
            controller.run()

    assert controller._determine_current_stage() == "validate"


def test_corrupt_validation_status_propagates_without_completion(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    status_file = controller.project_dir / "gpumd" / ".validation_status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text("{malformed", encoding="utf-8")

    with patch.object(controller, "_validate") as validate_mock:
        with pytest.raises(StateError):
            controller.run()

    validate_mock.assert_called_once_with()
    assert controller._determine_current_stage() == "validate"


def test_earlier_stage_progression_is_unchanged(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="generate")

    with patch.object(controller, "_generate") as generate_mock:
        controller.run()

    generate_mock.assert_called_once_with()
    assert controller._determine_current_stage() == "select"


def test_debug_completion_uses_persisted_validation_evidence(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, debug=True)
    write_validation_status(
        controller.project_dir,
        validation_complete=True,
        analysis_complete=True,
    )

    with (
        patch.object(controller, "_generate"),
        patch.object(controller, "_select"),
        patch.object(controller, "_run_vasp"),
        patch.object(controller, "_train_nep"),
        patch.object(controller, "_validate"),
    ):
        controller.run()

    assert controller._determine_current_stage() == "completed"


def test_corrupt_workflow_stage_is_not_reset_to_init(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="not-a-stage")

    with pytest.raises(StateError, match="Invalid workflow stage"):
        controller._determine_current_stage()


def test_corrupt_vasp_status_is_not_reported_as_pending(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="run_vasp")
    status_file = controller.project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status"
    status_file.parent.mkdir(parents=True)
    status_file.write_text("{malformed", encoding="utf-8")

    with pytest.raises(StateError):
        controller._print_status_summary()
