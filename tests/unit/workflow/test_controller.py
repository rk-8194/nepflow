from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from nepflow.errors import StateError
from nepflow.state import StateStore
from nepflow.workflow import (
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


def write_valid_config(project_dir: Path) -> None:
    config_path = project_dir / "config" / "project.config"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(VALID_PROJECT_CONFIG, encoding="utf-8")


def make_controller(tmp_path: Path, marker: str | None = None) -> WorkflowController:
    project_dir = tmp_path / "outputs" / "project_demo"
    write_valid_config(project_dir)
    if marker is not None:
        (project_dir / ".project").write_text(marker, encoding="utf-8")
    with StateStore(project_dir / "state.db") as store:
        store.upsert_project(
            "demo",
            name="demo",
            root_path=str(project_dir),
        )
    return WorkflowController("demo", tmp_path / "outputs")


def test_normal_startup_with_config_but_no_state_fails_without_creating_state(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "outputs" / "project_demo"
    write_valid_config(project_dir)

    with pytest.raises(StateError, match="state database is missing"):
        WorkflowController("demo", tmp_path / "outputs")

    assert not (project_dir / "state.db").exists()


def test_normal_startup_with_state_but_no_project_fails_without_inserting_one(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "outputs" / "project_demo"
    write_valid_config(project_dir)
    state_path = project_dir / "state.db"
    with StateStore(state_path):
        pass

    with pytest.raises(StateError, match="no project record"):
        WorkflowController("demo", tmp_path / "outputs")

    with StateStore(state_path) as store:
        assert store.get_project("demo") is None


def test_normal_startup_with_corrupt_state_fails_without_replacing_it(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "outputs" / "project_demo"
    write_valid_config(project_dir)
    state_path = project_dir / "state.db"
    state_path.write_bytes(b"not a sqlite database")

    with pytest.raises(StateError, match="state database"):
        WorkflowController("demo", tmp_path / "outputs")

    assert state_path.read_bytes() == b"not a sqlite database"


def test_controller_persists_reconciled_stage_run(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "validate")

    assert controller.current_stage() is WorkflowStage.VALIDATE
    with StateStore(controller.state_file) as store:
        row = store.get_latest_stage_run("demo")

    assert row is not None
    assert row["stage"] == "validate"


def test_stale_marker_is_repaired_from_authoritative_store(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "generate")
    assert controller.current_stage() is WorkflowStage.GENERATE
    with StateStore(controller.state_file) as store:
        before = store.get_stage_run("demo:generate")

    controller.project_file.write_text("validate", encoding="utf-8")
    result = controller.reconcile_stage()

    assert result.stage is WorkflowStage.GENERATE
    assert result.changed is True
    assert result.reason == "repaired stale legacy marker from StateStore"
    assert controller.project_file.read_text(encoding="utf-8") == "generate"
    with StateStore(controller.state_file) as store:
        assert store.get_stage_run("demo:generate") == before


def test_missing_marker_is_repaired_from_authoritative_store(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "generate")
    assert controller.current_stage() is WorkflowStage.GENERATE
    controller.project_file.unlink()

    result = controller.reconcile_stage()

    assert result.stage is WorkflowStage.GENERATE
    assert result.changed is True
    assert result.reason == "repaired missing legacy marker from StateStore"
    assert controller.project_file.read_text(encoding="utf-8") == "generate"


def test_transition_persists_completed_and_running_statuses(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "generate")
    assert controller.current_stage() is WorkflowStage.GENERATE

    controller.transition_to(WorkflowStage.SELECT)

    with StateStore(controller.state_file) as store:
        generate = store.get_stage_run("demo:generate")
        select = store.get_stage_run("demo:select")
        latest = store.get_latest_stage_run("demo")

    assert generate is not None and generate["status"] == "completed"
    assert select is not None and select["status"] == "running"
    assert latest is not None and latest["stage"] == "select"
    assert controller.project_file.read_text(encoding="utf-8") == "select"


def test_completion_persists_terminal_status_and_skips_validation(
    tmp_path: Path,
) -> None:
    controller = make_controller(tmp_path, "validate")
    assert controller.current_stage() is WorkflowStage.VALIDATE

    controller.transition_to(WorkflowStage.COMPLETED)

    with StateStore(controller.state_file) as store:
        validate = store.get_stage_run("demo:validate")
        completed = store.get_stage_run("demo:completed")
        latest = store.get_latest_stage_run("demo")

    assert validate is not None and validate["status"] == "completed"
    assert completed is not None and completed["status"] == "completed"
    assert latest is not None and latest["stage"] == "completed"
    assert controller.project_file.read_text(encoding="utf-8") == "completed"

    validation_handler = Mock(
        return_value=StageRunResult(
            stage=WorkflowStage.VALIDATE,
            status=StageRunState.COMPLETED,
            advanced_to=WorkflowStage.COMPLETED,
            completed=True,
        )
    )
    controller.stage_registry.register(WorkflowStage.VALIDATE, validation_handler)
    with patch.object(
        controller.stage_registry,
        "execute",
        wraps=controller.stage_registry.execute,
    ) as execute_mock:
        controller.run()
    execute_mock.assert_not_called()
    validation_handler.assert_not_called()


def test_completed_controller_rerun_stays_terminal(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "completed")

    assert controller.current_stage() is WorkflowStage.COMPLETED
    with pytest.raises(StateError):
        controller.transition_to(WorkflowStage.VALIDATE)


def test_invalid_marker_does_not_default_to_init(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "not-a-stage")

    with pytest.raises(StateError, match="Invalid workflow stage"):
        controller.current_stage()


def test_failed_transition_validation_leaves_previous_stage(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "generate")
    assert controller.current_stage() is WorkflowStage.GENERATE

    with pytest.raises(StateError):
        controller.transition_to(WorkflowStage.VALIDATE)

    assert controller.current_stage() is WorkflowStage.GENERATE
    assert controller.project_file.read_text(encoding="utf-8") == "generate"


def test_state_write_failure_rolls_back_previous_authority(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, "generate")
    assert controller.current_stage() is WorkflowStage.GENERATE

    with patch.object(
        controller._state_store,
        "upsert_stage_run",
        side_effect=RuntimeError("ledger write failed"),
    ):
        with pytest.raises(RuntimeError, match="ledger write failed"):
            controller.transition_to(WorkflowStage.SELECT)

    assert controller.current_stage() is WorkflowStage.GENERATE
    assert controller.project_file.read_text(encoding="utf-8") == "generate"
