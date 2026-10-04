from pathlib import Path
from types import SimpleNamespace

import pytest

from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.errors import StateError
from nepflow.stages.dft.orchestrator import DftPreparationResult
from nepflow.stages.dft.reconciliation import DftExecutionRecord, DftReconciliationResult
from nepflow.stages.dft.stage import DftStage
from nepflow.stages.training.campaign import CampaignReconciliationResult
from nepflow.stages.training.stage import TrainingStage
from nepflow.stages.validation.reconciliation import (
    ValidationExecutionRecord,
    ValidationReconciliationResult,
)
from nepflow.stages.validation.stage import ValidationStageResult
from nepflow.state import StateStore
from nepflow.workflow import (
    StageRegistry,
    StageRunResult,
    StageRunState,
    WorkflowController,
    WorkflowStage,
)

VALID_PROJECT_CONFIG = (
    """
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
""".strip()
    + "\n"
)


def build_test_registry(
    *,
    stage: str = "validate",
    stage_handler=None,
) -> tuple[StageRegistry, dict[WorkflowStage, object]]:
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
        return StageRunResult(
            stage=WorkflowStage.VALIDATE,
            status=StageRunState.RUNNING,
            completed=False,
        )

    registry.register(WorkflowStage.VALIDATE, validate_handler, replace=True)
    if stage_handler is not None:
        registry.register(
            WorkflowStage.from_legacy(stage),
            stage_handler,
            replace=True,
        )
    return registry, handlers


def make_controller(
    tmp_path: Path,
    stage: str = "validate",
    *,
    debug: bool = False,
    stage_handler=None,
) -> WorkflowController:
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
    registry, handlers = build_test_registry(
        stage=stage,
        stage_handler=stage_handler,
    )
    controller = WorkflowController(
        project_name="demo",
        output_dir=output_dir,
        debug=debug,
        stage_registry=registry,
    )
    controller.test_handlers = handlers
    return controller


def reopen_controller(
    tmp_path: Path,
    *,
    debug: bool = False,
) -> WorkflowController:
    output_dir = tmp_path / "outputs"
    project_dir = output_dir / "project_demo"
    assert (project_dir / "config" / "project.config").is_file()
    assert (project_dir / "state.db").is_file()
    registry, _ = build_test_registry()
    return WorkflowController(
        project_name="demo",
        output_dir=output_dir,
        debug=debug,
        stage_registry=registry,
    )


def test_completed_stage_is_read_from_project_file(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="completed")

    assert controller.current_stage() is WorkflowStage.COMPLETED


def test_completed_rerun_invokes_no_stage(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="completed")

    controller.run()

    assert controller.current_stage() is WorkflowStage.COMPLETED


def test_validation_error_does_not_mark_completed(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    error = RuntimeError("validation failed")

    controller.test_handlers[WorkflowStage.VALIDATE] = lambda _context: (_ for _ in ()).throw(error)
    with pytest.raises(RuntimeError, match="validation failed"):
        controller.run()

    assert controller.current_stage() is WorkflowStage.VALIDATE


def test_earlier_stage_progression_is_unchanged(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="generate")

    controller.run()

    assert controller.current_stage() is WorkflowStage.SELECT


def test_corrupt_workflow_stage_is_not_reset_to_init(tmp_path: Path) -> None:
    controller = make_controller(tmp_path, stage="not-a-stage")

    with pytest.raises(StateError, match="Invalid workflow stage"):
        controller.current_stage()


def _assert_failed_controller_run_does_not_advance(
    tmp_path: Path,
    stage: WorkflowStage,
    result: StageRunResult,
    downstream: WorkflowStage,
) -> None:
    controller = make_controller(
        tmp_path,
        stage=stage.value,
        stage_handler=lambda _context: result,
    )

    controller.run()

    assert controller.current_stage() is stage
    with StateStore(controller.state_file) as store:
        stage_row = store.get_stage_run(f"demo:{stage.value}")
        assert stage_row is not None
        assert stage_row["status"] == StageRunState.FAILED.value
        assert store.get_stage_run(f"demo:{downstream.value}") is None

    controller.project_file.unlink()
    reopened = reopen_controller(tmp_path)
    assert reopened.current_stage() is stage
    with StateStore(reopened.state_file) as store:
        assert store.get_stage_run(f"demo:{stage.value}")["status"] == (StageRunState.FAILED.value)
        assert store.get_stage_run(f"demo:{downstream.value}") is None


def test_terminal_dft_failure_stays_failed_after_controller_reopen(tmp_path: Path) -> None:
    class FailedPreparation:
        def prepare_calculations(self, **_kwargs):
            return DftPreparationResult(())

    class FailedExecution:
        def reconcile_once(self, _records):
            return DftReconciliationResult(
                (
                    DftExecutionRecord(
                        inputs=SimpleNamespace(),
                        attempt_id="dft-attempt-failed",
                        status="failed",
                    ),
                )
            )

    results = []

    def failed_dft_handler(context):
        result = (
            DftStage(
                orchestrator=FailedPreparation(),
                execution_orchestrator=FailedExecution(),
            )
            .run(context)
            .as_workflow_result()
        )
        results.append(result)
        return result

    controller = make_controller(
        tmp_path,
        stage=WorkflowStage.RUN_VASP.value,
        stage_handler=failed_dft_handler,
    )
    controller.run()

    assert results[0].status is StageRunState.FAILED
    assert results[0].advanced_to is None
    assert controller.current_stage() is WorkflowStage.RUN_VASP
    with StateStore(controller.state_file) as store:
        assert store.get_stage_run("demo:run_vasp")["status"] == StageRunState.FAILED.value
        assert store.get_stage_run("demo:train_nep") is None

    controller.project_file.unlink()
    reopened = reopen_controller(tmp_path)
    assert reopened.current_stage() is WorkflowStage.RUN_VASP
    with StateStore(reopened.state_file) as store:
        assert store.get_stage_run("demo:run_vasp")["status"] == StageRunState.FAILED.value
        assert store.get_stage_run("demo:train_nep") is None


def test_terminal_training_failure_stays_failed_after_controller_reopen(tmp_path: Path) -> None:
    dataset = TrainingDatasetManifest(
        DatasetIdentity.from_identity_payload(
            {"schema_version": "nepflow.dataset.v1", "records": []}
        ),
        (),
    )

    class FailedCampaign:
        def __init__(self, **_kwargs):
            pass

        def ensure(self, _specification):
            return None

        def snapshot(self):
            return CampaignReconciliationResult(
                campaign_id="campaign-failed",
                status="failed",
                candidates=(SimpleNamespace(status="failed", terminal=True),),
            )

    results = []

    def failed_training_handler(context):
        context.state_store.upsert_dataset(dataset)
        stage = TrainingStage(
            backend=SimpleNamespace(),
            campaign_factory=FailedCampaign,
        )
        stage._assemble_dataset = lambda _context, _config, _store: (
            dataset,
            context.project_dir / "dataset",
        )
        stage._prepare_candidates = lambda *_args: None
        result = stage.run(context)
        results.append(result)
        return result

    controller = make_controller(
        tmp_path,
        stage=WorkflowStage.TRAIN_NEP.value,
        stage_handler=failed_training_handler,
    )
    controller.run()

    assert results[0].status is StageRunState.FAILED
    assert results[0].advanced_to is None
    assert controller.current_stage() is WorkflowStage.TRAIN_NEP
    with StateStore(controller.state_file) as store:
        assert store.get_stage_run("demo:train_nep")["status"] == StageRunState.FAILED.value
        assert store.get_stage_run("demo:validate") is None

    controller.project_file.unlink()
    reopened = reopen_controller(tmp_path)
    assert reopened.current_stage() is WorkflowStage.TRAIN_NEP
    with StateStore(reopened.state_file) as store:
        assert store.get_stage_run("demo:train_nep")["status"] == StageRunState.FAILED.value
        assert store.get_stage_run("demo:validate") is None


def test_terminal_validation_failure_stays_failed_after_controller_reopen(tmp_path: Path) -> None:
    validation_result = ValidationStageResult(
        preparation=SimpleNamespace(),
        execution=ValidationReconciliationResult(
            validation_run_id="validation-failed",
            status="failed",
            cases=(
                ValidationExecutionRecord(
                    validation_run_id="validation-failed",
                    case=SimpleNamespace(),
                    attempt_id="validation-attempt-failed",
                    attempt_number=1,
                    status="failed",
                ),
            ),
        ),
    ).as_workflow_result()

    assert validation_result.status is StageRunState.FAILED
    assert validation_result.advanced_to is None
    _assert_failed_controller_run_does_not_advance(
        tmp_path,
        WorkflowStage.VALIDATE,
        validation_result,
        WorkflowStage.COMPLETED,
    )
