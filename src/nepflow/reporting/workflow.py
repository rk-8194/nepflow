"""Presentation of workflow state for CLI/reporting callers."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from nepflow.workflow.stages import StageRunStatus, WorkflowStage
from nepflow.workflow.state import WorkflowState


_STAGE_LABELS = {
    WorkflowStage.INIT: "1/6  init",
    WorkflowStage.GENERATE: "2/6  generate",
    WorkflowStage.SELECT: "3/6  select",
    WorkflowStage.RUN_VASP: "4/6  run_vasp",
    WorkflowStage.TRAIN_NEP: "5/6  train_nep",
    WorkflowStage.VALIDATE: "6/6  validate",
    WorkflowStage.COMPLETED: "completed",
}

@dataclass(frozen=True, slots=True)
class WorkflowStatus:
    """A presentation-safe snapshot of authoritative workflow state."""

    stage: WorkflowStage
    stage_run: StageRunStatus
    label: str


class WorkflowStatusPresenter:
    """Render workflow state without owning orchestration or stage policy."""

    def __init__(
        self,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.logger = logger_ or logging.getLogger(__name__)

    def snapshot(self, workflow_state: WorkflowState) -> WorkflowStatus:
        stage = workflow_state.current_stage()
        row = workflow_state.store.get_latest_stage_run(workflow_state.project_id)
        if row is None:  # pragma: no cover - current_stage already guards this
            raise RuntimeError("Workflow stage disappeared while presenting status")
        return WorkflowStatus(
            stage=stage,
            stage_run=StageRunStatus.from_mapping(row),
            label=_STAGE_LABELS[stage],
        )

    def log(self, workflow_state: WorkflowState) -> WorkflowStatus:
        status = self.snapshot(workflow_state)
        self.logger.info("NEPFlow · project: %s", workflow_state.project_id)
        self.logger.info("Stage: %s", status.label)
        self.logger.info("Stage status: %s", status.stage_run.status.value)
        return status


__all__ = [
    "WorkflowStatus",
    "WorkflowStatusPresenter",
]
