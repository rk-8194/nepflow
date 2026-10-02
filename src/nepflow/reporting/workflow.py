"""Presentation of workflow state for CLI/reporting callers."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Mapping
import logging
from pathlib import Path
from typing import Any

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

StatusDetailsProvider = Callable[[Path, WorkflowStage], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class WorkflowStatus:
    """A presentation-safe snapshot of authoritative workflow state."""

    stage: WorkflowStage
    stage_run: StageRunStatus
    label: str
    details: Mapping[str, Any] = field(default_factory=dict)


class WorkflowStatusPresenter:
    """Render workflow state without owning orchestration or stage policy."""

    def __init__(
        self,
        logger_: logging.Logger | None = None,
        *,
        details_provider: StatusDetailsProvider | None = None,
    ) -> None:
        self.logger = logger_ or logging.getLogger(__name__)
        self.details_provider = details_provider

    def snapshot(self, workflow_state: WorkflowState) -> WorkflowStatus:
        stage = workflow_state.current_stage()
        row = workflow_state.store.get_latest_stage_run(workflow_state.project_id)
        if row is None:  # pragma: no cover - current_stage already guards this
            raise RuntimeError("Workflow stage disappeared while presenting status")
        details = (
            {}
            if self.details_provider is None
            else dict(self.details_provider(workflow_state.marker_path.parent, stage))
        )
        return WorkflowStatus(
            stage=stage,
            stage_run=StageRunStatus.from_mapping(row),
            label=_STAGE_LABELS[stage],
            details=details,
        )

    def log(self, workflow_state: WorkflowState) -> WorkflowStatus:
        status = self.snapshot(workflow_state)
        self.logger.info("NEPFlow · project: %s", workflow_state.project_id)
        self.logger.info("Stage: %s", status.label)
        self.logger.info("Stage status: %s", status.stage_run.status.value)
        for dataset, counts in status.details.get("vasp_jobs", {}).items():
            self.logger.info(
                "VASP %-8s total=%d done=%d running=%d pending=%d failed=%d",
                dataset,
                counts["total"],
                counts["completed"],
                counts["submitted"],
                counts["pending"],
                counts["failed"],
            )
        return status


def summarize_legacy_vasp_jobs(
    project_dir: Path,
    stage: WorkflowStage,
    read_status: Callable[[Path], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Build the old VASP job-count view through an injected status reader."""

    if stage is not WorkflowStage.RUN_VASP:
        return {}
    jobs_dir = project_dir / "vasp" / "jobs"
    counts: dict[str, dict[str, int]] = {}
    for dataset in ("train", "test"):
        dataset_dir = jobs_dir / dataset
        if not dataset_dir.exists():
            continue
        tally = {"completed": 0, "submitted": 0, "pending": 0, "failed": 0}
        for structure_dir in (
            directory
            for directory in dataset_dir.iterdir()
            if directory.is_dir() and directory.name.startswith("struct_")
        ):
            status = read_status(structure_dir).get("status", "pending")
            if status == "reused":
                status = "completed"
            tally[status if status in tally else "pending"] += 1
        failed_dir = dataset_dir / "failed"
        if failed_dir.exists():
            tally["failed"] += sum(
                1
                for directory in failed_dir.iterdir()
                if directory.is_dir() and directory.name.startswith("struct_")
            )
        counts[dataset] = tally
    return {"vasp_jobs": counts} if counts else {}


__all__ = [
    "StatusDetailsProvider",
    "WorkflowStatus",
    "WorkflowStatusPresenter",
    "summarize_legacy_vasp_jobs",
]
