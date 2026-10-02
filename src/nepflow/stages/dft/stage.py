"""Thin backend-neutral DFT workflow stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex

from nepflow.config import NepflowConfig, load_config
from nepflow.dft.backend import DftBackend
from nepflow.hpc.slurm import SlurmScheduler
from nepflow.workflow.controller import StageContext
from nepflow.workflow.stages import StageRunResult, StageRunState, WorkflowStage

from .orchestrator import (
    DftPreparationOrchestrator,
    DftPreparationResult,
    VaspPreparationOrchestrator,
)


@dataclass(frozen=True, slots=True)
class DftStageResult:
    """Typed result returned after DFT preparation."""

    preparation: DftPreparationResult
    stage: WorkflowStage = WorkflowStage.RUN_VASP
    status: StageRunState = StageRunState.RUNNING

    def as_workflow_result(self) -> StageRunResult:
        return StageRunResult(
            stage=self.stage,
            status=self.status,
            completed=False,
            message=(
                f"Prepared {self.preparation.prepared_count} calculations; "
                f"{self.preparation.reused_count} reused"
            ),
        )


class DftStage:
    """Validate context, prepare calculations, and delegate orchestration."""

    def __init__(
        self,
        context: StageContext | None = None,
        *,
        backend: DftBackend | None = None,
        scheduler: object | None = None,
        orchestrator: DftPreparationOrchestrator | None = None,
        datasets: tuple[str, ...] = ("train", "test"),
    ) -> None:
        self.context = context
        self.backend = backend
        self.scheduler = scheduler
        self.orchestrator = orchestrator
        self.datasets = datasets

    def run(self, context: StageContext | None = None) -> DftStageResult:
        active_context = context or self.context
        if active_context is None:
            raise ValueError("DftStage requires a StageContext")
        config = self._validate_inputs(active_context)
        state_store = active_context.state_store
        if state_store is None:
            raise TypeError("DftStage requires the controller StateStore")

        orchestrator = self.orchestrator or VaspPreparationOrchestrator(
            backend=self.backend,
            scheduler=self.scheduler or SlurmScheduler(),
        )
        preparation = orchestrator.prepare_calculations(
            config=config,
            project_dir=active_context.project_dir,
            project_name=active_context.project_name,
            state_store=state_store,
            datasets=self.datasets,
        )
        return DftStageResult(preparation=preparation)

    @staticmethod
    def _validate_inputs(context: StageContext) -> NepflowConfig:
        config = context.config
        if config is None:
            config = load_config(
                Path(context.config_file),
                project_name=context.project_name,
            )
        if not isinstance(config, NepflowConfig):
            raise TypeError("DftStage requires typed NepflowConfig context")
        if not shlex.split(config.hpc.vasp_command):
            raise ValueError("Required configuration hpc.vasp_command must not be blank")
        if context.project_name != config.project.name:
            raise ValueError(
                "DFT stage project context does not match the typed configuration"
            )
        return config


__all__ = ["DftStage", "DftStageResult"]
