"""Thin backend-neutral DFT workflow stage."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from nepflow.config import NepflowConfig
from nepflow.dft.backend import DftBackend
from nepflow.dft.vasp.backend import VaspBackend
from nepflow.dft.vasp.recovery import VaspRecoveryPolicy
from nepflow.errors import StateError
from nepflow.hpc.resources import JobResources
from nepflow.hpc.scheduler import Scheduler
from nepflow.workflow.controller import StageContext
from nepflow.workflow.stages import StageRunResult, StageRunState, WorkflowStage

from .orchestrator import (
    DftPreparationOrchestrator,
    DftPreparationResult,
    PreparedCalculation,
    VaspPreparationOrchestrator,
    _attempt_number,
    _scientific_attempts,
)
from .reconciliation import (
    DftExecutionRecord,
    DftReconciliationOrchestrator,
    DftReconciliationResult,
    DftRecoveryDecision,
)
from .reports import DftPerformanceRecorder


@dataclass(frozen=True, slots=True)
class DftStageResult:
    """Typed result returned after DFT preparation."""

    preparation: DftPreparationResult
    execution: DftReconciliationResult | None = None
    stage: WorkflowStage = WorkflowStage.RUN_VASP
    status: StageRunState = StageRunState.RUNNING

    def as_workflow_result(self) -> StageRunResult:
        execution = self.execution
        failed = execution is not None and execution.any_failed
        complete = execution is not None and execution.all_successful
        status = (
            StageRunState.FAILED if failed else StageRunState.COMPLETED if complete else self.status
        )
        return StageRunResult(
            stage=self.stage,
            status=status,
            advanced_to=WorkflowStage.TRAIN_NEP if complete else None,
            completed=complete,
            message=(
                f"Prepared {self.preparation.prepared_count} calculations; "
                f"{self.preparation.reused_count} reused"
                + (
                    "; reconciliation failed"
                    if failed
                    else "; reconciliation complete"
                    if complete
                    else "; reconciliation pending"
                )
            ),
        )


class DftStage:
    """Validate context, prepare calculations, and delegate orchestration."""

    def __init__(
        self,
        context: StageContext | None = None,
        *,
        backend: DftBackend | None = None,
        scheduler: Scheduler | None = None,
        orchestrator: DftPreparationOrchestrator | None = None,
        execution_orchestrator: DftReconciliationOrchestrator | None = None,
        datasets: tuple[str, ...] = ("train", "test"),
    ) -> None:
        self.context = context
        self.backend = backend
        self.scheduler = scheduler
        self.orchestrator = orchestrator
        self.execution_orchestrator = execution_orchestrator
        self.datasets = datasets

    def run(self, context: StageContext | None = None) -> DftStageResult:
        active_context = context or self.context
        if active_context is None:
            raise ValueError("DftStage requires a StageContext")
        config = self._validate_inputs(active_context)
        state_store = active_context.state_store
        if state_store is None:
            raise TypeError("DftStage requires the controller StateStore")

        scheduler = self.scheduler
        backend = self.backend
        if backend is None and self.orchestrator is None:
            backend = VaspPreparationOrchestrator._build_backend(
                config,
                active_context.project_dir,
            )
        orchestrator = self.orchestrator
        if orchestrator is None:
            orchestrator = VaspPreparationOrchestrator(
                backend=backend,
                scheduler=scheduler,
            )
        preparation = orchestrator.prepare_calculations(
            config=config,
            project_dir=active_context.project_dir,
            project_name=active_context.project_name,
            state_store=state_store,
            datasets=self.datasets,
        )
        if self.orchestrator is not None and self.execution_orchestrator is None:
            return DftStageResult(preparation=preparation)

        execution_orchestrator = self.execution_orchestrator
        if execution_orchestrator is None:
            if scheduler is None:
                raise StateError("DftStage requires an injected scheduler for execution")
            if not isinstance(backend, VaspBackend):
                raise TypeError("Default DFT execution requires a backend with a runner renderer")
            runner_path = active_context.project_dir / "vasp" / "run_vasp.sh"
            runner_path.parent.mkdir(parents=True, exist_ok=True)
            runner_path.write_text(
                "#!/usr/bin/env bash\nset -e\n" + backend.render_runner_script(),
                encoding="utf-8",
            )
            execution_orchestrator = DftReconciliationOrchestrator(
                scheduler=scheduler,
                backend=backend,
                state_store=state_store,
                max_concurrent=config.slurm.max_concurrent,
                retry_limit=config.dft_recovery.max_retry_level,
                script_path=runner_path,
                job_name_prefix=f"nf_{active_context.project_name}_",
                recovery_policy=self._build_recovery_policy(config),
                performance_recorder=DftPerformanceRecorder(state_store),
            )

        resources = JobResources(
            nodes=1,
            gpus_per_node=1,
            mpi_ranks=1,
            walltime=config.dft_recovery.vasp_walltime,
        )
        records = tuple(
            self._execution_record(item, state_store, resources)
            for item in preparation.calculations
        )
        execution = execution_orchestrator.reconcile_once(records)
        return DftStageResult(preparation=preparation, execution=execution)

    @staticmethod
    def _build_recovery_policy(config: NepflowConfig):
        vasp_policy = VaspRecoveryPolicy(config)

        def decide(record, failure) -> DftRecoveryDecision:
            if not failure.recoverable:
                return DftRecoveryDecision(
                    retry=False,
                    retry_level=record.retry_level,
                    reason=failure.reason,
                )
            starting_gpu = record.resources.gpus_per_node if record.resources is not None else 1
            initial_ncore = record.inputs.calculation.ncore or max(
                2,
                config.hpc.cores_per_node // max(1, starting_gpu),
            )
            initial_kpar = record.inputs.calculation.kpar or 1
            decision = vasp_policy.decide(
                starting_gpu=starting_gpu,
                initial_ncore=initial_ncore,
                initial_kpar=initial_kpar,
                retry_level=record.retry_level,
            )
            if not decision.retry:
                return DftRecoveryDecision(
                    retry=False,
                    retry_level=decision.retry_level,
                    reason=decision.reason,
                )
            vasp_policy.apply(decision, record.inputs.working_directory)
            resources = JobResources(
                nodes=decision.nodes or 1,
                gpus_per_node=decision.gpus or starting_gpu,
                mpi_ranks=decision.gpus or starting_gpu,
                walltime=config.dft_recovery.vasp_walltime,
            )
            return DftRecoveryDecision(
                retry=True,
                retry_level=decision.retry_level,
                reason=decision.reason,
                resources=resources,
            )

        return decide

    @staticmethod
    def _execution_record(
        item: PreparedCalculation,
        state_store,
        resources: JobResources,
    ) -> DftExecutionRecord:
        attempts = _scientific_attempts(state_store, item.calculation.calculation_id)
        latest = attempts[-1] if attempts else None
        status = item.status
        if item.reused:
            status = "reused"
        elif latest is not None:
            status = str(latest.get("status", status))
        if status == "prepared":
            status = "pending"
        raw_job_id = None if latest is None else latest.get("job_id")
        job_id = raw_job_id if isinstance(raw_job_id, str) else None
        return DftExecutionRecord(
            inputs=item.artifacts,
            attempt_id=(
                str(latest["attempt_id"])
                if latest is not None
                else f"{item.calculation.calculation_id}:attempt:1"
            ),
            status=status,
            job_id=job_id,
            job_name=(
                f"nf_{item.spec.project_name}_{item.spec.dataset}_{item.spec.selected_index:04d}"
            ),
            retry_level=(max(0, _attempt_number(latest) - 1) if latest is not None else 0),
            resources=resources,
        )

    @staticmethod
    def _validate_inputs(context: StageContext) -> NepflowConfig:
        config = context.config
        if not isinstance(config, NepflowConfig):
            raise TypeError("DftStage requires the injected typed NepflowConfig context")
        if not shlex.split(config.hpc.vasp_command):
            raise ValueError("Required configuration hpc.vasp_command must not be blank")
        if context.project_name != config.project.name:
            raise ValueError("DFT stage project context does not match the typed configuration")
        return config


__all__ = ["DftStage", "DftStageResult"]
