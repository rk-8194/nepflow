"""Thin backend-neutral DFT workflow stage."""

from __future__ import annotations

import logging
import shlex
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from nepflow.config import NepflowConfig
from nepflow.dft.backend import DftBackend
from nepflow.dft.vasp.backend import VaspBackend
from nepflow.dft.vasp.resource_memory import (
    VaspMemoryObservation,
    default_vasp_resources,
    extract_resource_features,
    load_vasp_memory,
    predict_vasp_params,
    resolve_memory_path,
)
from nepflow.dft.vasp.recovery import VaspRecoveryPolicy
from nepflow.errors import StateError, VaspError
from nepflow.hpc.resources import JobResources
from nepflow.hpc.scheduler import Scheduler
from nepflow.io.atomic import atomic_write_text
from nepflow.workflow.controller import StageContext
from nepflow.workflow.stages import StageRunResult, StageRunState, WorkflowStage

from .orchestrator import (
    DftPreparationOrchestrator,
    DftPreparationResult,
    PreparedCalculation,
    VaspPreparationOrchestrator,
    _attempt_number,
    _attempt_retry_level,
    _scientific_attempts,
)
from .reconciliation import (
    DftExecutionRecord,
    DftReconciliationOrchestrator,
    DftReconciliationResult,
    DftRecoveryDecision,
)
from .reports import DftPerformanceRecorder

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DftStageResult:
    """Typed result returned after DFT preparation."""

    preparation: DftPreparationResult
    execution: DftReconciliationResult | None = None
    stage: WorkflowStage = WorkflowStage.RUN_VASP
    status: StageRunState = StageRunState.RUNNING

    def as_workflow_result(self) -> StageRunResult:
        execution = self.execution
        failed = execution is not None and execution.all_terminal and execution.any_failed
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

        memory_path = resolve_memory_path(active_context.project_dir)
        memory: tuple[VaspMemoryObservation, ...] | None = None

        def get_memory() -> tuple[VaspMemoryObservation, ...]:
            nonlocal memory
            if memory is None:
                memory = load_vasp_memory(memory_path)
                logger.info(
                    "Resolved VASP resource memory %s (%d observations; %d successful)",
                    memory_path,
                    len(memory),
                    sum(item.oom != 1 for item in memory),
                )
            return memory

        if any(
            (
                not item.reused
                and (attempts := _scientific_attempts(state_store, item.calculation.calculation_id))
                and _attempt_retry_level(attempts[-1]) == 0
                and str(attempts[-1].get("status", "")) in {"pending", "prepared"}
            )
            for item in preparation.calculations
        ):
            get_memory()

        records = tuple(
            self._execution_record_for_item(
                item,
                state_store,
                config=config,
                project_root=active_context.project_dir,
                memory_loader=get_memory,
            )
            for item in preparation.calculations
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
            atomic_write_text(
                runner_path,
                "#!/usr/bin/env bash\n" + backend.render_runner_script(),
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
                performance_recorder=DftPerformanceRecorder(
                    state_store,
                    memory_path=memory_path,
                    project_root=active_context.project_dir,
                    config=config,
                ),
            )
        reconcile_until_terminal = getattr(
            execution_orchestrator,
            "reconcile_until_terminal",
            None,
        )
        if callable(reconcile_until_terminal):
            execution = reconcile_until_terminal(
                records,
                poll_interval_seconds=config.slurm.poll_interval,
                slurm_deadline=active_context.slurm_deadline,
            )
        else:
            # Keep injected legacy test doubles preparation-only/non-executing
            # compatible.  Production composition always supplies the typed
            # DftReconciliationOrchestrator above.
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
            metadata = getattr(record, "metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            decision_metadata = metadata.get("resource_decision", {})
            if not isinstance(decision_metadata, Mapping):
                decision_metadata = {}
            starting_gpus_per_node = (
                int(
                    decision_metadata.get(
                        "initial_gpus_per_node",
                        record.resources.gpus_per_node if record.resources is not None else 1,
                    )
                )
            )
            initial_ncore = int(
                decision_metadata.get(
                    "initial_ncore",
                    record.inputs.calculation.ncore
                    or max(2, config.hpc.cores_per_node // max(1, starting_gpus_per_node)),
                )
            )
            initial_kpar = int(decision_metadata.get("initial_kpar", record.inputs.calculation.kpar or 1))
            initial_nodes = int(
                decision_metadata.get(
                    "initial_nodes", record.resources.nodes if record.resources is not None else 1
                )
            )
            try:
                decision = vasp_policy.decide(
                    starting_gpu=starting_gpus_per_node,
                    initial_ncore=initial_ncore,
                    initial_kpar=initial_kpar,
                    initial_nodes=initial_nodes,
                    retry_level=record.retry_level,
                )
            except TypeError as exc:
                if "initial_nodes" not in str(exc):
                    raise
                decision = vasp_policy.decide(
                    starting_gpu=starting_gpus_per_node,
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
            nodes = decision.nodes or 1
            gpus_per_node = decision.gpus_per_node or starting_gpus_per_node
            resources = JobResources(
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                mpi_ranks=nodes * gpus_per_node,
                walltime=config.dft_recovery.vasp_walltime,
                account=config.slurm.account,
                partition=config.slurm.dft_partition,
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
            retry_level=(_attempt_retry_level(latest) if latest is not None else 0),
            attempt_number=(_attempt_number(latest) if latest is not None else 1),
            resources=resources,
        )

    @classmethod
    def _execution_record_for_item(
        cls,
        item: PreparedCalculation,
        state_store,
        *,
        config: NepflowConfig,
        project_root,
        memory_loader,
    ) -> DftExecutionRecord:
        """Build one record, selecting and persisting resources for new attempts."""

        attempts = _scientific_attempts(state_store, item.calculation.calculation_id)
        latest = attempts[-1] if attempts else None
        persisted = cls._persisted_resource_decision(latest)
        if (
            persisted is None
            and latest is not None
            and _attempt_retry_level(latest) == 0
            and str(latest.get("status", "")) in {"pending", "prepared"}
            and not item.reused
        ):
            try:
                features = extract_resource_features(
                    item.artifacts.working_directory,
                    project_root,
                    config=config,
                )
                observations = memory_loader()
                prediction = predict_vasp_params(
                    observations,
                    n_atoms=features[0],
                    n_electrons=features[1],
                    n_kpoints_irr=features[2],
                    config=config,
                )
                reason = "sufficient .vasp_memory data" if prediction is not None else "insufficient .vasp_memory data"
            except (OSError, UnicodeError, StateError, VaspError, ValueError) as exc:
                prediction = None
                features = None
                reason = f"feature estimation unavailable: {exc}"
                logger.warning(
                    "Could not derive VASP resource features for %s (%s): %s",
                    item.calculation.calculation_id,
                    item.artifacts.working_directory,
                    exc,
                )
            if prediction is None:
                prediction = default_vasp_resources(config)
                source = f"default: {reason}"
            else:
                source = prediction.source
            from nepflow.dft.vasp.recovery import write_incar_resource_parameters

            write_incar_resource_parameters(
                item.artifacts.working_directory,
                prediction.ncore,
                prediction.kpar,
            )
            resources = JobResources(
                nodes=prediction.nodes,
                gpus_per_node=prediction.gpus_per_node,
                mpi_ranks=prediction.mpi_ranks,
                walltime=config.dft_recovery.vasp_walltime,
                account=config.slurm.account,
                partition=config.slurm.dft_partition,
            )
            metadata = dict(latest.get("metadata", {})) if isinstance(latest.get("metadata"), Mapping) else {}
            metadata["resource_decision"] = {
                "initial_ncore": prediction.ncore,
                "initial_kpar": prediction.kpar,
                "initial_nodes": prediction.nodes,
                "initial_gpus_per_node": prediction.gpus_per_node,
                "resource_source": prediction.source,
                "n_atoms": None if features is None else features[0],
                "n_electrons": None if features is None else features[1],
                "n_kpoints_irr": None if features is None else features[2],
                "features": (
                    {
                        "n_atoms": features[0],
                        "n_electrons": features[1],
                        "n_kpoints_irr": features[2],
                    }
                    if features is not None
                    else None
                ),
            }
            state_store.update_dft_attempt(
                str(latest["attempt_id"]),
                resources=asdict(resources),
                metadata=metadata,
            )
            logger.info(
                "%s/%s: NCORE=%d KPAR=%d nodes=%d gpus=%d%s (%s)",
                item.spec.dataset,
                item.spec.working_directory.name,
                prediction.ncore,
                prediction.kpar,
                prediction.nodes,
                prediction.gpus_per_node,
                f" kpts={features[2]}" if features is not None else "",
                source,
            )
            persisted = (resources, metadata)
        elif persisted is None:
            metadata = dict(latest.get("metadata", {})) if latest and isinstance(latest.get("metadata"), Mapping) else {}
            resources = cls._resources_from_attempt(latest, config)
            persisted = (resources, metadata)

        resources, metadata = persisted
        return cls._execution_record(item, state_store, resources).with_metadata(metadata)

    @staticmethod
    def _persisted_resource_decision(latest):
        if not latest or not isinstance(latest.get("resources"), Mapping):
            return None
        metadata = latest.get("metadata", {})
        decision = metadata.get("resource_decision") if isinstance(metadata, Mapping) else None
        if not isinstance(decision, Mapping):
            # A resource row from an older state is still safer to reuse than
            # silently predicting a different submitted attempt.
            return None
        return DftStage._resources_from_attempt(latest, None), dict(metadata)

    @staticmethod
    def _resources_from_attempt(latest, config) -> JobResources:
        if latest and isinstance(latest.get("resources"), Mapping):
            values = dict(latest["resources"])
            values.pop("total_gpus", None)
            return JobResources(**values)
        if config is None:
            return JobResources()
        return JobResources(
            nodes=1,
            gpus_per_node=1,
            mpi_ranks=1,
            walltime=config.dft_recovery.vasp_walltime,
            account=config.slurm.account,
            partition=config.slurm.dft_partition,
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
