"""Thin backend-neutral DFT workflow stage."""

from __future__ import annotations

import logging
import shlex
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

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
from nepflow.hpc.resources import (
    SITE_ENVIRONMENT_PROLOGUE,
    JobResources,
    load_site_environment_prologue,
    render_shell_script,
    require_resolvable_command_without_prologue,
)
from nepflow.hpc.scheduler import Scheduler
from nepflow.io.atomic import atomic_write_text
from nepflow.io.json import to_jsonable, write_json
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


def _write_failure_report(
    project_dir: Path,
    state_store: object,
    preparation: DftPreparationResult,
    execution: DftReconciliationResult,
) -> Path | None:
    """Persist terminal DFT failures using the authoritative StateStore rows."""

    failed_records = tuple(
        record
        for record in execution.records
        if record.status in {"failed", "cancelled"}
    )
    if not failed_records:
        return None

    prepared_by_calculation = {
        item.calculation.calculation_id: item for item in preparation.calculations
    }
    get_calculation = getattr(state_store, "get_dft_calculation", None)
    list_attempts = getattr(state_store, "list_dft_attempts", None)
    get_attempt = getattr(state_store, "get_dft_attempt", None)
    failures: list[dict[str, object]] = []
    for record in failed_records:
        calculation_id = record.inputs.calculation.calculation_id
        calculation = get_calculation(calculation_id) if callable(get_calculation) else None
        if callable(get_calculation) and calculation is None:
            raise StateError(f"Missing authoritative DFT calculation state: {calculation_id}")
        if calculation is not None and not isinstance(calculation, Mapping):
            raise StateError(f"Malformed DFT calculation state: {calculation_id}")
        item = prepared_by_calculation.get(calculation_id)
        calculation_metadata = (
            calculation.get("metadata", {})
            if isinstance(calculation, Mapping)
            else {}
        )
        if not isinstance(calculation_metadata, Mapping):
            raise StateError(f"Malformed DFT calculation metadata: {calculation_id}")

        attempt = get_attempt(record.attempt_id) if callable(get_attempt) else None
        if attempt is not None and not isinstance(attempt, Mapping):
            raise StateError(f"Malformed DFT attempt state: {record.attempt_id}")
        if attempt is None and callable(list_attempts):
            attempts = list_attempts(calculation_id)
            if not isinstance(attempts, (list, tuple)):
                raise StateError(f"Malformed DFT attempt history: {calculation_id}")
            attempt = next(
                (
                    candidate
                    for candidate in attempts
                    if isinstance(candidate, Mapping)
                    and candidate.get("attempt_id") == record.attempt_id
                ),
                None,
            )
        if callable(get_attempt) and attempt is None:
            raise StateError(f"Missing authoritative DFT attempt state: {record.attempt_id}")
        if callable(list_attempts) and attempt is None:
            raise StateError(f"Missing authoritative DFT attempt state: {record.attempt_id}")

        evidence = record.failure_evidence
        if isinstance(attempt, Mapping) and attempt.get("failure_evidence") is not None:
            evidence = attempt.get("failure_evidence")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise StateError(f"Malformed DFT failure evidence: {record.attempt_id}")
        evidence = {} if evidence is None else dict(evidence)
        attempt_metadata = (
            attempt.get("metadata", {}) if isinstance(attempt, Mapping) else {}
        )
        if not isinstance(attempt_metadata, Mapping):
            raise StateError(f"Malformed DFT attempt metadata: {record.attempt_id}")

        dataset = calculation_metadata.get("dataset")
        selected_index = calculation_metadata.get("selected_index")
        working_directory = attempt_metadata.get("working_directory")
        if dataset is None and item is not None:
            dataset = item.spec.dataset
        if selected_index is None and item is not None:
            selected_index = item.spec.selected_index
        if working_directory is None:
            working_directory = str(record.inputs.working_directory)
        failures.append(
            {
                "dataset": dataset,
                "selected_structure_index": selected_index,
                "structure_id": record.inputs.calculation.structure_id,
                "calculation_id": calculation_id,
                "status": record.status,
                "final_attempt_id": record.attempt_id,
                "attempt_number": (
                    attempt.get("attempt_number")
                    if isinstance(attempt, Mapping)
                    else record.attempt_number
                ),
                "retry_level": attempt_metadata.get("retry_level", record.retry_level),
                "failure_kind": evidence.get("kind"),
                "failure_reason": evidence.get("reason"),
                "scheduler_state": evidence.get("scheduler_state"),
                "scheduler_reason": evidence.get("scheduler_reason"),
                "scheduler_exit_code": evidence.get("scheduler_exit_code"),
                "working_directory": working_directory,
            }
        )

    failures.sort(
        key=lambda item: (
            str(item.get("dataset", "")),
            (
                -1
                if item.get("selected_structure_index") is None
                else int(item["selected_structure_index"])
            ),
            str(item.get("calculation_id", "")),
        )
    )
    report_path = Path(project_dir) / "reports" / "dft_campaign_failures.json"
    write_json(
        report_path,
        {
            "schema_version": "nepflow.dft_campaign_failure_report.v1",
            "summary": {
                "total": execution.total_count,
                "completed": execution.completed_count,
                "reused": execution.reused_count,
                "failed": execution.failed_count,
                "cancelled": execution.cancelled_count,
                "successful": execution.successful_count,
            },
            "failures": to_jsonable(failures),
        },
    )
    logger.warning(
        "Persisted DFT campaign failure report with %d terminal failures: %s",
        len(failures),
        report_path,
    )
    return report_path


@dataclass(frozen=True, slots=True)
class DftStageResult:
    """Typed result returned after DFT preparation."""

    preparation: DftPreparationResult
    execution: DftReconciliationResult | None = None
    stage: WorkflowStage = WorkflowStage.RUN_VASP
    status: StageRunState = StageRunState.RUNNING
    failure_report_path: Path | None = None

    def as_workflow_result(self) -> StageRunResult:
        execution = self.execution
        if execution is None or not execution.all_terminal:
            status = StageRunState.RUNNING
            complete = False
            advanced_to = None
            reconciliation_message = "reconciliation pending"
        elif execution.successful_count:
            status = StageRunState.COMPLETED
            complete = True
            advanced_to = WorkflowStage.TRAIN_NEP
            reconciliation_message = "reconciliation complete"
            if execution.failed_count or execution.cancelled_count:
                logger.warning(
                    "DFT campaign completed with %d terminal calculation failures; "
                    "proceeding with %d successful calculations",
                    execution.failed_count + execution.cancelled_count,
                    execution.successful_count,
                )
        else:
            status = StageRunState.FAILED
            complete = False
            advanced_to = None
            reconciliation_message = (
                "reconciliation complete but no usable completed/reused DFT results remain"
            )

        summary = (
            f"Resolved {0 if execution is None else execution.total_count} DFT calculations; "
            f"{0 if execution is None else execution.completed_count} completed; "
            f"{0 if execution is None else execution.reused_count} reused; "
            f"{0 if execution is None else execution.failed_count} failed; "
            f"{0 if execution is None else execution.cancelled_count} cancelled; "
            f"{0 if execution is None else execution.non_terminal_count} non-terminal; "
            f"{reconciliation_message}"
        )
        if (
            execution is not None
            and execution.all_terminal
            and not execution.successful_count
        ):
            summary += "; cannot advance to training without usable DFT data"
        if self.failure_report_path is not None:
            summary += f"; failure report: {self.failure_report_path}"
        return StageRunResult(
            stage=self.stage,
            status=status,
            advanced_to=advanced_to,
            completed=complete,
            message=summary,
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
        site_prologue = load_site_environment_prologue(active_context.project_dir)
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
            require_resolvable_command_without_prologue(
                backend.executable_identity,
                site_prologue,
                source=active_context.project_dir / SITE_ENVIRONMENT_PROLOGUE,
            )
            runner_path = active_context.project_dir / "vasp" / "run_vasp.sh"
            runner_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                runner_path,
                render_shell_script(
                    backend.render_runner_script(),
                    site_prologue=site_prologue,
                ),
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
        failure_report_path = _write_failure_report(
            active_context.project_dir,
            state_store,
            preparation,
            execution,
        )
        return DftStageResult(
            preparation=preparation,
            execution=execution,
            failure_report_path=failure_report_path,
        )

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
