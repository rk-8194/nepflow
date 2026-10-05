"""Canonical validation-stage orchestration over StateStore case events."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from nepflow.domain.identities import ValidationRunIdentity
from nepflow.errors import StateError
from nepflow.hpc.resources import JobResources
from nepflow.hpc.scheduler import Scheduler
from nepflow.mlip.gpumd import GpumdBackend
from nepflow.mlip.simulation import StaticPrediction, StaticPredictionBackend
from nepflow.workflow import StageContext, StageRunResult, StageRunState, WorkflowStage

from .metrics import ValidationMetrics, calculate_metrics, pair_prediction
from .preparation import prepare_validation_cases
from .protocols import VALIDATION_RESULT_SCHEMA, ValidationPreparation
from .reconciliation import (
    ValidationReconciliationOrchestrator,
    ValidationReconciliationResult,
)
from .resolution import resolve_model_dataset


@dataclass(frozen=True, slots=True)
class ValidationStageResult:
    """Typed result of preparation, reconciliation, and optional metrics."""

    preparation: ValidationPreparation
    execution: ValidationReconciliationResult
    metrics: ValidationMetrics | None = None
    stage: WorkflowStage = WorkflowStage.VALIDATE

    def as_workflow_result(self) -> StageRunResult:
        failed = self.execution.any_failed
        complete = self.execution.all_successful and self.metrics is not None
        return StageRunResult(
            stage=self.stage,
            status=(
                StageRunState.FAILED
                if failed
                else StageRunState.COMPLETED
                if complete
                else StageRunState.RUNNING
            ),
            advanced_to=WorkflowStage.COMPLETED if complete else None,
            completed=complete,
            message=(
                "Validation reconciliation failed"
                if failed
                else "Validation reconciliation and metrics complete"
                if complete
                else "Validation reconciliation pending"
            ),
        )


class ValidationStage:
    """Resolve explicit identities, prepare cases, and reconcile GPUMD jobs."""

    def __init__(
        self,
        context: StageContext | None = None,
        *,
        project_dir: Path | None = None,
        model_run_id: str | None = None,
        dataset_id: str | None = None,
        state_store: Any | None = None,
        backend: StaticPredictionBackend | Any | None = None,
        scheduler: Scheduler | None = None,
        orchestrator: ValidationReconciliationOrchestrator | None = None,
        max_concurrent: int | None = None,
        max_attempts: int = 1,
    ) -> None:
        self.context = context
        self.project_dir = None if project_dir is None else Path(project_dir)
        self.model_run_id = model_run_id
        self.dataset_id = dataset_id
        self.state_store = state_store
        self.backend = backend
        self.scheduler = scheduler
        self.orchestrator = orchestrator
        self.max_concurrent = max_concurrent
        self.max_attempts = max_attempts

    def _active_context(self, context: StageContext | None) -> StageContext | None:
        return context or self.context

    def _identities(self, context: StageContext | None) -> tuple[Path, str, str | None, Any]:
        active = self._active_context(context)
        project_dir = self.project_dir or (None if active is None else active.project_dir)
        if project_dir is None:
            raise StateError("ValidationStage requires a project directory")
        state_store = self.state_store or (None if active is None else active.state_store)
        if state_store is None:
            raise StateError("ValidationStage requires the authoritative StateStore")
        config = None if active is None else active.config
        configured_model = (
            None
            if config is None
            else getattr(getattr(config, "validation", None), "model_run_id", None)
        )
        options = {} if active is None else dict(active.options)
        model_run_id = self.model_run_id or options.get("model_run_id") or configured_model
        if not isinstance(model_run_id, str) or not model_run_id.strip():
            raise StateError("Validation requires an explicit model_run_id")
        dataset_id = self.dataset_id or options.get("dataset_id")
        if dataset_id is not None and (not isinstance(dataset_id, str) or not dataset_id.strip()):
            raise StateError("Validation dataset_id must be a non-empty string when supplied")
        return Path(project_dir), model_run_id, dataset_id, state_store

    def _backend(self, context: StageContext | None) -> Any:
        if self.backend is not None:
            return self.backend
        active = self._active_context(context)
        config = None if active is None else active.config
        command = "gpumd" if config is None else getattr(config.hpc, "gpumd_command", "gpumd")
        return GpumdBackend(shlex.split(command))

    def prepare(self, context: StageContext | None = None) -> ValidationPreparation:
        project_dir, model_run_id, dataset_id, state_store = self._identities(context)
        backend = self._backend(context)
        if not callable(getattr(backend, "prepare_inputs", None)):
            raise StateError("Validation backend must implement prepare_inputs")
        return prepare_validation_cases(
            project_dir,
            model_run_id,
            dataset_id=dataset_id,
            state_store=state_store,
            backend=backend,
        )

    def parse_completed(
        self,
        preparation: ValidationPreparation,
        context: StageContext | None = None,
    ) -> tuple[StaticPrediction, ...]:
        project_dir, model_run_id, dataset_id, state_store = self._identities(context)
        resolved = resolve_model_dataset(
            project_dir,
            model_run_id,
            dataset_id=dataset_id or preparation.dataset_id,
            state_store=state_store,
        )
        backend = self._backend(context)
        predictions: list[StaticPrediction] = []
        for case in preparation.cases:
            request = case.static_prediction_request(resolved.model_run)
            parser = getattr(backend, "parse_prediction", None)
            if callable(parser):
                parse = cast(Callable[..., StaticPrediction], parser)
                predictions.append(parse(request, case.output_path))
                continue
            predictor = getattr(backend, "predict", None)
            if not callable(predictor):
                raise StateError("Validation backend must implement parse_prediction or predict")
            predict = cast(Callable[..., StaticPrediction], predictor)
            predictions.append(predict(request))
        return tuple(predictions)

    def _resources(self, context: StageContext | None) -> JobResources | None:
        active = self._active_context(context)
        config = None if active is None else active.config
        if config is None:
            return None
        return JobResources(
            nodes=int(config.slurm.gpumd_nodes),
            gpus_per_node=int(config.slurm.gpumd_gpus),
            mpi_ranks=max(1, int(config.slurm.gpumd_gpus)),
            walltime=config.slurm.gpumd_walltime,
        )

    def run(self, context: StageContext | None = None) -> ValidationStageResult:
        active = self._active_context(context)
        project_dir, model_run_id, dataset_id, state_store = self._identities(active)
        backend = self._backend(active)
        resolved = resolve_model_dataset(
            project_dir,
            model_run_id,
            dataset_id=dataset_id,
            state_store=state_store,
        )
        preparation = prepare_validation_cases(
            project_dir,
            model_run_id,
            dataset_id=resolved.dataset_id,
            state_store=state_store,
            backend=backend,
        )
        if self.orchestrator is not None:
            orchestrator = self.orchestrator
        else:
            scheduler = self.scheduler
            if scheduler is None:
                raise StateError("ValidationStage requires an injected scheduler for execution")
            raw_command = getattr(backend, "command", ("gpumd",))
            command = (
                tuple(shlex.split(raw_command))
                if isinstance(raw_command, str)
                else tuple(raw_command)
            )
            validation_run = ValidationRunIdentity(
                resolved.model_run_id,
                resolved.dataset_id,
                {"backend_command": list(command), "case_schema": preparation.schema_version},
            )
            orchestrator = ValidationReconciliationOrchestrator(
                validation_run=validation_run,
                model=resolved.model_run,
                cases=preparation.cases,
                state_store=state_store,
                scheduler=scheduler,
                backend=backend,
                max_concurrent=self.max_concurrent or self._max_concurrent(active),
                max_attempts=self.max_attempts,
                resources=self._resources(active),
                job_name_prefix=f"nf_validation_{active.project_name}_" if active else None,
            )
        execution = orchestrator.reconcile_once()
        metrics = None
        if execution.all_successful:
            predictions = tuple(
                record.prediction for record in execution.cases if record.prediction is not None
            )
            if len(predictions) != len(execution.cases):
                predictions = self.parse_completed(preparation, active)
            metrics = calculate_metrics(
                tuple(
                    pair_prediction(case, prediction)
                    for case, prediction in zip(preparation.cases, predictions)
                )
            )
            self._persist_metrics(state_store, execution.validation_run_id, metrics)
        return ValidationStageResult(preparation, execution, metrics)

    @staticmethod
    def _max_concurrent(context: StageContext | None) -> int:
        config = None if context is None else context.config
        return 1 if config is None else int(config.slurm.max_concurrent)

    @staticmethod
    def _persist_metrics(
        state_store: Any, validation_run_id: str, metrics: ValidationMetrics
    ) -> None:
        recorder = getattr(state_store, "record_validation_result", None)
        if not callable(recorder):
            return
        for name, value in metrics.to_dict().items():
            if value is None:
                continue
            recorder(
                validation_run_id,
                name,
                metric_name=name,
                observed_value=float(value),
                passed=None,
                metadata={
                    "schema_version": VALIDATION_RESULT_SCHEMA,
                    "metric_name": name,
                    "report": metrics.to_report(),
                },
            )


__all__ = ["ValidationStage", "ValidationStageResult"]
