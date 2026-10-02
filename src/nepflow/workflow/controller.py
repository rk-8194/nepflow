"""Canonical workflow composition and orchestration boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Mapping
import logging
import time
from typing import Any, Protocol

from nepflow.config import canonical_config_path, load_config
from nepflow.errors import ConfigurationError, StateError
from nepflow.state import StateStore

from .resubmission import ReconciliationResult, SelfResubmitExit
from .stages import StageRunResult, StageRunState, StageRunStatus, WorkflowStage
from .state import WorkflowState


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StageContext:
    """Dependencies and runtime context supplied to a composed stage."""

    project_name: str
    project_dir: Path
    config_file: Path
    state_file: Path
    state_store: StateStore | None
    workflow_state: WorkflowState | None
    config: Any = None
    debug: bool = False
    local_mode: bool = False
    memory_mode: bool = False
    seeds_only: bool = False
    slurm_deadline: float | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


class StageHandler(Protocol):
    """Callable stage boundary used by the workflow registry."""

    def __call__(self, context: StageContext) -> StageRunResult | None: ...


StageReconciler = Callable[[StageContext], None]


@dataclass(frozen=True, slots=True)
class StageBinding:
    """One stage's execution and optional resume-reconciliation hooks."""

    stage: WorkflowStage
    handler: StageHandler
    next_stage: WorkflowStage | None
    reconcile: StageReconciler | None = None


_ORDERED_RUNNABLE_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage.INIT,
    WorkflowStage.GENERATE,
    WorkflowStage.SELECT,
    WorkflowStage.RUN_VASP,
    WorkflowStage.TRAIN_NEP,
    WorkflowStage.VALIDATE,
)


class StageRegistry:
    """Registry of stage handlers composed by the application entry point.

    The canonical workflow package owns this seam but not the concrete stage
    implementations.  That dependency direction lets Phase 4 replace legacy
    handlers one at a time without creating a second controller.
    """

    def __init__(self) -> None:
        self._bindings: dict[WorkflowStage, StageBinding] = {}
        self._auxiliary: dict[str, StageHandler] = {}

    def register(
        self,
        stage: WorkflowStage | str,
        handler: StageHandler,
        *,
        next_stage: WorkflowStage | str | None = None,
        reconcile: StageReconciler | None = None,
        replace: bool = False,
    ) -> None:
        typed_stage = WorkflowStage.from_legacy(stage)
        if typed_stage is WorkflowStage.COMPLETED:
            raise StateError("Completed workflow stage has no executable handler")
        if typed_stage in self._bindings and not replace:
            raise StateError(f"Stage is already registered: {typed_stage.value}")
        typed_next = (
            None
            if next_stage is None
            else WorkflowStage.from_legacy(next_stage)
        )
        if typed_next is None:
            index = _ORDERED_RUNNABLE_STAGES.index(typed_stage)
            typed_next = (
                WorkflowStage.COMPLETED
                if typed_stage is WorkflowStage.VALIDATE
                else _ORDERED_RUNNABLE_STAGES[index + 1]
            )
        self._bindings[typed_stage] = StageBinding(
            stage=typed_stage,
            handler=handler,
            next_stage=typed_next,
            reconcile=reconcile,
        )

    def register_auxiliary(self, name: str, handler: StageHandler) -> None:
        """Register a non-progressing operation such as memory benchmarking."""

        key = name.strip()
        if not key:
            raise StateError("Auxiliary workflow handler name is blank")
        if key in self._auxiliary:
            raise StateError(f"Auxiliary workflow handler is already registered: {key}")
        self._auxiliary[key] = handler

    def binding(self, stage: WorkflowStage | str) -> StageBinding:
        typed_stage = WorkflowStage.from_legacy(stage)
        try:
            return self._bindings[typed_stage]
        except KeyError as exc:
            raise StateError(f"No workflow handler is registered for {typed_stage.value}") from exc

    def auxiliary(self, name: str) -> StageHandler:
        try:
            return self._auxiliary[name]
        except KeyError as exc:
            raise StateError(f"No auxiliary workflow handler is registered for {name}") from exc

    def stages(self) -> tuple[WorkflowStage, ...]:
        """Return registered executable stages in workflow order."""

        return tuple(stage for stage in _ORDERED_RUNNABLE_STAGES if stage in self._bindings)

    def reconcile(self, stage: WorkflowStage, context: StageContext) -> None:
        hook = self.binding(stage).reconcile
        if hook is not None:
            hook(context)

    def execute(self, stage: WorkflowStage, context: StageContext) -> StageRunResult:
        binding = self.binding(stage)
        result = binding.handler(context)
        if result is None:
            return StageRunResult(
                stage=stage,
                status=StageRunState.COMPLETED,
                advanced_to=binding.next_stage,
                completed=True,
            )
        if not isinstance(result, StageRunResult):
            raise StateError(
                f"Stage handler for {stage.value} returned an unsupported result: "
                f"{type(result).__name__}"
            )
        if result.stage is not stage:
            raise StateError(
                f"Stage handler returned {result.stage.value!r} for {stage.value!r}"
            )
        if (
            result.advanced_to is not None
            and result.status is not StageRunState.COMPLETED
        ):
            raise StateError(
                f"Stage {stage.value} cannot advance with status "
                f"{result.status.value}"
            )
        return result

    def execute_auxiliary(self, name: str, context: StageContext) -> StageRunResult | None:
        return self.auxiliary(name)(context)


class WorkflowController:
    """Orchestrate workflow stages through a composed ``StageRegistry``."""

    def __init__(
        self,
        project_name: str,
        output_dir: Path,
        init_mode: bool = False,
        debug: bool = False,
        stage_override: str | None = None,
        local_mode: bool = False,
        memory_mode: bool = False,
        slurm_deadline: float | None = None,
        *,
        stage_registry: StageRegistry | None = None,
        registry: StageRegistry | None = None,
    ) -> None:
        if stage_registry is not None and registry is not None:
            raise ValueError("Provide either stage_registry or registry, not both")
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.init_mode = init_mode
        self.debug = debug
        self.stage_override = stage_override
        self.local_mode = local_mode
        self.memory_mode = memory_mode
        self.slurm_deadline = slurm_deadline
        self.stage_registry = stage_registry or registry or StageRegistry()

        self.project_dir = self.output_dir / f"project_{project_name}"
        self.config_dir = self.project_dir / "config"
        self.config_file = canonical_config_path(self.project_dir)
        self.project_file = self.project_dir / ".project"
        self.state_file = self.project_dir / "state.db"
        self.log_dir = self.project_dir / "logs"

        self._state_store: StateStore | None = None
        self._workflow_state: WorkflowState | None = None
        self.config: Any = None
        self._startup_validated = False
        if not self.init_mode:
            self._validate_project_foundations()

        logger.info("Initialized controller for project: %s", project_name)

    @property
    def state_store(self) -> StateStore:
        if self._state_store is None:
            raise StateError("Workflow state store is not available before initialization")
        return self._state_store

    @property
    def workflow_state(self) -> WorkflowState:
        if self._workflow_state is None:
            raise StateError("Workflow state is not available before initialization")
        return self._workflow_state

    def _validate_project_foundations(self) -> None:
        """Validate initialized project foundations without manufacturing state."""

        if not self.config_file.is_file():
            raise ConfigurationError(
                f"Canonical project configuration is missing: {self.config_file}; "
                "run --init for a new project or perform an explicit migration"
            )
        config = load_config(self.config_file, project_name=self.project_name)
        if config.project.name != self.project_name:
            raise ConfigurationError(
                "Project config name does not match the requested project: "
                f"{config.project.name!r} != {self.project_name!r}"
            )
        if not self.state_file.exists():
            raise StateError(
                f"Authoritative state database is missing: {self.state_file}; "
                "run --init only for a new project or restore state.db through an "
                "explicit migration"
            )

        store = StateStore(self.state_file)
        try:
            if store.get_project(self.project_name) is None:
                raise StateError(
                    f"Authoritative state.db has no project record for {self.project_name!r}; "
                    "perform an explicit state migration"
                )
            latest = store.get_latest_stage_run(self.project_name)
            if latest is None and not self.project_file.exists():
                raise StateError(
                    "Authoritative state.db has no stage history and no legacy .project "
                    "marker; perform an explicit state migration"
                )
            if latest is not None:
                # Validate persisted vocabulary before publishing the store.
                StageRunStatus.from_mapping(latest)
        except BaseException:
            store.close()
            raise

        if self._state_store is not None:
            self._state_store.close()
        self._state_store = store
        self._workflow_state = WorkflowState(
            self.project_name,
            store,
            self.project_file,
        )
        self.config = config
        self._startup_validated = True

    def _context(self, *, seeds_only: bool = False, mode: str = "normal") -> StageContext:
        return StageContext(
            project_name=self.project_name,
            project_dir=self.project_dir,
            config_file=self.config_file,
            state_file=self.state_file,
            state_store=self._state_store,
            workflow_state=self._workflow_state,
            config=self.config,
            debug=self.debug,
            local_mode=self.local_mode,
            memory_mode=self.memory_mode,
            seeds_only=seeds_only,
            slurm_deadline=self.slurm_deadline,
            options={"mode": mode},
        )

    def current_stage(self) -> WorkflowStage:
        return self.workflow_state.current_stage()

    def reconcile_stage(self) -> ReconciliationResult:
        return self.workflow_state.reconcile()

    def transition_to(self, stage: WorkflowStage | str) -> StageRunStatus:
        return self.workflow_state.transition_to(stage)

    def _check_deadline(self) -> None:
        if self.slurm_deadline is None:
            return
        current_time = time.time()
        grace_period = 300
        if current_time >= self.slurm_deadline:
            raise SelfResubmitExit(
                "SLURM walltime deadline reached, resubmitting workflow"
            )
        if current_time >= self.slurm_deadline - grace_period:
            logger.info(
                "Approaching SLURM deadline (%.1fs remaining)",
                self.slurm_deadline - current_time,
            )

    def _execute_stage(
        self,
        stage: WorkflowStage,
        *,
        seeds_only: bool = False,
        mode: str = "normal",
        advance: bool = True,
    ) -> StageRunResult:
        context = self._context(seeds_only=seeds_only, mode=mode)
        self.stage_registry.reconcile(stage, context)
        try:
            result = self.stage_registry.execute(stage, context)
        except SelfResubmitExit:
            raise
        except BaseException as exc:
            try:
                self.workflow_state.mark_failed(exc)
            except BaseException:
                logger.exception("Could not record failed stage %s", stage.value)
            raise

        if result.status is StageRunState.RUNNING or result.advanced_to is None or not advance:
            self.workflow_state.set_status(stage, result.status)
        if advance and result.advanced_to is not None:
            self.workflow_state.transition_to(result.advanced_to)
        return result

    def _run_debug(self) -> None:
        current = self.current_stage()
        if current is WorkflowStage.COMPLETED:
            return
        stages = self.stage_registry.stages()
        if current is WorkflowStage.INIT:
            start_index = 1
        else:
            try:
                start_index = stages.index(current)
            except ValueError as exc:
                raise StateError(f"No workflow handler is registered for {current.value}") from exc

        for stage in stages[start_index:]:
            result = self._execute_stage(stage, mode="debug")
            if result.advanced_to is None:
                return

    def _run_local(self) -> None:
        stage = self.current_stage()
        if stage is WorkflowStage.INIT:
            self._execute_stage(stage, mode="local")
            stage = self.current_stage()
        if stage is WorkflowStage.GENERATE:
            # Local mode deliberately leaves generation running so an HPC run
            # can resume with perturbations.
            self._execute_stage(
                stage,
                seeds_only=True,
                mode="local",
                advance=False,
            )
            self.workflow_state.set_status(stage, StageRunState.RUNNING)

    def run(self) -> None:
        """Run one workflow unit, or all units in debug mode."""

        if self.init_mode:
            self.stage_registry.execute(
                WorkflowStage.INIT,
                self._context(mode="init"),
            )
            self._validate_project_foundations()
            current = self.current_stage()
            if current is WorkflowStage.INIT:
                self.transition_to(WorkflowStage.GENERATE)
            elif current is not WorkflowStage.GENERATE:
                raise StateError(
                    "Project initialization cannot move an existing workflow backward; "
                    f"current stage is {current.value!r}"
                )
            if not self.local_mode and not self.debug:
                return

        if not self._startup_validated:
            raise StateError(
                f"Project '{self.project_name}' is not initialized; run the initialization stage first"
            )

        if self.stage_override is not None:
            self.transition_to(self.stage_override)

        stage = self.current_stage()
        if stage is WorkflowStage.COMPLETED:
            logger.info("Workflow is already complete; no stage will be run")
            return

        if self.memory_mode:
            self.stage_registry.execute_auxiliary("memory", self._context(mode="memory"))
            return

        if self.debug:
            self._run_debug()
            return

        if self.local_mode:
            self._run_local()
            return

        self._check_deadline()
        self._execute_stage(stage)


__all__ = [
    "StageBinding",
    "StageContext",
    "StageHandler",
    "StageRegistry",
    "WorkflowController",
]
