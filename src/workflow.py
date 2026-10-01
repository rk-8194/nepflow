"""
Workflow controller and stage orchestration.
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging
import time

# Note: src/ is added to sys.path dynamically by nepflow_cli.py
# pylint: disable=import-error
from modules import (
    InitStage,
    GenerateStage,
    SelectStage,
    RunVaspStage,
    TrainNepStage,
    ValidateStage,
    MemoryStage,
)
from modules.validate.launcher import read_validation_status
from modules.run_vasp._common import read_status
from nepflow.io.atomic import atomic_write_text
from nepflow.errors import StateError
from nepflow.state import StateStore
from nepflow.workflow.resubmission import (
    ReconciliationResult,
    SelfResubmitExit,
)
from nepflow.workflow.stages import (
    StageRunStatus,
    StageRunState,
    WorkflowStage,
    parse_legacy_stage,
    stage_to_legacy,
    validate_transition,
)

logger = logging.getLogger("nepflow.workflow")


class WorkflowController:
    """
    Main workflow controller for project orchestration.
    
    Handles:
    - Reconciling StateStore stage state with the legacy .project marker
    - Automatically determining current workflow stage
    - Stage execution and state progression
    - Slurm job submission and tracking
    - Resumable workflow logic
    """
    
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
    ):
        """
        Initialize workflow controller.
        
        Args:
            project_name: Name of the project (scopes config, state, outputs)
            output_dir: Base directory for project outputs
            init_mode: If True, initialize a new project (skip validation)
            debug: Enable debug mode (no external dependencies)
            stage_override: Override current stage
            local_mode: Run in local mode
            memory_mode: Run VASP memory benchmarking mode
            slurm_deadline: Unix timestamp of SLURM walltime deadline (None if no SLURM limit)
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.init_mode = init_mode
        self.debug = debug
        self.stage_override = stage_override
        self.local_mode = local_mode
        self.memory_mode = memory_mode
        self.slurm_deadline = slurm_deadline
        
        # Project directory (contains config, state, logs, outputs)
        self.project_dir = self.output_dir / f"project_{project_name}"
        
        # Derived paths - config is now per-project
        self.config_dir = self.project_dir / "config"
        self.config_file = self.config_dir / f"{project_name}.yaml"
        self.project_file = self.project_dir / ".project"
        self.state_file = self.project_dir / "state.db"
        self.log_dir = self.project_dir / "logs"

        # The controller remains in its legacy location until Phase 4, but
        # stage identity is already ledger-backed.  This bridge does not read
        # or migrate any legacy state file beyond the .project cache marker.
        self._state_store = StateStore(self.state_file)
        if self._state_store.get_project(self.project_name) is None:
            self._state_store.upsert_project(
                self.project_name,
                name=self.project_name,
                root_path=str(self.project_dir),
            )
        
        logger.info("Initialized controller for project: %s", project_name)
        if self.debug:
            logger.debug("Debug mode enabled")
            logger.debug("Config dir: %s", self.config_dir)
            logger.debug("Config file: %s", self.config_file)
            logger.debug("Project file: %s", self.project_file)
            logger.debug("State file: %s", self.state_file)
            logger.debug("Project dir: %s", self.project_dir)
        
        if self.slurm_deadline:
            remaining = self.slurm_deadline - time.time()
            logger.debug("SLURM deadline set: %.1fs remaining", remaining)
        
        # Debug mode: trigger deadline immediately to test self-resubmit path
        if self.debug and self.slurm_deadline:
            logger.info("[DEBUG] Triggering deadline immediately to test self-resubmit path")
            self.slurm_deadline = time.time() - 1  # Already past deadline
    
    def _check_deadline(self) -> None:
        """
        Check if SLURM deadline is approaching (within 10 minutes).
        
        Raises SelfResubmitExit if deadline reached to trigger resubmission.
        """
        if not self.slurm_deadline:
            return  # No deadline, continue
        
        current_time = time.time()
        grace_period = 300  # 5 minutes before deadline
        
        if current_time >= self.slurm_deadline:
            if self.debug:
                logger.info("[DEBUG] Approaching walltime — triggering workflow self-resubmit")
            else:
                logger.warning("SLURM deadline reached — resubmitting workflow")
            raise SelfResubmitExit(
                "SLURM walltime deadline reached, resubmitting workflow"
            )
        
        if current_time >= self.slurm_deadline - grace_period:
            remaining = self.slurm_deadline - current_time
            logger.info("Approaching SLURM deadline (%.1fs remaining) — will resubmit after this stage", remaining)
    
    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read_marker_stage(self) -> WorkflowStage | None:
        """Read and canonically parse the optional legacy stage marker."""

        if not self.project_file.exists():
            return None
        try:
            marker = self.project_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise StateError(
                f"Failed to read workflow marker {self.project_file}"
            ) from exc
        try:
            return parse_legacy_stage(marker)
        except StateError as exc:
            raise StateError(
                f"Invalid workflow stage marker in {self.project_file}: {exc}"
            ) from exc

    def _read_authoritative_stage(
        self,
    ) -> tuple[
        WorkflowStage | None,
        WorkflowStage | None,
        StageRunStatus | None,
        StateError | None,
    ]:
        """Read StateStore and marker values without choosing an authority."""

        row = self._state_store.get_latest_stage_run(self.project_name)
        stored_run = StageRunStatus.from_mapping(row) if row is not None else None
        stored_stage = stored_run.stage if stored_run is not None else None
        marker_stage: WorkflowStage | None = None
        marker_error: StateError | None = None
        try:
            marker_stage = self._read_marker_stage()
        except StateError as exc:
            # Reconciliation decides whether this malformed cache can be
            # repaired from an existing authoritative StateStore row.
            marker_error = exc
        return stored_stage, marker_stage, stored_run, marker_error

    def _stage_run_id(self, stage: WorkflowStage) -> str:
        """Return the stable ledger identity for this project's stage."""

        return f"{self.project_name}:{stage.value}"

    def _record_stage(
        self,
        stage: WorkflowStage,
        *,
        status: StageRunState,
        metadata: dict[str, str] | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        """Persist one typed stage record inside the caller's transaction."""

        self._state_store.upsert_stage_run(
            self._stage_run_id(stage),
            self.project_name,
            stage.value,
            status=status.value,
            started_at=started_at,
            completed_at=completed_at,
            metadata=metadata or {},
        )

    def _write_stage_marker(self, stage: WorkflowStage) -> None:
        """Write the compatibility marker from the canonical enum only."""

        try:
            self.project_file.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                self.project_file,
                stage_to_legacy(stage),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to write project file %s: %s", self.project_file, exc)
            raise

    def _reconcile_stage(self) -> ReconciliationResult:
        """Reconcile legacy marker state with the StateStore ledger."""

        stored_stage, marker_stage, stored_run, marker_error = (
            self._read_authoritative_stage()
        )
        if stored_run is not None:
            marker_needs_repair = marker_error is not None or marker_stage is not stored_stage
            if marker_needs_repair:
                # The marker is only a compatibility cache; repair missing,
                # malformed, and stale values from authoritative state.
                self._write_stage_marker(stored_run.stage)
            if marker_error is not None:
                reason = "repaired invalid legacy marker from StateStore"
            elif marker_stage is None:
                reason = "repaired missing legacy marker from StateStore"
            elif marker_stage is not stored_stage:
                reason = "repaired stale legacy marker from StateStore"
            else:
                reason = None
            return ReconciliationResult(
                stage=stored_run.stage,
                source="state_store",
                marker_stage=marker_stage,
                authoritative_stage=stored_run.stage,
                changed=marker_needs_repair,
                reason=reason,
            )

        if marker_error is not None:
            # Without authoritative state there is nothing safe to repair
            # from, so preserve the explicit corruption failure.
            raise marker_error

        if marker_stage is not None:
            # Existing projects have no stage-run history yet.  This one-time
            # import is explicit reconciliation, not folder-based inference.
            now = self._timestamp()
            status = (
                StageRunState.COMPLETED
                if marker_stage.is_terminal
                else StageRunState.RUNNING
            )
            with self._state_store.transaction():
                self._record_stage(
                    marker_stage,
                    status=status,
                    started_at=now,
                    completed_at=now if marker_stage.is_terminal else None,
                    metadata={"source": "legacy_project_marker"},
                )
            return ReconciliationResult(
                stage=marker_stage,
                source="legacy_marker",
                marker_stage=marker_stage,
                authoritative_stage=marker_stage,
                changed=True,
            )

        # A brand-new project has a typed initial state.  There is no
        # arbitrary-string fallback and no reconstruction from directories.
        now = self._timestamp()
        with self._state_store.transaction():
            self._record_stage(
                WorkflowStage.INIT,
                status=StageRunState.RUNNING,
                started_at=now,
                metadata={"source": "initial_state"},
            )
        return ReconciliationResult(
            stage=WorkflowStage.INIT,
            source="initial_state",
            authoritative_stage=WorkflowStage.INIT,
            changed=True,
        )

    def _determine_current_stage(self) -> WorkflowStage:
        """Return the typed current stage after explicit reconciliation."""

        result = self._reconcile_stage()
        logger.debug("Current workflow stage: %s (%s)", result.stage.value, result.source)
        return result.stage

    def _set_current_stage(self, stage: WorkflowStage | str) -> None:
        """Validate and persist one adjacent stage transition.

        The StateStore transaction is committed before the marker cache is
        updated.  A failed state write therefore leaves both the previous
        authoritative record and its marker untouched.
        """

        target = parse_legacy_stage(stage)
        current = self._reconcile_stage().stage
        row = self._state_store.get_latest_stage_run(self.project_name)
        stored_run = StageRunStatus.from_mapping(row) if row is not None else None
        if stored_run is None:
            raise StateError("Workflow stage reconciliation did not create authoritative state")
        validate_transition(current, target)

        now = self._timestamp()
        with self._state_store.transaction():
            if current is not None and current is not target:
                self._record_stage(
                    current,
                    status=StageRunState.COMPLETED,
                    started_at=stored_run.started_at,
                    completed_at=now,
                    metadata={"source": "workflow_controller"},
                )
                # Make the new stage the most recent row even when both rows
                # are transitioned within the same microsecond.
                target_started_at = self._timestamp()
                if target_started_at <= now:
                    target_started_at = (
                        datetime.fromisoformat(now) + timedelta(microseconds=1)
                    ).isoformat()
            else:
                target_started_at = now
            self._record_stage(
                target,
                status=(
                    StageRunState.COMPLETED
                    if target.is_terminal
                    else StageRunState.RUNNING
                ),
                started_at=target_started_at,
                completed_at=target_started_at if target.is_terminal else None,
                metadata={"source": "workflow_controller"},
            )

        self._write_stage_marker(target)
        logger.debug("Updated project stage to: %s", target.value)
    
    def _initialize(self) -> None:
        """Initialize a new project."""
        stage = InitStage(
            project_name=self.project_name,
            config_file=self.config_file,
            state_file=self.state_file,
            project_dir=self.project_dir,
            debug=self.debug
        )
        stage.run()
    
    def _generate(self, seeds_only: bool = False) -> None:
        """Generate base structures and variants."""
        stage = GenerateStage(
            project_name=self.project_name,
            config_file=self.config_file,
            state_file=self.state_file,
            project_dir=self.project_dir,
            debug=self.debug
        )
        stage.run(seeds_only=seeds_only)
    
    def _select(self) -> None:
        """Select representative subset from candidates."""
        stage = SelectStage(
            project_name=self.project_name,
            config_file=self.config_file,
            state_file=self.state_file,
            project_dir=self.project_dir,
            debug=self.debug
        )
        stage.run()
    
    def _run_vasp(self) -> None:
        """Run VASP DFT calculations adaptively."""
        stage = RunVaspStage(
            project_name=self.project_name,
            config_file=self.config_file,
            state_file=self.state_file,
            project_dir=self.project_dir,
            debug=self.debug,
            slurm_deadline=self.slurm_deadline,
        )
        stage.run()
    
    def _train_nep(self) -> None:
        """Train NEP models on results."""
        stage = TrainNepStage(
            project_name=self.project_name,
            config_file=self.config_file,
            state_file=self.state_file,
            project_dir=self.project_dir,
            debug=self.debug
        )
        stage.run()
    
    def _validate(self) -> None:
        """Validate with GPUMD simulations."""
        stage = ValidateStage(
            project_name=self.project_name,
            config_file=self.config_file,
            state_file=self.state_file,
            project_dir=self.project_dir,
            debug=self.debug,
            slurm_deadline=self.slurm_deadline,
        )
        stage.run()

    def _validation_is_complete(self) -> bool:
        """Return whether persisted validation and analysis both completed."""
        status = read_validation_status(self.project_dir)
        return (
            status.get("validation_complete") is True
            and status.get("analysis_complete") is True
        )
    
    def _memory(self) -> None:
        """Run VASP memory benchmarks."""
        stage = MemoryStage(
            project_name=self.project_name,
            config_file=self.config_file,
            state_file=self.state_file,
            project_dir=self.project_dir,
            debug=self.debug
        )
        stage.run()
    
    def _check_project_initialized(self) -> bool:
        """
        Check if project has been initialized.
        
        Returns:
            True if project is initialized, False otherwise
        """
        # Project is considered initialized if config directory exists
        return self.config_dir.exists()
    
    def _run_local(self) -> None:
        """
        Fetch base structures locally, then stop.
        
        Used when the HPC has no web access. Runs init (if needed)
        and the seed-generation part of generate (composition grid +
        configurational generators including Materials Project fetch),
        but skips perturbations. The stage stays at 'generate' so
        the HPC can resume with perturbations → select → …
        """
        stage = self._determine_current_stage()
        
        # If we haven't initialized yet, do that first
        if stage is WorkflowStage.INIT:
            logger.info("Local mode — initializing project")
            self._initialize()
            self._set_current_stage(WorkflowStage.GENERATE)
            stage = WorkflowStage.GENERATE
        
        if stage is WorkflowStage.GENERATE:
            logger.info("Local mode — generating base structures (seeds only)")
            self._generate(seeds_only=True)
            # Stage stays at 'generate' — HPC resumes with perturbations
            logger.info("Local mode complete — seeds generated, stage remains at 'generate'")
            print("✓ Base structures generated (seeds only, no perturbations)")
            print("  Transfer the project directory to HPC and resume with:")
            print("  python nepflow_cli.py --project %s" % self.project_name)
        else:
            logger.info(
                "Local mode — nothing to do, current stage is '%s'. "
                "Base structures were already generated.", stage.value
            )
            print("✓ Local stages already complete (current stage: '%s')" % stage.value)
            print("  Resume on HPC with:")
            print("  python nepflow_cli.py --project %s" % self.project_name)

    def _print_status_summary(self) -> None:
        """Log a summary of the current workflow state."""
        stage = self._determine_current_stage()

        STAGE_LABELS = {
            WorkflowStage.INIT: "1/6  init",
            WorkflowStage.GENERATE: "2/6  generate",
            WorkflowStage.SELECT: "3/6  select",
            WorkflowStage.RUN_VASP: "4/6  run_vasp",
            WorkflowStage.TRAIN_NEP: "5/6  train_nep",
            WorkflowStage.VALIDATE: "6/6  validate",
            WorkflowStage.COMPLETED: "completed",
        }
        label = STAGE_LABELS.get(stage, stage.value)

        logger.info("═" * 55)
        logger.info("  NEPFlow  ·  project: %s", self.project_name)
        logger.info("  Stage    :  %s", label)

        if stage is WorkflowStage.RUN_VASP:
            jobs_dir = self.project_dir / "vasp" / "jobs"
            counts: dict[str, dict[str, int]] = {}
            for ds in ("train", "test"):
                ds_dir = jobs_dir / ds
                if not ds_dir.exists():
                    continue
                tally: dict[str, int] = {"completed": 0, "submitted": 0, "pending": 0, "failed": 0}
                for struct_dir in (d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("struct_")):
                    s = read_status(struct_dir).get("status", "pending")
                    if s == "reused":
                        s = "completed"
                    tally[s if s in tally else "pending"] += 1
                failed_dir = ds_dir / "failed"
                if failed_dir.exists():
                    tally["failed"] += sum(1 for d in failed_dir.iterdir() if d.is_dir() and d.name.startswith("struct_"))
                counts[ds] = tally

            if counts:
                logger.info("  %-8s  %5s  %5s  %7s  %7s  %6s", "Dataset", "total", "done", "running", "pending", "failed")
                for ds, t in counts.items():
                    total = sum(t.values())
                    logger.info("  %-8s  %5d  %5d  %7d  %7d  %6d",
                                ds, total, t["completed"], t["submitted"], t["pending"], t["failed"])

        logger.info("═" * 55)

    def _run_debug(self) -> None:
        """Run all stages sequentially with simulated external calls."""
        all_stages = [
            (WorkflowStage.GENERATE,  self._generate),
            (WorkflowStage.SELECT,    self._select),
            (WorkflowStage.RUN_VASP,  self._run_vasp),
            (WorkflowStage.TRAIN_NEP, self._train_nep),
            (WorkflowStage.VALIDATE,  self._validate),
        ]
        current = self._determine_current_stage()
        if current is WorkflowStage.INIT:
            start_index = 0
        else:
            start_index = next(
                index for index, (stage, _) in enumerate(all_stages) if stage is current
            )
        stages = all_stages[start_index:]

        for name, stage_fn in stages:
            logger.info("")
            logger.info("=" * 60)
            logger.info("[DEBUG] Stage: %s", name.value)
            logger.info("=" * 60)
            self._set_current_stage(name)

            if name is WorkflowStage.RUN_VASP:
                # Clean marker so the first launcher invocation exercises
                # the self-resubmit path.
                marker = self.project_dir / "vasp" / ".debug_resubmit_done"
                if marker.exists():
                    marker.unlink()
                # The debug launcher self-resubmits once; catch it and
                # re-invoke to simulate the resumed launcher job.
                try:
                    stage_fn()
                except SelfResubmitExit:
                    logger.info("[DEBUG] Launcher self-resubmitted — resuming (second invocation)")
                    stage_fn()
            else:
                stage_fn()

        if self._validation_is_complete():
            self._set_current_stage(WorkflowStage.COMPLETED)

        logger.info("")
        logger.info("=" * 60)
        logger.info("[DEBUG] All stages completed successfully")
        logger.info("=" * 60)
    
    def run(self) -> None:
        """
        Run workflow controller.
        
        If --init flag was used, only initializes the project and exits.
        Otherwise, validates that project is initialized, then reconciles the
        authoritative ledger with the compatibility marker before executing it.
        
        In debug mode, all stages run sequentially in one invocation.
        """
        # If in init mode, only run initialization
        if self.init_mode:
            logger.info("Running in init mode - initializing project only")
            self._determine_current_stage()
            self._initialize()
            self._set_current_stage(WorkflowStage.GENERATE)
            logger.info("Project initialization complete. Initialization stage finished.")
            if not self.local_mode and not self.debug:
                return
        
        # Check if project is initialized
        if not self._check_project_initialized():
            logger.error(
                "Project '%s' is not initialized. "
                "Run with --init flag first: python3 nepflow_cli.py --project %s --init",
                self.project_name,
                self.project_name
            )
            raise ValueError(
                f"Project '{self.project_name}' is not initialized. "
                f"Run: python3 nepflow_cli.py --project {self.project_name} --init"
            )
        
        if self.stage_override:
            override = parse_legacy_stage(self.stage_override)
            self._set_current_stage(override)
            logger.info("Stage overridden to: %s", override.value)

        self._print_status_summary()

        stage = self._determine_current_stage()
        if stage is WorkflowStage.COMPLETED:
            logger.info("Workflow is already complete; no stage will be run")
            print("✓ Workflow already complete")
            return

        # Memory mode: run VASP benchmarks, no stage progression
        if self.memory_mode:
            logger.info("Running VASP memory benchmarks")
            self._memory()
            logger.info("Memory benchmarks complete")
            return

        # Debug mode: run all stages sequentially in one invocation
        if self.debug:
            self._run_debug()
            return

        # Local mode: run all stages before select, then stop
        if self.local_mode:
            self._run_local()
            return

        logger.info("Executing stage: %s", stage.value)
        
        if stage is WorkflowStage.INIT:
            logger.debug("Initializing project")
            self._initialize()
            self._set_current_stage(WorkflowStage.GENERATE)
        elif stage is WorkflowStage.GENERATE:
            logger.debug("Running structure generation")
            self._check_deadline()
            self._generate()
            self._set_current_stage(WorkflowStage.SELECT)
        elif stage is WorkflowStage.SELECT:
            logger.debug("Running selection algorithm")
            self._check_deadline()
            self._select()
            self._set_current_stage(WorkflowStage.RUN_VASP)
        elif stage is WorkflowStage.RUN_VASP:
            logger.debug("Running VASP calculations")
            self._check_deadline()
            try:
                self._run_vasp()
            except SelfResubmitExit:
                logger.info("VASP launcher deadline reached — resubmitting workflow")
                raise
            self._set_current_stage(WorkflowStage.TRAIN_NEP)
        elif stage is WorkflowStage.TRAIN_NEP:
            logger.debug("Training NEP models")
            self._check_deadline()
            try:
                self._train_nep()
            except SelfResubmitExit:
                logger.info("NEP training deadline reached — resubmitting workflow")
                raise
            self._set_current_stage(WorkflowStage.VALIDATE)
        elif stage is WorkflowStage.VALIDATE:
            logger.debug("Running GPUMD validation")
            self._check_deadline()
            try:
                self._validate()
            except SelfResubmitExit:
                logger.info("GPUMD validation deadline reached — resubmitting workflow")
                raise
            if self._validation_is_complete():
                self._set_current_stage(WorkflowStage.COMPLETED)
        else:
            logger.error("Unknown stage: %s", stage.value)
            raise StateError(f"Unknown workflow stage: {stage.value}")

