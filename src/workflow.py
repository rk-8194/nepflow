"""
Workflow controller and stage orchestration.
"""

from pathlib import Path
import logging

# Note: src/ is added to sys.path dynamically by nepflow.py
# pylint: disable=import-error
from modules import (
    SelfResubmitExit,
    InitStage,
    GenerateStage,
    SelectStage,
    RunVaspStage,
    TrainNepStage,
    ValidateStage,
)

logger = logging.getLogger(__name__)


class WorkflowController:
    """
    Main workflow controller for project orchestration.
    
    Handles:
    - Reading project state from .project file
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
    ):
        """
        Initialize workflow controller.
        
        Args:
            project_name: Name of the project (scopes config, state, outputs)
            output_dir: Base directory for project outputs
            init_mode: If True, initialize a new project (skip validation)
            debug: Enable debug mode (no external dependencies)
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.init_mode = init_mode
        self.debug = debug
        self.stage_override = stage_override
        self.local_mode = local_mode
        
        # Project directory (contains config, state, logs, outputs)
        self.project_dir = self.output_dir / f"project_{project_name}"
        
        # Derived paths - config is now per-project
        self.config_dir = self.project_dir / "config"
        self.config_file = self.config_dir / f"{project_name}.yaml"
        self.project_file = self.project_dir / ".project"
        self.state_file = self.project_dir / "state.db"
        self.log_dir = self.project_dir / "logs"
        
        logger.info("Initialized controller for project: %s", project_name)
        if self.debug:
            logger.debug("Debug mode enabled")
            logger.debug("Config dir: %s", self.config_dir)
            logger.debug("Config file: %s", self.config_file)
            logger.debug("Project file: %s", self.project_file)
            logger.debug("State file: %s", self.state_file)
            logger.debug("Project dir: %s", self.project_dir)
    
    def _determine_current_stage(self) -> str:
        """
        Determine the current workflow stage from .project file.
        
        Returns:
            Current stage name: 'init', 'generate', 'select', 'run_vasp', 'train_nep', 'validate'
            Defaults to 'init' if .project file doesn't exist.
        """
        if not self.project_file.exists():
            logger.debug("Project file not found: %s, defaulting to 'init'", self.project_file)
            return "init"
        
        try:
            with open(self.project_file, "r", encoding="utf-8") as f:
                stage = f.read().strip()
            
            # Validate stage name
            valid_stages = {"init", "generate", "select", "run_vasp", "train_nep", "validate"}
            if stage not in valid_stages:
                logger.warning("Invalid stage '%s' in %s, resetting to 'init'", stage, self.project_file)
                return "init"
            
            logger.debug("Read stage from %s: %s", self.project_file, stage)
            return stage
            
        except IOError as e:
            logger.warning("Failed to read project file %s: %s, defaulting to 'init'", self.project_file, e)
            return "init"
    
    def _set_current_stage(self, stage: str) -> None:
        """
        Update the current workflow stage in .project file.
        
        Args:
            stage: Stage name to set
        """
        valid_stages = {"init", "generate", "select", "run_vasp", "train_nep", "validate"}
        if stage not in valid_stages:
            logger.error("Invalid stage '%s' - must be one of %s", stage, valid_stages)
            raise ValueError("Invalid stage: %s" % stage)
        
        try:
            self.project_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.project_file, "w", encoding="utf-8") as f:
                f.write(stage)
            logger.debug("Updated project stage to: %s", stage)
        except IOError as e:
            logger.error("Failed to write project file %s: %s", self.project_file, e)
            raise
    
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
            debug=self.debug
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
        if stage == "init":
            logger.info("Local mode — initializing project")
            self._initialize()
            self._set_current_stage("generate")
            stage = "generate"
        
        if stage == "generate":
            logger.info("Local mode — generating base structures (seeds only)")
            self._generate(seeds_only=True)
            # Stage stays at 'generate' — HPC resumes with perturbations
            logger.info("Local mode complete — seeds generated, stage remains at 'generate'")
            print("✓ Base structures generated (seeds only, no perturbations)")
            print("  Transfer the project directory to HPC and resume with:")
            print("  python nepflow.py --project %s" % self.project_name)
        else:
            logger.info(
                "Local mode — nothing to do, current stage is '%s'. "
                "Base structures were already generated.", stage
            )
            print("✓ Local stages already complete (current stage: '%s')" % stage)
            print("  Resume on HPC with:")
            print("  python nepflow.py --project %s" % self.project_name)
    
    def run(self) -> None:
        """
        Run workflow controller.
        
        If --init flag was used, only initializes the project and exits.
        Otherwise, validates that project is initialized, then reads .project file
        to determine current stage and executes it.
        """
        # If in init mode, only run initialization
        if self.init_mode:
            logger.info("Running in init mode - initializing project only")
            self._initialize()
            self._set_current_stage("generate")
            logger.info("Project initialization complete. Initialization stage finished.")
            if not self.local_mode:
                return
        
        # Check if project is initialized
        if not self._check_project_initialized():
            logger.error(
                "Project '%s' is not initialized. "
                "Run with --init flag first: python3 nepflow.py --project %s --init",
                self.project_name,
                self.project_name
            )
            raise ValueError(
                f"Project '{self.project_name}' is not initialized. "
                f"Run: python3 nepflow.py --project {self.project_name} --init"
            )
        
        if self.stage_override:
            self._set_current_stage(self.stage_override)
            logger.info("Stage overridden to: %s", self.stage_override)

        # Local mode: run all stages before select, then stop
        if self.local_mode:
            self._run_local()
            return

        stage = self._determine_current_stage()
        logger.info("Executing stage: %s", stage)
        
        if stage == "init":
            logger.debug("Initializing project")
            self._initialize()
            self._set_current_stage("generate")
        elif stage == "generate":
            logger.debug("Running structure generation")
            self._generate()
            self._set_current_stage("select")
        elif stage == "select":
            logger.debug("Running selection algorithm")
            self._select()
            self._set_current_stage("run_vasp")
        elif stage == "run_vasp":
            logger.debug("Running VASP calculations")
            try:
                self._run_vasp()
            except SelfResubmitExit:
                logger.info("Launcher resubmitted itself — exiting without advancing stage")
                return
            self._set_current_stage("train_nep")
        elif stage == "train_nep":
            logger.debug("Training NEP models")
            self._train_nep()
            self._set_current_stage("validate")
        elif stage == "validate":
            logger.debug("Running GPUMD validation")
            self._validate()
        else:
            logger.error("Unknown stage: %s", stage)
            raise ValueError("Unknown stage: %s" % stage)

