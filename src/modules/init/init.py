"""Project initialization stage."""

import logging

from ..base import Stage

logger = logging.getLogger(__name__)


class InitStage(Stage):
    """Initialize a new project."""
    
    def run(self) -> None:
        """Execute initialization."""
        logger.info("Initializing project")
        
        # Create project directory structure
        self._create_directories()
        
        # Load or create config
        self._setup_config()
        
        logger.info("Project initialization complete")
    
    def _create_directories(self) -> None:
        """Create project directory structure."""
        dirs = [
            self.project_dir / "config" / "slurm",
            self.project_dir / "config" / "nep",
            self.project_dir / "config" / "gpumd",
            self.project_dir / "config" / "vasp",
            self.project_dir / "structures" / "seeds",
            self.project_dir / "structures" / "generated",
            self.project_dir / "structures" / "selected",
            self.project_dir / "vasp" / "jobs",
            self.project_dir / "vasp" / "results",
            self.project_dir / "nep" / "datasets",
            self.project_dir / "nep" / "runs",
            self.project_dir / "gpumd" / "validation",
            self.project_dir / "logs",
            self.project_dir / "reports",
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory: %s", d)
    
    def _setup_config(self) -> None:
        """Load or create configuration."""
        config_dir = self.project_dir / "config"
        project_config_file = config_dir / "project.config"
        
        # Create default project.config if it doesn't exist
        if not project_config_file.exists():
            default_config = f"""# Project Configuration File
# Project: {self.project_name}

[project]
name={self.project_name}
description=NEPFlow project for atomic structure generation and validation
status=initialized

[paths]
structures_path=structures
vasp_path=vasp
nep_path=nep
gpumd_path=gpumd
reports_path=reports

[vasp]
# VASP calculation settings
enabled=true

[nep]
# NEP model training settings
enabled=true

[gpumd]
# GPUMD validation settings
enabled=true

[slurm]
# SLURM job submission settings
enabled=false
"""
            try:
                with open(project_config_file, "w", encoding="utf-8") as f:
                    f.write(default_config)
                logger.info("Created default project config: %s", project_config_file)
            except IOError as e:
                logger.error("Failed to create project config file: %s", e)
                raise
        else:
            logger.debug("Project config file already exists: %s", project_config_file)
        
        # Check for per-project YAML config
        if self.config_file.exists():
            logger.debug("Config file already exists: %s", self.config_file)
        else:
            logger.debug("Config file not found: %s", self.config_file)
            logger.info("Please create config file at: %s", self.config_file)
        
        # Log template directory location
        logger.info("Project config directory: %s", config_dir)
        logger.info("  - Project config: %s", project_config_file)
        logger.info("  - SLURM templates: %s", config_dir / "slurm")
        logger.info("  - NEP templates: %s", config_dir / "nep")
        logger.info("  - GPUMD templates: %s", config_dir / "gpumd")
        logger.info("  - VASP templates: %s", config_dir / "vasp")
