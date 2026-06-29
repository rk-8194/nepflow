"""Base class for workflow stages."""

from pathlib import Path
from abc import ABC, abstractmethod
from configparser import ConfigParser
import logging

logger = logging.getLogger(__name__)


class SelfResubmitExit(Exception):
    """Raised when a stage resubmits itself via SLURM and needs to exit without advancing."""


class Stage(ABC):
    """Base class for all workflow stages."""
    
    def __init__(
        self,
        project_name: str,
        config_file: Path,
        state_file: Path,
        project_dir: Path,
        debug: bool = False,
        slurm_deadline: float | None = None,
    ):
        """
        Initialize stage.
        
        Args:
            project_name: Name of the project
            config_file: Path to project config file (YAML/TOML)
            state_file: Path to project state database
            project_dir: Base directory for project outputs
            debug: Enable debug mode
            slurm_deadline: Unix timestamp of SLURM walltime deadline
        """
        self.project_name = project_name
        self.config_file = Path(config_file)
        self.state_file = Path(state_file)
        self.project_dir = Path(project_dir)
        self.debug = debug
        self.slurm_deadline = slurm_deadline
    
    @abstractmethod
    def run(self) -> None:
        """Execute this stage."""

    def _find_config_file(self) -> Path:
        """Resolve the project config file using the shared stage search order."""
        candidates = [
            self.project_dir / "config" / "project.config",
            self.config_file,
            self.project_dir / "config" / f"{self.project_name}.ini",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        tried = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(f"Config file not found. Tried:\n{tried}")

    def _load_config(self) -> ConfigParser:
        """Load and return the project config using the shared resolution logic."""
        config_path = self._find_config_file()
        config = ConfigParser()
        config.read(config_path)
        return config
    
    @property
    def stage_name(self) -> str:
        """Return the name of this stage."""
        return self.__class__.__name__.replace("Stage", "").lower()
