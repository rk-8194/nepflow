"""Base class for workflow stages."""

from pathlib import Path
from abc import ABC, abstractmethod
import logging

from configparser import ConfigParser

from nepflow.config.loader import find_config_path, load_legacy_config

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
            config_file: Legacy path hint; the shared boundary resolves project.config
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
        """Resolve the one supported project configuration path."""
        return find_config_path(self.project_dir, explicit_path=self.config_file)

    def _load_config(self) -> ConfigParser:
        """Load the typed config through the temporary legacy stage adapter."""
        config_path = self._find_config_file()
        return load_legacy_config(
            config_path,
            project_name=self.project_name,
            require_scientific_fields=False,
        )
    
    @property
    def stage_name(self) -> str:
        """Return the name of this stage."""
        return self.__class__.__name__.replace("Stage", "").lower()
