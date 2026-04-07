"""Base class for workflow stages."""

from pathlib import Path
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Stage(ABC):
    """Base class for all workflow stages."""
    
    def __init__(
        self,
        project_name: str,
        config_file: Path,
        state_file: Path,
        project_dir: Path,
        debug: bool = False
    ):
        """
        Initialize stage.
        
        Args:
            project_name: Name of the project
            config_file: Path to project config file (YAML/TOML)
            state_file: Path to project state database
            project_dir: Base directory for project outputs
            debug: Enable debug mode
        """
        self.project_name = project_name
        self.config_file = Path(config_file)
        self.state_file = Path(state_file)
        self.project_dir = Path(project_dir)
        self.debug = debug
    
    @abstractmethod
    def run(self) -> None:
        """Execute this stage."""
    
    @property
    def stage_name(self) -> str:
        """Return the name of this stage."""
        return self.__class__.__name__.replace("Stage", "").lower()
