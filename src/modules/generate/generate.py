"""Structure generation stage."""

import logging
from configparser import ConfigParser
from pathlib import Path
from typing import Dict, List

from ..base import Stage
from .materials_project import get_materials_project_fetcher

logger = logging.getLogger("nepflow.generate")


class GenerateStage(Stage):
    """Generate base structures and variants."""
    
    def run(self) -> None:
        """Execute structure generation."""
        logger.info("Running structure generation")
        
        try:
            # Find and parse configuration
            config_path = self._find_config_file()
            config = ConfigParser()
            config.read(config_path)
            logger.debug(f"Loaded config from {config_path}")
            
            # Verify required sections exist
            if not config.has_section("generation"):
                logger.error("Missing [generation] section in config")
                raise ValueError("Config missing [generation] section")
            
            elements = self._parse_config_list(config, "generation", "elements")
            structures = self._parse_config_list(config, "generation", "crystal_structures")
            
            logger.info(f"Target elements: {elements}")
            logger.info(f"Target structures: {structures}")
            
            # Fetch lattice parameters from Materials Project
            self._fetch_and_report_lattice_parameters(config, elements, structures)
            
            logger.info("Structure generation complete")
            
        except Exception as e:
            logger.error(f"Error during structure generation: {e}")
            raise
    
    def _find_config_file(self) -> Path:
        """
        Find the project config file.
        
        Tries to locate project.config (INI format) in the project directory.
        Falls back to the configured config_file path if it exists.
        
        Returns:
            Path to the config file
            
        Raises:
            FileNotFoundError: If no config file is found
        """
        # Try project.config in the project directory config folder
        project_config = self.project_dir / "config" / "project.config"
        if project_config.exists():
            logger.debug(f"Found project config at {project_config}")
            return project_config
        
        # Fall back to configured config_file
        if self.config_file.exists():
            logger.debug(f"Found config at {self.config_file}")
            return self.config_file
        
        raise FileNotFoundError(
            f"Config file not found. Tried:\n"
            f"  - {project_config}\n"
            f"  - {self.config_file}"
        )
    
    @staticmethod
    def _parse_config_list(config: ConfigParser, section: str, option: str) -> List[str]:
        """Parse comma-separated list from config."""
        value = config.get(section, option)
        return [item.strip() for item in value.split(",") if item.strip()]
    
    def _fetch_and_report_lattice_parameters(
        self, config: ConfigParser, elements: List[str], structures: List[str]
    ) -> None:
        """Fetch and report lattice parameters for selected systems."""
        logger.info("")
        logger.info("Fetching lattice parameters from Materials Project")
        
        try:
            fetcher = get_materials_project_fetcher(dict(config))
            
            # Clear cache for fresh data on each run
            fetcher.clear_cache()
            
            results = fetcher.fetch_structures(elements, structures, use_cache=True)
            
            if not results:
                logger.warning("No structures found matching the criteria")
                return
            
            # Filter results to only include requested structures
            filtered_results = [r for r in results if r.get("structure") in structures]
            
            if not filtered_results:
                logger.warning(f"No structures found for requested types: {structures}")
                return
            
            logger.info(f"✓ Found {len(filtered_results)} structure(s)")
            
            # Group by element and structure for reporting
            for element in sorted(set(r["elements"][0] for r in filtered_results)):
                element_results = [r for r in filtered_results if r["elements"][0] == element]
                
                for struct_type in sorted(set(r["structure"] for r in element_results)):
                    type_results = [r for r in element_results if r["structure"] == struct_type]
                    if type_results:
                        result = type_results[0]  # Take the most stable (first in MP results)
                        self._log_structure_summary(element, struct_type, result)
            
            logger.info("")
            
        except ImportError as e:
            logger.error(f"Materials Project API not available: {e}")
            logger.info("Install with: pip install mp-api")
            raise
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            logger.info("Please set api_key in [materialsproject] section of project config or set MP_API_KEY environment variable")
            raise
    
    @staticmethod
    def _group_by_structure_type(results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group results by crystal structure type."""
        grouped = {}
        for result in results:
            crystal_system = result["symmetry"]["crystal_system"]
            if crystal_system not in grouped:
                grouped[crystal_system] = []
            grouped[crystal_system].append(result)
        return grouped
    
    @staticmethod
    def _log_structure_summary(element: str, struct_type: str, structure: Dict) -> None:
        """Log a one-line summary of structure."""
        lattice = structure["lattice"]
        logger.info(
            f"  {element:3s}-{struct_type:12s}  a={lattice['a']:.6f}Å  V={lattice['volume']:8.4f}ų  {structure['material_id']}"
        )
    
    @staticmethod
    def _log_structure_info(structure: Dict) -> None:
        """Log detailed information about a structure (for log file)."""
        logger.debug(f"Material ID: {structure['material_id']}")
        logger.debug(f"Formula: {structure['formula']}")
        
        lattice = structure["lattice"]
        logger.debug(f"Lattice Parameters (Å):")
        logger.debug(f"  a={lattice['a']:.6f}  b={lattice['b']:.6f}  c={lattice['c']:.6f}")
        logger.debug(f"  α={lattice['alpha']:.2f}°  β={lattice['beta']:.2f}°  γ={lattice['gamma']:.2f}°")
        logger.debug(f"  Volume={lattice['volume']:.4f} ų")
        
        symmetry = structure["symmetry"]
        logger.debug(f"Symmetry: {symmetry['crystal_system']} (Space group #{symmetry['space_group']})")
