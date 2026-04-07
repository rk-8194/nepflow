"""
Example script: Fetch structures from Materials Project and cache them.

This demonstrates how to use the MaterialsProjectFetcher to get lattice
parameters for elements and crystal structures specified in project.config.
"""

import sys
import logging
from pathlib import Path
from configparser import ConfigParser

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.generate.materials_project import get_materials_project_fetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Fetch structures based on project config."""
    # Load project config
    config_path = Path(__file__).parent.parent.parent / "projects" / "project_test" / "config" / "project.config"
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    config = ConfigParser()
    config.read(config_path)

    # Get generation parameters
    elements = [e.strip() for e in config.get("generation", "elements").split(",")]
    structures = [s.strip() for s in config.get("generation", "crystal_structures").split(",")]

    logger.info(f"Elements: {elements}")
    logger.info(f"Structures: {structures}")

    # Create fetcher
    try:
        fetcher = get_materials_project_fetcher(dict(config))
    except ValueError as e:
        logger.error(f"Failed to initialize fetcher: {e}")
        logger.info("Please set MP_API_KEY environment variable or add api_key to project.config")
        return

    # Fetch structures
    try:
        results = fetcher.fetch_structures(elements, structures)
        
        logger.info(f"\nFound {len(results)} structures:")
        print("\n" + "="*80)
        for result in results:
            print(f"\nMaterial ID: {result['material_id']}")
            print(f"Formula: {result['formula']}")
            print(f"Crystal System: {result['symmetry']['crystal_system']}")
            lattice = result['lattice']
            print(f"Lattice Parameters:")
            print(f"  a = {lattice['a']:.4f} Å")
            print(f"  b = {lattice['b']:.4f} Å")
            print(f"  c = {lattice['c']:.4f} Å")
            print(f"  α = {lattice['alpha']:.2f}°")
            print(f"  β = {lattice['beta']:.2f}°")
            print(f"  γ = {lattice['gamma']:.2f}°")
            print(f"  Volume = {lattice['volume']:.4f} Ų")
        print("="*80)

    except Exception as e:
        logger.error(f"Error fetching structures: {e}")
        raise


if __name__ == "__main__":
    main()
