"""Example: fetch Materials Project structures using canonical generation APIs."""

from __future__ import annotations

import logging
from pathlib import Path

from nepflow.config.loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Fetch structures based on the example project configuration."""

    from nepflow.stages.generation.generators.materials_project import (
        build_materials_project_fetcher,
    )

    config_path = (
        Path(__file__).resolve().parents[1]
        / "projects"
        / "project_test"
        / "config"
        / "project.config"
    )
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return

    config = load_config(config_path)
    elements = list(config.composition.elements)
    structures = list(config.generation.crystal_structures)
    logger.info("Elements: %s", elements)
    logger.info("Structures: %s", structures)

    try:
        fetcher = build_materials_project_fetcher(
            api_key=config.materials_project.api_key,
        )
    except ValueError as exc:
        logger.error("Failed to initialize fetcher: %s", exc)
        logger.info("Set MP_API_KEY in the runtime environment")
        return

    results = fetcher.fetch_structures(elements, structures)
    logger.info("Found %d structures:", len(results))
    print("\n" + "=" * 80)
    for result in results:
        print(f"\nMaterial ID: {result['material_id']}")
        print(f"Formula: {result['formula']}")
        print(f"Crystal System: {result['symmetry']['crystal_system']}")
        lattice = result["lattice"]
        print("Lattice Parameters:")
        print(f"  a = {lattice['a']:.4f} Å")
        print(f"  b = {lattice['b']:.4f} Å")
        print(f"  c = {lattice['c']:.4f} Å")
        print(f"  α = {lattice['alpha']:.2f}°")
        print(f"  β = {lattice['beta']:.2f}°")
        print(f"  γ = {lattice['gamma']:.2f}°")
        print(f"  Volume = {lattice['volume']:.4f} Å³")
    print("=" * 80)


if __name__ == "__main__":
    main()
