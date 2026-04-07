"""
Structure generation stage — unified orchestration.

Pipeline:
  1. Parse config → build composition grid
  2. For each composition → run configurational generators → base structures
  3. For each base structure → PerturbationEngine.process() → perturbed variants
  4. Save all structures + summary
"""

import logging
from configparser import ConfigParser
from pathlib import Path
from typing import List

from ase import Atoms
from ase.io import write

from ..base import Stage
from .composition import CompositionGrid
from .configurational import (
    MaterialsProjectGenerator,
    RandomSolidSolutionGenerator,
    SegregatedGenerator,
    SQSGenerator,
)
from .materials_project import get_materials_project_fetcher
from .structure_generation import PerturbationEngine

logger = logging.getLogger("nepflow.generate")


class GenerateStage(Stage):
    """Generate structures spanning the full phase space."""

    def run(self, seeds_only: bool = False) -> None:
        logger.info("Running structure generation")

        config_path = self._find_config_file()
        config = ConfigParser()
        config.read(config_path)
        logger.debug(f"Loaded config from {config_path}")

        for section in ("composition", "generation"):
            if not config.has_section(section):
                raise ValueError(f"Config missing [{section}] section")

        # --- global settings ---
        random_seed = config.getint("project", "random_seed", fallback=42)
        elements = self._parse_list(config, "composition", "elements")
        crystal_structures = self._parse_list(config, "generation", "crystal_structures")
        target_n_atoms = config.getint("generation", "target_n_atoms", fallback=250)

        logger.info(f"Elements: {elements}")
        logger.info(f"Crystal structures: {crystal_structures}")

        seeds_dir = self.project_dir / "structures" / "seeds"
        seeds_file = seeds_dir / "base_structures.xyz"

        # Resume: if seeds already exist, load them and skip to perturbations
        if not seeds_only and seeds_file.exists():
            from ase.io import read as ase_read
            logger.info("Found existing seeds — loading from %s", seeds_file)
            all_bases = ase_read(str(seeds_file), index=":")
            logger.info(f"  Loaded {len(all_bases)} base structures from seeds")
        else:
            # ----------------------------------------------------------
            # Step 1: composition grid
            # ----------------------------------------------------------
            logger.info("")
            logger.info("Step 1: Building composition grid")
            grid = CompositionGrid(
                elements=elements,
                step=config.getfloat("composition", "composition_step", fallback=0.1),
                include_pure=config.getboolean("composition", "include_pure_elements", fallback=True),
                include_binaries=config.getboolean("composition", "include_binaries", fallback=True),
                include_ternaries=config.getboolean("composition", "include_ternaries", fallback=True),
            )
            compositions = grid.generate()
            logger.info(f"  {len(compositions)} compositions")

            # ----------------------------------------------------------
            # Step 2: configurational generators
            # ----------------------------------------------------------
            logger.info("")
            logger.info("Step 2: Generating base structures (configurational generators)")
            generators = self._build_generators(config, elements, random_seed)
            if not generators:
                logger.warning("No configurational generators enabled")
                return

            all_bases: List[Atoms] = []
            for comp in compositions:
                label = CompositionGrid.format_composition(comp)
                for gen_name, gen in generators:
                    bases = gen.generate(comp, crystal_structures, target_n_atoms)
                    for b in bases:
                        b.info.setdefault("composition", comp)
                        b.info.setdefault("elements", elements)
                    all_bases.extend(bases)
                    if bases:
                        logger.debug(f"  {label} / {gen_name}: {len(bases)} structures")

            logger.info(f"  Total base structures: {len(all_bases)}")

            if not all_bases:
                logger.warning("No base structures generated — aborting")
                return

            # Save seeds
            seeds_dir.mkdir(parents=True, exist_ok=True)
            write(str(seeds_file), all_bases)
            logger.info(f"  Saved seeds to {seeds_file}")

        # In seeds_only mode, stop here (no perturbations)
        if seeds_only:
            logger.info("Seeds-only mode — skipping perturbations")
            return

        # ----------------------------------------------------------
        # Step 3: perturbations
        # ----------------------------------------------------------
        logger.info("")
        logger.info("Step 3: Applying perturbations")
        engine = self._build_engine(config, target_n_atoms, random_seed)

        generated_dir = self.project_dir / "structures" / "generated"
        engine.process(
            all_bases,
            output_dir=generated_dir,
            n_rattled=config.getint("generation", "n_rattled", fallback=10),
            n_strained=config.getint("generation", "n_strained", fallback=10),
            n_deformed=config.getint("generation", "n_deformed", fallback=10),
            n_vacancies=config.getint("generation", "n_vacancies", fallback=10),
            n_interstitials=config.getint("generation", "n_interstitials", fallback=10),
            n_workers=config.getint("generation", "n_workers", fallback=0),
        )

        # ----------------------------------------------------------
        # Step 4: report
        # ----------------------------------------------------------
        summary = engine.get_summary()
        logger.info("")
        logger.info(f"Total structures: {summary['total']}")
        logger.info("By perturbation type:")
        for ptype, count in sorted(summary["by_type"].items()):
            logger.info(f"  {ptype:20s}: {count:5d}")
        logger.info("By configurational type:")
        for ctype, count in sorted(summary["by_config"].items()):
            logger.info(f"  {ctype:25s}: {count:5d}")

        logger.info("")
        logger.info("Structure generation complete")

    # ==================================================================
    # helpers
    # ==================================================================

    def _find_config_file(self) -> Path:
        project_config = self.project_dir / "config" / "project.config"
        if project_config.exists():
            return project_config
        if self.config_file.exists():
            return self.config_file
        raise FileNotFoundError(
            f"Config file not found. Tried:\n"
            f"  - {project_config}\n"
            f"  - {self.config_file}"
        )

    @staticmethod
    def _parse_list(config: ConfigParser, section: str, option: str) -> List[str]:
        value = config.get(section, option)
        return [item.strip() for item in value.split(",") if item.strip()]

    def _build_generators(
        self, config: ConfigParser, elements: List[str], random_seed: int  # noqa: ARG002
    ) -> List[tuple]:
        """Instantiate enabled configurational generators."""
        generators: List[tuple] = []

        if config.getboolean("generation", "use_materials_project", fallback=True):
            try:
                fetcher = get_materials_project_fetcher(dict(config))
                generators.append((
                    "MaterialsProject",
                    MaterialsProjectGenerator(fetcher, max_per_composition=5),
                ))
            except Exception as e:
                logger.warning(f"Cannot initialise MP fetcher: {e}")

        n_rss = config.getint("generation", "n_random_solid_solution", fallback=3)
        if config.getboolean("generation", "use_random_solid_solution", fallback=True):
            generators.append((
                "RandomSolidSolution",
                RandomSolidSolutionGenerator(n_structures=n_rss, random_seed=random_seed),
            ))

        n_sqs = config.getint("generation", "n_sqs", fallback=3)
        if config.getboolean("generation", "use_sqs", fallback=True):
            generators.append((
                "SQS",
                SQSGenerator(n_structures=n_sqs, random_seed=random_seed),
            ))

        n_seg = config.getint("generation", "n_segregated", fallback=3)
        if config.getboolean("generation", "use_segregated", fallback=True):
            generators.append((
                "Segregated",
                SegregatedGenerator(n_structures=n_seg, random_seed=random_seed),
            ))

        logger.info(f"  Active generators: {[name for name, _ in generators]}")
        return generators

    @staticmethod
    def _build_engine(
        config: ConfigParser, target_n_atoms: int, random_seed: int
    ) -> PerturbationEngine:
        return PerturbationEngine(
            rattle_std=config.getfloat("generation", "rattle_std", fallback=0.03),
            rattle_d_min=config.getfloat("generation", "rattle_d_min", fallback=1.5),
            strain_limit=(
                config.getfloat("generation", "strain_min", fallback=-0.02),
                config.getfloat("generation", "strain_max", fallback=0.02),
            ),
            vacancy_range=(
                config.getfloat("generation", "vacancy_min", fallback=0.0),
                config.getfloat("generation", "vacancy_max", fallback=0.1),
            ),
            interstitial_range=(
                config.getfloat("generation", "interstitial_min", fallback=0.05),
                config.getfloat("generation", "interstitial_max", fallback=0.1),
            ),
            interstitial_d_min=config.getfloat("generation", "interstitial_d_min", fallback=1.65),
            volume_scale_range=(
                config.getfloat("generation", "volume_scale_min", fallback=0.8),
                config.getfloat("generation", "volume_scale_max", fallback=1.2),
            ),
            n_volume_points=config.getint("generation", "n_volume_points", fallback=11),
            target_n_atoms=target_n_atoms,
            random_seed=random_seed,
        )
