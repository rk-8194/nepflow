"""
Structure generation stage - unified orchestration.

Pipeline:
  1. Parse config -> build composition grid
  2. For each composition -> run configurational generators -> base structures
  3. For each base structure -> PerturbationEngine.process() -> perturbed variants
  4. Save all structures + summary
"""

import logging
import random
import subprocess
from configparser import ConfigParser
from pathlib import Path
from typing import List

from ase import Atoms
from ase.build import bulk
from ase.io import write

from common.structure_identity import annotate_structure_hashes
from ..base import Stage
from .generators import (
    CompositionGrid,
    MaterialsProjectGenerator,
    PerturbationEngine,
    RandomSolidSolutionGenerator,
    SegregatedGenerator,
    SQSGenerator,
    get_materials_project_fetcher,
)

logger = logging.getLogger("nepflow.generate")


class GenerateStage(Stage):
    """Generate structures spanning the full phase space."""

    def run(self, seeds_only: bool = False) -> None:
        logger.info("Running structure generation")

        config, settings = self.load_config()

        if self.debug:
            return self._run_debug(
                settings["elements"],
                settings["random_seed"],
                gas_elements=settings["gas_elements"],
            )

        all_bases = self.resume_if_needed(config, settings, seeds_only=seeds_only)
        if all_bases is None:
            return

        if seeds_only:
            logger.info("Seeds-only mode - skipping perturbations")
            self._offer_project_upload(config)
            return

        summary = self.execute(config, settings, all_bases)
        self.finalize(summary)

        logger.info("")
        logger.info("Structure generation complete")

    # ==================================================================
    # stage lifecycle
    # ==================================================================

    def load_config(self) -> tuple[ConfigParser, dict]:
        """Load, validate, and summarise the stage configuration."""
        config_path = self._find_config_file()
        config = self._load_config()
        logger.debug(f"Loaded config from {config_path}")

        self._validate_config(config)
        settings = self._load_settings(config)
        self._log_settings(settings)
        return config, settings

    def resume_if_needed(
        self,
        config: ConfigParser,
        settings: dict,
        seeds_only: bool = False,
    ) -> List[Atoms] | None:
        """Reuse saved seeds when possible, otherwise prepare fresh bases."""
        seeds_file = self._seeds_file()
        if not seeds_only and seeds_file.exists():
            return self._load_saved_bases(seeds_file)
        return self.prepare(config, settings)

    def prepare(
        self,
        config: ConfigParser,
        settings: dict,
    ) -> List[Atoms] | None:
        """Build or fetch the base structures for later perturbation."""
        compositions = self._build_compositions(config, settings["elements"])
        generators = self._build_generators(
            config,
            settings["elements"],
            settings["random_seed"],
            gas_elements=settings["gas_elements"],
        )
        if not generators:
            logger.warning("No configurational generators enabled")
            return None

        all_bases: List[Atoms] = []
        seed_index = 0
        logger.info("")
        logger.info("Step 2: Generating base structures (configurational generators)")
        for composition in compositions:
            label = CompositionGrid.format_composition(composition)
            for gen_name, gen in generators:
                bases = gen.generate(
                    composition,
                    settings["crystal_structures"],
                    settings["target_n_atoms"],
                )
                self._annotate_base_structures(
                    bases,
                    composition,
                    settings["elements"],
                    settings["gas_elements"],
                )
                seed_index = self._assign_seed_ids(bases, seed_index)
                all_bases.extend(bases)
                if bases:
                    logger.debug(f"  {label} / {gen_name}: {len(bases)} structures")

        if settings["gas_elements"]:
            self._extend_with_gas_phase_bases(all_bases, generators, settings)

        logger.info(f"  Total base structures: {len(all_bases)}")
        if not all_bases:
            logger.warning("No base structures generated - aborting")
            return None

        seeds_file = self._seeds_file()
        seeds_file.parent.mkdir(parents=True, exist_ok=True)
        annotate_structure_hashes(all_bases)
        write(str(seeds_file), all_bases)
        logger.info(f"  Saved seeds to {seeds_file}")
        return all_bases

    def execute(
        self,
        config: ConfigParser,
        settings: dict,
        all_bases: List[Atoms],
    ) -> dict:
        """Apply perturbations to the prepared base structures."""
        logger.info("")
        logger.info("Step 3: Applying perturbations")
        engine = self._build_engine(
            config,
            settings["target_n_atoms"],
            settings["random_seed"],
            gas_elements=settings["gas_elements"],
        )

        generated_dir = self.project_dir / "structures" / "generated"
        engine.process(
            all_bases,
            output_dir=generated_dir,
            n_rattled=config.getint("generation", "n_rattled", fallback=10),
            n_strained=config.getint("generation", "n_strained", fallback=10),
            n_deformed=config.getint("generation", "n_deformed", fallback=10),
            n_vacancies=config.getint("generation", "n_vacancies", fallback=10),
            n_interstitials=config.getint("generation", "n_interstitials", fallback=10),
            n_gas_interstitials=(
                config.getint("generation", "n_gas_interstitials", fallback=10)
                if settings["gas_elements"]
                else 0
            ),
            n_vacancy_interstitial=(
                config.getint("generation", "n_vacancy_interstitial", fallback=10)
                if settings["gas_elements"]
                else 0
            ),
            n_gas_in_vacancy=(
                config.getint("generation", "n_gas_in_vacancy", fallback=10)
                if settings["gas_elements"]
                else 0
            ),
            n_workers=config.getint("generation", "n_workers", fallback=0),
        )
        return engine.get_summary()

    def finalize(self, summary: dict) -> None:
        """Log the stage summary after successful execution."""
        logger.info("")
        logger.info(f"Total structures: {summary['total']}")
        logger.info("By perturbation type:")
        for ptype, count in sorted(summary["by_type"].items()):
            logger.info(f"  {ptype:20s}: {count:5d}")
        logger.info("By configurational type:")
        for ctype, count in sorted(summary["by_config"].items()):
            logger.info(f"  {ctype:25s}: {count:5d}")

    def _offer_project_upload(self, config: ConfigParser) -> None:
        """Offer to copy the project to the configured remote NEPFlow folder."""
        scp_address = config.get("hpc", "scp_address", fallback="").strip()
        if not scp_address:
            logger.debug("No hpc.scp_address configured - skipping upload prompt")
            return

        remote_target = f"{scp_address.rstrip('/')}/projects/{self.project_dir.name}"
        print()
        print("Local seed generation is complete.")
        response = input(
            f"Upload project to the remote projects folder at {remote_target}? [y/N]: "
        ).strip().lower()
        if response not in {"y", "yes"}:
            logger.info("Project upload skipped")
            return

        logger.info("Uploading project to %s", remote_target)
        try:
            subprocess.run(
                ["scp", "-r", str(self.project_dir), remote_target],
                check=True,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError("scp was not found on PATH") from e
        logger.info("Project upload complete")

    @staticmethod
    def _validate_config(config: ConfigParser) -> None:
        for section in ("composition", "generation"):
            if not config.has_section(section):
                raise ValueError(f"Config missing [{section}] section")

    def _load_settings(self, config: ConfigParser) -> dict:
        return {
            "random_seed": config.getint("project", "random_seed", fallback=42),
            "elements": self._parse_list(config, "composition", "elements"),
            "gas_elements": (
                self._parse_list(config, "composition", "gasElements")
                if config.has_option("composition", "gasElements")
                else []
            ),
            "crystal_structures": self._parse_list(
                config, "generation", "crystal_structures"
            ),
            "target_n_atoms": config.getint("generation", "target_n_atoms", fallback=250),
        }

    @staticmethod
    def _log_settings(settings: dict) -> None:
        logger.info(f"Elements: {settings['elements']}")
        if settings["gas_elements"]:
            logger.info(f"Gas elements: {settings['gas_elements']}")
        logger.info(f"Crystal structures: {settings['crystal_structures']}")

    def _seeds_file(self) -> Path:
        return self.project_dir / "structures" / "seeds" / "base_structures.xyz"

    @staticmethod
    def _load_saved_bases(seeds_file: Path) -> List[Atoms]:
        from ase.io import read as ase_read

        logger.info("Found existing seeds - loading from %s", seeds_file)
        all_bases = ase_read(str(seeds_file), index=":")
        logger.info(f"  Loaded {len(all_bases)} base structures from seeds")
        if isinstance(all_bases, list):
            return all_bases
        return [all_bases]

    def _build_compositions(
        self,
        config: ConfigParser,
        elements: List[str],
    ) -> List[dict[str, float]]:
        logger.info("")
        logger.info("Step 1: Building composition grid")
        grid = CompositionGrid(
            elements=elements,
            step=config.getfloat("composition", "composition_step", fallback=0.1),
            include_pure=config.getboolean(
                "composition", "include_pure_elements", fallback=True
            ),
            include_binaries=config.getboolean(
                "composition", "include_binaries", fallback=True
            ),
            include_ternaries=config.getboolean(
                "composition", "include_ternaries", fallback=True
            ),
        )
        compositions = grid.generate()
        logger.info(f"  {len(compositions)} compositions")
        return compositions

    @staticmethod
    def _annotate_base_structures(
        bases: List[Atoms],
        composition: dict[str, float],
        elements: List[str],
        gas_elements: List[str],
    ) -> None:
        for base in bases:
            base.info.setdefault("composition", composition)
            base.info.setdefault("elements", elements)
            if gas_elements:
                base.info.setdefault("gas_elements", gas_elements)

    @staticmethod
    def _assign_seed_ids(bases: List[Atoms], start_index: int) -> int:
        for offset, base in enumerate(bases):
            base.info.setdefault("seed_id", f"seed_{start_index + offset:06d}")
        return start_index + len(bases)

    def _extend_with_gas_phase_bases(
        self,
        all_bases: List[Atoms],
        generators: List[tuple],
        settings: dict,
    ) -> None:
        logger.info("")
        logger.info("Step 2b: Fetching gas-phase structures from Materials Project")
        mp_gen = self._get_mp_generator(generators)
        if mp_gen is None:
            logger.warning("  MP generator not available - skipping gas-phase fetch")
            return

        gas_bases = mp_gen.generate_gas_phases(
            metal_elements=settings["elements"],
            gas_elements=settings["gas_elements"],
            target_n_atoms=settings["target_n_atoms"],
        )
        for base in gas_bases:
            base.info.setdefault("elements", settings["elements"])
            base.info.setdefault("gas_elements", settings["gas_elements"])
        self._assign_seed_ids(gas_bases, len(all_bases))
        all_bases.extend(gas_bases)
        logger.info(f"  Gas-phase base structures: {len(gas_bases)}")

    # ==================================================================
    # generation helpers
    # ==================================================================

    @staticmethod
    def _parse_list(config: ConfigParser, section: str, option: str) -> List[str]:
        value = config.get(section, option)
        return [item.strip() for item in value.split(",") if item.strip()]

    def _build_generators(
        self, config: ConfigParser, elements: List[str], random_seed: int,  # noqa: ARG002
        gas_elements: List[str] | None = None,
    ) -> List[tuple]:
        """Instantiate enabled configurational generators."""
        generators: List[tuple] = []

        if config.getboolean("generation", "use_materials_project", fallback=True):
            try:
                fetcher = get_materials_project_fetcher(dict(config))
                generators.append((
                    "MaterialsProject",
                    MaterialsProjectGenerator(
                        fetcher, max_per_composition=5,
                        gas_elements=gas_elements,
                    ),
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
    def _get_mp_generator(generators: List[tuple]):
        """Extract the MaterialsProjectGenerator from the generators list."""
        for name, gen in generators:
            if isinstance(gen, MaterialsProjectGenerator):
                return gen
        return None

    @staticmethod
    def _build_engine(
        config: ConfigParser, target_n_atoms: int, random_seed: int,
        gas_elements: List[str] | None = None,
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
            gas_elements=gas_elements or [],
            gas_interstitial_d_min=config.getfloat("generation", "gas_interstitial_d_min", fallback=1.2),
            max_gas_occupancy=config.getint("generation", "max_gas_occupancy", fallback=3),
            elastic_stress_enabled=config.getboolean(
                "generation", "elastic_stress_enabled", fallback=True
            ),
            elastic_strain_amplitudes=GenerateStage._parse_float_list(
                config,
                "generation",
                "elastic_strain_amplitudes",
                fallback="-0.02,-0.01,-0.005,0.005,0.01,0.02",
            ),
        )

    @staticmethod
    def _parse_float_list(
        config: ConfigParser,
        section: str,
        option: str,
        fallback: str,
    ) -> List[float]:
        value = config.get(section, option, fallback=fallback)
        floats: List[float] = []
        for item in value.split(","):
            stripped = item.strip()
            if stripped:
                floats.append(float(stripped))
        return floats

    def _run_debug(self, elements: List[str], random_seed: int,
                    gas_elements: List[str] | None = None) -> None:
        """Generate 10 synthetic structures for debug/simulation mode."""
        if not elements:
            elements = ["Si", "Ge"]
            logger.info("[DEBUG] No elements in config - using defaults: %s", elements)
        logger.info("[DEBUG] Generating 10 synthetic structures (no external calls)")
        if gas_elements:
            logger.info("[DEBUG] Gas elements: %s", gas_elements)
        rng = random.Random(random_seed)
        n_debug = 10

        structures: List[Atoms] = []
        crystal_types = ["bcc", "fcc"]
        lattice_a = {"bcc": 3.16, "fcc": 3.80}
        for i in range(n_debug):
            crystal = crystal_types[i % len(crystal_types)]
            elem = elements[0]
            atoms = bulk(elem, crystal, a=lattice_a[crystal], cubic=True) * (2, 2, 2)

            # Assign random element mix across all sites
            symbols = [elements[rng.randrange(len(elements))]
                       for _ in range(len(atoms))]
            atoms.set_chemical_symbols(symbols)

            # Apply small random rattle so structures aren't identical
            atoms.rattle(stdev=0.01, seed=random_seed + i)

            atoms.info["config_type"] = f"debug_{crystal}_{i:04d}"
            atoms.info["generator"] = "debug"
            atoms.info["elements"] = elements
            atoms.info["seed_id"] = f"seed_{i:06d}"
            if gas_elements:
                atoms.info["gas_elements"] = gas_elements
            structures.append(atoms)

        # Generate a few gas-interstitial debug structures
        if gas_elements:
            from ase import Atom
            n_gas_debug = 5
            logger.info("[DEBUG] Generating %d gas-interstitial debug structures", n_gas_debug)
            for i in range(n_gas_debug):
                base = structures[i % len(structures)].copy()
                # Insert 1-3 gas atoms at random positions
                n_insert = rng.randint(1, 3)
                for _ in range(n_insert):
                    gas_elem = gas_elements[rng.randrange(len(gas_elements))]
                    frac = [rng.random() for _ in range(3)]
                    pos = [
                        sum(frac[j] * float(base.cell[j][axis]) for j in range(3))
                        for axis in range(3)
                    ]
                    base.append(Atom(symbol=gas_elem, position=pos))
                base.info["config_type"] = f"debug_gas_interstitial_{i:04d}"
                base.info["generator"] = "debug"
                base.info["perturbation_type"] = "gas_interstitial"
                base.info["elements"] = elements
                base.info["gas_elements"] = gas_elements
                base.info["seed_id"] = f"seed_{n_gas_debug + i:06d}"
                structures.append(base)

        # Save seeds
        seeds_dir = self.project_dir / "structures" / "seeds"
        seeds_dir.mkdir(parents=True, exist_ok=True)
        seeds_file = seeds_dir / "base_structures.xyz"
        annotate_structure_hashes(structures)
        write(str(seeds_file), structures)
        logger.info(f"[DEBUG] Saved {len(structures)} seed structures to {seeds_file}")

        # Save generated (same as seeds in debug - no perturbation step)
        generated_dir = self.project_dir / "structures" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        generated_file = generated_dir / "generated_structures.xyz"
        write(str(generated_file), structures)
        logger.info(f"[DEBUG] Saved {len(structures)} generated structures to {generated_file}")

        logger.info("[DEBUG] Structure generation complete")
