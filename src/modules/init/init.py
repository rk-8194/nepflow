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
# Random seed for reproducibility (used across all stages)
random_seed=42

[paths]
structures_path=structures
vasp_path=vasp
nep_path=nep
gpumd_path=gpumd
reports_path=reports

[materialsproject]
# Materials Project API key - get from https://next-gen.materialsproject.org/dashboard
# Can also be set via MP_API_KEY environment variable
api_key=

[composition]
# Elements to include (comma-separated)
elements=

# Composition step size (atomic fraction)
# Controls granularity of the simplex grid
# step=0.1 → 11 points per binary edge, 66 total for 3 elements
# step=0.05 → 21 points per binary edge, ~250 total for 3 elements
composition_step=0.125

# Which subsystems to include
include_pure_elements=true
include_binaries=true
include_ternaries=true

[generation]
# Crystal structures to use as base lattices
# Applied to all compositions (pure element lattices are substituted for alloys)
crystal_structures=bcc,fcc,hcp

# Target number of atoms per supercell for DFT calculations
target_n_atoms=250

# Parallel workers for perturbation generation (0 = auto-detect CPU count, 1 = serial)
n_workers=0

# --- Configurational generators (enable/disable) ---
use_materials_project=true
use_random_solid_solution=true
use_sqs=true
use_segregated=true

# Number of configurations per generator per composition
n_random_solid_solution=3
n_sqs=1
n_segregated=3

# --- Volume profile ---
# Isotropic volume scaling for E-V curves (applied to unperturbed supercells)
volume_scale_min=0.8
volume_scale_max=1.2
n_volume_points=11

# --- Perturbation counts (per base structure at equilibrium) ---
n_rattled=10
n_strained=10
n_deformed=10
n_vacancies=10
n_interstitials=10

# --- Perturbation parameters ---
# Rattling (thermal disorder via hiphive MC)
rattle_std=0.03
rattle_d_min=1.5

# Strain (isotropic)
strain_min=-0.02
strain_max=0.02

# Vacancies (fraction of atoms to remove)
vacancy_min=0.0
vacancy_max=0.1

# Interstitials (atoms inserted at random valid positions)
interstitial_d_min=1.65
interstitial_min=0.05
interstitial_max=0.1

[selection]
# NEP model file for descriptor computation (in config/nep/ directory)
# Download NEP89 from: https://github.com/brucefan1983/GPUMD/tree/master/potentials/nep/nep89_20250409
# Place this in the config/nep folder.
nep_model_file=nep89.txt

# Farthest-point sampling parameters
# Binary search adjusts min_distance to hit these counts (±target_tolerance)
target_train_count=1000
target_test_count=200
target_tolerance=50

# Descriptor aggregation type
# structure: mean of per-atom descriptors → one vector per structure (recommended)
# atomic: per-atom descriptors with frame deduplication
descriptor_type=structure

[vasp]
enabled=true

[nep]
enabled=true

[gpumd]
enabled=true

[slurm]
enabled=false

# Maximum concurrent VASP jobs for this project (squeue-filtered by project name)
max_concurrent=20

# Launcher walltime fallback (HH:MM:SS) — normally obtained from SLURM submission script
# If launcher runs under SLURM, actual time comes from SLURM_JOB_END_TIME or SLURM_JOB_TIMELIMIT env vars
walltime=03:00:00

# Individual VASP job walltime (HH:MM:SS format)
vasp_walltime=00:30:00

# Seconds between squeue polls during the launcher loop
poll_interval=30

# Max OOM retry escalation level (0-6, see adaptive_healing levels)
max_retry_level=6

[hpc]
# Node architecture — used to generate valid NCORE/KPAR retry levels
cores_per_node=64
gpus_per_node=4
max_nodes=16

# VASP execution command template ({{ntasks}} is replaced at runtime)
vasp_command=mpirun -np {{ntasks}} vasp_std
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
