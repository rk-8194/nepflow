"""Project initialization stage."""

from dataclasses import dataclass
import logging
import os
from typing import Callable, Iterable

from ase.data import chemical_symbols

from ..base import Stage

logger = logging.getLogger(__name__)


KNOWN_ELEMENT_SYMBOLS = {symbol for symbol in chemical_symbols if symbol}
ALLOWED_CRYSTAL_STRUCTURES = {
    "bcc",
    "fcc",
    "hcp",
    "diamond",
    "simple_cubic",
}


def _identity(value: str) -> str:
    """Return *value* unchanged."""
    return value


def _normalize_element_list(raw: str, *, allow_blank: bool) -> str:
    """Normalize and validate a comma-separated list of chemical symbols."""
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        if allow_blank:
            return ""
        raise ValueError("At least one element is required")

    normalized: list[str] = []
    for item in items:
        symbol = item.capitalize()
        if symbol not in KNOWN_ELEMENT_SYMBOLS:
            raise ValueError(f"Unknown element: {item}")
        normalized.append(symbol)
    return ",".join(normalized)


def _normalize_required_elements(raw: str) -> str:
    return _normalize_element_list(raw, allow_blank=False)


def _normalize_optional_elements(raw: str) -> str:
    return _normalize_element_list(raw, allow_blank=True)


def _normalize_crystal_structures(raw: str) -> str:
    """Normalize and validate a comma-separated list of structure names."""
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one crystal structure is required")

    normalized: list[str] = []
    for item in items:
        if item not in ALLOWED_CRYSTAL_STRUCTURES:
            raise ValueError(f"Unknown crystal structure: {item}")
        normalized.append(item)
    return ",".join(normalized)


def _normalize_target_n_atoms(raw: str) -> str:
    """Normalize target_n_atoms, defaulting to 128 when left blank."""
    value = raw.strip()
    if not value:
        return "128"
    try:
        target = int(value)
    except ValueError as exc:
        raise ValueError("target_n_atoms must be a positive integer") from exc
    if target <= 0:
        raise ValueError("target_n_atoms must be a positive integer")
    return str(target)


@dataclass(frozen=True)
class ConfigPrompt:
    """Definition of one interactive initialization prompt."""

    key: str
    label: str
    message: str
    default: str = ""
    env_var: str | None = None
    normalize: Callable[[str], str] = _identity
    required: bool = False


CONFIG_PROMPTS: tuple[ConfigPrompt, ...] = (
    ConfigPrompt(
        key="materialsproject_api_key",
        label="Materials Project API key",
        message="Enter your Materials Project API key",
        default="",
        env_var="MP_API_KEY",
    ),
    ConfigPrompt(
        key="elements",
        label="Elements",
        message="Enter one or more chemical symbols, comma-separated",
        normalize=_normalize_required_elements,
        required=True,
    ),
    ConfigPrompt(
        key="gasElements",
        label="Gas elements",
        message="Enter gas-phase elements, comma-separated",
        normalize=_normalize_optional_elements,
    ),
    ConfigPrompt(
        key="crystal_structures",
        label="Crystal structures",
        message="Enter crystal structures, comma-separated",
        default="bcc,fcc,hcp",
        normalize=_normalize_crystal_structures,
        required=True,
    ),
    ConfigPrompt(
        key="target_n_atoms",
        label="Target atoms",
        message="Enter the target number of atoms per supercell",
        default="128",
        normalize=_normalize_target_n_atoms,
    ),
    ConfigPrompt(
        key="scp_address",
        label="Remote NEPFlow directory",
        message="Enter the remote directory where nepflow.py is located (e.g. user@host:/path/to/nepflow)",
    ),
)


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
            self._print_init_header()
            prompt_values = self._collect_prompt_values(CONFIG_PROMPTS)
            default_config = self._render_default_config(prompt_values)
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

    def _collect_prompt_values(self, prompts: Iterable[ConfigPrompt]) -> dict[str, str]:
        """
        Collect interactive config values.

        The prompt registry keeps the initialization flow extensible:
        add a new ConfigPrompt to CONFIG_PROMPTS and the value will
        be collected and injected into the generated config template.
        """
        values: dict[str, str] = {}
        for index, prompt in enumerate(prompts, start=1):
            env_value = os.environ.get(prompt.env_var) if prompt.env_var else None
            if env_value is not None and env_value != "":
                values[prompt.key] = prompt.normalize(env_value)
                logger.debug("Using %s from environment variable %s", prompt.key, prompt.env_var)
                continue

            prompt_text = self._format_prompt_text(prompt, index)

            response = input(prompt_text)

            if response == "":
                response = prompt.default

            values[prompt.key] = prompt.normalize(response)
            logger.debug("Collected value for %s", prompt.key)

        return values

    def _print_init_header(self) -> None:
        """Print a friendly header for interactive project setup."""
        print()
        print("=" * 72)
        print(f"NEPFlow setup for project: {self.project_name}")
        print("We'll ask for a few initial config values to build the project file.")
        print("Press Enter to accept any shown default.")
        print("=" * 72)
        print()
        print("1. Materials Project API key")
        print("2. Elements to include")
        print("3. Gas elements to include")
        print("4. Crystal structures")
        print("5. Target number of atoms per supercell")
        print("6. Remote NEPFlow directory")
        print()

    @staticmethod
    def _format_prompt_text(prompt: ConfigPrompt, index: int) -> str:
        """Format a consistent one-line prompt."""
        qualifiers: list[str] = []
        if prompt.required:
            qualifiers.append("required")
        if prompt.default:
            qualifiers.append(f"default: {prompt.default}")
        if qualifiers:
            return f"{index}. {prompt.label} ({', '.join(qualifiers)}): "
        return f"{index}. {prompt.label}: "

    def _render_default_config(self, prompt_values: dict[str, str]) -> str:
        """Render the default project config using collected prompt values."""
        return """# Project Configuration File
# Project: {project_name}

[project]
name={project_name}
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
api_key={materialsproject_api_key}

[composition]
# Elements to include (comma-separated)
elements={elements}
# Gas elements to include (comma-separated, optional)
gasElements={gasElements}

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
crystal_structures={crystal_structures}

# Target number of atoms per supercell for DFT calculations
target_n_atoms={target_n_atoms}

# Parallel workers for perturbation generation (0 = auto-detect CPU count, 1 = serial)
n_workers=0

# --- Configurational generators (enable/disable) ---
use_materials_project=true
use_random_solid_solution=true
use_sqs=true
use_segregated=true
use_liquid=false

# Number of configurations per generator per composition
n_random_solid_solution=3
n_sqs=1
n_segregated=3

# Liquid perturbation (applied during the perturbation stage, not as a seed generator)
n_liquid_configurations=2
n_liquid_snapshots=5

# Liquid perturbation (ASE Langevin MD with Lennard-Jones)
liquid_temperature=3000
liquid_timestep_fs=1.0
liquid_equilibration_steps=200
liquid_steps_between_snapshots=100
liquid_friction=0.02

# --- Volume profile ---
# Isotropic volume scaling for E-V curves (applied to unperturbed supercells)
volume_scale_min=0.8
volume_scale_max=1.2
n_volume_points=11

# --- Elastic stress sets ---
# Deterministic normal, coupled-normal, and shear strain series for elastic constants
elastic_stress_enabled=true
elastic_strain_amplitudes=-0.02,-0.01,-0.005,0.005,0.01,0.02

# --- Perturbation counts (per base structure at equilibrium) ---
n_rattled=10
n_vacancies=10
n_interstitials=10

# --- Perturbation parameters ---
# Rattling (thermal disorder via hiphive MC)
rattle_std=0.03
rattle_std_min=0.015
rattle_std_max=0.06
rattle_d_min=1.5

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

# Include seed structures from structures/seeds/base_structures.xyz as
# fixed training anchors before FPS fills the remaining target_train_count.
include_seed_structures=false

# Include generated elastic stress structures for unary/single-element seeds
# as fixed training anchors before FPS. Useful for elemental elastic benchmarks.
include_single_element_elastic_stress_structures=false

# Include all generated elastic stress structures as fixed training anchors before FPS.
# This can be expensive for large alloy datasets.
include_elastic_stress_structures=false

# Balance descriptor-space novelty with sparse binary/ternary composition coverage
# when filling the non-anchor portion of the training set.
composition_aware_fps=false
composition_aware_fps_frontier_fraction=0.10
composition_aware_fps_ternary_weight=1.0
composition_aware_fps_adaptive_retries=4
composition_aware_fps_descriptor_floor_fraction=0.95

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

[train_nep]
# NEP training parameters (all optional with sensible defaults)
# Training hyperparameters
population=50
batch=3000
generation=250000

# Charge mode for NEP training
# 0 = NEP
# 1 = qNEP
charge_mode=0

# Element type configuration
# weights: relative weight for each element (comma-separated, optional)
# Example: weights=1,1,1,1 (for W,Cr,Y,Zr)
weights=

# ZBL cutoff distance (outer radius)
outerZBL=2.0

# Loss function weights
lambda_e=1.0
lambda_f=1.0
lambda_v=1.0
lambda_shear=1.0

# Include virial tensor in training data (requires VASP STRESS calculation)
train_virial=false

# Maximum number of resubmission attempts if training job fails
max_resubmit=3

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
vasp_walltime=01:00:00

# Seconds between squeue polls during the launcher loop
poll_interval=20

# Max OOM retry escalation level (0-6, see adaptive_healing levels)
max_retry_level=100

[hpc]
# Node architecture — used to generate valid NCORE/KPAR retry levels
cores_per_node=64
gpus_per_node=4
max_nodes=16

# Remote directory where nepflow.py is located (e.g. user@host:/path/to/nepflow)
scp_address={scp_address}

# VASP execution command template ({{ntasks}} is replaced at runtime)
vasp_command=mpirun -np {{ntasks}} vasp_std

# NEP training command or executable path
nep_command=mpirun --bind-to none $HOME/src/GPUMD/src/nep
""".format(project_name=self.project_name, **prompt_values)
