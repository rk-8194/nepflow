"""Config rendering and validated installation for new projects."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Mapping

from nepflow.errors import ConfigurationError
from nepflow.io.atomic import atomic_write_text

from .loader import load_config
from .models import NepflowConfig

logger = logging.getLogger(__name__)


def validate_project_identity(config: NepflowConfig, project_name: str) -> None:
    """Ensure a typed config belongs to the requested project."""
    if config.project.name != project_name:
        raise ConfigurationError(
            "Project config name does not match the requested project: "
            f"{config.project.name!r} != {project_name!r}"
        )


def render_default_config(project_name: str, prompt_values: Mapping[str, str]) -> str:
    """Render the canonical default config used by project initialization."""
    render_values = dict(prompt_values)
    render_values["gas_elements"] = prompt_values.get(
        "gas_elements", prompt_values.get("gasElements", "")
    )
    # Credentials are runtime-only inputs.  Even if an older caller supplies
    # a prompted value, never serialize it into the project configuration.
    render_values["materialsproject_api_key"] = ""
    return """# Project Configuration File
# Project: {project_name}

[project]
name={project_name}
description=NEPFlow project for atomic structure generation and validation
status=initialized
schema_version=1
config_version=1
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
# Set at runtime via MP_API_KEY; credentials are never persisted here
api_key={materialsproject_api_key}

[composition]
# Elements to include (comma-separated)
elements={elements}
# Gas elements to include (comma-separated, optional)
gas_elements={gas_elements}

# Composition step size (atomic fraction)
# Controls granularity of the simplex grid
# step=0.1 â†’ 11 points per binary edge, 66 total for 3 elements
# step=0.05 â†’ 21 points per binary edge, ~250 total for 3 elements
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

# Maximum allowed absolute atomic-fraction composition error during planning
composition_tolerance=0.05

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

# Liquid-like geometry disorder (ASE Langevin + Lennard-Jones; not a
# material-specific or physically validated liquid trajectory)
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
# Explicit source scope; "all" preserves the schema-v1 all-base behaviour.
volume_sources=all

# --- Elastic stress sets ---
# Deterministic normal, coupled-normal, and shear strain series for elastic constants
elastic_stress_enabled=true
elastic_strain_amplitudes=-0.02,-0.01,-0.005,0.005,0.01,0.02
elastic_sources=all

# --- Perturbation counts (per base structure at equilibrium) ---
n_rattled=10
rattle_sources=all
n_vacancies=10
vacancy_sources=all
n_interstitials=10
interstitial_sources=all
n_gas_interstitials=10
gas_interstitial_sources=all
n_substitutions=0
substitution_sources=all
n_antisites=0
antisite_sources=all
n_vacancy_interstitial=10
vacancy_interstitial_sources=all
n_gas_in_vacancy=10
gas_in_vacancy_sources=all
surface_enabled=false
# Deprecated compatibility field; surface multiplicity comes from the
# requested Miller orientations and termination settings.
n_surfaces=0
surface_sources=all

# --- Surface/slab perturbations ---
# Ordered Miller indices use h,k,l;h,k,l notation.  A blank thickness uses
# surface_layers; otherwise thickness is the minimum slab-size target in Å.
surface_miller_indices=1,0,0
surface_layers=3
surface_thickness=
surface_vacuum=10.0
surface_min_half_depth=6.0
surface_bulk_environment_radius=5.0
surface_min_bulk_core_atoms=1
surface_bulk_environment_distance_tolerance=0.05
surface_termination_policy=all
surface_max_terminations=0
# Surface planning target defaults to generation.target_n_atoms when blank.
surface_target_n_atoms=
surface_target_tolerance=0.20
surface_max_n_atoms=512
# Existing repeat is an expert minimum; the planner searches up to these bounds.
surface_in_plane_repeat=1,1
surface_max_in_plane_repeat=4,4
surface_max_normal_repeat=16
surface_min_in_plane_dimensions=0.0,0.0
surface_symmetric=false

# --- Grain-boundary perturbations ---
# Supported initial relationship: Sigma 5 [001] symmetric tilt, (210), 36.8699°.
grain_boundary_enabled=false
n_grain_boundaries=0
grain_boundary_sources=all
grain_boundary_rotation_axis=0,0,1
grain_boundary_misorientation_angle=36.86989764584402
grain_boundary_sigma=5
grain_boundary_plane=2,1,0
grain_boundary_expand_times=2
grain_boundary_min_thickness=0.0
# Pymatgen overlap ratio relative to the bulk nearest-neighbour distance.
grain_boundary_overlap_tolerance=0.7
liquid_sources=all

# --- Perturbation parameters ---
# Rattling (thermal disorder via hiphive MC)
rattle_std=0.03
rattle_std_min=0.015
rattle_std_max=0.06
rattle_d_min=1.5

# Vacancies (fraction of atoms to remove; about 1-3 defects in a 128-atom cell)
# The fraction is applied to the actual generated supercell atom count, so
# realised integer counts may differ when the cell is not exactly 128 atoms.
vacancy_min=0.008
vacancy_max=0.025
vacancy_species=

# Interstitials (about 1-3 defects in a 128-atom cell; see vacancy note above)
interstitial_d_min=1.65
interstitial_min=0.008
interstitial_max=0.025
interstitial_sites=
crystallographic_interstitial_sites=
# Minimum separation between inserted defect sites (Angstrom).
defect_defect_d_min=1.65
# Minimum shortest periodic cell translation checked for defect placement (Angstrom);
# this is not a full relaxed defect-image interaction-distance calculation.
periodic_image_d_min=6.0
# Stochastic placement attempts per requested interstitial site.
interstitial_max_attempts=1000

# Substitution and antisite defects. Pairs use source->target notation.
substitution_pairs=
substitution_min=0.008
substitution_max=0.025
antisite_pairs=
antisite_min=0.008
antisite_max=0.025

[magnetism]
# Magnetic expansion is opt-in. Set target_potential_magnetic=true before
# enabling FM/AFM; enabled=true alone does not imply FM or AFM generation.
enabled=false
target_potential_magnetic=false
include_non_magnetic=true
include_ferromagnetic=false
include_antiferromagnetic=false
# Leave moment_sets blank unless scientifically justified project values are
# available. Example only - not a default or recommendation for arbitrary chemistry:
# moment_sets={{"low":{{"Fe":2.5,"Cr":1.5}},"nominal":{{"Fe":3.5,"Cr":2.5}},"high":{{"Fe":4.5,"Cr":3.5}}}}
moment_sets=
# Topology/symmetry matching tolerance.
symmetry_tolerance=0.001
# Numerical tolerance for AFM phase reality and commensurability checks.
phase_tolerance=1e-8
# AFM enumeration budget retained per structural parent; this is AFM-only.
max_afm_orderings=16
# Omit AFM for unsupported topology rather than guessing magnetic signs.
unmapped_site_policy=skip_afm
# "all" makes every supported configurational source eligible.
magnetic_sources=all
# Blank means pristine/unperturbed structures only, not all defects.
defect_families=
# Zero means unlimited eligible defect parents, but only for explicitly enabled
# families listed in defect_families.
max_defect_parents=0
# Final total magnetic-variant budgets per structural parent and defect parent.
# max_afm_orderings above limits AFM enumeration specifically.
max_magnetic_variants_per_parent=16
max_magnetic_variants_per_defect=16

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
# Binary search adjusts min_distance to hit these counts (Â±target_tolerance)
target_train_count=1000
target_test_count=200
target_tolerance=50

# Descriptor aggregation type (the current selector consumes one vector per structure)
# structure: mean of per-atom descriptors â†’ one vector per structure (recommended)
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

# Permit an explicitly partial dataset; false fails creation when any selected
# structure is rejected during DFT extraction.
allow_partial_dataset=false

# Maximum number of resubmission attempts if training job fails
max_resubmit=3

# Optional deterministic Cartesian sweep over scientific NEP fields.
# Separate ordered candidates with |, e.g. lambda_f=1.0|2.0 or
# cutoff=6 5|7 5.  Runtime-only scheduler/resource settings are not allowed.
[training_sweep]
# lambda_f=1.0|2.0
# cutoff=6 5|7 5

[gpumd]
enabled=true
# Explicit model_run_id to validate; generated after NEP training completes.
model_run_id=

[slurm]
enabled=false

# Maximum concurrent VASP jobs for this project (squeue-filtered by project name)
max_concurrent=20

# Launcher walltime fallback (HH:MM:SS) â€” normally obtained from SLURM submission script
# If launcher runs under SLURM, actual time comes from SLURM_JOB_END_TIME or SLURM_JOB_TIMELIMIT env vars
walltime=03:00:00

# Individual VASP job walltime (HH:MM:SS format)
vasp_walltime=01:00:00

# Seconds between squeue polls during the launcher loop
poll_interval=20

# Max OOM retry escalation level (0-6, see adaptive_healing levels)
max_retry_level=100

[hpc]
# Node architecture â€” used to generate valid NCORE/KPAR retry levels
cores_per_node=64
gpus_per_node=4
max_nodes=16

# Remote directory where the installed nepflow command is located (e.g. user@host:/path/to/nepflow)
scp_address={scp_address}

# VASP execution command template ({{ntasks}} is replaced at runtime)
vasp_command=mpirun -np {{ntasks}} vasp_std

# NEP training command or executable path
nep_command=mpirun --bind-to none $HOME/src/GPUMD/src/nep

# GPUMD validation command or executable path
gpumd_command=mpirun -np 1 --bind-to none $HOME/src/GPUMD/src/gpumd
""".format(project_name=project_name, **render_values)


def write_validated_config(
    config_path: Path,
    rendered: str,
    *,
    project_name: str,
) -> NepflowConfig:
    """Validate rendered text before atomically installing the config."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    file_descriptor: int | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            dir=config_path.parent,
        )
        os.close(file_descriptor)
        file_descriptor = None
        temporary_path = Path(temporary_name)
        atomic_write_text(temporary_path, rendered, encoding="utf-8")
        config = load_config(temporary_path, project_name=project_name)
        validate_project_identity(config, project_name)
        atomic_write_text(config_path, rendered, encoding="utf-8")
        config = load_config(config_path, project_name=project_name)
        validate_project_identity(config, project_name)
        return config
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temporary config: %s", temporary_path)


__all__ = [
    "render_default_config",
    "validate_project_identity",
    "write_validated_config",
]
