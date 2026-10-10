"""Immutable typed models for the project configuration schema."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from nepflow.domain.magnetism import MagneticMomentSet

CONFIG_SCHEMA_VERSION = 1

ALL_SOURCES = "all"
SUPPORTED_CONFIGURATIONAL_SOURCES = frozenset(
    {
        "mp_phase",
        "mp_gas_phase",
        "random_solid_solution",
        "sqs",
        "segregated",
    }
)
SourceScope = tuple[str, ...]
DEFAULT_SOURCE_SCOPE: SourceScope = (ALL_SOURCES,)
SUPPORTED_SURFACE_MILLER_INDICES = frozenset(
    {
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
    }
)
MAGNETIC_DEFECT_FAMILIES = (
    "vacancy",
    "interstitial",
    "substitution",
    "antisite",
    "vacancy_interstitial",
    "gas_in_vacancy",
)
GENERATION_SOURCE_SCOPE_FIELDS = (
    "volume_sources",
    "elastic_sources",
    "rattle_sources",
    "liquid_sources",
    "vacancy_sources",
    "interstitial_sources",
    "gas_interstitial_sources",
    "substitution_sources",
    "antisite_sources",
    "vacancy_interstitial_sources",
    "gas_in_vacancy_sources",
    "surface_sources",
    "grain_boundary_sources",
)


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Project identity and reproducibility settings."""

    name: str = ""
    description: str = ""
    status: str = "initialized"
    random_seed: int = 42
    config_version: int = CONFIG_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Project-relative output directories."""

    structures_path: Path = Path("structures")
    vasp_path: Path = Path("vasp")
    nep_path: Path = Path("nep")
    gpumd_path: Path = Path("gpumd")
    reports_path: Path = Path("reports")


@dataclass(frozen=True, slots=True)
class MaterialsProjectConfig:
    """Materials Project settings; environment credentials are not loaded here."""

    api_key: str | None = None


@dataclass(frozen=True, slots=True)
class CompositionConfig:
    """Chemical composition and phase-space controls."""

    elements: tuple[str, ...] = ()
    gas_elements: tuple[str, ...] = ()
    composition_step: float = 0.125
    include_pure_elements: bool = True
    include_binaries: bool = True
    include_ternaries: bool = True


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Structure-generation switches, counts, and scientific parameters."""

    crystal_structures: tuple[str, ...] = ("bcc", "fcc", "hcp")
    target_n_atoms: int = 128
    composition_tolerance: float = 0.05
    n_workers: int = 0
    use_materials_project: bool = True
    use_random_solid_solution: bool = True
    use_sqs: bool = True
    use_segregated: bool = True
    use_liquid: bool = False
    n_random_solid_solution: int = 3
    n_sqs: int = 1
    n_segregated: int = 3
    n_liquid_configurations: int = 2
    n_liquid_snapshots: int = 5
    liquid_temperature: float = 3000.0
    liquid_timestep_fs: float = 1.0
    liquid_equilibration_steps: int = 200
    liquid_steps_between_snapshots: int = 100
    liquid_friction: float = 0.02
    volume_scale_min: float = 0.8
    volume_scale_max: float = 1.2
    n_volume_points: int = 11
    volume_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    elastic_stress_enabled: bool = True
    elastic_strain_amplitudes: tuple[float, ...] = (
        -0.02,
        -0.01,
        -0.005,
        0.005,
        0.01,
        0.02,
    )
    elastic_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    n_rattled: int = 10
    rattle_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    n_vacancies: int = 10
    vacancy_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    n_interstitials: int = 10
    interstitial_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    n_gas_interstitials: int = 10
    gas_interstitial_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    n_substitutions: int = 0
    substitution_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    n_antisites: int = 0
    antisite_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    n_vacancy_interstitial: int = 10
    vacancy_interstitial_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    n_gas_in_vacancy: int = 10
    gas_in_vacancy_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    surface_enabled: bool = False
    # Deprecated compatibility field; surface multiplicity is orientation and
    # termination driven when the surface feature is enabled.
    n_surfaces: int = 0
    surface_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    surface_miller_indices: tuple[tuple[int, int, int], ...] = ((1, 0, 0),)
    surface_layers: int = 3
    surface_thickness: float | None = None
    # Minimum physical vacuum thickness in Angstrom; never an hkl-plane count.
    surface_vacuum: float = 10.0
    surface_min_half_depth: float = 6.0
    surface_bulk_environment_radius: float = 5.0
    surface_min_bulk_core_atoms: int = 1
    surface_bulk_environment_distance_tolerance: float = 0.05
    surface_termination_policy: str = "all"
    surface_max_terminations: int = 0
    # Surface planning controls.  The global target remains the default when
    # ``surface_target_n_atoms`` is unset.
    surface_target_n_atoms: int | None = None
    surface_target_tolerance: float = 0.20
    surface_max_n_atoms: int = 512
    surface_in_plane_repeat: tuple[int, int] = (1, 1)
    surface_max_in_plane_repeat: tuple[int, int] = (4, 4)
    surface_max_normal_repeat: int = 16
    surface_min_in_plane_dimensions: tuple[float, float] = (0.0, 0.0)
    # Require the backend to return a genuinely symmetric slab when enabled.
    surface_symmetric: bool = False
    surface_stoichiometry_policy: str = "allow"
    surface_polarity_policy: str = "allow"
    grain_boundary_enabled: bool = False
    n_grain_boundaries: int = 0
    grain_boundary_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    grain_boundary_rotation_axis: tuple[int, int, int] = (0, 0, 1)
    grain_boundary_misorientation_angle: float = 36.86989764584402
    grain_boundary_sigma: int = 5
    grain_boundary_plane: tuple[int, int, int] = (2, 1, 0)
    grain_boundary_expand_times: int = 2
    grain_boundary_min_thickness: float = 0.0
    grain_boundary_overlap_tolerance: float = 0.7
    liquid_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    rattle_std: float = 0.03
    rattle_std_min: float = 0.015
    rattle_std_max: float = 0.06
    rattle_d_min: float = 1.5
    vacancy_min: float = 0.008
    vacancy_max: float = 0.025
    vacancy_species: tuple[str, ...] = ()
    interstitial_d_min: float = 1.65
    interstitial_min: float = 0.008
    interstitial_max: float = 0.025
    interstitial_sites: tuple[Any, ...] = ()
    crystallographic_interstitial_sites: tuple[Any, ...] = ()
    defect_defect_d_min: float = 1.65
    periodic_image_d_min: float = 6.0
    interstitial_max_attempts: int = 1000
    substitution_pairs: tuple[tuple[str, str], ...] = ()
    substitution_min: float = 0.008
    substitution_max: float = 0.025
    antisite_pairs: tuple[tuple[str, str], ...] = ()
    antisite_min: float = 0.008
    antisite_max: float = 0.025
    gas_interstitial_d_min: float = 1.2
    max_gas_occupancy: int = 3


@dataclass(frozen=True, slots=True)
class MagnetismConfig:
    """Cross-stage magnetic candidate and downstream capability settings."""

    enabled: bool = False
    target_potential_magnetic: bool = False
    include_non_magnetic: bool = True
    include_ferromagnetic: bool = False
    include_antiferromagnetic: bool = False
    moment_sets: tuple[MagneticMomentSet, ...] = ()
    symmetry_tolerance: float = 1.0e-3
    phase_tolerance: float = 1.0e-8
    max_afm_orderings: int = 16
    unmapped_site_policy: str = "skip_afm"
    magnetic_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    defect_families: tuple[str, ...] = ()
    max_defect_parents: int = 0
    max_magnetic_variants_per_parent: int = 16
    max_magnetic_variants_per_defect: int = 16

    def __post_init__(self) -> None:
        raw_sets = self.moment_sets
        if isinstance(raw_sets, Mapping):
            raw_sets = tuple(
                MagneticMomentSet.from_mapping(str(name), values)
                for name, values in raw_sets.items()
            )
        else:
            raw_sets = tuple(
                item
                if isinstance(item, MagneticMomentSet)
                else MagneticMomentSet.from_mapping(str(item["name"]), item["moments"])
                for item in raw_sets
            )
        object.__setattr__(self, "moment_sets", tuple(raw_sets))
        object.__setattr__(
            self,
            "magnetic_sources",
            tuple(str(source).strip().lower() for source in self.magnetic_sources),
        )
        object.__setattr__(
            self,
            "defect_families",
            tuple(str(family).strip().lower() for family in self.defect_families),
        )
        object.__setattr__(
            self,
            "unmapped_site_policy",
            str(self.unmapped_site_policy).strip().lower(),
        )

    @property
    def target_potential_supports_magnetism(self) -> bool:
        """Readable alias for the target-potential capability guard."""

        return self.target_potential_magnetic

    @property
    def target_mlip_magnetic(self) -> bool:
        """Compatibility alias for callers using MLIP terminology."""

        return self.target_potential_magnetic

    @property
    def target_mlip_supports_magnetism(self) -> bool:
        """Compatibility alias for the documented target capability name."""

        return self.target_potential_magnetic

    @property
    def include_nonmagnetic(self) -> bool:
        """Compatibility alias for the documented non-magnetic switch."""

        return self.include_non_magnetic

    @property
    def include_fm(self) -> bool:
        """Compatibility alias for the short FM switch."""

        return self.include_ferromagnetic

    @property
    def include_afm(self) -> bool:
        """Compatibility alias for the short AFM switch."""

        return self.include_antiferromagnetic

    @property
    def max_afm_orderings_per_parent(self) -> int:
        """Compatibility alias for the AFM parent budget."""

        return self.max_afm_orderings

    @property
    def unmapped_afm_policy(self) -> str:
        """Compatibility alias for the documented unmapped-site policy."""

        return self.unmapped_site_policy

    @property
    def source_scope(self) -> SourceScope:
        """Readable alias for the magnetic source scope."""

        return self.magnetic_sources


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Sparse selection and descriptor controls."""

    algorithm: str = "information_entropy"
    nep_model_file: str = "nep89.txt"
    include_seed_structures: bool = False
    include_single_element_elastic_stress_structures: bool = False
    include_elastic_stress_structures: bool = False
    composition_aware_fps: bool = False
    composition_aware_fps_frontier_fraction: float = 0.1
    composition_aware_fps_ternary_weight: float = 1.0
    composition_aware_fps_adaptive_retries: int = 4
    composition_aware_fps_descriptor_floor_fraction: float = 0.95
    target_train_count: int = 1000
    target_test_count: int = 200
    target_tolerance: int = 50
    descriptor_type: str = "structure"
    batch_size: int = 500
    max_search_iterations: int = 30
    test_pool_factor: float = 0.5
    local_magnetic_mode: str = "structural"
    local_descriptor_workers: int = 0
    background_mass: float = 1.0e-12


@dataclass(frozen=True, slots=True)
class VaspConfig:
    """VASP scientific input settings used by shared preparation helpers."""

    enabled: bool = True
    kspacing: float = 0.30
    kgamma: bool = True


@dataclass(frozen=True, slots=True)
class DftRecoveryConfig:
    """VASP recovery and per-job resource policy."""

    max_retry_level: int = 100
    vasp_walltime: str = "01:00:00"


@dataclass(frozen=True, slots=True)
class NepConfig:
    """NEP backend enablement settings."""

    enabled: bool = True


@dataclass(frozen=True, slots=True)
class NepTrainingConfig:
    """NEP model and training parameters."""

    population: int = 50
    batch: int = 3000
    generations: int = 250000
    charge_mode: int = 0
    weights: tuple[float, ...] = ()
    outer_zbl: float = 2.0
    cutoff: tuple[str, ...] = ("6", "5")
    n_max: tuple[str, ...] = ("4", "4")
    basis_size: tuple[str, ...] = ("8", "8")
    l_max: tuple[str, ...] = ("4", "2", "1")
    neuron: tuple[str, ...] = ("80",)
    lambda_e: float = 1.0
    lambda_f: float = 1.0
    lambda_v: float = 1.0
    lambda_shear: float = 1.0
    train_virial: bool = False
    allow_partial_dataset: bool = False
    max_resubmit: int = 3
    # A deterministic, typed candidate sweep.  Each item is a field name and
    # its ordered candidate values; the optimisation layer validates that the
    # names are scientific NEP fields before constructing candidates.
    sweep: tuple[tuple[str, tuple[Any, ...]], ...] = ()

    def __post_init__(self) -> None:
        source = self.sweep.items() if isinstance(self.sweep, Mapping) else self.sweep
        normalized = tuple(
            (str(name), tuple(values))
            for name, values in sorted(source, key=lambda item: str(item[0]))
        )
        if len({name for name, _values in normalized}) != len(normalized):
            raise ValueError("NEP training sweep fields must be unique")
        object.__setattr__(self, "sweep", normalized)

    def sweep_mapping(self) -> dict[str, tuple[Any, ...]]:
        """Return the configured sweep in the mapping form used by ``ControlledSweep``."""

        return {str(name): tuple(values) for name, values in self.sweep}


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """GPUMD validation settings and explicit model identity."""

    enabled: bool = True
    model_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class HpcConfig:
    """Site resources, commands, and optional seed-upload destination."""

    cores_per_node: int = 64
    gpus_per_node: int = 4
    max_nodes: int = 16
    scp_address: str = ""
    vasp_command: str = "mpirun -np {ntasks} vasp_std"
    nep_command: str = "mpirun --bind-to none $HOME/src/GPUMD/src/nep"
    gpumd_command: str = "mpirun -np 1 --bind-to none $HOME/src/GPUMD/src/gpumd"


@dataclass(frozen=True, slots=True)
class SlurmConfig:
    """Shared scheduler settings for all stage launchers."""

    enabled: bool = False
    max_concurrent: int = 20
    walltime: str = "03:00:00"
    train_nep_walltime: str | None = None
    poll_interval: int = 20
    gpumd_walltime: str = "00:10:00"
    gpumd_nodes: int = 1
    gpumd_gpus: int = 1
    memory_poll_interval: int = 10
    memory_walltime: str = "00:10:00"


@dataclass(frozen=True, slots=True)
class NepflowConfig:
    """Canonical immutable root configuration returned by the loader."""

    schema_version: int = 1
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    materials_project: MaterialsProjectConfig = field(default_factory=MaterialsProjectConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    vasp: VaspConfig = field(default_factory=VaspConfig)
    dft_recovery: DftRecoveryConfig = field(default_factory=DftRecoveryConfig)
    nep: NepConfig = field(default_factory=NepConfig)
    train_nep: NepTrainingConfig = field(default_factory=NepTrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    hpc: HpcConfig = field(default_factory=HpcConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    source_path: Path | None = field(default=None, compare=False, repr=False)
    raw_sections: frozenset[str] = field(
        default_factory=frozenset,
        compare=False,
        repr=False,
    )
    magnetism: MagnetismConfig = field(default_factory=MagnetismConfig)

    def effective_mapping(self, *, redact_secrets: bool = True) -> dict[str, object]:
        """Return a deterministic, serializable view of the effective config."""
        mapping = _normalize_mapping(self)
        if not isinstance(mapping, dict):
            raise TypeError("effective configuration must normalize to a mapping")
        mapping.pop("source_path", None)
        mapping.pop("raw_sections", None)
        if redact_secrets:
            materials_project = mapping["materials_project"]
            if isinstance(materials_project, dict):
                materials_project["api_key"] = None
        normalized = _normalize_mapping(mapping)
        if not isinstance(normalized, dict):
            raise TypeError("effective configuration must normalize to a mapping")
        return {str(key): value for key, value in normalized.items()}


def _normalize_mapping(value: object) -> object:
    """Normalize dataclass values for deterministic effective-config output."""
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _normalize_mapping(to_dict())
    if is_dataclass(value):
        return {item.name: _normalize_mapping(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_mapping(item) for item in value]
    if isinstance(value, list):
        return [_normalize_mapping(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize_mapping(value[key]) for key in sorted(value)}
    return value


RootConfig = NepflowConfig
