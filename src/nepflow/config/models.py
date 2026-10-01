"""Immutable typed models for the project configuration schema."""

from dataclasses import asdict, dataclass, field
from pathlib import Path


CONFIG_SCHEMA_VERSION = 1


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
    elastic_stress_enabled: bool = True
    elastic_strain_amplitudes: tuple[float, ...] = (
        -0.02,
        -0.01,
        -0.005,
        0.005,
        0.01,
        0.02,
    )
    n_rattled: int = 10
    n_vacancies: int = 10
    n_interstitials: int = 10
    n_gas_interstitials: int = 10
    n_vacancy_interstitial: int = 10
    n_gas_in_vacancy: int = 10
    rattle_std: float = 0.03
    rattle_std_min: float = 0.015
    rattle_std_max: float = 0.06
    rattle_d_min: float = 1.5
    vacancy_min: float = 0.0
    vacancy_max: float = 0.1
    interstitial_d_min: float = 1.65
    interstitial_min: float = 0.05
    interstitial_max: float = 0.1
    volume_scale_minimum: float | None = None
    interstitial_d_max: float | None = None
    gas_interstitial_d_min: float = 1.2
    max_gas_occupancy: int = 3


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Sparse selection and descriptor controls."""

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

    def effective_mapping(self, *, redact_secrets: bool = True) -> dict[str, object]:
        """Return a deterministic, serializable view of the effective config."""
        mapping = asdict(self)
        mapping.pop("source_path", None)
        if redact_secrets:
            materials_project = mapping["materials_project"]
            if isinstance(materials_project, dict):
                materials_project["api_key"] = None
        return _normalize_mapping(mapping)


def _normalize_mapping(value: object) -> object:
    """Normalize dataclass values for deterministic effective-config output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_mapping(item) for item in value]
    if isinstance(value, list):
        return [_normalize_mapping(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_mapping(value[key])
            for key in sorted(value)
        }
    return value


RootConfig = NepflowConfig
