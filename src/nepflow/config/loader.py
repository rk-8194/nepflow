"""Read canonical project configs and coordinate section-specific parsing."""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path

from nepflow.errors import ConfigurationError

from .legacy import load_legacy_config, to_legacy_config
from .models import (
    CONFIG_SCHEMA_VERSION,
    MaterialsProjectConfig,
    NepflowConfig,
)
from .section_parsers import (
    parse_composition,
    parse_generation,
    parse_hpc,
    parse_magnetism,
    parse_nep,
    parse_paths,
    parse_project,
    parse_selection,
    parse_slurm,
    parse_training,
    parse_validation,
    parse_vasp,
)
from .validation import validate_config

CANONICAL_CONFIG_NAME = "project.config"

_ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "project": frozenset(
        {"name", "description", "status", "random_seed", "schema_version", "config_version"}
    ),
    "paths": frozenset({"structures_path", "vasp_path", "nep_path", "gpumd_path", "reports_path"}),
    "materialsproject": frozenset({"api_key"}),
    "composition": frozenset(
        {
            "elements",
            "gas_elements",
            "gaselements",
            "composition_step",
            "include_pure_elements",
            "include_binaries",
            "include_ternaries",
        }
    ),
    "generation": frozenset(
        {
            "elements",
            "crystal_structures",
            "target_n_atoms",
            "composition_tolerance",
            "n_workers",
            "use_materials_project",
            "use_random_solid_solution",
            "use_sqs",
            "use_segregated",
            "use_liquid",
            "n_random_solid_solution",
            "n_sqs",
            "n_segregated",
            "n_liquid_configurations",
            "n_liquid_snapshots",
            "liquid_temperature",
            "liquid_timestep_fs",
            "liquid_equilibration_steps",
            "liquid_steps_between_snapshots",
            "liquid_friction",
            "volume_scale_min",
            "volume_scale_max",
            "n_volume_points",
            "volume_sources",
            "elastic_stress_enabled",
            "elastic_strain_amplitudes",
            "elastic_sources",
            "n_rattled",
            "rattle_sources",
            "n_vacancies",
            "vacancy_sources",
            "n_interstitials",
            "interstitial_sources",
            "n_gas_interstitials",
            "gas_interstitial_sources",
            "n_substitutions",
            "substitution_sources",
            "n_antisites",
            "antisite_sources",
            "n_vacancy_interstitial",
            "vacancy_interstitial_sources",
            "n_gas_in_vacancy",
            "gas_in_vacancy_sources",
            "surface_enabled",
            "n_surfaces",
            "surface_sources",
            "surface_miller_indices",
            "surface_layers",
            "surface_thickness",
            "surface_vacuum",
            "surface_min_half_depth",
            "surface_bulk_environment_radius",
            "surface_min_bulk_core_atoms",
            "surface_bulk_environment_distance_tolerance",
            "surface_termination_policy",
            "surface_max_terminations",
            "surface_target_n_atoms",
            "surface_target_tolerance",
            "surface_max_n_atoms",
            "surface_in_plane_repeat",
            "surface_max_in_plane_repeat",
            "surface_max_normal_repeat",
            "surface_min_in_plane_dimensions",
            "surface_symmetric",
            "surface_stoichiometry_policy",
            "surface_polarity_policy",
            "grain_boundary_enabled",
            "n_grain_boundaries",
            "grain_boundary_sources",
            "grain_boundary_rotation_axis",
            "grain_boundary_misorientation_angle",
            "grain_boundary_sigma",
            "grain_boundary_plane",
            "grain_boundary_expand_times",
            "grain_boundary_min_thickness",
            "grain_boundary_overlap_tolerance",
            "liquid_sources",
            "rattle_std",
            "rattle_std_min",
            "rattle_std_max",
            "rattle_d_min",
            "vacancy_min",
            "vacancy_max",
            "vacancy_species",
            "interstitial_d_min",
            "interstitial_min",
            "interstitial_max",
            "interstitial_sites",
            "crystallographic_interstitial_sites",
            "defect_defect_d_min",
            "periodic_image_d_min",
            "interstitial_max_attempts",
            "substitution_pairs",
            "substitution_min",
            "substitution_max",
            "antisite_pairs",
            "antisite_min",
            "antisite_max",
            "gas_interstitial_d_min",
            "max_gas_occupancy",
        }
    ),
    "magnetism": frozenset(
        {
            "enabled",
            "target_potential_magnetic",
            "target_potential_supports_magnetism",
            "target_mlip_magnetic",
            "target_mlip_supports_magnetism",
            "include_non_magnetic",
            "include_nonmagnetic",
            "include_nm",
            "include_ferromagnetic",
            "include_fm",
            "include_antiferromagnetic",
            "include_afm",
            "moment_sets",
            "named_moment_sets",
            "symmetry_tolerance",
            "phase_tolerance",
            "max_afm_orderings",
            "max_afm_orderings_per_parent",
            "max_afm_orderings_per_structure",
            "unmapped_site_policy",
            "unmapped_afm_policy",
            "magnetic_sources",
            "source_scopes",
            "defect_families",
            "max_defect_parents",
            "max_magnetic_variants_per_parent",
            "max_magnetic_variants_per_defect",
        }
    ),
    "selection": frozenset(
        {
            "algorithm",
            "nep_model_file",
            "include_seed_structures",
            "include_single_element_elastic_stress_structures",
            "include_elastic_stress_structures",
            "composition_aware_fps",
            "composition_aware_fps_frontier_fraction",
            "composition_aware_fps_ternary_weight",
            "composition_aware_fps_adaptive_retries",
            "composition_aware_fps_descriptor_floor_fraction",
            "target_train_count",
            "target_test_count",
            "target_tolerance",
            "descriptor_type",
            "batch_size",
            "max_search_iterations",
            "test_pool_factor",
            "local_magnetic_mode",
            "local_descriptor_workers",
            "max_local_descriptor_inflight_bytes",
            "background_mass",
            "entropy_beta",
            "entropy_optimizer_method",
            "entropy_local_cutoff",
            "entropy_local_radial_bins",
            "entropy_local_angular_bins",
            "entropy_local_radial_sigma",
            "entropy_local_angular_sigma",
            "entropy_local_species",
            "entropy_whitening_tolerance",
            "entropy_whitening_regularization",
            "entropy_whitening_singular_policy",
            "entropy_bandwidth_mode",
            "entropy_bandwidth_k",
            "entropy_bandwidth_c",
            "entropy_bandwidth_k_candidates",
            "entropy_bandwidth_c_candidates",
            "entropy_bandwidth_backend",
            "entropy_bandwidth_metric",
            "entropy_bandwidth_chunk_size",
            "entropy_bandwidth_max_neighbour_entries",
            "entropy_bandwidth_max_index_bytes",
            "entropy_bandwidth_max_radius_query_bytes",
            "entropy_bandwidth_max_calibration_work_bytes",
            "entropy_bandwidth_calibration_batch_size",
            "entropy_max_edges",
            "entropy_max_graph_bytes",
            "entropy_max_graph_spool_bytes",
            "entropy_max_entries",
            "entropy_max_contribution_bytes",
            "entropy_max_contribution_spool_bytes",
            "beta",
            "optimizer_method",
            "local_cutoff",
            "local_radial_bins",
            "local_angular_bins",
            "local_radial_sigma",
            "local_angular_sigma",
            "local_species",
            "whitening_tolerance",
            "whitening_regularization",
            "whitening_singular_policy",
            "bandwidth_mode",
            "bandwidth_k",
            "bandwidth_c",
            "k_candidates",
            "c_candidates",
            "neighbour_backend",
            "neighbour_metric",
            "bandwidth_chunk_size",
            "max_neighbour_entries",
            "max_index_bytes",
            "max_radius_query_bytes",
            "max_calibration_work_bytes",
            "calibration_batch_size",
            "max_edges",
            "max_graph_bytes",
            "max_graph_spool_bytes",
            "max_entries",
            "max_contribution_bytes",
            "max_contribution_spool_bytes",
        }
    ),
    "vasp": frozenset({"enabled", "kspacing", "kgamma"}),
    "nep": frozenset({"enabled"}),
    "train_nep": frozenset(
        {
            "population",
            "batch",
            "generation",
            "cutoff",
            "n_max",
            "basis_size",
            "l_max",
            "neuron",
            "outerzbl",
            "charge_mode",
            "weights",
            "lambda_e",
            "lambda_f",
            "lambda_v",
            "lambda_shear",
            "train_virial",
            "allow_partial_dataset",
            "max_resubmit",
            "sweep",
            "sweep_population",
            "sweep_batch",
            "sweep_generations",
            "sweep_charge_mode",
            "sweep_weights",
            "sweep_outer_zbl",
            "sweep_cutoff",
            "sweep_n_max",
            "sweep_basis_size",
            "sweep_l_max",
            "sweep_neuron",
            "sweep_lambda_e",
            "sweep_lambda_f",
            "sweep_lambda_v",
            "sweep_lambda_shear",
        }
    ),
    "training_sweep": frozenset(
        {
            "population",
            "batch",
            "generations",
            "charge_mode",
            "weights",
            "outer_zbl",
            "cutoff",
            "n_max",
            "basis_size",
            "l_max",
            "neuron",
            "lambda_e",
            "lambda_f",
            "lambda_v",
            "lambda_shear",
        }
    ),
    "gpumd": frozenset({"enabled", "model_run_id"}),
    "validate": frozenset({"model_run_id"}),
    "slurm": frozenset(
        {
            "enabled",
            "max_concurrent",
            "walltime",
            "train_nep_walltime",
            "poll_interval",
            "max_retry_level",
            "vasp_walltime",
            "gpumd_walltime",
            "gpumd_nodes",
            "gpumd_gpus",
            "memory_poll_interval",
            "memory_walltime",
            "vasp_command",
        }
    ),
    "hpc": frozenset(
        {
            "cores_per_node",
            "gpus_per_node",
            "max_nodes",
            "scp_address",
            "vasp_command",
            "nep_command",
            "gpumd_command",
        }
    ),
}


def canonical_config_path(project_dir: Path) -> Path:
    """Return the only supported project configuration path."""

    return Path(project_dir) / "config" / CANONICAL_CONFIG_NAME


def find_config_path(project_dir: Path, explicit_path: Path | None = None) -> Path:
    """Locate the canonical config deterministically and fail if absent."""

    canonical_path = canonical_config_path(project_dir)
    if canonical_path.is_file():
        return canonical_path
    if explicit_path is not None and Path(explicit_path).resolve() == canonical_path.resolve():
        raise ConfigurationError(f"Project configuration not found: {canonical_path}")
    if explicit_path is not None and Path(explicit_path).exists():
        raise ConfigurationError(
            f"Only {CANONICAL_CONFIG_NAME} is supported; legacy config path is not accepted: "
            f"{explicit_path}"
        )
    raise ConfigurationError(f"Project configuration not found: {canonical_path}")


def load_config(
    config_path: Path,
    *,
    project_name: str | None = None,
    require_scientific_fields: bool = True,
) -> NepflowConfig:
    """Load, parse, validate, and return the canonical typed root config."""

    path = Path(config_path)
    parser = _read_parser(path)
    _reject_unknown_keys(parser)
    sections = {section.lower(): _section(parser, section) for section in parser.sections()}

    schema_version, project = parse_project(
        sections.get("project", {}),
        project_name=project_name,
        derived_name=_derived_project_name(path),
        schema_default=CONFIG_SCHEMA_VERSION,
    )
    composition_values = sections.get("composition", {})
    generation_values = sections.get("generation", {})
    composition = parse_composition(composition_values, generation_values)
    generation = parse_generation(generation_values)
    dft_recovery, slurm = parse_slurm(sections.get("slurm", {}))

    config = NepflowConfig(
        schema_version=schema_version,
        project=project,
        paths=parse_paths(sections.get("paths", {})),
        materials_project=MaterialsProjectConfig(api_key=None),
        composition=composition,
        generation=generation,
        magnetism=parse_magnetism(sections.get("magnetism", {})),
        selection=parse_selection(sections.get("selection", {})),
        vasp=parse_vasp(sections.get("vasp", {})),
        dft_recovery=dft_recovery,
        nep=parse_nep(sections.get("nep", {})),
        train_nep=parse_training(
            sections.get("train_nep", {}),
            sections.get("training_sweep", {}),
            composition.elements + composition.gas_elements,
        ),
        validation=parse_validation(sections.get("validate", {}), sections.get("gpumd", {})),
        hpc=parse_hpc(sections.get("hpc", {})),
        slurm=slurm,
        source_path=path,
        raw_sections=frozenset(sections),
    )
    if require_scientific_fields and not sections.get("hpc", {}).get("vasp_command", "").strip():
        raise ConfigurationError("Required configuration hpc.vasp_command is missing or blank")
    return validate_config(config, require_scientific_fields=require_scientific_fields)


def _read_parser(path: Path) -> ConfigParser:
    if not path.is_file():
        raise ConfigurationError(f"Project configuration not found: {path}")
    parser = ConfigParser(interpolation=None)
    try:
        read_files = parser.read(path, encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Could not read project configuration: {path}") from exc
    if not read_files:
        raise ConfigurationError(f"Could not read project configuration: {path}")
    return parser


def _reject_unknown_keys(parser: ConfigParser) -> None:
    if parser.defaults():
        unknown = ", ".join(sorted(parser.defaults()))
        raise ConfigurationError(f"Unknown configuration DEFAULT keys: {unknown}")
    for section in parser.sections():
        normalized_section = section.lower()
        if normalized_section not in _ALLOWED_KEYS:
            raise ConfigurationError(f"Unknown configuration section [{section}]")
        for key in parser[section]:
            normalized_key = key.lower()
            if normalized_key == "project_dir" and normalized_section == "paths":
                raise ConfigurationError(
                    "paths.project_dir is not supported; project_dir is runtime context"
                )
            if normalized_section == "slurm" and normalized_key == "vasp_command":
                raise ConfigurationError(
                    "slurm.vasp_command is not supported; use the single hpc.vasp_command source"
                )
            if normalized_key not in _ALLOWED_KEYS[normalized_section]:
                raise ConfigurationError(f"Unknown configuration key {section}.{key}")


def _section(parser: ConfigParser, name: str) -> dict[str, str]:
    actual_name = next(
        (section for section in parser.sections() if section.lower() == name),
        None,
    )
    if actual_name is None:
        return {}
    return {key.lower(): value.strip() for key, value in parser.items(actual_name)}


def _derived_project_name(path: Path) -> str:
    project_dir = path.parent.parent.name
    return project_dir.removeprefix("project_") or project_dir


__all__ = [
    "CANONICAL_CONFIG_NAME",
    "canonical_config_path",
    "find_config_path",
    "load_config",
    "load_legacy_config",
    "to_legacy_config",
]
