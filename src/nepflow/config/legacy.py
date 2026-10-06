"""Compatibility rendering for unmigrated ConfigParser consumers."""

from __future__ import annotations

import json
from configparser import ConfigParser
from pathlib import Path

from .models import NepflowConfig

_CANONICAL_SECTIONS = frozenset(
    {
        "project",
        "paths",
        "materialsproject",
        "composition",
        "generation",
        "selection",
        "vasp",
        "nep",
        "train_nep",
        "training_sweep",
        "gpumd",
        "validate",
        "slurm",
        "hpc",
    }
)


def load_legacy_config(
    config_path: Path,
    *,
    project_name: str | None = None,
    require_scientific_fields: bool = False,
) -> ConfigParser:
    """Adapt a validated canonical config for remaining legacy consumers.

    This bridge is retained for Phase-4 operators and tests while callers
    migrate to typed configuration access. It has no independent file parser.
    """

    from .loader import load_config

    typed = load_config(
        config_path,
        project_name=project_name,
        require_scientific_fields=require_scientific_fields,
    )
    return to_legacy_config(typed)


def to_legacy_config(config: NepflowConfig) -> ConfigParser:
    """Render a typed root config using the temporary legacy key names."""

    parser = ConfigParser(interpolation=None)
    present_sections = config.raw_sections or _CANONICAL_SECTIONS
    for name, values in _section_values(config, present_sections).items():
        if name in present_sections:
            parser[name] = values
    return parser


def _section_values(
    config: NepflowConfig,
    present_sections: frozenset[str],
) -> dict[str, dict[str, str]]:
    sections = {
        "project": _project_values(config),
        "paths": _path_values(config),
        "materialsproject": {"api_key": ""},
        "composition": _composition_values(config),
        "generation": _generation_values(config),
        "selection": _selection_values(config),
        "vasp": _vasp_values(config),
        "nep": {"enabled": _bool_text(config.nep.enabled)},
        "train_nep": _training_values(config),
    }
    if config.train_nep.sweep:
        sections["training_sweep"] = {
            name: "|".join(_format_sweep_value(value) for value in values)
            for name, values in config.train_nep.sweep
        }
    sections["gpumd"] = {"enabled": _bool_text(config.validation.enabled)}
    if "gpumd" in present_sections and config.validation.model_run_id:
        sections["gpumd"]["model_run_id"] = config.validation.model_run_id
    if "validate" in present_sections and config.validation.model_run_id:
        sections["validate"] = {"model_run_id": config.validation.model_run_id}
    sections["slurm"] = _slurm_values(config)
    sections["hpc"] = _hpc_values(config)
    return sections


def _project_values(config: NepflowConfig) -> dict[str, str]:
    return {
        "name": config.project.name,
        "description": config.project.description,
        "status": config.project.status,
        "random_seed": str(config.project.random_seed),
        "schema_version": str(config.schema_version),
        "config_version": str(config.project.config_version),
    }


def _path_values(config: NepflowConfig) -> dict[str, str]:
    return {
        "structures_path": str(config.paths.structures_path),
        "vasp_path": str(config.paths.vasp_path),
        "nep_path": str(config.paths.nep_path),
        "gpumd_path": str(config.paths.gpumd_path),
        "reports_path": str(config.paths.reports_path),
    }


def _composition_values(config: NepflowConfig) -> dict[str, str]:
    return {
        "elements": ",".join(config.composition.elements),
        "gasElements": ",".join(config.composition.gas_elements),
        "composition_step": str(config.composition.composition_step),
        "include_pure_elements": _bool_text(config.composition.include_pure_elements),
        "include_binaries": _bool_text(config.composition.include_binaries),
        "include_ternaries": _bool_text(config.composition.include_ternaries),
    }


def _generation_values(config: NepflowConfig) -> dict[str, str]:
    generation = config.generation
    return {
        "crystal_structures": ",".join(generation.crystal_structures),
        "target_n_atoms": str(generation.target_n_atoms),
        "composition_tolerance": str(generation.composition_tolerance),
        "n_workers": str(generation.n_workers),
        "use_materials_project": _bool_text(generation.use_materials_project),
        "use_random_solid_solution": _bool_text(generation.use_random_solid_solution),
        "use_sqs": _bool_text(generation.use_sqs),
        "use_segregated": _bool_text(generation.use_segregated),
        "use_liquid": _bool_text(generation.use_liquid),
        "n_random_solid_solution": str(generation.n_random_solid_solution),
        "n_sqs": str(generation.n_sqs),
        "n_segregated": str(generation.n_segregated),
        "n_liquid_configurations": str(generation.n_liquid_configurations),
        "n_liquid_snapshots": str(generation.n_liquid_snapshots),
        "liquid_temperature": str(generation.liquid_temperature),
        "liquid_timestep_fs": str(generation.liquid_timestep_fs),
        "liquid_equilibration_steps": str(generation.liquid_equilibration_steps),
        "liquid_steps_between_snapshots": str(generation.liquid_steps_between_snapshots),
        "liquid_friction": str(generation.liquid_friction),
        "volume_scale_min": str(generation.volume_scale_min),
        "volume_scale_max": str(generation.volume_scale_max),
        "n_volume_points": str(generation.n_volume_points),
        "volume_sources": ",".join(generation.volume_sources),
        "elastic_stress_enabled": _bool_text(generation.elastic_stress_enabled),
        "elastic_strain_amplitudes": ",".join(map(str, generation.elastic_strain_amplitudes)),
        "elastic_sources": ",".join(generation.elastic_sources),
        "n_rattled": str(generation.n_rattled),
        "rattle_sources": ",".join(generation.rattle_sources),
        "n_vacancies": str(generation.n_vacancies),
        "vacancy_sources": ",".join(generation.vacancy_sources),
        "n_interstitials": str(generation.n_interstitials),
        "interstitial_sources": ",".join(generation.interstitial_sources),
        "n_gas_interstitials": str(generation.n_gas_interstitials),
        "gas_interstitial_sources": ",".join(generation.gas_interstitial_sources),
        "n_substitutions": str(generation.n_substitutions),
        "substitution_sources": ",".join(generation.substitution_sources),
        "n_antisites": str(generation.n_antisites),
        "antisite_sources": ",".join(generation.antisite_sources),
        "n_vacancy_interstitial": str(generation.n_vacancy_interstitial),
        "vacancy_interstitial_sources": ",".join(generation.vacancy_interstitial_sources),
        "n_gas_in_vacancy": str(generation.n_gas_in_vacancy),
        "gas_in_vacancy_sources": ",".join(generation.gas_in_vacancy_sources),
        "surface_enabled": _bool_text(generation.surface_enabled),
        "n_surfaces": str(generation.n_surfaces),
        "surface_sources": ",".join(generation.surface_sources),
        "surface_miller_indices": ";".join(
            ",".join(map(str, index)) for index in generation.surface_miller_indices
        ),
        "surface_layers": str(generation.surface_layers),
        "surface_thickness": (
            "" if generation.surface_thickness is None else str(generation.surface_thickness)
        ),
        "surface_vacuum": str(generation.surface_vacuum),
        "surface_termination_policy": generation.surface_termination_policy,
        "surface_max_terminations": str(generation.surface_max_terminations),
        "surface_in_plane_repeat": ",".join(map(str, generation.surface_in_plane_repeat)),
        "surface_min_in_plane_dimensions": ",".join(
            map(str, generation.surface_min_in_plane_dimensions)
        ),
        "surface_symmetric": _bool_text(generation.surface_symmetric),
        "grain_boundary_enabled": _bool_text(generation.grain_boundary_enabled),
        "n_grain_boundaries": str(generation.n_grain_boundaries),
        "grain_boundary_sources": ",".join(generation.grain_boundary_sources),
        "grain_boundary_rotation_axis": ",".join(map(str, generation.grain_boundary_rotation_axis)),
        "grain_boundary_misorientation_angle": str(generation.grain_boundary_misorientation_angle),
        "grain_boundary_sigma": str(generation.grain_boundary_sigma),
        "grain_boundary_plane": ",".join(map(str, generation.grain_boundary_plane)),
        "grain_boundary_expand_times": str(generation.grain_boundary_expand_times),
        "grain_boundary_min_thickness": str(generation.grain_boundary_min_thickness),
        "grain_boundary_overlap_tolerance": str(generation.grain_boundary_overlap_tolerance),
        "liquid_sources": ",".join(generation.liquid_sources),
        "rattle_std": str(generation.rattle_std),
        "rattle_std_min": str(generation.rattle_std_min),
        "rattle_std_max": str(generation.rattle_std_max),
        "rattle_d_min": str(generation.rattle_d_min),
        "vacancy_min": str(generation.vacancy_min),
        "vacancy_max": str(generation.vacancy_max),
        "vacancy_species": ",".join(generation.vacancy_species),
        "interstitial_d_min": str(generation.interstitial_d_min),
        "interstitial_min": str(generation.interstitial_min),
        "interstitial_max": str(generation.interstitial_max),
        "interstitial_sites": json.dumps(generation.interstitial_sites),
        "crystallographic_interstitial_sites": json.dumps(
            generation.crystallographic_interstitial_sites
        ),
        "defect_defect_d_min": str(generation.defect_defect_d_min),
        "periodic_image_d_min": str(generation.periodic_image_d_min),
        "interstitial_max_attempts": str(generation.interstitial_max_attempts),
        "substitution_pairs": ";".join(
            f"{source}->{target}" for source, target in generation.substitution_pairs
        ),
        "substitution_min": str(generation.substitution_min),
        "substitution_max": str(generation.substitution_max),
        "antisite_pairs": ";".join(
            f"{source}->{target}" for source, target in generation.antisite_pairs
        ),
        "antisite_min": str(generation.antisite_min),
        "antisite_max": str(generation.antisite_max),
        "gas_interstitial_d_min": str(generation.gas_interstitial_d_min),
        "max_gas_occupancy": str(generation.max_gas_occupancy),
    }


def _selection_values(config: NepflowConfig) -> dict[str, str]:
    selection = config.selection
    return {
        "nep_model_file": selection.nep_model_file,
        "include_seed_structures": _bool_text(selection.include_seed_structures),
        "include_single_element_elastic_stress_structures": _bool_text(
            selection.include_single_element_elastic_stress_structures
        ),
        "include_elastic_stress_structures": _bool_text(
            selection.include_elastic_stress_structures
        ),
        "composition_aware_fps": _bool_text(selection.composition_aware_fps),
        "composition_aware_fps_frontier_fraction": str(
            selection.composition_aware_fps_frontier_fraction
        ),
        "composition_aware_fps_ternary_weight": str(selection.composition_aware_fps_ternary_weight),
        "composition_aware_fps_adaptive_retries": str(
            selection.composition_aware_fps_adaptive_retries
        ),
        "composition_aware_fps_descriptor_floor_fraction": str(
            selection.composition_aware_fps_descriptor_floor_fraction
        ),
        "target_train_count": str(selection.target_train_count),
        "target_test_count": str(selection.target_test_count),
        "target_tolerance": str(selection.target_tolerance),
        "descriptor_type": selection.descriptor_type,
        "batch_size": str(selection.batch_size),
        "max_search_iterations": str(selection.max_search_iterations),
        "test_pool_factor": str(selection.test_pool_factor),
    }


def _vasp_values(config: NepflowConfig) -> dict[str, str]:
    return {
        "enabled": _bool_text(config.vasp.enabled),
        "kspacing": str(config.vasp.kspacing),
        "kgamma": _vasp_bool_text(config.vasp.kgamma),
    }


def _training_values(config: NepflowConfig) -> dict[str, str]:
    training = config.train_nep
    return {
        "population": str(training.population),
        "batch": str(training.batch),
        "generation": str(training.generations),
        "cutoff": " ".join(training.cutoff),
        "n_max": " ".join(training.n_max),
        "basis_size": " ".join(training.basis_size),
        "l_max": " ".join(training.l_max),
        "neuron": " ".join(training.neuron),
        "outerZBL": str(training.outer_zbl),
        "charge_mode": str(training.charge_mode),
        "weights": ",".join(map(str, training.weights)),
        "lambda_e": str(training.lambda_e),
        "lambda_f": str(training.lambda_f),
        "lambda_v": str(training.lambda_v),
        "lambda_shear": str(training.lambda_shear),
        "train_virial": _bool_text(training.train_virial),
        "allow_partial_dataset": _bool_text(training.allow_partial_dataset),
        "max_resubmit": str(training.max_resubmit),
    }


def _slurm_values(config: NepflowConfig) -> dict[str, str]:
    slurm = config.slurm
    values = {
        "enabled": _bool_text(slurm.enabled),
        "max_concurrent": str(slurm.max_concurrent),
        "walltime": slurm.walltime,
        "poll_interval": str(slurm.poll_interval),
        "max_retry_level": str(config.dft_recovery.max_retry_level),
        "vasp_walltime": config.dft_recovery.vasp_walltime,
        "gpumd_walltime": slurm.gpumd_walltime,
        "gpumd_nodes": str(slurm.gpumd_nodes),
        "gpumd_gpus": str(slurm.gpumd_gpus),
        "memory_poll_interval": str(slurm.memory_poll_interval),
        "memory_walltime": slurm.memory_walltime,
    }
    if slurm.train_nep_walltime:
        values["train_nep_walltime"] = slurm.train_nep_walltime
    return values


def _hpc_values(config: NepflowConfig) -> dict[str, str]:
    return {
        "cores_per_node": str(config.hpc.cores_per_node),
        "gpus_per_node": str(config.hpc.gpus_per_node),
        "max_nodes": str(config.hpc.max_nodes),
        "scp_address": config.hpc.scp_address,
        "vasp_command": config.hpc.vasp_command,
        "nep_command": config.hpc.nep_command,
        "gpumd_command": config.hpc.gpumd_command,
    }


def _format_sweep_value(value: object) -> str:
    if isinstance(value, (tuple, list)):
        return " ".join(str(item) for item in value)
    return str(value)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _vasp_bool_text(value: bool) -> str:
    return ".TRUE." if value else ".FALSE."
