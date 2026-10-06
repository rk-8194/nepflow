"""Parse normalized canonical configuration sections into typed models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nepflow.errors import ConfigurationError

from .models import (
    ALL_SOURCES,
    GENERATION_SOURCE_SCOPE_FIELDS,
    SUPPORTED_CONFIGURATIONAL_SOURCES,
    CompositionConfig,
    DftRecoveryConfig,
    GenerationConfig,
    HpcConfig,
    NepConfig,
    NepTrainingConfig,
    PathsConfig,
    ProjectConfig,
    SelectionConfig,
    SlurmConfig,
    ValidationConfig,
    VaspConfig,
)

_SWEEP_FIELDS = (
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
)
_SWEEP_TUPLE_FIELDS = frozenset({"weights", "cutoff", "n_max", "basis_size", "l_max", "neuron"})


def parse_project(
    values: Mapping[str, str],
    *,
    project_name: str | None,
    derived_name: str,
    schema_default: int,
) -> tuple[int, ProjectConfig]:
    """Parse project identity values and return the root schema version."""

    schema_version = _parse_int(
        values.get("schema_version", str(schema_default)),
        "project.schema_version",
    )
    config_version = _parse_int(
        values.get("config_version", str(schema_version)),
        "project.config_version",
    )
    project = ProjectConfig(
        name=values.get("name", project_name or derived_name).strip(),
        description=values.get("description", "").strip(),
        status=values.get("status", "initialized").strip(),
        random_seed=_parse_int(values.get("random_seed", "42"), "project.random_seed"),
        config_version=config_version,
    )
    return schema_version, project


def parse_paths(values: Mapping[str, str]) -> PathsConfig:
    """Parse project-relative output paths."""

    return PathsConfig(
        structures_path=_parse_path(
            values.get("structures_path", "structures"), "paths.structures_path"
        ),
        vasp_path=_parse_path(values.get("vasp_path", "vasp"), "paths.vasp_path"),
        nep_path=_parse_path(values.get("nep_path", "nep"), "paths.nep_path"),
        gpumd_path=_parse_path(values.get("gpumd_path", "gpumd"), "paths.gpumd_path"),
        reports_path=_parse_path(values.get("reports_path", "reports"), "paths.reports_path"),
    )


def parse_composition(
    values: Mapping[str, str], generation_values: Mapping[str, str]
) -> CompositionConfig:
    """Parse composition settings, including the supported legacy aliases."""

    elements = _parse_symbols(
        _resolve_alias(
            values,
            "elements",
            generation_values,
            "elements",
            "composition.elements",
        ),
        "composition.elements",
    )
    gas_elements = _parse_symbols(
        _resolve_alias(
            values,
            "gas_elements",
            values,
            "gaselements",
            "composition.gas_elements",
        ),
        "composition.gas_elements",
    )
    return CompositionConfig(
        elements=elements,
        gas_elements=gas_elements,
        composition_step=_parse_float(
            values.get("composition_step", "0.125"),
            "composition.composition_step",
        ),
        include_pure_elements=_parse_bool(
            values.get("include_pure_elements", "true"),
            "composition.include_pure_elements",
        ),
        include_binaries=_parse_bool(
            values.get("include_binaries", "true"),
            "composition.include_binaries",
        ),
        include_ternaries=_parse_bool(
            values.get("include_ternaries", "true"),
            "composition.include_ternaries",
        ),
    )


def parse_generation(values: Mapping[str, str]) -> GenerationConfig:
    """Parse structure-generation switches, counts, and scientific values."""

    return GenerationConfig(
        **_parse_generation_modes(values),
        **_parse_generation_liquid(values),
        **_parse_generation_scopes(values),
        **_parse_generation_perturbations(values),
    )


def _parse_generation_modes(values: Mapping[str, str]) -> dict[str, Any]:
    return {
        "crystal_structures": _parse_list(
            values.get("crystal_structures", ""),
            "generation.crystal_structures",
        ),
        "target_n_atoms": _parse_int(
            values.get("target_n_atoms", "128"), "generation.target_n_atoms"
        ),
        "composition_tolerance": _parse_float(
            values.get("composition_tolerance", "0.05"),
            "generation.composition_tolerance",
        ),
        "n_workers": _parse_int(values.get("n_workers", "0"), "generation.n_workers"),
        "use_materials_project": _parse_bool(
            values.get("use_materials_project", "true"),
            "generation.use_materials_project",
        ),
        "use_random_solid_solution": _parse_bool(
            values.get("use_random_solid_solution", "true"),
            "generation.use_random_solid_solution",
        ),
        "use_sqs": _parse_bool(values.get("use_sqs", "true"), "generation.use_sqs"),
        "use_segregated": _parse_bool(
            values.get("use_segregated", "true"), "generation.use_segregated"
        ),
        "use_liquid": _parse_bool(values.get("use_liquid", "false"), "generation.use_liquid"),
        "n_random_solid_solution": _parse_int(
            values.get("n_random_solid_solution", "3"),
            "generation.n_random_solid_solution",
        ),
        "n_sqs": _parse_int(values.get("n_sqs", "1"), "generation.n_sqs"),
        "n_segregated": _parse_int(values.get("n_segregated", "3"), "generation.n_segregated"),
        "n_rattled": _parse_int(values.get("n_rattled", "10"), "generation.n_rattled"),
        "n_vacancies": _parse_int(values.get("n_vacancies", "10"), "generation.n_vacancies"),
        "n_interstitials": _parse_int(
            values.get("n_interstitials", "10"), "generation.n_interstitials"
        ),
        "n_gas_interstitials": _parse_int(
            values.get("n_gas_interstitials", "10"),
            "generation.n_gas_interstitials",
        ),
        "n_substitutions": _parse_int(
            values.get("n_substitutions", "0"), "generation.n_substitutions"
        ),
        "n_antisites": _parse_int(values.get("n_antisites", "0"), "generation.n_antisites"),
        "n_vacancy_interstitial": _parse_int(
            values.get("n_vacancy_interstitial", "10"),
            "generation.n_vacancy_interstitial",
        ),
        "n_gas_in_vacancy": _parse_int(
            values.get("n_gas_in_vacancy", "10"),
            "generation.n_gas_in_vacancy",
        ),
        "n_surfaces": _parse_int(
            values.get("n_surfaces", "0"), "generation.n_surfaces"
        ),
    }


def _parse_generation_liquid(values: Mapping[str, str]) -> dict[str, Any]:
    return {
        "n_liquid_configurations": _parse_int(
            values.get("n_liquid_configurations", "2"),
            "generation.n_liquid_configurations",
        ),
        "n_liquid_snapshots": _parse_int(
            values.get("n_liquid_snapshots", "5"),
            "generation.n_liquid_snapshots",
        ),
        "liquid_temperature": _parse_float(
            values.get("liquid_temperature", "3000"),
            "generation.liquid_temperature",
        ),
        "liquid_timestep_fs": _parse_float(
            values.get("liquid_timestep_fs", "1.0"),
            "generation.liquid_timestep_fs",
        ),
        "liquid_equilibration_steps": _parse_int(
            values.get("liquid_equilibration_steps", "200"),
            "generation.liquid_equilibration_steps",
        ),
        "liquid_steps_between_snapshots": _parse_int(
            values.get("liquid_steps_between_snapshots", "100"),
            "generation.liquid_steps_between_snapshots",
        ),
        "liquid_friction": _parse_float(
            values.get("liquid_friction", "0.02"),
            "generation.liquid_friction",
        ),
        "volume_scale_min": _parse_float(
            values.get("volume_scale_min", "0.8"), "generation.volume_scale_min"
        ),
        "volume_scale_max": _parse_float(
            values.get("volume_scale_max", "1.2"), "generation.volume_scale_max"
        ),
        "n_volume_points": _parse_int(
            values.get("n_volume_points", "11"), "generation.n_volume_points"
        ),
        "elastic_stress_enabled": _parse_bool(
            values.get("elastic_stress_enabled", "true"),
            "generation.elastic_stress_enabled",
        ),
        "elastic_strain_amplitudes": _parse_float_list(
            values.get("elastic_strain_amplitudes", "-0.02,-0.01,-0.005,0.005,0.01,0.02"),
            "generation.elastic_strain_amplitudes",
        ),
    }


def _parse_generation_perturbations(values: Mapping[str, str]) -> dict[str, Any]:
    rattle_std = _parse_float(values.get("rattle_std", "0.03"), "generation.rattle_std")
    return {
        "rattle_std": rattle_std,
        "rattle_std_min": _parse_float(
            values.get("rattle_std_min", str(0.5 * rattle_std)),
            "generation.rattle_std_min",
        ),
        "rattle_std_max": _parse_float(
            values.get("rattle_std_max", str(2.0 * rattle_std)),
            "generation.rattle_std_max",
        ),
        "rattle_d_min": _parse_float(values.get("rattle_d_min", "1.5"), "generation.rattle_d_min"),
        "vacancy_min": _parse_float(values.get("vacancy_min", "0.0"), "generation.vacancy_min"),
        "vacancy_max": _parse_float(values.get("vacancy_max", "0.1"), "generation.vacancy_max"),
        "interstitial_d_min": _parse_float(
            values.get("interstitial_d_min", "1.65"),
            "generation.interstitial_d_min",
        ),
        "interstitial_min": _parse_float(
            values.get("interstitial_min", "0.05"), "generation.interstitial_min"
        ),
        "interstitial_max": _parse_float(
            values.get("interstitial_max", "0.1"),
            "generation.interstitial_max",
        ),
        "gas_interstitial_d_min": _parse_float(
            values.get("gas_interstitial_d_min", "1.2"),
            "generation.gas_interstitial_d_min",
        ),
        "defect_defect_d_min": _parse_float(
            values.get("defect_defect_d_min", "0.0"),
            "generation.defect_defect_d_min",
        ),
        "periodic_image_d_min": _parse_float(
            values.get("periodic_image_d_min", "0.0"),
            "generation.periodic_image_d_min",
        ),
        "interstitial_max_attempts": _parse_int(
            values.get("interstitial_max_attempts", "500"),
            "generation.interstitial_max_attempts",
        ),
        "max_gas_occupancy": _parse_int(
            values.get("max_gas_occupancy", "3"),
            "generation.max_gas_occupancy",
        ),
        "vacancy_species": _parse_symbols(
            values.get("vacancy_species", ""), "generation.vacancy_species"
        ),
        "substitution_pairs": _parse_species_pairs(
            values.get("substitution_pairs", ""), "generation.substitution_pairs"
        ),
        "substitution_min": _parse_float(
            values.get("substitution_min", "0.0"), "generation.substitution_min"
        ),
        "substitution_max": _parse_float(
            values.get("substitution_max", "0.1"), "generation.substitution_max"
        ),
        "antisite_pairs": _parse_species_pairs(
            values.get("antisite_pairs", ""), "generation.antisite_pairs"
        ),
        "antisite_min": _parse_float(values.get("antisite_min", "0.0"), "generation.antisite_min"),
        "antisite_max": _parse_float(values.get("antisite_max", "0.1"), "generation.antisite_max"),
        "interstitial_sites": _parse_interstitial_sites(
            values.get("interstitial_sites", ""), "generation.interstitial_sites"
        ),
        "crystallographic_interstitial_sites": _parse_interstitial_sites(
            values.get("crystallographic_interstitial_sites", ""),
            "generation.crystallographic_interstitial_sites",
        ),
        "surface_enabled": _parse_bool(
            values.get("surface_enabled", "false"), "generation.surface_enabled"
        ),
        "surface_miller_indices": _parse_miller_indices(
            values.get("surface_miller_indices", "1,0,0"),
            "generation.surface_miller_indices",
        ),
        "surface_layers": _parse_int(
            values.get("surface_layers", "3"), "generation.surface_layers"
        ),
        "surface_thickness": _parse_optional_float(
            values.get("surface_thickness", ""), "generation.surface_thickness"
        ),
        "surface_vacuum": _parse_float(
            values.get("surface_vacuum", "10.0"), "generation.surface_vacuum"
        ),
        "surface_termination_policy": values.get(
            "surface_termination_policy", "all"
        ).strip().lower(),
        "surface_max_terminations": _parse_int(
            values.get("surface_max_terminations", "0"),
            "generation.surface_max_terminations",
        ),
        "surface_in_plane_repeat": _parse_int_pair(
            values.get("surface_in_plane_repeat", "1,1"),
            "generation.surface_in_plane_repeat",
        ),
        "surface_min_in_plane_dimensions": _parse_float_pair(
            values.get("surface_min_in_plane_dimensions", "0.0,0.0"),
            "generation.surface_min_in_plane_dimensions",
        ),
        "surface_symmetric": _parse_bool(
            values.get("surface_symmetric", "false"), "generation.surface_symmetric"
        ),
    }


def _parse_generation_scopes(values: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    return {
        field_name: _parse_source_scope(values, field_name)
        for field_name in GENERATION_SOURCE_SCOPE_FIELDS
    }


def _parse_source_scope(values: Mapping[str, str], field_name: str) -> tuple[str, ...]:
    key = f"generation.{field_name}"
    raw = values.get(field_name, ALL_SOURCES)
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigurationError(f"{key} must explicitly name one or more sources")
    entries = tuple(item.strip().lower() for item in raw.split(","))
    if any(not item for item in entries):
        raise ConfigurationError(f"{key} must be a comma-separated source list")
    if len(set(entries)) != len(entries):
        raise ConfigurationError(f"{key} must not contain duplicate sources")
    if ALL_SOURCES in entries and len(entries) != 1:
        raise ConfigurationError(f"{key} cannot combine 'all' with named sources")
    unknown = set(entries) - SUPPORTED_CONFIGURATIONAL_SOURCES - {ALL_SOURCES}
    if unknown:
        raise ConfigurationError(
            f"{key} contains unsupported sources: {', '.join(sorted(unknown))}"
        )
    return entries


def parse_selection(values: Mapping[str, str]) -> SelectionConfig:
    """Parse selection and descriptor settings."""

    return SelectionConfig(
        nep_model_file=values.get("nep_model_file", "nep89.txt").strip(),
        include_seed_structures=_parse_bool(
            values.get("include_seed_structures", "false"),
            "selection.include_seed_structures",
        ),
        include_single_element_elastic_stress_structures=_parse_bool(
            values.get("include_single_element_elastic_stress_structures", "false"),
            "selection.include_single_element_elastic_stress_structures",
        ),
        include_elastic_stress_structures=_parse_bool(
            values.get("include_elastic_stress_structures", "false"),
            "selection.include_elastic_stress_structures",
        ),
        composition_aware_fps=_parse_bool(
            values.get("composition_aware_fps", "false"),
            "selection.composition_aware_fps",
        ),
        composition_aware_fps_frontier_fraction=_parse_float(
            values.get("composition_aware_fps_frontier_fraction", "0.1"),
            "selection.composition_aware_fps_frontier_fraction",
        ),
        composition_aware_fps_ternary_weight=_parse_float(
            values.get("composition_aware_fps_ternary_weight", "1.0"),
            "selection.composition_aware_fps_ternary_weight",
        ),
        composition_aware_fps_adaptive_retries=_parse_int(
            values.get("composition_aware_fps_adaptive_retries", "4"),
            "selection.composition_aware_fps_adaptive_retries",
        ),
        composition_aware_fps_descriptor_floor_fraction=_parse_float(
            values.get("composition_aware_fps_descriptor_floor_fraction", "0.95"),
            "selection.composition_aware_fps_descriptor_floor_fraction",
        ),
        target_train_count=_parse_int(
            values.get("target_train_count", "1000"),
            "selection.target_train_count",
        ),
        target_test_count=_parse_int(
            values.get("target_test_count", "200"), "selection.target_test_count"
        ),
        target_tolerance=_parse_int(
            values.get("target_tolerance", "50"), "selection.target_tolerance"
        ),
        descriptor_type=values.get("descriptor_type", "structure").strip().lower(),
        batch_size=_parse_int(values.get("batch_size", "500"), "selection.batch_size"),
        max_search_iterations=_parse_int(
            values.get("max_search_iterations", "30"),
            "selection.max_search_iterations",
        ),
        test_pool_factor=_parse_float(
            values.get("test_pool_factor", "0.5"), "selection.test_pool_factor"
        ),
    )


def parse_vasp(values: Mapping[str, str]) -> VaspConfig:
    """Parse VASP enablement and k-point settings."""

    return VaspConfig(
        enabled=_parse_bool(values.get("enabled", "true"), "vasp.enabled"),
        kspacing=_parse_float(values.get("kspacing", "0.30"), "vasp.kspacing"),
        kgamma=_parse_bool(values.get("kgamma", "true"), "vasp.kgamma"),
    )


def parse_nep(values: Mapping[str, str]) -> NepConfig:
    """Parse NEP enablement settings."""

    return NepConfig(enabled=_parse_bool(values.get("enabled", "true"), "nep.enabled"))


def parse_training(
    values: Mapping[str, str],
    sweep_values: Mapping[str, str],
    train_elements: tuple[str, ...],
) -> NepTrainingConfig:
    """Parse NEP training settings and the optional scientific sweep."""

    weights = _parse_float_list(values.get("weights", ""), "train_nep.weights")
    if not weights and train_elements:
        weights = tuple(1.0 for _ in train_elements)
    return NepTrainingConfig(
        population=_parse_int(values.get("population", "50"), "train_nep.population"),
        batch=_parse_int(values.get("batch", "3000"), "train_nep.batch"),
        generations=_parse_int(values.get("generation", "250000"), "train_nep.generation"),
        charge_mode=_parse_int(values.get("charge_mode", "0"), "train_nep.charge_mode"),
        weights=weights,
        outer_zbl=_parse_float(values.get("outerzbl", "2.0"), "train_nep.outer_zbl"),
        cutoff=_parse_tokens(values.get("cutoff", "6 5"), "train_nep.cutoff"),
        n_max=_parse_tokens(values.get("n_max", "4 4"), "train_nep.n_max"),
        basis_size=_parse_tokens(values.get("basis_size", "8 8"), "train_nep.basis_size"),
        l_max=_parse_tokens(values.get("l_max", "4 2 1"), "train_nep.l_max"),
        neuron=_parse_tokens(values.get("neuron", "80"), "train_nep.neuron"),
        lambda_e=_parse_float(values.get("lambda_e", "1.0"), "train_nep.lambda_e"),
        lambda_f=_parse_float(values.get("lambda_f", "1.0"), "train_nep.lambda_f"),
        lambda_v=_parse_float(values.get("lambda_v", "1.0"), "train_nep.lambda_v"),
        lambda_shear=_parse_float(values.get("lambda_shear", "1.0"), "train_nep.lambda_shear"),
        train_virial=_parse_bool(values.get("train_virial", "false"), "train_nep.train_virial"),
        allow_partial_dataset=_parse_bool(
            values.get("allow_partial_dataset", "false"),
            "train_nep.allow_partial_dataset",
        ),
        max_resubmit=_parse_int(values.get("max_resubmit", "3"), "train_nep.max_resubmit"),
        sweep=_parse_training_sweep(values, sweep_values),
    )


def parse_slurm(
    values: Mapping[str, str],
) -> tuple[DftRecoveryConfig, SlurmConfig]:
    """Parse scheduler settings and DFT retry policy."""

    recovery = DftRecoveryConfig(
        max_retry_level=_parse_int(values.get("max_retry_level", "100"), "slurm.max_retry_level"),
        vasp_walltime=values.get("vasp_walltime", "01:00:00").strip(),
    )
    slurm = SlurmConfig(
        enabled=_parse_bool(values.get("enabled", "false"), "slurm.enabled"),
        max_concurrent=_parse_int(values.get("max_concurrent", "20"), "slurm.max_concurrent"),
        walltime=values.get("walltime", "03:00:00").strip(),
        train_nep_walltime=_optional_string(values.get("train_nep_walltime")),
        poll_interval=_parse_int(values.get("poll_interval", "20"), "slurm.poll_interval"),
        gpumd_walltime=values.get("gpumd_walltime", "00:10:00").strip(),
        gpumd_nodes=_parse_int(values.get("gpumd_nodes", "1"), "slurm.gpumd_nodes"),
        gpumd_gpus=_parse_int(values.get("gpumd_gpus", "1"), "slurm.gpumd_gpus"),
        memory_poll_interval=_parse_int(
            values.get("memory_poll_interval", "10"),
            "slurm.memory_poll_interval",
        ),
        memory_walltime=values.get("memory_walltime", "00:10:00").strip(),
    )
    return recovery, slurm


def parse_hpc(values: Mapping[str, str]) -> HpcConfig:
    """Parse site resource and command settings."""

    return HpcConfig(
        cores_per_node=_parse_int(values.get("cores_per_node", "64"), "hpc.cores_per_node"),
        gpus_per_node=_parse_int(values.get("gpus_per_node", "4"), "hpc.gpus_per_node"),
        max_nodes=_parse_int(values.get("max_nodes", "16"), "hpc.max_nodes"),
        scp_address=values.get("scp_address", "").strip(),
        vasp_command=values.get("vasp_command", "mpirun -np {ntasks} vasp_std").strip(),
        nep_command=values.get(
            "nep_command", "mpirun --bind-to none $HOME/src/GPUMD/src/nep"
        ).strip(),
        gpumd_command=values.get(
            "gpumd_command", "mpirun -np 1 --bind-to none $HOME/src/GPUMD/src/gpumd"
        ).strip(),
    )


def parse_validation(
    values: Mapping[str, str], gpumd_values: Mapping[str, str]
) -> ValidationConfig:
    """Parse validation enablement and canonical model identity aliases."""

    model_run_id = _resolve_alias(
        values,
        "model_run_id",
        gpumd_values,
        "model_run_id",
        "validation.model_run_id",
    )
    return ValidationConfig(
        enabled=_parse_bool(gpumd_values.get("enabled", "true"), "gpumd.enabled"),
        model_run_id=_optional_string(model_run_id),
    )


def _parse_training_sweep(
    train_values: Mapping[str, str],
    sweep_values: Mapping[str, str],
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Parse the canonical, typed training sweep declaration."""

    declarations: dict[str, Any] = {
        name: _parse_sweep_options(name, raw) for name, raw in sweep_values.items()
    }
    compact = train_values.get("sweep")
    if compact:
        try:
            decoded = json.loads(compact)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                "train_nep.sweep must be a JSON object of scientific field candidates"
            ) from exc
        if not isinstance(decoded, dict):
            raise ConfigurationError("train_nep.sweep must be a JSON object")
        for name, values in decoded.items():
            if name in declarations:
                raise ConfigurationError(f"Training sweep field is declared twice: {name}")
            declarations[str(name)] = values

    for key, raw in train_values.items():
        if not key.startswith("sweep_") or key == "sweep":
            continue
        name = key.removeprefix("sweep_")
        if name in declarations:
            raise ConfigurationError(f"Training sweep field is declared twice: {name}")
        declarations[name] = _parse_sweep_options(name, raw)

    unknown = sorted(set(declarations) - set(_SWEEP_FIELDS))
    if unknown:
        raise ConfigurationError(
            "Training sweeps may vary only scientific NEP settings: " + ", ".join(unknown)
        )

    base = NepTrainingConfig()
    normalized: list[tuple[str, tuple[Any, ...]]] = []
    for name in _SWEEP_FIELDS:
        if name not in declarations:
            continue
        raw_values = declarations[name]
        if not isinstance(raw_values, (list, tuple)) or not raw_values:
            raise ConfigurationError(f"training sweep {name} must contain candidates")
        normalized.append(
            (
                name,
                tuple(_parse_sweep_value(name, value, getattr(base, name)) for value in raw_values),
            )
        )
    return tuple(normalized)


def _parse_sweep_options(name: str, raw: str) -> list[Any]:
    text = raw.strip()
    if not text:
        raise ConfigurationError(f"training sweep {name} must not be blank")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, list):
        return list(decoded)
    if "|" in text:
        return [item.strip() for item in text.split("|") if item.strip()]
    if name in _SWEEP_TUPLE_FIELDS:
        return [text]
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_sweep_value(name: str, value: Any, current: Any) -> Any:
    if name in _SWEEP_TUPLE_FIELDS:
        if isinstance(value, (list, tuple)):
            items = tuple(value)
        elif isinstance(value, str):
            items = tuple(item for item in value.replace(",", " ").split() if item)
        else:
            raise ConfigurationError(f"training sweep {name} contains an invalid candidate")
        if name == "weights":
            try:
                return tuple(float(item) for item in items)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"training sweep {name} must contain numbers") from exc
        return tuple(str(item) for item in items)
    try:
        if isinstance(current, bool):
            if not isinstance(value, bool):
                raise ValueError
            return value
        if isinstance(current, int):
            return int(value)
        if isinstance(current, float):
            return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"training sweep {name} has an invalid candidate") from exc
    return value


def _parse_int(value: str, name: str) -> int:
    try:
        return int(value.strip())
    except (AttributeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _parse_float(value: str, name: str) -> float:
    try:
        return float(value.strip())
    except (AttributeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a float") from exc


def _parse_optional_float(value: str, name: str) -> float | None:
    if not value.strip():
        return None
    return _parse_float(value, name)


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "yes", "true", "on"}:
        return True
    if normalized in {"0", "no", "false", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


def _parse_list(value: str, name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if value.strip() and not items:
        raise ConfigurationError(f"{name} must be a comma-separated list")
    return items


def _parse_symbols(value: str, name: str) -> tuple[str, ...]:
    return _parse_list(value, name)


def _parse_species_pairs(value: str, name: str) -> tuple[tuple[str, str], ...]:
    if not value.strip():
        return ()
    pairs: list[tuple[str, str]] = []
    for item in value.replace(";", ",").split(","):
        parts = tuple(part.strip() for part in item.split("->"))
        if len(parts) != 2 or not all(parts):
            raise ConfigurationError(f"{name} must use source->target pairs")
        pairs.append((parts[0], parts[1]))
    return tuple(pairs)


def _parse_interstitial_sites(value: str, name: str) -> tuple[Any, ...]:
    if not value.strip():
        return ()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{name} must be JSON fractional-coordinate sites") from exc
    if not isinstance(decoded, list):
        raise ConfigurationError(f"{name} must be a JSON list of sites")
    result: list[Any] = []
    for site in decoded:
        if isinstance(site, Mapping):
            coordinates = site.get("fractional", site.get("position"))
        else:
            coordinates = site
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3:
            raise ConfigurationError(f"{name} sites must contain three coordinates")
        try:
            result.append(tuple(float(value) for value in coordinates))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{name} sites must contain numeric coordinates") from exc
    return tuple(result)


def _parse_miller_indices(value: str, name: str) -> tuple[tuple[int, int, int], ...]:
    """Parse ``h,k,l;h,k,l`` or a JSON list of Miller triples."""

    text = value.strip()
    if not text:
        return ()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if decoded is not None:
        raw_indices = decoded
    else:
        raw_indices = [item for item in text.replace("|", ";").split(";") if item.strip()]
    if not isinstance(raw_indices, (list, tuple)):
        raise ConfigurationError(f"{name} must contain Miller-index triples")
    result: list[tuple[int, int, int]] = []
    for raw_index in raw_indices:
        if isinstance(raw_index, str):
            values = tuple(item for item in raw_index.replace(",", " ").split() if item)
        elif isinstance(raw_index, (list, tuple)):
            values = tuple(raw_index)
        else:
            raise ConfigurationError(f"{name} must contain Miller-index triples")
        if len(values) != 3:
            raise ConfigurationError(f"{name} must contain Miller-index triples")
        try:
            index = tuple(int(item) for item in values)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{name} must contain integer Miller indices") from exc
        result.append(index)  # type: ignore[arg-type]
    return tuple(result)


def _parse_int_pair(value: str, name: str) -> tuple[int, int]:
    items = tuple(item for item in value.replace(",", " ").split() if item)
    if len(items) != 2:
        raise ConfigurationError(f"{name} must contain exactly two integers")
    try:
        return int(items[0]), int(items[1])
    except ValueError as exc:
        raise ConfigurationError(f"{name} must contain exactly two integers") from exc


def _parse_float_pair(value: str, name: str) -> tuple[float, float]:
    items = tuple(item for item in value.replace(",", " ").split() if item)
    if len(items) != 2:
        raise ConfigurationError(f"{name} must contain exactly two floats")
    try:
        return float(items[0]), float(items[1])
    except ValueError as exc:
        raise ConfigurationError(f"{name} must contain exactly two floats") from exc


def _parse_float_list(value: str, name: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    result: list[float] = []
    for item in value.replace(" ", ",").split(","):
        if item.strip():
            result.append(_parse_float(item, name))
    return tuple(result)


def _parse_tokens(value: str, name: str) -> tuple[str, ...]:
    tokens = tuple(value.replace(",", " ").split())
    if not tokens:
        raise ConfigurationError(f"{name} must not be empty")
    return tokens


def _parse_path(value: str, name: str) -> Path:
    if not value.strip():
        raise ConfigurationError(f"{name} must not be blank")
    return Path(value.strip())


def _optional_string(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _resolve_alias(
    primary_section: Mapping[str, str],
    primary_key: str,
    legacy_section: Mapping[str, str],
    legacy_key: str,
    public_name: str,
) -> str:
    primary = primary_section.get(primary_key)
    legacy = legacy_section.get(legacy_key)
    if primary is not None and legacy is not None:
        raise ConfigurationError(
            f"{public_name} has both canonical and legacy keys; keep one source"
        )
    return primary if primary is not None else (legacy if legacy is not None else "")
