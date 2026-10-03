"""Canonical project-config loading and the temporary legacy adapter."""

from configparser import ConfigParser
import json
from pathlib import Path
from typing import Any, Mapping

from nepflow.errors import ConfigurationError

from .models import (
    CONFIG_SCHEMA_VERSION,
    CompositionConfig,
    DftRecoveryConfig,
    GenerationConfig,
    HpcConfig,
    MaterialsProjectConfig,
    NepConfig,
    NepTrainingConfig,
    NepflowConfig,
    PathsConfig,
    ProjectConfig,
    SelectionConfig,
    SlurmConfig,
    ValidationConfig,
    VaspConfig,
)
from .validation import validate_config


CANONICAL_CONFIG_NAME = "project.config"

_ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "project": frozenset({"name", "description", "status", "random_seed", "schema_version", "config_version"}),
    "paths": frozenset({"structures_path", "vasp_path", "nep_path", "gpumd_path", "reports_path"}),
    "materialsproject": frozenset({"api_key"}),
    "composition": frozenset({
        "elements",
        "gas_elements",
        "gaselements",
        "composition_step",
        "include_pure_elements",
        "include_binaries",
        "include_ternaries",
    }),
    "generation": frozenset({
        "elements",
        "crystal_structures",
        "target_n_atoms",
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
        "elastic_stress_enabled",
        "elastic_strain_amplitudes",
        "n_rattled",
        "n_vacancies",
        "n_interstitials",
        "n_gas_interstitials",
        "n_vacancy_interstitial",
        "n_gas_in_vacancy",
        "rattle_std",
        "rattle_std_min",
        "rattle_std_max",
        "rattle_d_min",
        "vacancy_min",
        "vacancy_max",
        "interstitial_d_min",
        "interstitial_min",
        "interstitial_max",
        "gas_interstitial_d_min",
        "max_gas_occupancy",
    }),
    "selection": frozenset({
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
    }),
    "vasp": frozenset({"enabled", "kspacing", "kgamma"}),
    "nep": frozenset({"enabled"}),
    "train_nep": frozenset({
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
    }),
    "training_sweep": frozenset({
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
    }),
    "gpumd": frozenset({"enabled", "model_run_id"}),
    "validate": frozenset({"model_run_id"}),
    "slurm": frozenset({
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
    }),
    "hpc": frozenset({
        "cores_per_node",
        "gpus_per_node",
        "max_nodes",
        "scp_address",
        "vasp_command",
        "nep_command",
        "gpumd_command",
    }),
}


def canonical_config_path(project_dir: Path) -> Path:
    """Return the only supported project configuration path."""
    return Path(project_dir) / "config" / CANONICAL_CONFIG_NAME


def find_config_path(project_dir: Path, explicit_path: Path | None = None) -> Path:
    """Locate the canonical config deterministically and fail if it is absent."""
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

    project_section = _section(parser, "project")
    derived_name = _derived_project_name(path)
    schema_version = _parse_int(
        project_section.get("schema_version", str(CONFIG_SCHEMA_VERSION)),
        "project.schema_version",
    )
    config_version = _parse_int(
        project_section.get("config_version", str(schema_version)),
        "project.config_version",
    )
    project = ProjectConfig(
        name=project_section.get("name", project_name or derived_name).strip(),
        description=project_section.get("description", "").strip(),
        status=project_section.get("status", "initialized").strip(),
        random_seed=_parse_int(
            project_section.get("random_seed", "42"), "project.random_seed"
        ),
        config_version=config_version,
    )

    paths_section = _section(parser, "paths")
    paths = PathsConfig(
        structures_path=_parse_path(paths_section.get("structures_path", "structures"), "paths.structures_path"),
        vasp_path=_parse_path(paths_section.get("vasp_path", "vasp"), "paths.vasp_path"),
        nep_path=_parse_path(paths_section.get("nep_path", "nep"), "paths.nep_path"),
        gpumd_path=_parse_path(paths_section.get("gpumd_path", "gpumd"), "paths.gpumd_path"),
        reports_path=_parse_path(paths_section.get("reports_path", "reports"), "paths.reports_path"),
    )

    materials_section = _section(parser, "materialsproject")
    materials_project = MaterialsProjectConfig(
        api_key=_optional_string(materials_section.get("api_key"))
    )

    composition_section = _section(parser, "composition")
    generation_section = _section(parser, "generation")
    elements = _parse_symbols(
        _resolve_alias(
            composition_section,
            "elements",
            generation_section,
            "elements",
            "composition.elements",
        ),
        "composition.elements",
    )
    gas_elements = _parse_symbols(
        _resolve_alias(
            composition_section,
            "gas_elements",
            composition_section,
            "gaselements",
            "composition.gas_elements",
        ),
        "composition.gas_elements",
    )
    composition = CompositionConfig(
        elements=elements,
        gas_elements=gas_elements,
        composition_step=_parse_float(
            composition_section.get("composition_step", "0.125"),
            "composition.composition_step",
        ),
        include_pure_elements=_parse_bool(
            composition_section.get("include_pure_elements", "true"),
            "composition.include_pure_elements",
        ),
        include_binaries=_parse_bool(
            composition_section.get("include_binaries", "true"),
            "composition.include_binaries",
        ),
        include_ternaries=_parse_bool(
            composition_section.get("include_ternaries", "true"),
            "composition.include_ternaries",
        ),
    )

    rattle_std = _parse_float(
        generation_section.get("rattle_std", "0.03"), "generation.rattle_std"
    )
    generation = GenerationConfig(
        crystal_structures=_parse_list(
            generation_section.get("crystal_structures", ""),
            "generation.crystal_structures",
        ),
        target_n_atoms=_parse_int(
            generation_section.get("target_n_atoms", "128"), "generation.target_n_atoms"
        ),
        n_workers=_parse_int(generation_section.get("n_workers", "0"), "generation.n_workers"),
        use_materials_project=_parse_bool(generation_section.get("use_materials_project", "true"), "generation.use_materials_project"),
        use_random_solid_solution=_parse_bool(generation_section.get("use_random_solid_solution", "true"), "generation.use_random_solid_solution"),
        use_sqs=_parse_bool(generation_section.get("use_sqs", "true"), "generation.use_sqs"),
        use_segregated=_parse_bool(generation_section.get("use_segregated", "true"), "generation.use_segregated"),
        use_liquid=_parse_bool(generation_section.get("use_liquid", "false"), "generation.use_liquid"),
        n_random_solid_solution=_parse_int(generation_section.get("n_random_solid_solution", "3"), "generation.n_random_solid_solution"),
        n_sqs=_parse_int(generation_section.get("n_sqs", "1"), "generation.n_sqs"),
        n_segregated=_parse_int(generation_section.get("n_segregated", "3"), "generation.n_segregated"),
        n_liquid_configurations=_parse_int(generation_section.get("n_liquid_configurations", "2"), "generation.n_liquid_configurations"),
        n_liquid_snapshots=_parse_int(generation_section.get("n_liquid_snapshots", "5"), "generation.n_liquid_snapshots"),
        liquid_temperature=_parse_float(generation_section.get("liquid_temperature", "3000"), "generation.liquid_temperature"),
        liquid_timestep_fs=_parse_float(generation_section.get("liquid_timestep_fs", "1.0"), "generation.liquid_timestep_fs"),
        liquid_equilibration_steps=_parse_int(generation_section.get("liquid_equilibration_steps", "200"), "generation.liquid_equilibration_steps"),
        liquid_steps_between_snapshots=_parse_int(generation_section.get("liquid_steps_between_snapshots", "100"), "generation.liquid_steps_between_snapshots"),
        liquid_friction=_parse_float(generation_section.get("liquid_friction", "0.02"), "generation.liquid_friction"),
        volume_scale_min=_parse_float(generation_section.get("volume_scale_min", "0.8"), "generation.volume_scale_min"),
        volume_scale_max=_parse_float(generation_section.get("volume_scale_max", "1.2"), "generation.volume_scale_max"),
        n_volume_points=_parse_int(generation_section.get("n_volume_points", "11"), "generation.n_volume_points"),
        elastic_stress_enabled=_parse_bool(generation_section.get("elastic_stress_enabled", "true"), "generation.elastic_stress_enabled"),
        elastic_strain_amplitudes=_parse_float_list(generation_section.get("elastic_strain_amplitudes", "-0.02,-0.01,-0.005,0.005,0.01,0.02"), "generation.elastic_strain_amplitudes"),
        n_rattled=_parse_int(generation_section.get("n_rattled", "10"), "generation.n_rattled"),
        n_vacancies=_parse_int(generation_section.get("n_vacancies", "10"), "generation.n_vacancies"),
        n_interstitials=_parse_int(generation_section.get("n_interstitials", "10"), "generation.n_interstitials"),
        n_gas_interstitials=_parse_int(generation_section.get("n_gas_interstitials", "10"), "generation.n_gas_interstitials"),
        n_vacancy_interstitial=_parse_int(generation_section.get("n_vacancy_interstitial", "10"), "generation.n_vacancy_interstitial"),
        n_gas_in_vacancy=_parse_int(generation_section.get("n_gas_in_vacancy", "10"), "generation.n_gas_in_vacancy"),
        rattle_std=rattle_std,
        rattle_std_min=_parse_float(generation_section.get("rattle_std_min", str(0.5 * rattle_std)), "generation.rattle_std_min"),
        rattle_std_max=_parse_float(generation_section.get("rattle_std_max", str(2.0 * rattle_std)), "generation.rattle_std_max"),
        rattle_d_min=_parse_float(generation_section.get("rattle_d_min", "1.5"), "generation.rattle_d_min"),
        vacancy_min=_parse_float(generation_section.get("vacancy_min", "0.0"), "generation.vacancy_min"),
        vacancy_max=_parse_float(generation_section.get("vacancy_max", "0.1"), "generation.vacancy_max"),
        interstitial_d_min=_parse_float(generation_section.get("interstitial_d_min", "1.65"), "generation.interstitial_d_min"),
        interstitial_min=_parse_float(generation_section.get("interstitial_min", "0.05"), "generation.interstitial_min"),
        interstitial_max=_parse_float(generation_section.get("interstitial_max", "0.1"), "generation.interstitial_max"),
        gas_interstitial_d_min=_parse_float(generation_section.get("gas_interstitial_d_min", "1.2"), "generation.gas_interstitial_d_min"),
        max_gas_occupancy=_parse_int(generation_section.get("max_gas_occupancy", "3"), "generation.max_gas_occupancy"),
    )

    selection_section = _section(parser, "selection")
    selection = SelectionConfig(
        nep_model_file=selection_section.get("nep_model_file", "nep89.txt").strip(),
        include_seed_structures=_parse_bool(selection_section.get("include_seed_structures", "false"), "selection.include_seed_structures"),
        include_single_element_elastic_stress_structures=_parse_bool(selection_section.get("include_single_element_elastic_stress_structures", "false"), "selection.include_single_element_elastic_stress_structures"),
        include_elastic_stress_structures=_parse_bool(selection_section.get("include_elastic_stress_structures", "false"), "selection.include_elastic_stress_structures"),
        composition_aware_fps=_parse_bool(selection_section.get("composition_aware_fps", "false"), "selection.composition_aware_fps"),
        composition_aware_fps_frontier_fraction=_parse_float(selection_section.get("composition_aware_fps_frontier_fraction", "0.1"), "selection.composition_aware_fps_frontier_fraction"),
        composition_aware_fps_ternary_weight=_parse_float(selection_section.get("composition_aware_fps_ternary_weight", "1.0"), "selection.composition_aware_fps_ternary_weight"),
        composition_aware_fps_adaptive_retries=_parse_int(selection_section.get("composition_aware_fps_adaptive_retries", "4"), "selection.composition_aware_fps_adaptive_retries"),
        composition_aware_fps_descriptor_floor_fraction=_parse_float(selection_section.get("composition_aware_fps_descriptor_floor_fraction", "0.95"), "selection.composition_aware_fps_descriptor_floor_fraction"),
        target_train_count=_parse_int(selection_section.get("target_train_count", "1000"), "selection.target_train_count"),
        target_test_count=_parse_int(selection_section.get("target_test_count", "200"), "selection.target_test_count"),
        target_tolerance=_parse_int(selection_section.get("target_tolerance", "50"), "selection.target_tolerance"),
        descriptor_type=selection_section.get("descriptor_type", "structure").strip().lower(),
        batch_size=_parse_int(selection_section.get("batch_size", "500"), "selection.batch_size"),
        max_search_iterations=_parse_int(selection_section.get("max_search_iterations", "30"), "selection.max_search_iterations"),
        test_pool_factor=_parse_float(selection_section.get("test_pool_factor", "0.5"), "selection.test_pool_factor"),
    )

    vasp_section = _section(parser, "vasp")
    vasp = VaspConfig(
        enabled=_parse_bool(vasp_section.get("enabled", "true"), "vasp.enabled"),
        kspacing=_parse_float(vasp_section.get("kspacing", "0.30"), "vasp.kspacing"),
        kgamma=_parse_bool(vasp_section.get("kgamma", "true"), "vasp.kgamma"),
    )
    nep_section = _section(parser, "nep")
    nep = NepConfig(enabled=_parse_bool(nep_section.get("enabled", "true"), "nep.enabled"))

    train_section = _section(parser, "train_nep")
    train_elements = elements + gas_elements
    weights = _parse_float_list(train_section.get("weights", ""), "train_nep.weights")
    if not weights and train_elements:
        weights = tuple(1.0 for _ in train_elements)
    train_nep = NepTrainingConfig(
        population=_parse_int(train_section.get("population", "50"), "train_nep.population"),
        batch=_parse_int(train_section.get("batch", "3000"), "train_nep.batch"),
        generations=_parse_int(train_section.get("generation", "250000"), "train_nep.generation"),
        charge_mode=_parse_int(train_section.get("charge_mode", "0"), "train_nep.charge_mode"),
        weights=weights,
        outer_zbl=_parse_float(train_section.get("outerzbl", "2.0"), "train_nep.outer_zbl"),
        cutoff=_parse_tokens(train_section.get("cutoff", "6 5"), "train_nep.cutoff"),
        n_max=_parse_tokens(train_section.get("n_max", "4 4"), "train_nep.n_max"),
        basis_size=_parse_tokens(train_section.get("basis_size", "8 8"), "train_nep.basis_size"),
        l_max=_parse_tokens(train_section.get("l_max", "4 2 1"), "train_nep.l_max"),
        neuron=_parse_tokens(train_section.get("neuron", "80"), "train_nep.neuron"),
        lambda_e=_parse_float(train_section.get("lambda_e", "1.0"), "train_nep.lambda_e"),
        lambda_f=_parse_float(train_section.get("lambda_f", "1.0"), "train_nep.lambda_f"),
        lambda_v=_parse_float(train_section.get("lambda_v", "1.0"), "train_nep.lambda_v"),
        lambda_shear=_parse_float(train_section.get("lambda_shear", "1.0"), "train_nep.lambda_shear"),
        train_virial=_parse_bool(train_section.get("train_virial", "false"), "train_nep.train_virial"),
        allow_partial_dataset=_parse_bool(train_section.get("allow_partial_dataset", "false"), "train_nep.allow_partial_dataset"),
        max_resubmit=_parse_int(train_section.get("max_resubmit", "3"), "train_nep.max_resubmit"),
        sweep=_parse_training_sweep(parser, train_section),
    )

    slurm_section = _section(parser, "slurm")
    dft_recovery = DftRecoveryConfig(
        max_retry_level=_parse_int(slurm_section.get("max_retry_level", "100"), "slurm.max_retry_level"),
        vasp_walltime=slurm_section.get("vasp_walltime", "01:00:00").strip(),
    )
    slurm = SlurmConfig(
        enabled=_parse_bool(slurm_section.get("enabled", "false"), "slurm.enabled"),
        max_concurrent=_parse_int(slurm_section.get("max_concurrent", "20"), "slurm.max_concurrent"),
        walltime=slurm_section.get("walltime", "03:00:00").strip(),
        train_nep_walltime=_optional_string(slurm_section.get("train_nep_walltime")),
        poll_interval=_parse_int(slurm_section.get("poll_interval", "20"), "slurm.poll_interval"),
        gpumd_walltime=slurm_section.get("gpumd_walltime", "00:10:00").strip(),
        gpumd_nodes=_parse_int(slurm_section.get("gpumd_nodes", "1"), "slurm.gpumd_nodes"),
        gpumd_gpus=_parse_int(slurm_section.get("gpumd_gpus", "1"), "slurm.gpumd_gpus"),
        memory_poll_interval=_parse_int(slurm_section.get("memory_poll_interval", "10"), "slurm.memory_poll_interval"),
        memory_walltime=slurm_section.get("memory_walltime", "00:10:00").strip(),
    )

    hpc_section = _section(parser, "hpc")
    hpc = HpcConfig(
        cores_per_node=_parse_int(hpc_section.get("cores_per_node", "64"), "hpc.cores_per_node"),
        gpus_per_node=_parse_int(hpc_section.get("gpus_per_node", "4"), "hpc.gpus_per_node"),
        max_nodes=_parse_int(hpc_section.get("max_nodes", "16"), "hpc.max_nodes"),
        scp_address=hpc_section.get("scp_address", "").strip(),
        vasp_command=hpc_section.get("vasp_command", "mpirun -np {ntasks} vasp_std").strip(),
        nep_command=hpc_section.get("nep_command", "mpirun --bind-to none $HOME/src/GPUMD/src/nep").strip(),
        gpumd_command=hpc_section.get("gpumd_command", "mpirun -np 1 --bind-to none $HOME/src/GPUMD/src/gpumd").strip(),
    )

    validation_section = _section(parser, "validate")
    gpumd_section = _section(parser, "gpumd")
    model_run_id = _resolve_alias(
        validation_section,
        "model_run_id",
        gpumd_section,
        "model_run_id",
        "validation.model_run_id",
    )
    validation = ValidationConfig(
        enabled=_parse_bool(gpumd_section.get("enabled", "true"), "gpumd.enabled"),
        model_run_id=_optional_string(model_run_id),
    )

    config = NepflowConfig(
        schema_version=schema_version,
        project=project,
        paths=paths,
        materials_project=materials_project,
        composition=composition,
        generation=generation,
        selection=selection,
        vasp=vasp,
        dft_recovery=dft_recovery,
        nep=nep,
        train_nep=train_nep,
        validation=validation,
        hpc=hpc,
        slurm=slurm,
        source_path=path,
        raw_sections=frozenset(section.lower() for section in parser.sections()),
    )
    if require_scientific_fields and (
        not hpc_section.get("vasp_command", "").strip()
    ):
        raise ConfigurationError("Required configuration hpc.vasp_command is missing or blank")
    return validate_config(config, require_scientific_fields=require_scientific_fields)


def load_legacy_config(
    config_path: Path,
    *,
    project_name: str | None = None,
    require_scientific_fields: bool = False,
) -> ConfigParser:
    """Return a ConfigParser adapter produced from the validated typed root.

    This adapter exists only while Phase 4 migrates legacy stages. It is the
    sole place where typed values are rendered back into legacy section/key
    names; it never reads the file independently.
    """
    typed = load_config(
        config_path,
        project_name=project_name,
        require_scientific_fields=require_scientific_fields,
    )
    return to_legacy_config(typed)


def to_legacy_config(config: NepflowConfig) -> ConfigParser:
    """Render a typed root config for unmigrated ConfigParser consumers."""
    parser = ConfigParser(interpolation=None)

    # Loaded configs preserve their source sections so the temporary bridge
    # retains the partial-config behavior used by unmigrated stages. A root
    # assembled directly in code is complete and should render every section.
    present_sections = config.raw_sections or frozenset(_ALLOWED_KEYS)

    def add_section(name: str, values: Mapping[str, str]) -> None:
        if name in present_sections:
            parser[name] = dict(values)

    add_section("project", {
        "name": config.project.name,
        "description": config.project.description,
        "status": config.project.status,
        "random_seed": str(config.project.random_seed),
        "schema_version": str(config.schema_version),
        "config_version": str(config.project.config_version),
    })
    add_section("paths", {
        "structures_path": str(config.paths.structures_path),
        "vasp_path": str(config.paths.vasp_path),
        "nep_path": str(config.paths.nep_path),
        "gpumd_path": str(config.paths.gpumd_path),
        "reports_path": str(config.paths.reports_path),
    })
    material_values = {}
    if config.materials_project.api_key:
        material_values["api_key"] = config.materials_project.api_key
    add_section("materialsproject", material_values)
    add_section("composition", {
        "elements": ",".join(config.composition.elements),
        "gasElements": ",".join(config.composition.gas_elements),
        "composition_step": str(config.composition.composition_step),
        "include_pure_elements": _bool_text(config.composition.include_pure_elements),
        "include_binaries": _bool_text(config.composition.include_binaries),
        "include_ternaries": _bool_text(config.composition.include_ternaries),
    })
    add_section("generation", {
        "crystal_structures": ",".join(config.generation.crystal_structures),
        "target_n_atoms": str(config.generation.target_n_atoms),
        "n_workers": str(config.generation.n_workers),
        "use_materials_project": _bool_text(config.generation.use_materials_project),
        "use_random_solid_solution": _bool_text(config.generation.use_random_solid_solution),
        "use_sqs": _bool_text(config.generation.use_sqs),
        "use_segregated": _bool_text(config.generation.use_segregated),
        "use_liquid": _bool_text(config.generation.use_liquid),
        "n_random_solid_solution": str(config.generation.n_random_solid_solution),
        "n_sqs": str(config.generation.n_sqs),
        "n_segregated": str(config.generation.n_segregated),
        "n_liquid_configurations": str(config.generation.n_liquid_configurations),
        "n_liquid_snapshots": str(config.generation.n_liquid_snapshots),
        "liquid_temperature": str(config.generation.liquid_temperature),
        "liquid_timestep_fs": str(config.generation.liquid_timestep_fs),
        "liquid_equilibration_steps": str(config.generation.liquid_equilibration_steps),
        "liquid_steps_between_snapshots": str(config.generation.liquid_steps_between_snapshots),
        "liquid_friction": str(config.generation.liquid_friction),
        "volume_scale_min": str(config.generation.volume_scale_min),
        "volume_scale_max": str(config.generation.volume_scale_max),
        "n_volume_points": str(config.generation.n_volume_points),
        "elastic_stress_enabled": _bool_text(config.generation.elastic_stress_enabled),
        "elastic_strain_amplitudes": ",".join(map(str, config.generation.elastic_strain_amplitudes)),
        "n_rattled": str(config.generation.n_rattled),
        "n_vacancies": str(config.generation.n_vacancies),
        "n_interstitials": str(config.generation.n_interstitials),
        "n_gas_interstitials": str(config.generation.n_gas_interstitials),
        "n_vacancy_interstitial": str(config.generation.n_vacancy_interstitial),
        "n_gas_in_vacancy": str(config.generation.n_gas_in_vacancy),
        "rattle_std": str(config.generation.rattle_std),
        "rattle_std_min": str(config.generation.rattle_std_min),
        "rattle_std_max": str(config.generation.rattle_std_max),
        "rattle_d_min": str(config.generation.rattle_d_min),
        "vacancy_min": str(config.generation.vacancy_min),
        "vacancy_max": str(config.generation.vacancy_max),
        "interstitial_d_min": str(config.generation.interstitial_d_min),
        "interstitial_min": str(config.generation.interstitial_min),
        "interstitial_max": str(config.generation.interstitial_max),
        "gas_interstitial_d_min": str(config.generation.gas_interstitial_d_min),
        "max_gas_occupancy": str(config.generation.max_gas_occupancy),
    })
    add_section("selection", {
        "nep_model_file": config.selection.nep_model_file,
        "include_seed_structures": _bool_text(config.selection.include_seed_structures),
        "include_single_element_elastic_stress_structures": _bool_text(config.selection.include_single_element_elastic_stress_structures),
        "include_elastic_stress_structures": _bool_text(config.selection.include_elastic_stress_structures),
        "composition_aware_fps": _bool_text(config.selection.composition_aware_fps),
        "composition_aware_fps_frontier_fraction": str(config.selection.composition_aware_fps_frontier_fraction),
        "composition_aware_fps_ternary_weight": str(config.selection.composition_aware_fps_ternary_weight),
        "composition_aware_fps_adaptive_retries": str(config.selection.composition_aware_fps_adaptive_retries),
        "composition_aware_fps_descriptor_floor_fraction": str(config.selection.composition_aware_fps_descriptor_floor_fraction),
        "target_train_count": str(config.selection.target_train_count),
        "target_test_count": str(config.selection.target_test_count),
        "target_tolerance": str(config.selection.target_tolerance),
        "descriptor_type": config.selection.descriptor_type,
        "batch_size": str(config.selection.batch_size),
        "max_search_iterations": str(config.selection.max_search_iterations),
        "test_pool_factor": str(config.selection.test_pool_factor),
    })
    add_section("vasp", {
        "enabled": _bool_text(config.vasp.enabled),
        "kspacing": str(config.vasp.kspacing),
        "kgamma": _vasp_bool_text(config.vasp.kgamma),
    })
    add_section("nep", {"enabled": _bool_text(config.nep.enabled)})
    add_section("train_nep", {
        "population": str(config.train_nep.population),
        "batch": str(config.train_nep.batch),
        "generation": str(config.train_nep.generations),
        "cutoff": " ".join(config.train_nep.cutoff),
        "n_max": " ".join(config.train_nep.n_max),
        "basis_size": " ".join(config.train_nep.basis_size),
        "l_max": " ".join(config.train_nep.l_max),
        "neuron": " ".join(config.train_nep.neuron),
        "outerZBL": str(config.train_nep.outer_zbl),
        "charge_mode": str(config.train_nep.charge_mode),
        "weights": ",".join(map(str, config.train_nep.weights)),
        "lambda_e": str(config.train_nep.lambda_e),
        "lambda_f": str(config.train_nep.lambda_f),
        "lambda_v": str(config.train_nep.lambda_v),
        "lambda_shear": str(config.train_nep.lambda_shear),
        "train_virial": _bool_text(config.train_nep.train_virial),
        "allow_partial_dataset": _bool_text(config.train_nep.allow_partial_dataset),
        "max_resubmit": str(config.train_nep.max_resubmit),
    })
    if config.train_nep.sweep:
        add_section(
            "training_sweep",
            {
                name: "|".join(_format_sweep_value(value) for value in values)
                for name, values in config.train_nep.sweep
            },
        )
    gpumd_values = {"enabled": _bool_text(config.validation.enabled)}
    if "gpumd" in present_sections and config.validation.model_run_id:
        gpumd_values["model_run_id"] = config.validation.model_run_id
    add_section("gpumd", gpumd_values)
    if "validate" in present_sections and config.validation.model_run_id:
        add_section("validate", {"model_run_id": config.validation.model_run_id})
    add_section("slurm", {
        "enabled": _bool_text(config.slurm.enabled),
        "max_concurrent": str(config.slurm.max_concurrent),
        "walltime": config.slurm.walltime,
        "poll_interval": str(config.slurm.poll_interval),
        "max_retry_level": str(config.dft_recovery.max_retry_level),
        "vasp_walltime": config.dft_recovery.vasp_walltime,
        "gpumd_walltime": config.slurm.gpumd_walltime,
        "gpumd_nodes": str(config.slurm.gpumd_nodes),
        "gpumd_gpus": str(config.slurm.gpumd_gpus),
        "memory_poll_interval": str(config.slurm.memory_poll_interval),
        "memory_walltime": config.slurm.memory_walltime,
        **({"train_nep_walltime": config.slurm.train_nep_walltime} if config.slurm.train_nep_walltime else {}),
    })
    add_section("hpc", {
        "cores_per_node": str(config.hpc.cores_per_node),
        "gpus_per_node": str(config.hpc.gpus_per_node),
        "max_nodes": str(config.hpc.max_nodes),
        "scp_address": config.hpc.scp_address,
        "vasp_command": config.hpc.vasp_command,
        "nep_command": config.hpc.nep_command,
        "gpumd_command": config.hpc.gpumd_command,
    })
    return parser


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
_SWEEP_TUPLE_FIELDS = frozenset(
    {"weights", "cutoff", "n_max", "basis_size", "l_max", "neuron"}
)


def _parse_training_sweep(
    parser: ConfigParser,
    train_section: Mapping[str, str],
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Parse the canonical, typed sweep declaration.

    The preferred form is a ``[training_sweep]`` section whose values use
    ``|`` between ordered candidates, for example ``lambda_f = 1.0|2.0`` or
    ``cutoff = 6 5|7 5``.  A JSON object in ``train_nep.sweep`` is accepted as
    a compact equivalent for programmatic/project generators.
    """

    section_values = _section(parser, "training_sweep")
    declarations: dict[str, Any] = {}
    for name, raw in section_values.items():
        declarations[name] = _parse_sweep_options(name, raw)

    compact = train_section.get("sweep")
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
                raise ConfigurationError(
                    f"Training sweep field is declared twice: {name}"
                )
            declarations[str(name)] = values

    for key, raw in train_section.items():
        if not key.startswith("sweep_") or key == "sweep":
            continue
        name = key.removeprefix("sweep_")
        if name in declarations:
            raise ConfigurationError(f"Training sweep field is declared twice: {name}")
        declarations[name] = _parse_sweep_options(name, raw)

    unknown = sorted(set(declarations) - set(_SWEEP_FIELDS))
    if unknown:
        raise ConfigurationError(
            "Training sweeps may vary only scientific NEP settings: "
            + ", ".join(unknown)
        )

    base = NepTrainingConfig()
    normalized: list[tuple[str, tuple[Any, ...]]] = []
    for name in _SWEEP_FIELDS:
        if name not in declarations:
            continue
        raw_values = declarations[name]
        if not isinstance(raw_values, (list, tuple)) or not raw_values:
            raise ConfigurationError(f"training sweep {name} must contain candidates")
        values: list[Any] = []
        for value in raw_values:
            values.append(_parse_sweep_value(name, value, getattr(base, name)))
        normalized.append((name, tuple(values)))
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
    # Tuple-valued fields need an explicit candidate separator.  For scalar
    # fields a comma-separated declaration is the convenient INI form.
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


def _format_sweep_value(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return " ".join(str(item) for item in value)
    return str(value)


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
                raise ConfigurationError(
                    f"Unknown configuration key {section}.{key}"
                )


def _section(parser: ConfigParser, name: str) -> dict[str, str]:
    actual_name = next(
        (section for section in parser.sections() if section.lower() == name),
        None,
    )
    if actual_name is None:
        return {}
    return {key.lower(): value.strip() for key, value in parser.items(actual_name)}


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


def _parse_float_list(value: str, name: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    result: list[float] = []
    for item in value.replace(" ", ",").split(","):
        if not item.strip():
            continue
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


def _derived_project_name(path: Path) -> str:
    project_dir = path.parent.parent.name
    return project_dir.removeprefix("project_") or project_dir


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _vasp_bool_text(value: bool) -> str:
    return ".TRUE." if value else ".FALSE."
