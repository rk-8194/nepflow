"""Compose workflow stages and their external services at the application boundary."""

from __future__ import annotations

import logging
from collections.abc import Callable

from nepflow.cli_wizard import CONFIG_PROMPTS, ConfigWizard
from nepflow.errors import ValidationError
from nepflow.hpc.process import ProcessRunner
from nepflow.hpc.slurm import SlurmScheduler
from nepflow.stages.dft import DftStage
from nepflow.stages.generation import GenerationStage
from nepflow.stages.generation.debug import run_debug
from nepflow.stages.generation.generators.base import ConfigurationalGenerator
from nepflow.stages.generation.stage import PerturbationCoordinator
from nepflow.stages.selection import SelectionStage
from nepflow.stages.training import TrainingStage
from nepflow.stages.validation import ValidationStage
from nepflow.workflow import (
    StageContext,
    StageRegistry,
    StageRunResult,
    StageRunState,
    WorkflowStage,
)


def build_generation_stage(
    context: StageContext,
    *,
    logger: logging.Logger,
) -> GenerationStage:
    """Build the generation stage from the validated application config."""

    config = context.config
    if config is None:
        raise ValidationError("Generation requires the validated typed project config")
    if context.debug:
        return GenerationStage(
            generators=(),
            coordinator=None,
            state_store=context.state_store,
            debug_runner=run_debug,
            logger=logger,
        )

    generators = _build_generators(context)
    coordinator = _build_perturbation_coordinator(context)
    return GenerationStage(
        generators=generators,
        coordinator=coordinator,
        state_store=context.state_store,
        debug_runner=run_debug,
        logger=logger,
    )


def _build_generators(context: StageContext) -> list[tuple[str, ConfigurationalGenerator]]:
    """Build enabled configurational generators from typed settings."""

    config = context.config
    assert config is not None
    generation = config.generation
    configured: list[tuple[str, ConfigurationalGenerator]] = []
    if generation.use_materials_project:
        configured.append(_build_materials_project_generator(context))
    if generation.use_random_solid_solution:
        configured.append(_build_random_solid_solution_generator(context))
    if generation.use_sqs:
        configured.append(_build_sqs_generator(context))
    if generation.use_segregated:
        configured.append(_build_segregated_generator(context))
    return configured


def _build_materials_project_generator(
    context: StageContext,
) -> tuple[str, ConfigurationalGenerator]:
    from nepflow.stages.generation.generators import (
        MaterialsProjectGenerator,
        build_materials_project_fetcher,
    )

    config = context.config
    assert config is not None
    try:
        fetcher = build_materials_project_fetcher(api_key=config.materials_project.api_key)
        generator = MaterialsProjectGenerator(
            fetcher,
            max_per_composition=5,
            gas_elements=list(config.composition.gas_elements),
        )
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("Cannot initialise the enabled Materials Project generator") from exc
    return "MaterialsProject", generator


def _build_random_solid_solution_generator(
    context: StageContext,
) -> tuple[str, ConfigurationalGenerator]:
    from nepflow.stages.generation.generators import RandomSolidSolutionGenerator

    config = context.config
    assert config is not None
    generator = RandomSolidSolutionGenerator(
        n_structures=config.generation.n_random_solid_solution,
        random_seed=config.project.random_seed,
        composition_tolerance=config.generation.composition_tolerance,
    )
    return "RandomSolidSolution", generator


def _build_sqs_generator(context: StageContext) -> tuple[str, ConfigurationalGenerator]:
    from nepflow.stages.generation.generators import SQSGenerator

    config = context.config
    assert config is not None
    generator = SQSGenerator(
        n_structures=config.generation.n_sqs,
        random_seed=config.project.random_seed,
        composition_tolerance=config.generation.composition_tolerance,
    )
    return "SQS", generator


def _build_segregated_generator(context: StageContext) -> tuple[str, ConfigurationalGenerator]:
    from nepflow.stages.generation.generators import SegregatedGenerator

    config = context.config
    assert config is not None
    generator = SegregatedGenerator(
        n_structures=config.generation.n_segregated,
        random_seed=config.project.random_seed,
        composition_tolerance=config.generation.composition_tolerance,
    )
    return "Segregated", generator


def _build_perturbation_coordinator(context: StageContext) -> PerturbationCoordinator:
    """Build the configured perturbation coordinator."""

    config = context.config
    assert config is not None
    composition = config.composition
    generation = config.generation
    from nepflow.stages.generation.perturbations import (
        PerturbationCoordinator as ConcreteCoordinator,
    )

    return ConcreteCoordinator(
        rattle_std=generation.rattle_std,
        rattle_std_min=generation.rattle_std_min,
        rattle_std_max=generation.rattle_std_max,
        rattle_d_min=generation.rattle_d_min,
        vacancy_range=(generation.vacancy_min, generation.vacancy_max),
        interstitial_range=(generation.interstitial_min, generation.interstitial_max),
        interstitial_d_min=generation.interstitial_d_min,
        defect_defect_d_min=generation.defect_defect_d_min,
        periodic_image_d_min=generation.periodic_image_d_min,
        interstitial_max_attempts=generation.interstitial_max_attempts,
        volume_scale_range=(generation.volume_scale_min, generation.volume_scale_max),
        n_volume_points=generation.n_volume_points,
        volume_sources=generation.volume_sources,
        target_n_atoms=generation.target_n_atoms,
        random_seed=config.project.random_seed,
        gas_elements=list(composition.gas_elements),
        gas_interstitial_d_min=generation.gas_interstitial_d_min,
        max_gas_occupancy=generation.max_gas_occupancy,
        vacancy_species=list(generation.vacancy_species),
        substitution_pairs=list(generation.substitution_pairs),
        antisite_pairs=list(generation.antisite_pairs),
        substitution_range=(generation.substitution_min, generation.substitution_max),
        antisite_range=(generation.antisite_min, generation.antisite_max),
        interstitial_sites=list(generation.interstitial_sites),
        crystallographic_interstitial_sites=list(generation.crystallographic_interstitial_sites),
        elastic_stress_enabled=generation.elastic_stress_enabled,
        elastic_strain_amplitudes=list(generation.elastic_strain_amplitudes),
        elastic_sources=generation.elastic_sources,
        liquid_enabled=generation.use_liquid,
        liquid_temperature_k=generation.liquid_temperature,
        liquid_timestep_fs=generation.liquid_timestep_fs,
        liquid_equilibration_steps=generation.liquid_equilibration_steps,
        liquid_steps_between_snapshots=generation.liquid_steps_between_snapshots,
        liquid_friction=generation.liquid_friction,
        liquid_sources=generation.liquid_sources,
        rattle_sources=generation.rattle_sources,
        vacancy_sources=generation.vacancy_sources,
        interstitial_sources=generation.interstitial_sources,
        gas_interstitial_sources=generation.gas_interstitial_sources,
        substitution_sources=generation.substitution_sources,
        antisite_sources=generation.antisite_sources,
        vacancy_interstitial_sources=generation.vacancy_interstitial_sources,
        gas_in_vacancy_sources=generation.gas_in_vacancy_sources,
    )


def offer_project_upload(
    context: StageContext,
    *,
    process_runner: ProcessRunner,
    confirm: Callable[[str], str],
) -> None:
    """Offer the local seeds-only result to the configured remote project."""

    config = context.config
    scp_address = "" if config is None else config.hpc.scp_address
    if not scp_address:
        return
    answer = confirm("Upload project to remote NEPFlow folder? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        return
    process_runner.run(
        [
            "scp",
            "-r",
            str(context.project_dir),
            f"{scp_address}/projects/{context.project_dir.name}",
        ],
        check=True,
        capture_output=False,
    )


def compose_stage_registry(
    *,
    logger: logging.Logger,
    scheduler: SlurmScheduler,
    offer_upload: Callable[[StageContext], None] | None = None,
) -> StageRegistry:
    """Compose workflow stages behind the application composition boundary."""

    registry = StageRegistry()
    _register_initialization(registry)
    _register_generation(
        registry,
        logger=logger,
        offer_upload=offer_upload,
    )
    _register_selection(registry)
    _register_scheduler_stages(registry, scheduler=scheduler)
    return registry


def _register_initialization(registry: StageRegistry) -> None:
    def run_initialization(context: StageContext) -> None:
        from nepflow.workflow.initialization import ProjectCreationService

        service = ProjectCreationService(
            context.project_name,
            context.config_file,
            context.state_file,
            context.project_dir,
        )
        prompt_values = None
        if service.requires_prompt_values():
            wizard = ConfigWizard(context.project_name)
            wizard.present_header()
            prompt_values = wizard.collect_prompt_values(CONFIG_PROMPTS)
        service.run(prompt_values=prompt_values)

    registry.register(WorkflowStage.INIT, run_initialization)


def _register_generation(
    registry: StageRegistry,
    *,
    logger: logging.Logger,
    offer_upload: Callable[[StageContext], None] | None,
) -> None:
    def run_generation(context: StageContext) -> StageRunResult:
        result = build_generation_stage(context, logger=logger).run(context)
        if context.seeds_only and result.base_structures and offer_upload is not None:
            offer_upload(context)
        return StageRunResult(
            stage=WorkflowStage.GENERATE,
            status=StageRunState.COMPLETED,
            advanced_to=WorkflowStage.SELECT,
            completed=result.completed,
            message=result.status,
        )

    registry.register(WorkflowStage.GENERATE, run_generation)


def _register_selection(registry: StageRegistry) -> None:
    def run_selection(context: StageContext) -> StageRunResult:
        SelectionStage(context=context).run()
        return StageRunResult(
            stage=WorkflowStage.SELECT,
            status=StageRunState.COMPLETED,
            advanced_to=WorkflowStage.RUN_VASP,
            completed=True,
        )

    registry.register(WorkflowStage.SELECT, run_selection)


def _register_scheduler_stages(
    registry: StageRegistry,
    *,
    scheduler: SlurmScheduler,
) -> None:
    def run_dft(context: StageContext) -> StageRunResult:
        return DftStage(scheduler=scheduler).run(context).as_workflow_result()

    def run_validation(context: StageContext) -> StageRunResult:
        return ValidationStage(context=context, scheduler=scheduler).run().as_workflow_result()

    registry.register(WorkflowStage.RUN_VASP, run_dft)
    registry.register(
        WorkflowStage.TRAIN_NEP,
        lambda context: TrainingStage(scheduler=scheduler).run(context),
    )
    registry.register(WorkflowStage.VALIDATE, run_validation)
