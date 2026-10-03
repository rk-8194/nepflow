#!/usr/bin/env python3
"""
NEPFlow: HPC workflow system for atomistic datasets.

Entry point for the workflow controller.
"""

import sys
import argparse
import logging
import os
import time
from pathlib import Path

# Check required libraries before doing anything else
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "ase": "ase",
    "hiphive": "hiphive",
    "mp_api": "mp-api",
    "pymatgen": "pymatgen",
    "icet": "icet",
    "NepTrainKit": "NepTrainKit",
}

_missing = []
for _mod, _pkg in REQUIRED_PACKAGES.items():
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print(
        f"Missing required packages: {', '.join(_missing)}\n"
        f"Install with:  pip install {' '.join(_missing)}",
        file=sys.stderr,
    )
    sys.exit(1)

# Package root used for the project-output default below.
_SOURCE_ROOT = Path(__file__).resolve().parents[1]

from nepflow.errors import SchedulerError, StateError, ValidationError
from nepflow.cli_wizard import CONFIG_PROMPTS, ConfigWizard
from nepflow.hpc.process import ProcessError, ProcessRunner
from nepflow.hpc.slurm import SlurmScheduler
from nepflow.logging import configure_logging
from nepflow.config.loader import canonical_config_path
from nepflow.reporting import WorkflowStatusPresenter, summarize_legacy_vasp_jobs
from nepflow.stages.dft import DftStage
from nepflow.stages.generation import GenerationStage
from nepflow.stages.generation.debug import run_debug
from nepflow.stages.selection import SelectionStage
from nepflow.stages.training import TrainingStage
from nepflow.stages.validation import ValidationStage
from nepflow.workflow import (
    StageContext,
    StageRegistry,
    StageRunResult,
    StageRunState,
    SelfResubmitExit,
    WorkflowController,
    WorkflowStage,
    resolve_resubmit_command as canonical_resolve_resubmit_command,
)

logger = logging.getLogger("nepflow")
process_runner = ProcessRunner(logger=logger)
scheduler = SlurmScheduler(process_runner=process_runner)


def _build_generation_stage(context: StageContext) -> GenerationStage:
    """Compose legacy scientific generators behind the canonical stage seam."""

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

    # These imports stay at the application composition root.  GenerationStage
    # itself owns only orchestration and provenance.
    from nepflow.stages.generation.generators import (
        MaterialsProjectGenerator,
        RandomSolidSolutionGenerator,
        SQSGenerator,
        SegregatedGenerator,
        build_materials_project_fetcher,
    )
    from nepflow.stages.generation.perturbations import PerturbationCoordinator

    composition = config.composition
    generation = config.generation
    configured_generators = []
    if generation.use_materials_project:
        try:
            fetcher = build_materials_project_fetcher(
                api_key=config.materials_project.api_key,
            )
            configured_generators.append(
                (
                    "MaterialsProject",
                    MaterialsProjectGenerator(
                        fetcher,
                        max_per_composition=5,
                        gas_elements=list(composition.gas_elements),
                    ),
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Cannot initialise the enabled Materials Project generator"
            ) from exc

    if generation.use_random_solid_solution:
        configured_generators.append(
            (
                "RandomSolidSolution",
                RandomSolidSolutionGenerator(
                    n_structures=generation.n_random_solid_solution,
                    random_seed=config.project.random_seed,
                ),
            )
        )
    if generation.use_sqs:
        configured_generators.append(
            (
                "SQS",
                SQSGenerator(
                    n_structures=generation.n_sqs,
                    random_seed=config.project.random_seed,
                ),
            )
        )
    if generation.use_segregated:
        configured_generators.append(
            (
                "Segregated",
                SegregatedGenerator(
                    n_structures=generation.n_segregated,
                    random_seed=config.project.random_seed,
                ),
            )
        )

    coordinator = None
    if not context.debug:
        coordinator = PerturbationCoordinator(
            rattle_std=generation.rattle_std,
            rattle_std_min=generation.rattle_std_min,
            rattle_std_max=generation.rattle_std_max,
            rattle_d_min=generation.rattle_d_min,
            vacancy_range=(generation.vacancy_min, generation.vacancy_max),
            interstitial_range=(generation.interstitial_min, generation.interstitial_max),
            interstitial_d_min=generation.interstitial_d_min,
            volume_scale_range=(generation.volume_scale_min, generation.volume_scale_max),
            n_volume_points=generation.n_volume_points,
            target_n_atoms=generation.target_n_atoms,
            random_seed=config.project.random_seed,
            gas_elements=list(composition.gas_elements),
            gas_interstitial_d_min=generation.gas_interstitial_d_min,
            max_gas_occupancy=generation.max_gas_occupancy,
            elastic_stress_enabled=generation.elastic_stress_enabled,
            elastic_strain_amplitudes=list(generation.elastic_strain_amplitudes),
            liquid_enabled=generation.use_liquid,
            liquid_temperature_k=generation.liquid_temperature,
            liquid_timestep_fs=generation.liquid_timestep_fs,
            liquid_equilibration_steps=generation.liquid_equilibration_steps,
            liquid_steps_between_snapshots=generation.liquid_steps_between_snapshots,
            liquid_friction=generation.liquid_friction,
        )

    return GenerationStage(
        generators=configured_generators,
        coordinator=coordinator,
        state_store=context.state_store,
        debug_runner=run_debug,
        logger=logger,
    )


def _offer_project_upload(context: StageContext) -> None:
    """Offer the local seeds-only result to the configured remote project."""

    config = context.config
    scp_address = "" if config is None else config.hpc.scp_address
    if not scp_address:
        return
    answer = input("Upload project to remote NEPFlow folder? [y/N]: ").strip().lower()
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


def compose_stage_registry() -> StageRegistry:
    """Compose canonical workflow stages behind the application seam."""

    registry = StageRegistry()

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

    registry.register(
        WorkflowStage.INIT,
        run_initialization,
    )

    def run_generation(context: StageContext) -> StageRunResult:
        result = _build_generation_stage(context).run(context)
        if context.seeds_only and result.base_structures:
            _offer_project_upload(context)
        return StageRunResult(
            stage=WorkflowStage.GENERATE,
            status=StageRunState.COMPLETED,
            advanced_to=WorkflowStage.SELECT,
            completed=result.completed,
            message=result.status,
        )

    registry.register(WorkflowStage.GENERATE, run_generation)
    def run_selection(context: StageContext) -> StageRunResult:
        SelectionStage(context=context).run()
        return StageRunResult(
            stage=WorkflowStage.SELECT,
            status=StageRunState.COMPLETED,
            advanced_to=WorkflowStage.RUN_VASP,
            completed=True,
        )

    registry.register(WorkflowStage.SELECT, run_selection)
    def run_dft(context: StageContext) -> StageRunResult:
        return DftStage().run(context).as_workflow_result()

    registry.register(WorkflowStage.RUN_VASP, run_dft)
    registry.register(
        WorkflowStage.TRAIN_NEP,
        lambda context: TrainingStage().run(context),
    )

    def run_validation(context: StageContext) -> StageRunResult:
        return ValidationStage(context=context, scheduler=scheduler).run().as_workflow_result()

    registry.register(WorkflowStage.VALIDATE, run_validation)
    return registry


def _legacy_status_details(project_dir: Path, stage: WorkflowStage) -> dict:
    """Adapt legacy VASP status reporting to the package CLI presenter."""

    from nepflow.dft.vasp.registry import read_status

    return summarize_legacy_vasp_jobs(project_dir, stage, read_status)


def _resolve_project_config_path(project_name: str, output_dir: Path) -> Path:
    """Return the canonical project.config path used by the workflow."""
    project_dir = Path(output_dir) / f"project_{project_name}"
    return canonical_config_path(project_dir)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NEPFlow: HPC workflow orchestrator for VASP->NEP->GPUMD pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nepflow --project myproject
  nepflow --project myproject --local
  nepflow --project myproject --debug
        """,
    )

    # Global options
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project name (determines config, state, and output paths)",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize a new project (required on first run)",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Open the project config file in Vim and exit",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["init", "generate", "select", "run_vasp", "train_nep", "validate"],
        default=None,
        help="Set the workflow stage (overrides .project file)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Fetch base structures locally (seeds only, no perturbations), "
        "then stop. Use on a machine with web access before transferring "
        "to HPC.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (no external dependencies required)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SOURCE_ROOT.parent / "projects",
        help="Output base directory (default: ./projects)",
    )

    return parser


def _get_slurm_walltime_info() -> tuple[int | None, str]:
    """
    Get SLURM job walltime information.

    Returns:
        (remaining_seconds, source_description) or (None, "not_in_slurm") if not under SLURM
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    logger.debug("SLURM_JOB_ID env var: %s", slurm_job_id or "(not set)")

    if not slurm_job_id:
        return None, "not_in_slurm"

    start_time = time.time()

    # Try SLURM_JOB_END_TIME first (Unix timestamp, most accurate)
    slurm_job_end_time = os.environ.get("SLURM_JOB_END_TIME")
    logger.debug("SLURM_JOB_END_TIME env var: %s", slurm_job_end_time or "(not set)")

    if slurm_job_end_time:
        try:
            remaining = int(slurm_job_end_time) - int(start_time)
            logger.debug("Using SLURM_JOB_END_TIME: remaining=%d", remaining)
            return remaining, "SLURM_JOB_END_TIME"
        except (ValueError, TypeError) as e:
            logger.debug("Failed to parse SLURM_JOB_END_TIME: %s", e)

    # Fallback to SLURM_JOB_TIMELIMIT (in minutes)
    slurm_timelimit = os.environ.get("SLURM_JOB_TIMELIMIT")
    logger.debug("SLURM_JOB_TIMELIMIT env var: %s", slurm_timelimit or "(not set)")

    if slurm_timelimit:
        try:
            remaining = int(slurm_timelimit) * 60
            logger.debug(
                "Using SLURM_JOB_TIMELIMIT: %s min -> %d seconds",
                slurm_timelimit,
                remaining,
            )
            return remaining, "SLURM_JOB_TIMELIMIT"
        except (ValueError, TypeError) as e:
            logger.debug("Failed to parse SLURM_JOB_TIMELIMIT: %s", e)

    logger.debug("No SLURM walltime vars available")
    return None, "slurm_vars_unavailable"


def _resolve_resubmit_command() -> tuple[list[str], Path, str]:
    """Resolve self-resubmission through the canonical workflow boundary."""
    return canonical_resolve_resubmit_command(
        workdir=Path.cwd(),
        scheduler=scheduler,
    )


def _resubmit_slurm_job(debug: bool = False) -> None:
    """
    Resubmit nepflow job to SLURM via sbatch submit.slurm.

    If debug=True, simulates the resubmission without actually launching.
    """
    command, submit_cwd, submit_source = _resolve_resubmit_command()

    logger.info("SLURM walltime deadline approaching - resubmitting nepflow")

    if debug:
        logger.info(
            "[DEBUG] Would resubmit with: %s (cwd=%s, source=%s)",
            " ".join(command),
            submit_cwd,
            submit_source,
        )
        return

    try:
        result = scheduler.submit(command, cwd=submit_cwd)
        logger.info("Resubmission via sbatch: %s", result.stdout.strip())
    except SchedulerError as e:
        stderr = e.stderr or ""
        stdout = e.stdout or ""
        logger.error(
            "Could not resubmit job via %s: %s%s%s",
            submit_source,
            e,
            f" | stdout: {stdout.strip()}" if stdout.strip() else "",
            f" | stderr: {stderr.strip()}" if stderr.strip() else "",
        )
        raise


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.config:
        project_config = _resolve_project_config_path(args.project, args.output_dir)
        project_dir = args.output_dir / f"project_{args.project}"
        project_dir.mkdir(parents=True, exist_ok=True)
        try:
            process_runner.run(
                ["vim", str(project_config)],
                check=False,
                cwd=project_dir,
                capture_output=False,
            )
        except ProcessError as e:
            if e.kind != "not_found":
                raise
            raise FileNotFoundError("vim was not found on PATH") from e
        return

    # Setup logging for the project
    # Log directory is inside each project: projects/project_{name}/logs/
    project_dir = args.output_dir / f"project_{args.project}"
    log_dir = project_dir / "logs"

    configure_logging(
        project_name=args.project,
        log_dir=log_dir,
        debug=args.debug,
    )

    logger.info("NEPFlow started for project: %s", args.project)
    if args.debug:
        logger.debug("Debug mode enabled")

    print("[nepflow] Checking SLURM walltime (DEBUG)", file=sys.stderr)
    logger.debug("Checking for SLURM walltime...")

    # Get SLURM walltime info (None if not running under SLURM)
    try:
        walltime_remaining, walltime_source = _get_slurm_walltime_info()
        print(
            "[nepflow] SLURM walltime result: "
            f"remaining={walltime_remaining}, source={walltime_source}",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"[nepflow] ERROR detecting SLURM walltime: {e}", file=sys.stderr)
        logger.error("Error detecting SLURM walltime: %s", e)
        walltime_remaining, walltime_source = None, "error"

    margin_seconds = 300  # 5 minutes before deadline

    if walltime_remaining is None:
        logger.info("Not running under SLURM - no walltime limit, workflow will run indefinitely")
        slurm_deadline = None
    else:
        slurm_deadline = time.time() + walltime_remaining - margin_seconds
        logger.info(
            "SLURM walltime: %ds (source: %s), deadline in %ds",
            walltime_remaining,
            walltime_source,
            walltime_remaining - margin_seconds,
        )

    # Create workflow controller
    controller = WorkflowController(
        project_name=args.project,
        output_dir=args.output_dir,
        init_mode=args.init,
        debug=args.debug,
        stage_override=args.stage,
        local_mode=args.local,
        slurm_deadline=slurm_deadline,
        stage_registry=compose_stage_registry(),
    )

    # Run workflow: controller reconciles StateStore with the legacy marker
    # If under SLURM and approaching deadline, resubmit before running
    try:
        if not args.init:
            WorkflowStatusPresenter(
                logger,
                details_provider=_legacy_status_details,
            ).log(controller.workflow_state)
        if slurm_deadline and time.time() >= slurm_deadline:
            logger.warning("Already past SLURM deadline - resubmitting immediately")
            _resubmit_slurm_job(debug=args.debug)
            return

        controller.run()
        logger.info("Workflow completed successfully")
        print("Workflow completed successfully")

    except SelfResubmitExit as e:
        # Workflow reached SLURM deadline - resubmit
        logger.info("Workflow resubmit triggered: %s", e)
        _resubmit_slurm_job(debug=args.debug)
        logger.info("Resubmission initiated, exiting")
        return

    except (ValueError, IOError, RuntimeError, StateError, ValidationError) as e:
        # Log the error (message only, not traceback)
        logger.error("Workflow failed: %s", e)

        # Print formatted error message
        print("\n" + "=" * 70, file=sys.stderr)
        print("ERROR: Workflow failed", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("  %s: %s" % (type(e).__name__, e), file=sys.stderr)

        if args.debug:
            print("\nTraceback:", file=sys.stderr)
            print("-" * 70, file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            print("-" * 70, file=sys.stderr)
        else:
            print("\n  Use --debug for full traceback", file=sys.stderr)

        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
