#!/usr/bin/env python3
"""
NEPFlow: HPC workflow system for atomistic datasets.

Entry point for the workflow controller.
"""

import sys
import argparse
import logging
import os
import re
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

# Temporary Phase 4 composition bridge: legacy stage modules still live
# directly under src/. Keep this compatibility path at the composition root;
# the canonical workflow package remains independent of it.
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (_SOURCE_ROOT / "modules").is_dir() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from nepflow.errors import SchedulerError, StateError, ValidationError
from nepflow.cli_wizard import CONFIG_PROMPTS, ConfigWizard
from nepflow.hpc.process import ProcessError, ProcessRunner
from nepflow.hpc.slurm import SlurmScheduler
from nepflow.logging import configure_logging
from nepflow.config.loader import canonical_config_path
from nepflow.reporting import WorkflowStatusPresenter, summarize_legacy_vasp_jobs
from nepflow.workflow import (
    StageContext,
    StageRegistry,
    StageRunResult,
    StageRunState,
    SelfResubmitExit,
    WorkflowController,
    WorkflowStage,
)

logger = logging.getLogger("nepflow")
process_runner = ProcessRunner(logger=logger)
scheduler = SlurmScheduler(process_runner=process_runner)


def _legacy_stage_kwargs(context: StageContext) -> dict:
    """Adapt the canonical stage context to the remaining legacy stages."""

    return {
        "project_name": context.project_name,
        "config_file": context.config_file,
        "state_file": context.state_file,
        "project_dir": context.project_dir,
        "debug": context.debug,
        "slurm_deadline": context.slurm_deadline,
    }


def compose_stage_registry() -> StageRegistry:
    """Compose legacy stage implementations behind the canonical seam."""

    # These imports are intentionally confined to the composition root while
    # the legacy stage implementations are migrated into the package.
    # pylint: disable=import-error,import-outside-toplevel
    from modules import (
        GenerateStage,
        MemoryStage,
        RunVaspStage,
        SelectStage,
        TrainNepStage,
        ValidateStage,
    )
    from modules.validate.launcher import read_validation_status

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
    registry.register(
        WorkflowStage.GENERATE,
        lambda context: GenerateStage(**_legacy_stage_kwargs(context)).run(
            seeds_only=context.seeds_only
        ),
    )
    registry.register(
        WorkflowStage.SELECT,
        lambda context: SelectStage(**_legacy_stage_kwargs(context)).run(),
    )
    registry.register(
        WorkflowStage.RUN_VASP,
        lambda context: RunVaspStage(**_legacy_stage_kwargs(context)).run(),
    )
    registry.register(
        WorkflowStage.TRAIN_NEP,
        lambda context: TrainNepStage(**_legacy_stage_kwargs(context)).run(),
    )

    def run_validation(context: StageContext) -> StageRunResult:
        ValidateStage(**_legacy_stage_kwargs(context)).run()
        status = read_validation_status(context.project_dir)
        complete = (
            status.get("validation_complete") is True
            and status.get("analysis_complete") is True
        )
        return StageRunResult(
            stage=WorkflowStage.VALIDATE,
            status=StageRunState.COMPLETED if complete else StageRunState.RUNNING,
            advanced_to=WorkflowStage.COMPLETED if complete else None,
            completed=complete,
        )

    registry.register(WorkflowStage.VALIDATE, run_validation)
    registry.register_auxiliary(
        "memory",
        lambda context: MemoryStage(**_legacy_stage_kwargs(context)).run(),
    )
    return registry


def _legacy_status_details(project_dir: Path, stage: WorkflowStage) -> dict:
    """Adapt legacy VASP status reporting to the package CLI presenter."""

    # pylint: disable=import-error,import-outside-toplevel
    from modules.run_vasp._common import read_status

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
  nepflow --project myproject --memory
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
        "--memory",
        action="store_true",
        help="Run VASP memory/performance benchmarks on BCC W supercells "
        "to populate .vasp_memory with timing and OOM data. "
        "Does not advance the workflow stage.",
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


def _resolve_resubmit_from_scontrol(slurm_job_id: str) -> tuple[list[str], Path, str] | None:
    """Try to recover the original batch script path from `scontrol show job`."""
    try:
        output = scheduler.show_job(slurm_job_id)
    except SchedulerError as e:
        logger.debug("Could not inspect SLURM job %s via scontrol: %s", slurm_job_id, e)
        return None

    command_match = re.search(r"\bCommand=(\S+)", output)
    if not command_match:
        logger.debug("scontrol output for job %s did not include Command=", slurm_job_id)
        return None

    command_path = Path(command_match.group(1)).expanduser()
    if not command_path.is_absolute():
        workdir_match = re.search(r"\bWorkDir=(\S+)", output)
        if workdir_match:
            command_path = Path(workdir_match.group(1)) / command_path

    if not command_path.exists():
        logger.debug("Recovered SLURM command does not exist: %s", command_path)
        return None

    submit_cwd = command_path.parent
    return ["sbatch", str(command_path)], submit_cwd, f"scontrol job {slurm_job_id}"


def _resolve_resubmit_command() -> tuple[list[str], Path, str]:
    """Resolve the best available `sbatch` command for self-resubmission."""
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        resolved = _resolve_resubmit_from_scontrol(slurm_job_id)
        if resolved is not None:
            return resolved

    nepflow_root = Path(__file__).parent.resolve()
    candidate_dirs = []
    for directory in (
        os.environ.get("SLURM_SUBMIT_DIR"),
        os.getcwd(),
        str(nepflow_root),
    ):
        if not directory:
            continue
        path = Path(directory).resolve()
        if path not in candidate_dirs:
            candidate_dirs.append(path)

    tried = []
    for directory in candidate_dirs:
        submit_script = directory / "submit.slurm"
        tried.append(str(submit_script))
        if submit_script.exists():
            return ["sbatch", str(submit_script)], directory, f"submit.slurm in {directory}"

    tried_text = "\n".join(f"  - {candidate}" for candidate in tried)
    raise FileNotFoundError(
        "Could not find a resubmission script for nepflow. Tried:\n"
        f"{tried_text}"
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
        memory_mode=getattr(args, "memory", False),
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
