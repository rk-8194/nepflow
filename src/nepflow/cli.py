#!/usr/bin/env python3
"""
NEPFlow: HPC workflow system for atomistic datasets.

Entry point for the workflow controller.
"""

import argparse
import logging
import os
import sys
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

from nepflow.application.composition import (  # noqa: E402
    build_generation_stage as _compose_generation_stage,
    compose_stage_registry as _compose_stage_registry,
    offer_project_upload as _offer_project_upload_impl,
)
from nepflow.application.runtime import (  # noqa: E402
    create_process_runner,
    create_scheduler,
    resolve_resubmit_command as _resolve_runtime_resubmit_command,
    resubmit_slurm_job as _resubmit_runtime_job,
)
from nepflow.config.loader import canonical_config_path  # noqa: E402
from nepflow.errors import (  # noqa: E402
    ProcessError,
    SchedulerError as _SchedulerError,
    StateError,
    ValidationError,
)
from nepflow.logging import configure_logging  # noqa: E402
from nepflow.reporting import WorkflowStatusPresenter  # noqa: E402
from nepflow.stages.generation import GenerationStage  # noqa: E402
from nepflow.workflow import (  # noqa: E402
    SelfResubmitExit,
    StageContext,
    StageRegistry,
    WorkflowController,
)

logger = logging.getLogger("nepflow")
SchedulerError = _SchedulerError
process_runner = create_process_runner(logger)
scheduler = create_scheduler(process_runner)


def _build_generation_stage(context: StageContext) -> GenerationStage:
    """Preserve the historical CLI composition seam."""

    return _compose_generation_stage(context, logger=logger)


def _offer_project_upload(context: StageContext) -> None:
    """Offer the local seeds-only result to the configured remote project."""

    _offer_project_upload_impl(context, process_runner=process_runner)


def compose_stage_registry() -> StageRegistry:
    """Compose canonical workflow stages behind the application seam."""

    return _compose_stage_registry(
        logger=logger,
        process_runner=process_runner,
        scheduler=scheduler,
    )


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
    return _resolve_runtime_resubmit_command(
        workdir=Path.cwd(),
        scheduler=scheduler,
    )


def _resubmit_slurm_job(debug: bool = False) -> None:
    """
    Resubmit nepflow job to SLURM via sbatch submit.slurm.

    If debug=True, simulates the resubmission without actually launching.
    """
    command, submit_cwd, submit_source = _resolve_resubmit_command()
    _resubmit_runtime_job(
        command,
        submit_cwd,
        submit_source,
        debug=debug,
        logger=logger,
        scheduler=scheduler,
    )


def _open_project_config(args: argparse.Namespace) -> None:
    """Open the requested project config and preserve the CLI error contract."""

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
    except ProcessError as exc:
        if exc.kind != "not_found":
            raise
        raise FileNotFoundError("vim was not found on PATH") from exc


def _configure_project_logging(args: argparse.Namespace) -> None:
    """Configure project logging before workflow composition."""

    project_dir = args.output_dir / f"project_{args.project}"
    configure_logging(
        project_name=args.project,
        log_dir=project_dir / "logs",
        debug=args.debug,
    )
    logger.info("NEPFlow started for project: %s", args.project)
    if args.debug:
        logger.debug("Debug mode enabled")


def _resolve_slurm_deadline() -> float | None:
    """Resolve and log the deadline used to protect an active SLURM job."""

    print("[nepflow] Checking SLURM walltime (DEBUG)", file=sys.stderr)
    logger.debug("Checking for SLURM walltime...")
    try:
        walltime_remaining, walltime_source = _get_slurm_walltime_info()
        print(
            "[nepflow] SLURM walltime result: "
            f"remaining={walltime_remaining}, source={walltime_source}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"[nepflow] ERROR detecting SLURM walltime: {exc}", file=sys.stderr)
        logger.error("Error detecting SLURM walltime: %s", exc)
        walltime_remaining, walltime_source = None, "error"

    margin_seconds = 300
    if walltime_remaining is None:
        logger.info(
            "Not running under SLURM - no walltime limit, workflow will run indefinitely"
        )
        return None
    deadline = time.time() + walltime_remaining - margin_seconds
    logger.info(
        "SLURM walltime: %ds (source: %s), deadline in %ds",
        walltime_remaining,
        walltime_source,
        walltime_remaining - margin_seconds,
    )
    return deadline


def _create_controller(
    args: argparse.Namespace,
    slurm_deadline: float | None,
) -> WorkflowController:
    """Compose the workflow controller from parsed application arguments."""

    return WorkflowController(
        project_name=args.project,
        output_dir=args.output_dir,
        init_mode=args.init,
        debug=args.debug,
        stage_override=args.stage,
        local_mode=args.local,
        slurm_deadline=slurm_deadline,
        stage_registry=compose_stage_registry(),
    )


def _report_workflow_failure(exc: Exception, *, debug: bool) -> None:
    """Print the established user-facing workflow failure report."""

    logger.error("Workflow failed: %s", exc)
    print("\n" + "=" * 70, file=sys.stderr)
    print("ERROR: Workflow failed", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("  %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
    if debug:
        print("\nTraceback:", file=sys.stderr)
        print("-" * 70, file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        print("-" * 70, file=sys.stderr)
    else:
        print("\n  Use --debug for full traceback", file=sys.stderr)
    print("=" * 70 + "\n", file=sys.stderr)


def _run_controller(
    args: argparse.Namespace,
    controller: WorkflowController,
    slurm_deadline: float | None,
) -> None:
    """Run the controller and handle resubmission and user-facing failures."""

    try:
        if not args.init:
            WorkflowStatusPresenter(logger).log(controller.workflow_state)
        if slurm_deadline and time.time() >= slurm_deadline:
            logger.warning("Already past SLURM deadline - resubmitting immediately")
            _resubmit_slurm_job(debug=args.debug)
            return
        controller.run()
        logger.info("Workflow completed successfully")
        print("Workflow completed successfully")
    except SelfResubmitExit as exc:
        logger.info("Workflow resubmit triggered: %s", exc)
        _resubmit_slurm_job(debug=args.debug)
        logger.info("Resubmission initiated, exiting")
    except (ValueError, IOError, RuntimeError, StateError, ValidationError) as exc:
        _report_workflow_failure(exc, debug=args.debug)
        sys.exit(1)


def main() -> None:
    """Parse arguments, compose the application, and run the workflow."""

    args = create_parser().parse_args()
    if args.config:
        _open_project_config(args)
        return
    _configure_project_logging(args)
    slurm_deadline = _resolve_slurm_deadline()
    _run_controller(args, _create_controller(args, slurm_deadline), slurm_deadline)


if __name__ == "__main__":
    main()
