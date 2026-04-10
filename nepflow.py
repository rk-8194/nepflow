#!/usr/bin/env python3
"""
NEPFlow: HPC workflow system for atomistic datasets.

Entry point for the workflow controller.
"""

import sys
import argparse
import logging
import os
import subprocess
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Note: Imports below depend on src/ being in sys.path
# pylint: disable=import-error
from workflow import WorkflowController
from logging_config import setup_logging
from modules import SelfResubmitExit

logger = logging.getLogger("nepflow")


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NEPFlow: HPC workflow orchestrator for VASP→NEP→GPUMD pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nepflow.py --project myproject
  python nepflow.py --project myproject --local
  python nepflow.py --project myproject --memory
  python nepflow.py --project myproject --debug
        """
    )
    
    # Global options
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project name (determines config, state, and output paths)"
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize a new project (required on first run)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["init", "generate", "select", "run_vasp", "train_nep", "validate"],
        default=None,
        help="Set the workflow stage (overrides .project file)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Fetch base structures locally (seeds only, no perturbations), "
             "then stop. Use on a machine with web access before transferring "
             "to HPC."
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Run VASP memory/performance benchmarks on BCC W supercells "
             "to populate .vasp_memory with timing and OOM data. "
             "Does not advance the workflow stage."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (no external dependencies required)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "projects",
        help="Output base directory (default: ./projects)"
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
            logger.debug("Using SLURM_JOB_TIMELIMIT: %s min → %d seconds", slurm_timelimit, remaining)
            return remaining, "SLURM_JOB_TIMELIMIT"
        except (ValueError, TypeError) as e:
            logger.debug("Failed to parse SLURM_JOB_TIMELIMIT: %s", e)
    
    logger.debug("No SLURM walltime vars available")
    return None, "slurm_vars_unavailable"


def _resubmit_slurm_job(debug: bool = False) -> None:
    """
    Resubmit nepflow job to SLURM via sbatch submit.slurm.

    If debug=True, simulates the resubmission without actually launching.
    """
    nepflow_root = Path(__file__).parent
    submit_script = nepflow_root / "submit.slurm"

    logger.info("SLURM walltime deadline approaching — resubmitting nepflow")

    if debug:
        logger.info("[DEBUG] Would resubmit with: sbatch %s", submit_script)
        return

    try:
        result = subprocess.run(
            ["sbatch", str(submit_script)],
            capture_output=True, text=True, check=True,
            cwd=str(nepflow_root),
        )
        logger.info("Resubmission via sbatch: %s", result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error("Could not resubmit job: %s", e)
        raise


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging for the project
    # Log directory is inside each project: projects/project_{name}/logs/
    project_dir = args.output_dir / f"project_{args.project}"
    log_dir = project_dir / "logs"
    
    setup_logging(
        project_name=args.project,
        log_dir=log_dir,
        debug=args.debug
    )
    
    logger.info("NEPFlow started for project: %s", args.project)
    if args.debug:
        logger.debug("Debug mode enabled")
    
    print(f"[nepflow] Checking SLURM walltime (DEBUG)", file=sys.stderr)
    logger.debug("Checking for SLURM walltime...")
    
    # Get SLURM walltime info (None if not running under SLURM)
    try:
        walltime_remaining, walltime_source = _get_slurm_walltime_info()
        print(f"[nepflow] SLURM walltime result: remaining={walltime_remaining}, source={walltime_source}", file=sys.stderr)
    except Exception as e:
        print(f"[nepflow] ERROR detecting SLURM walltime: {e}", file=sys.stderr)
        logger.error("Error detecting SLURM walltime: %s", e)
        walltime_remaining, walltime_source = None, "error"
    
    margin_seconds = 300  # 5 minutes before deadline
    
    if walltime_remaining is None:
        logger.info("Not running under SLURM — no walltime limit, workflow will run indefinitely")
        slurm_deadline = None
    else:
        slurm_deadline = time.time() + walltime_remaining - margin_seconds
        logger.info("SLURM walltime: %ds (source: %s), deadline in %ds", walltime_remaining, walltime_source, walltime_remaining - margin_seconds)
    
    # Create workflow controller
    controller = WorkflowController(
        project_name=args.project,
        output_dir=args.output_dir,
        init_mode=args.init,
        debug=args.debug,
        stage_override=args.stage,
        local_mode=args.local,
        memory_mode=getattr(args, 'memory', False),
        slurm_deadline=slurm_deadline,
    )
    
    # Run workflow: controller determines current stage from .project file
    # If under SLURM and approaching deadline, resubmit before running
    try:
        if slurm_deadline and time.time() >= slurm_deadline:
            logger.warning("Already past SLURM deadline — resubmitting immediately")
            _resubmit_slurm_job(debug=args.debug)
            return
        
        controller.run()
        logger.info("Workflow completed successfully")
        print("✓ Workflow completed successfully")
        
    except SelfResubmitExit as e:
        # Workflow reached SLURM deadline — resubmit
        logger.info("Workflow resubmit triggered: %s", e)
        _resubmit_slurm_job(debug=args.debug)
        logger.info("Resubmission initiated, exiting")
        return
        
    except (ValueError, IOError, RuntimeError) as e:
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
