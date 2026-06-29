"""Launcher sub-stage: submit, monitor, and resubmit GPUMD validation jobs."""

import json
import logging
import os
import subprocess
import time
from configparser import ConfigParser
from pathlib import Path

from ..base import SelfResubmitExit

logger = logging.getLogger("nepflow.validate")


def read_validation_status(project_dir: Path) -> dict:
    """Read validation job status from .validation_status file."""
    status_file = project_dir / "gpumd" / ".validation_status"
    if status_file.exists():
        try:
            return json.loads(status_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read validation status file: {e}")
            return {}
    return {}


def write_validation_status(project_dir: Path, **kwargs) -> None:
    """Write validation job status to .validation_status file."""
    status_dir = project_dir / "gpumd"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / ".validation_status"
    
    status_data = read_validation_status(project_dir)
    status_data.update(kwargs)
    status_data["updated"] = time.time()
    
    status_file.write_text(json.dumps(status_data, indent=2))


def _check_struct_complete(struct_dir: Path) -> bool:
    """Check if GPUMD validation completed for a structure.
    
    Looks for out.xyz output file.
    """
    out_xyz = struct_dir / "out.xyz"
    if out_xyz.exists() and out_xyz.stat().st_size > 0:
        logger.debug(f"  {struct_dir.name}: out.xyz found")
        return True
    return False


def _generate_slurm_script(
    struct_dir: Path,
    job_name: str,
    config: ConfigParser,
) -> str:
    """Generate SLURM batch script for single GPUMD validation job.
    
    Args:
        struct_dir: Path to struct_XXXX folder
        job_name: SLURM job name
        config: ConfigParser with SLURM settings
        
    Returns:
        SLURM script content as string
    """
    # Read SLURM header template
    slurm_header_path = Path(config.get("paths", "project_dir")) / "config" / "slurm" / "header.slurm"
    
    if slurm_header_path.exists():
        header = slurm_header_path.read_text()
    else:
        logger.warning(f"SLURM header not found at {slurm_header_path}, using minimal template")
        header = "#!/bin/bash\n#SBATCH --job-name=gpumd_validate\n"
    
    # Add validation-specific settings
    gpumd_walltime = config.get("slurm", "gpumd_walltime", fallback="00:10:00")
    gpumd_nodes = config.getint("slurm", "gpumd_nodes", fallback=1)
    gpumd_gpus = config.getint("slurm", "gpumd_gpus", fallback=1)
    
    script_lines = [
        header.rstrip(),
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --time={gpumd_walltime}",
        f"#SBATCH --nodes={gpumd_nodes}",
        f"#SBATCH --gpus-per-node={gpumd_gpus}",
        "",
        "# Load environment",
        "module load cuda",
        "",
        f"cd {struct_dir}",
        "",
        "# Run GPUMD",
        'mpirun -np 1 --bind-to none "$HOME/src/GPUMD/src/gpumd" < run.in > gpumd.log 2>&1',
        "",
    ]
    
    return "\n".join(script_lines)


def submit_struct_validation_job(
    struct_dir: Path,
    config: ConfigParser,
    debug: bool = False,
) -> str:
    """Submit GPUMD validation job for a single structure.
    
    Args:
        struct_dir: Path to struct_XXXX folder (contains run.in, nep.txt, model.xyz)
        config: ConfigParser with SLURM settings
        debug: If True, simulate submission (don't actually submit)
        
    Returns:
        SLURM job ID as string (or "debug-job-id" if debug mode)
        
    Raises:
        RuntimeError: If batch submission fails
    """
    struct_name = struct_dir.name
    job_name = f"gpumd_val_{struct_name}"
    
    # Generate SLURM script
    script_content = _generate_slurm_script(struct_dir, job_name, config)
    
    # Write script to file
    script_path = struct_dir / "validate.slurm"
    script_path.write_text(script_content)
    logger.debug(f"Wrote SLURM script: {script_path}")
    
    if debug:
        logger.debug(f"[DEBUG] Would submit: sbatch {script_path}")
        return "debug-job-id"
    
    # Submit via sbatch
    try:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")
        
        # Parse job ID from output: "Submitted batch job 12345"
        output = result.stdout.strip()
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            logger.info(f"  {struct_name}: submitted as job {job_id}")
            return job_id
        else:
            raise RuntimeError(f"Unexpected sbatch output: {output}")
    
    except subprocess.TimeoutExpired:
        raise RuntimeError("sbatch submission timed out")
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        raise


def _get_running_job_ids() -> set:
    """Get set of currently running SLURM job IDs.
    
    Uses squeue to query the scheduler.
    """
    try:
        result = subprocess.run(
            ["squeue", "-ho", "%i"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode != 0:
            logger.warning("squeue failed, assuming no running jobs")
            return set()
        
        job_ids = {line.strip() for line in result.stdout.strip().split("\n") if line.strip()}
        return job_ids
    
    except subprocess.TimeoutExpired:
        logger.warning("squeue timed out")
        return set()
    except Exception as e:
        logger.warning(f"Could not query squeue: {e}")
        return set()


def run_validation_launcher(
    config: ConfigParser,
    preparation_state: dict,
    project_dir: Path,
    debug: bool = False,
    slurm_deadline: float | None = None,
) -> None:
    """Submit, monitor, and resubmit GPUMD validation jobs.
    
    Manages job queue up to max_concurrent limit, polls for completion,
    and resubmits failed jobs up to max_attempts.
    
    Args:
        config: ConfigParser with SLURM settings
        preparation_state: Dict from prepare_validation_structures()
        project_dir: Project root directory
        debug: If True, simulate job submission
        slurm_deadline: Unix timestamp of SLURM walltime deadline
        
    Raises:
        SelfResubmitExit: If walltime deadline approaching, triggers job resubmission
    """
    max_concurrent = config.getint("slurm", "max_concurrent", fallback=20)
    poll_interval = 0 if debug else config.getint("slurm", "poll_interval", fallback=30)
    max_attempts = config.getint("slurm", "max_retry_level", fallback=3)
    
    validation_root = Path(preparation_state["validation_root"])
    struct_folders = preparation_state["struct_folders"]
    struct_count = len(struct_folders)
    
    logger.info(f"Starting validation launcher for {struct_count} structures")
    logger.info(f"Max concurrent jobs: {max_concurrent}, poll interval: {poll_interval}s")
    
    # Load or initialize status
    status = read_validation_status(project_dir)
    
    if not status.get("struct_status"):
        # Initialize struct tracking
        status["struct_status"] = {
            info["name"]: {
                "status": "pending",  # pending, submitted, completed, failed
                "job_id": None,
                "attempts": 0,
            }
            for info in struct_folders
        }
        status["completed_count"] = 0
        logger.info("Initialized validation status tracking")
    
    # Main launcher loop
    iteration = 0
    while True:
        iteration += 1
        logger.debug(f"\n=== Launcher iteration {iteration} ===")
        
        # Check deadline
        if slurm_deadline and time.time() > slurm_deadline - 600:  # 10 min margin
            logger.warning("SLURM deadline approaching, resubmitting job")
            write_validation_status(project_dir, **status)
            raise SelfResubmitExit("SLURM walltime deadline approaching")
        
        struct_status = status["struct_status"]
        
        # Phase 1: Submit pending jobs (up to max_concurrent)
        running_jobs = _get_running_job_ids() if not debug else set()
        currently_submitted = {
            info["job_id"]
            for info in struct_status.values()
            if info["job_id"] and info["status"] == "submitted"
        }
        
        available_slots = max_concurrent - len(currently_submitted)
        pending_structs = [
            (name, info)
            for name, info in struct_status.items()
            if info["status"] == "pending" and info["attempts"] < max_attempts
        ]
        
        for struct_name, info in pending_structs[:available_slots]:
            struct_dir = validation_root / struct_name
            
            try:
                job_id = submit_struct_validation_job(struct_dir, config, debug=debug)
                info["job_id"] = job_id
                info["status"] = "submitted"
                info["attempts"] += 1
                logger.info(f"Submitted {struct_name} (attempt {info['attempts']}/{max_attempts})")
            except Exception as e:
                logger.error(f"Failed to submit {struct_name}: {e}")
                info["attempts"] += 1
                if info["attempts"] >= max_attempts:
                    info["status"] = "failed"
                    logger.error(f"{struct_name}: max attempts exceeded")
        
        # Phase 2: Check completed jobs
        for struct_name, info in struct_status.items():
            if info["status"] == "submitted":
                struct_dir = validation_root / struct_name
                
                # Check if Job has left queue
                if info["job_id"] and info["job_id"] not in running_jobs and not debug:
                    # Job finished (left queue), check for output
                    if _check_struct_complete(struct_dir):
                        info["status"] = "completed"
                        status["completed_count"] = len(
                            [s for s in struct_status.values() if s["status"] == "completed"]
                        )
                        logger.info(f"{struct_name}: COMPLETED")
                    else:
                        # Job finished but no output → failed
                        logger.warning(f"{struct_name}: job finished without output")
                        info["status"] = "failed"
                        if info["attempts"] < max_attempts:
                            info["status"] = "pending"
                            info["job_id"] = None
                            logger.info(f"{struct_name}: will retry")
                        else:
                            logger.error(f"{struct_name}: max attempts exceeded")
        
        # Phase 3: Check for completion
        completed = [s for s in struct_status.values() if s["status"] == "completed"]
        failed = [s for s in struct_status.values() if s["status"] == "failed"]
        
        logger.info(
            f"Status: {len(completed)}/{struct_count} completed, "
            f"{len(failed)} failed, {len(currently_submitted)} running"
        )
        
        if len(completed) == struct_count:
            logger.info("All validation jobs completed successfully!")
            status["validation_complete"] = True
            write_validation_status(project_dir, **status)
            break
        
        if len(failed) + len(completed) == struct_count:
            logger.warning(f"Validation finished with {len(failed)} failures")
            status["validation_complete"] = False
            write_validation_status(project_dir, **status)
            break
        
        # Save status and wait for next iteration
        write_validation_status(project_dir, **status)
        
        if not debug:
            logger.debug(f"Waiting {poll_interval}s before next poll...")
            time.sleep(poll_interval)
    
    logger.info("Validation launcher exiting")
