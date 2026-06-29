"""Launcher sub-stage: submit, monitor, and resubmit NEP training jobs."""

import json
import logging
import os
import re
import subprocess
import time
from configparser import ConfigParser
from pathlib import Path

from ..base import SelfResubmitExit
from ._common import logger

# Status file format: json dict with keys: potential_path, job_id, status, job_name, attempt, created, updated, error


def read_train_status(project_dir: Path) -> dict:
    """Read NEP training status from .train_nep_status file."""
    status_file = project_dir / "nep" / ".train_nep_status"
    if status_file.exists():
        try:
            return json.loads(status_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read status file: {e}")
            return {}
    return {}


def write_train_status(project_dir: Path, **kwargs) -> None:
    """Write NEP training status to .train_nep_status file."""
    status_dir = project_dir / "nep"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / ".train_nep_status"
    
    status_data = read_train_status(project_dir)
    status_data.update(kwargs)
    status_data["updated"] = time.time()
    
    status_file.write_text(json.dumps(status_data, indent=2))


def _get_job_name_from_potential(potential_path: Path) -> str:
    """Generate SLURM job name from potential folder name."""
    return f"nep_train_{potential_path.name}"


def _check_nep_complete(potential_path: Path) -> bool:
    """Check if NEP training completed successfully by looking for output files."""
    # NEP generates a nep.txt file when training completes
    nep_output = potential_path / "nep.txt"
    if nep_output.exists() and nep_output.stat().st_size > 0:
        logger.info(f"  Found NEP output: {nep_output.name}")
        return True
    
    # Check for nep_*.txt outputs (different NEP versions)
    nep_outputs = list(potential_path.glob("nep*.txt"))
    if nep_outputs:
        for f in nep_outputs:
            if f.stat().st_size > 0:
                logger.info(f"  Found NEP output: {f.name}")
                return True
    
    return False


def _get_training_generation(potential_path: Path) -> tuple[int, float] | None:
    """Read generation number and total loss from loss.out file.
    
    Extracts the first value (generation) and second value (total loss) from the last line.
    Returns tuple (generation, loss) or None if file doesn't exist or cannot be read.
    """
    loss_file = potential_path / "loss.out"
    if not loss_file.exists():
        return None
    
    try:
        lines = loss_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
        if lines:
            last_line = lines[-1].strip()
            if last_line:
                parts = last_line.split()
                # First value is generation, second is total loss
                if len(parts) >= 2:
                    try:
                        generation = int(parts[0])
                        loss = float(parts[1])
                        return (generation, loss)
                    except (ValueError, IndexError):
                        pass
    except OSError:
        pass
    
    return None


def _get_target_generations(dataset_path: Path) -> int | None:
    """Read target generation count from nep.in file."""
    nep_in = dataset_path / "nep.in"
    if not nep_in.exists():
        return None
    
    try:
        for line in nep_in.read_text().split("\n"):
            if line.strip().startswith("generation"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return int(parts[1])
                    except ValueError:
                        pass
    except OSError:
        pass
    
    return None


def _format_time_remaining(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 0:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def _check_training_error(potential_path: Path) -> str | None:
    """Check for training errors in log files."""
    log_file = potential_path / "train_nep_*.log"
    log_files = list(potential_path.glob("train_nep_*.log"))
    
    if not log_files:
        return None
    
    log_file = log_files[0]  # Get the SLURM log
    try:
        content = log_file.read_text(encoding="utf-8", errors="ignore")
        
        # Check for common error patterns
        if "CUDA Error" in content:
            return "CUDA error (GPU not available)"
        if "out of memory" in content.lower() or "oom" in content.lower():
            return "Out of memory"
        if "segmentation fault" in content.lower():
            return "Segmentation fault"
        if "Killed" in content or "terminated" in content.lower():
            return "Job terminated/killed"
        
        # Check last line for any error indication
        lines = content.strip().split("\n")
        if lines and ("error" in lines[-1].lower() or "failed" in lines[-1].lower()):
            return lines[-1][:100]
    except OSError as e:
        logger.debug(f"Could not read log file: {e}")
    
    return None


def _get_slurm_job_id(job_name: str) -> str | None:
    """Get SLURM job ID for a job by name."""
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_name, "-o", "%i"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()[0]
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def run_launcher(
    config: ConfigParser,
    dataset_path: Path,
    potential_path: Path,
    project_name: str,
    project_dir: Path,
    debug: bool = False,
    slurm_deadline: float | None = None,
) -> None:
    """Monitor and manage NEP training job submission.
    
    Monitors the job via squeue polling. When the job completes (leaves squeue),
    checks if training was successful. If it fails, can resubmit with incremented
    attempt counter. Raises SelfResubmitExit when walltime deadline is reached.
    
    Args:
        config: ConfigParser with NEP/SLURM settings
        dataset_path: Path to dataset folder (with train.xyz, test.xyz, nep.in)
        potential_path: Path to potential folder (where training runs)
        project_name: Name of the project
        project_dir: Project root directory
        debug: Whether to use simulated SLURM
        slurm_deadline: Unix timestamp of SLURM deadline
    """
    poll_interval = 0 if debug else config.getint("slurm", "poll_interval", fallback=30)
    max_attempts = config.getint("train_nep", "max_resubmit", fallback=3)
    
    logger.info("")
    logger.info("═" * 60)
    logger.info("NEP Training Launcher")
    logger.info("═" * 60)
    logger.info(f"  Potential    : {potential_path.name}")
    logger.info(f"  Poll interval: {poll_interval}s")
    logger.info(f"  Max attempts : {max_attempts}")
    logger.info("")
    
    # Load or initialize status
    status = read_train_status(project_dir)
    # If there's an existing job_id and attempt, continue with that attempt
    # Otherwise, this is the first attempt
    existing_job_id = status.get("job_id")
    if existing_job_id and status.get("attempt"):
        # Continuing to monitor existing job
        attempt = status.get("attempt")
    else:
        # New submission or no attempt tracked yet
        attempt = status.get("attempt", 0) + 1
    
    max_attempts_exceeded = attempt > max_attempts
    
    job_name = _get_job_name_from_potential(potential_path)
    
    if max_attempts_exceeded:
        logger.error(f"Training failed after {max_attempts} attempts")
        write_train_status(
            project_dir,
            potential_path=str(potential_path),
            status="failed",
            attempt=attempt,
            error="Max resubmit attempts exceeded",
        )
        return
    
    logger.info(f"Training attempt {attempt}/{max_attempts}")
    
    # Check if training script exists (should have been copied by submit.py)
    train_script = potential_path / "train_nep.sh"
    if not train_script.exists():
        logger.error(f"Training script not found: {train_script}")
        write_train_status(
            project_dir,
            potential_path=str(potential_path),
            status="failed",
            attempt=attempt,
            error="Training script not found",
        )
        return
    
    # Submit job if not already submitted this attempt
    job_id = status.get("job_id")
    # Only submit if: no job_id yet, OR this is a new attempt (attempt index changed)
    already_submitted_this_attempt = job_id and status.get("attempt") == attempt
    if not already_submitted_this_attempt:
        logger.info(f"Submitting training job: {job_name}")
        try:
            result = subprocess.run(
                ["sbatch", str(train_script)],
                capture_output=True,
                text=True,
                cwd=str(potential_path),
                timeout=10,
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"sbatch failed: {error_msg}")
                write_train_status(
                    project_dir,
                    potential_path=str(potential_path),
                    status="failed",
                    attempt=attempt,
                    error=f"sbatch failed: {error_msg}",
                )
                return
            
            # Extract job ID
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            job_id = match.group(1) if match else None
            if not job_id:
                logger.warning(f"Could not parse job ID from sbatch output: {result.stdout}")
        except subprocess.TimeoutExpired:
            logger.error("sbatch command timed out")
            return
        except OSError as e:
            logger.error(f"Failed to submit job: {e}")
            return
    
    if not job_id:
        logger.error("No job ID available for monitoring")
        return
    
    # Update status: job submitted
    write_train_status(
        project_dir,
        potential_path=str(potential_path),
        job_id=job_id,
        job_name=job_name,
        status="running",
        attempt=attempt,
        start_time=time.time(),
    )
    logger.info(f"Job submitted: {job_id}")
    
    # Get target generations for ETC calculation
    target_generations = _get_target_generations(dataset_path)
    if target_generations:
        logger.info(f"Target generations: {target_generations:,}")
    
    # Main monitoring loop
    poll_count = 0
    last_check_time = time.time()
    last_generation = 0  # Track previous generation for rate calculation
    
    while True:
        poll_count += 1
        
        # Check deadline
        if slurm_deadline is not None and time.time() >= slurm_deadline:
            logger.warning("SLURM deadline reached in NEP launcher — triggering resubmit")
            write_train_status(
                project_dir,
                potential_path=str(potential_path),
                status="running",
                attempt=attempt,
            )
            raise SelfResubmitExit("SLURM walltime deadline reached in NEP launcher")
        
        # Check if job is still in squeue
        current_time = time.time()
        if current_time - last_check_time >= poll_interval or poll_count == 1:
            last_check_time = current_time
            
            if not debug:
                try:
                    result = subprocess.run(
                        ["squeue", "-h", "-j", job_id, "-o", "%T"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    job_state = result.stdout.strip()
                except (subprocess.TimeoutExpired, OSError):
                    job_state = ""
            else:
                # Debug mode: assume job completes after a few polls
                job_state = "COMPLETED" if poll_count > 3 else "RUNNING"
            
            if job_state:
                logger.debug(f"  Poll {poll_count}: Job state = {job_state}")
                
                # Get current training generation/progress
                training_stats = _get_training_generation(potential_path)
                if training_stats is not None:
                    generation, loss = training_stats
                    
                    # Calculate ETA based on generations since last poll
                    eta_str = ""
                    if target_generations and generation < target_generations and poll_interval > 0:
                        gens_since_last_poll = generation - last_generation
                        
                        if gens_since_last_poll > 0:
                            gen_per_sec = gens_since_last_poll / poll_interval
                            remaining_gens = target_generations - generation
                            eta_seconds = remaining_gens / gen_per_sec
                            eta_str = f", ETA: {_format_time_remaining(eta_seconds)}"
                        
                        last_generation = generation
                    
                    logger.info(f"  Generation: {generation}/{target_generations or '?'}, Loss: {loss:.6f}{eta_str}")
                elif job_state == "RUNNING":
                    logger.info(f"  Job running (generation not yet available)")
                
                if job_state in ["COMPLETED", "FAILED", "CANCELLED", "NODE_FAIL"]:
                    logger.info(f"Job left queue: {job_state}")
                    
                    # Job finished — check for success
                    if _check_nep_complete(potential_path):
                        logger.info(f"  ✓ NEP training completed successfully")
                        write_train_status(
                            project_dir,
                            potential_path=str(potential_path),
                            job_id=job_id,
                            status="completed",
                            attempt=attempt,
                        )
                        return
                    
                    # Training failed — check error
                    error_msg = _check_training_error(potential_path)
                    if not error_msg:
                        error_msg = f"Training did not produce output (state: {job_state})"
                    
                    logger.warning(f"  ✗ Training failed: {error_msg}")
                    # Clear job_id so resubmission knows to increment attempt
                    write_train_status(
                        project_dir,
                        potential_path=str(potential_path),
                        job_id=None,
                        status="failed",
                        attempt=attempt,
                        error=error_msg,
                    )
                    
                    # Check if we should retry
                    if attempt < max_attempts:
                        logger.info(f"Resubmitting training (attempt {attempt + 1}/{max_attempts})")
                        # Recursively call launcher for next attempt
                        run_launcher(
                            config,
                            dataset_path,
                            potential_path,
                            project_name,
                            project_dir,
                            debug=debug,
                            slurm_deadline=slurm_deadline,
                        )
                    else:
                        logger.error(f"Training failed after {max_attempts} attempts")
                    
                    return
            else:
                logger.debug(f"  Poll {poll_count}: Job not found in squeue")
        
        # Wait before next poll
        if poll_interval > 0:
            time.sleep(min(poll_interval, 5))  # Max 5s between checks even with long intervals
