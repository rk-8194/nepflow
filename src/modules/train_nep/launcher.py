"""Launcher sub-stage: submit, monitor, and resubmit NEP training jobs."""

import os
import time
from configparser import ConfigParser
from pathlib import Path

from nepflow.errors import SchedulerError, StateError
from nepflow.hpc.jobs import SchedulerJobState
from nepflow.hpc.slurm import SlurmScheduler
from nepflow.io.json import read_json, write_json

from nepflow.workflow.resubmission import SelfResubmitExit
from nepflow.mlip.nep.artifacts import NepArtifactError, update_model_run_status
from nepflow.mlip.nep.metrics import classify_training_error, parse_progress
from nepflow.mlip.nep.outputs import parse_completion
from ._common import logger

scheduler = SlurmScheduler()

# Status file format: json dict with keys: potential_path, job_id, status, job_name, attempt, created, updated, error


def read_train_status(project_dir: Path) -> dict:
    """Read NEP training status from .train_nep_status file."""
    status_file = project_dir / "nep" / ".train_nep_status"
    if not status_file.exists():
        return {}
    return read_json(status_file, error_type=StateError, require_object=True)


def write_train_status(project_dir: Path, **kwargs) -> None:
    """Write NEP training status to .train_nep_status file."""
    status_dir = project_dir / "nep"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / ".train_nep_status"
    
    status_data = read_train_status(project_dir)
    status_data.update(kwargs)
    status_data["updated"] = time.time()
    write_json(status_file, status_data)


def _get_job_name_from_potential(potential_path: Path) -> str:
    """Generate SLURM job name from potential folder name."""
    return f"nep_train_{potential_path.name}"


def _check_nep_complete(potential_path: Path) -> bool:
    """Compatibility predicate backed by the canonical NEP output parser."""
    return parse_completion(potential_path).completed


def _get_training_generation(potential_path: Path) -> tuple[int, float] | None:
    """Compatibility tuple backed by the canonical NEP progress parser."""
    progress = parse_progress(potential_path)
    return None if progress is None else (progress.generation, progress.loss)


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
    """Compatibility error lookup backed by the canonical NEP classifier."""
    return classify_training_error(potential_path)


def _get_slurm_job_id(job_name: str) -> str | None:
    """Get SLURM job ID for a job by name."""
    job = scheduler.find_job_by_name(job_name, timeout=10)
    return job.job_id if job is not None else None


def _mark_model_run(
    potential_path: Path,
    status: str,
    *,
    error: str | None,
    state_store: object | None,
    model_run_id: str | None,
) -> None:
    """Persist model status before the legacy status-file projection."""

    if state_store is None or not model_run_id:
        return
    update_model_run_status(
        potential_path,
        status,
        error=error,
        state_store=state_store,
        model_run_id=model_run_id,
    )


def run_launcher(
    config: ConfigParser,
    dataset_path: Path,
    potential_path: Path,
    project_name: str,
    project_dir: Path,
    debug: bool = False,
    slurm_deadline: float | None = None,
    state_store: object | None = None,
    model_run_id: str | None = None,
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
        _mark_model_run(
            potential_path,
            "failed",
            error="Max resubmit attempts exceeded",
            state_store=state_store,
            model_run_id=model_run_id,
        )
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
        _mark_model_run(
            potential_path,
            "failed",
            error="Training script not found",
            state_store=state_store,
            model_run_id=model_run_id,
        )
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
            result = scheduler.submit(
                ["sbatch", str(train_script)],
                cwd=potential_path,
                timeout=10,
            )
            job_id = result.job_id
        except SchedulerError as exc:
            logger.error("Failed to submit training job: %s", exc)
            if exc.kind == "command_failed":
                error_msg = exc.stderr.strip() if exc.stderr else str(exc)
                _mark_model_run(
                    potential_path,
                    "failed",
                    error=f"sbatch failed: {error_msg}",
                    state_store=state_store,
                    model_run_id=model_run_id,
                )
                write_train_status(
                    project_dir,
                    potential_path=str(potential_path),
                    status="failed",
                    attempt=attempt,
                    error=f"sbatch failed: {error_msg}",
                )
            return
    
    if not job_id:
        logger.error("No job ID available for monitoring")
        _mark_model_run(
            potential_path,
            "failed",
            error="No job ID available for monitoring",
            state_store=state_store,
            model_run_id=model_run_id,
        )
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
                query = scheduler.queue_status(job_id, timeout=5)
                job_state = query.job.state if query.job is not None else None
            else:
                # Debug mode: assume job completes after a few polls
                job_state = (
                    SchedulerJobState.COMPLETED
                    if poll_count > 3
                    else SchedulerJobState.RUNNING
                )
            
            if job_state:
                logger.debug(f"  Poll {poll_count}: Job state = {job_state.value}")
                
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
                elif job_state is SchedulerJobState.RUNNING:
                    logger.info(f"  Job running (generation not yet available)")
                
                if job_state in {
                    SchedulerJobState.COMPLETED,
                    SchedulerJobState.FAILED,
                    SchedulerJobState.OOM,
                    SchedulerJobState.CANCELLED,
                    SchedulerJobState.TIMEOUT,
                }:
                    logger.info(f"Job left queue: {job_state.value}")
                    
                    # Job finished — check for success
                    if _check_nep_complete(potential_path):
                        try:
                            manifest = update_model_run_status(
                                potential_path,
                                "completed",
                                state_store=state_store,
                                model_run_id=model_run_id,
                            )
                        except NepArtifactError:
                            logger.exception("Training artifact could not be finalized")
                            raise
                        else:
                            logger.info(f"  ✓ NEP training completed successfully")
                            write_train_status(
                                project_dir,
                                potential_path=str(potential_path),
                                model_run_id=manifest["model_run_id"],
                                job_id=job_id,
                                status="completed",
                                attempt=attempt,
                            )
                            return
                    
                    # Training failed — check error
                    error_msg = _check_training_error(potential_path)
                    if not error_msg:
                        error_msg = f"Training did not produce output (state: {job_state.value})"
                    
                    logger.warning(f"  ✗ Training failed: {error_msg}")
                    try:
                        update_model_run_status(
                            potential_path,
                            "failed",
                            error=error_msg,
                            state_store=state_store,
                            model_run_id=model_run_id,
                        )
                    except NepArtifactError:
                        logger.exception("Could not update model-run manifest")
                        raise

                    # Clear job_id so resubmission knows to increment attempt.
                    # The legacy projection is written only after the
                    # authoritative StateStore transition succeeds.
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
                            state_store=state_store,
                            model_run_id=model_run_id,
                        )
                    else:
                        logger.error(f"Training failed after {max_attempts} attempts")
                    
                    return
            else:
                logger.debug(f"  Poll {poll_count}: Job not found in squeue")
        
        # Wait before next poll
        if poll_interval > 0:
            time.sleep(min(poll_interval, 5))  # Max 5s between checks even with long intervals
