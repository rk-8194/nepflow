"""Submit sub-stage: create and submit SLURM training job."""

import shutil
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path

from nepflow.hpc.resources import JobResources, render_slurm_header
from nepflow.hpc.slurm import SlurmScheduler

from ._common import logger

scheduler = SlurmScheduler()


def submit_training_job(
    config: ConfigParser,
    dataset_path: Path,
    potential_path: Path,
    project_name: str,
    project_dir: Path,
) -> str:
    """Create and submit SLURM batch script for NEP training.
    
    Copies XYZ and nep.in from dataset folder to potential folder, then submits.
    
    Follows the same pattern as run_vasp.py to ensure consistent SLURM settings
    (account, partition, etc.) are properly inherited from config/slurm/header.slurm.
    
    Args:
        config: ConfigParser with SLURM settings
        dataset_path: Path to dataset folder (where train.xyz, test.xyz, nep.in live)
        potential_path: Path to potential folder (where training will run)
        project_name: Name of the project
        project_dir: Project root directory
        
    Returns:
        SLURM job ID as string
        
    Raises:
        FileNotFoundError: If SLURM header not found
        SchedulerError: If sbatch submission fails or returns malformed output
    """
    # Copy dataset files to potential folder
    logger.info(f"Copying dataset files from {dataset_path.name} to {potential_path.name}")
    for filename in ["train.xyz", "test.xyz", "nep.in"]:
        src = dataset_path / filename
        dst = potential_path / filename
        if src.exists():
            shutil.copy2(src, dst)
            logger.debug(f"Copied {filename}")
        else:
            logger.warning(f"Source file not found: {src}")

    # Read SLURM header from project config
    slurm_header_path = project_dir / "config" / "slurm" / "header.slurm"
    if not slurm_header_path.exists():
        raise FileNotFoundError(
            f"SLURM header not found at {slurm_header_path}\n"
            f"Place your header.slurm in: {slurm_header_path.parent}/"
        )

    header_text = slurm_header_path.read_text(encoding="utf-8")
    
    # Get walltime from config (default 24h for NEP training)
    # Check for train_nep_walltime first, fall back to general walltime, then default
    walltime = config.get("slurm", "train_nep_walltime", fallback=None)
    if not walltime:
        walltime = config.get("slurm", "walltime", fallback="24:00:00")

    nep_command = config.get("hpc", "nep_command", fallback="").strip()
    if not nep_command:
        raise ValueError(
            "Required configuration hpc.nep_command is missing or blank"
        )
    
    job_name = f"nep_train_{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rendered_header = render_slurm_header(
        JobResources(nodes=1, gpus_per_node=1, mpi_ranks=1, walltime=walltime),
        base_header=header_text,
        job_name=job_name,
        stdout_path="train_nep_%j.log",
        stderr_path="train_nep_%j.err",
    )

    # Backend command body remains owned by the NEP training stage.
    script_lines = [
        rendered_header.rstrip(),
        "",
        f"cd {potential_path}",
        nep_command,
    ]
    
    script_content = "\n".join(script_lines)

    # Write script
    script_path = potential_path / "train_nep.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    logger.debug(f"Created SLURM script: {script_path}")

    # Submit script
    result = scheduler.submit(
        ["sbatch", str(script_path)],
        cwd=potential_path,
    )
    return result.job_id
