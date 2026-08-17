"""Submit sub-stage: create and submit SLURM training job."""

import shutil
import subprocess
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path

from ._common import logger


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
        RuntimeError: If sbatch submission fails
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
    
    # Separate SBATCH directives from other content
    sbatch_directives = []
    other_content = []
    shebang = ""
    
    for line in header_text.splitlines():
        if line.startswith("#!"):
            shebang = line
        elif line.strip().startswith("#SBATCH"):
            # Filter out resource-specific directives (we'll add our own)
            resource_flags = ("--nodes", "--ntasks-per-node", "--gres", "--time", "--output", "--error")
            if not any(flag in line for flag in resource_flags):
                sbatch_directives.append(line)
        else:
            other_content.append(line)
    
    # Ensure we have a shebang
    if not shebang:
        shebang = "#!/bin/bash"
    
    # Get walltime from config (default 24h for NEP training)
    # Check for train_nep_walltime first, fall back to general walltime, then default
    walltime = config.get("slurm", "train_nep_walltime", fallback=None)
    if not walltime:
        walltime = config.get("slurm", "walltime", fallback="24:00:00")

    nep_command = config.get(
        "hpc",
        "nep_command",
        fallback="mpirun --bind-to none $HOME/src/GPUMD/src/nep",
    ).strip()
    if not nep_command:
        nep_command = "mpirun --bind-to none $HOME/src/GPUMD/src/nep"
    
    # Build script with proper ordering: shebang → SBATCH directives → other commands → execution
    script_lines = [
        shebang,
        *sbatch_directives,
        f"#SBATCH --nodes=1",
        f"#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gres=gpu:1",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --job-name=nep_train_{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        f"#SBATCH --output=train_nep_%j.log",
        f"#SBATCH --error=train_nep_%j.err",
        "",
        *other_content,
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
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(potential_path),
    )

    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    # Extract job ID from output
    # sbatch outputs "Submitted batch job XXXXX"
    output = result.stdout.strip()
    job_id = output.split()[-1] if output else "UNKNOWN"

    return job_id
