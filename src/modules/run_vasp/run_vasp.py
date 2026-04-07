"""
VASP DFT calculation stage — orchestrator.

Sub-stages:
  prepare.py  — Create POSCAR/POTCAR/INCAR per structure
  prepare.py  — Write shared run_vasp.sh (self-healing adaptive script)
  launcher.py — Submit jobs, monitor via squeue, self-resubmit before walltime
"""

import logging
from configparser import ConfigParser
from pathlib import Path

from ..base import Stage
from .launcher import run_launcher
from .prepare import prepare_jobs, write_shared_vasp_script

logger = logging.getLogger("nepflow.run_vasp")


class RunVaspStage(Stage):
    """Prepare, submit, and monitor VASP DFT jobs."""

    def run(self) -> None:
        config_path = self._find_config_file()
        config = ConfigParser()
        config.read(config_path)

        vasp_config_dir = self.project_dir / "config" / "vasp"
        selected_dir = self.project_dir / "structures" / "selected"
        jobs_dir = self.project_dir / "vasp" / "jobs"
        vasp_dir = self.project_dir / "vasp"
        datasets = ["train", "test"]

        # Validate SLURM header + VASP command (needed for both phases)
        slurm_header_path = self.project_dir / "config" / "slurm" / "header.slurm"
        if not slurm_header_path.exists():
            raise FileNotFoundError(
                f"SLURM header not found at {slurm_header_path}\n"
                f"Place your header.slurm in: {slurm_header_path.parent}/"
            )
        slurm_header = slurm_header_path.read_text(encoding="utf-8")
        vasp_command = config.get("slurm", "vasp_command", fallback="mpirun -np {ntasks} vasp_std")

        # ----------------------------------------------------------
        # Phase 1a: Prepare job folders (idempotent — skips if done)
        # ----------------------------------------------------------
        prepare_jobs(config, vasp_config_dir, selected_dir, jobs_dir, datasets)

        # ----------------------------------------------------------
        # Phase 1b: Write/update shared run_vasp.sh (always, so code
        #           changes are picked up on resubmission)
        # ----------------------------------------------------------
        write_shared_vasp_script(vasp_dir, slurm_header, vasp_command)

        # ----------------------------------------------------------
        # Phase 2: Launcher loop (submit / monitor / resubmit)
        # ----------------------------------------------------------
        run_launcher(config, jobs_dir, vasp_dir, datasets,
                     self.project_name, self.project_dir)

    def _find_config_file(self) -> Path:
        project_config = self.project_dir / "config" / "project.config"
        if project_config.exists():
            return project_config
        if self.config_file.exists():
            return self.config_file
        raise FileNotFoundError(
            f"Config file not found. Tried:\n"
            f"  - {project_config}\n"
            f"  - {self.config_file}"
        )

