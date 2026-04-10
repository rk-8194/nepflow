"""
VASP DFT calculation stage — orchestrator.

Sub-stages:
  prepare.py  — Create POSCAR/POTCAR/INCAR per structure
  prepare.py  — Write shared run_vasp.sh
  launcher.py — Submit jobs, monitor via squeue, handle OOM escalation
"""

import logging
from configparser import ConfigParser
from pathlib import Path

from ase.io import iread, write as ase_write

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

        selected_dir = self.project_dir / "structures" / "selected"
        jobs_dir = self.project_dir / "vasp" / "jobs"
        vasp_dir = self.project_dir / "vasp"
        datasets = ["train", "test"]

        # === DEBUG: create stub job dirs + run launcher with simulated SLURM ===
        if self.debug:
            self._prepare_debug_jobs(selected_dir, jobs_dir, vasp_dir, datasets)
            run_launcher(config, jobs_dir, vasp_dir, datasets,
                         self.project_name, self.project_dir, debug=True)
            return
        vasp_config_dir = self.project_dir / "config" / "vasp"

        # Validate SLURM header + VASP command (needed for both phases)
        slurm_header_path = self.project_dir / "config" / "slurm" / "header.slurm"
        if not slurm_header_path.exists():
            raise FileNotFoundError(
                f"SLURM header not found at {slurm_header_path}\n"
                f"Place your header.slurm in: {slurm_header_path.parent}/"
            )
        slurm_header = slurm_header_path.read_text(encoding="utf-8")

        # Strip resource directives from header — we set these per-job
        # via sbatch command-line args (--nodes, --ntasks-per-node, etc.)
        resource_flags = (
            "--nodes", "--ntasks-per-node", "--gres",
            "--time", "--output", "--error",
        )
        slurm_header = "".join(
            line for line in slurm_header.splitlines(keepends=True)
            if not (
                line.strip().startswith("#SBATCH")
                and any(flag in line for flag in resource_flags)
            )
        )

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
                     self.project_name, self.project_dir,
                     slurm_deadline=self.slurm_deadline)

    def _prepare_debug_jobs(
        self, selected_dir: Path, jobs_dir: Path, vasp_dir: Path,
        datasets: list[str],
    ) -> None:
        """Create minimal stub job directories from selected structures."""
        # Idempotent — skip if job dirs already prepared
        first_ds_dir = jobs_dir / datasets[0]
        if first_ds_dir.exists() and any(first_ds_dir.glob("struct_*")):
            logger.info("[DEBUG] Stub job directories already exist — skipping preparation")
            return

        logger.info("[DEBUG] Preparing stub VASP job directories")

        for ds in datasets:
            xyz_path = selected_dir / f"{ds}.xyz"
            if not xyz_path.exists():
                logger.warning(f"[DEBUG] {xyz_path} not found — skipping {ds}")
                continue

            ds_jobs_dir = jobs_dir / ds
            ds_jobs_dir.mkdir(parents=True, exist_ok=True)

            for idx, atoms in enumerate(iread(str(xyz_path), format="extxyz")):
                struct_dir = ds_jobs_dir / f"struct_{idx:04d}"
                struct_dir.mkdir(parents=True, exist_ok=True)

                # Minimal POSCAR
                poscar_path = struct_dir / "POSCAR"
                ase_write(str(poscar_path), atoms, format="vasp")

                # Minimal INCAR (for performance-log parsing)
                incar_path = struct_dir / "INCAR"
                incar_path.write_text(
                    "NCORE = 16\nKPAR = 1\n",
                    encoding="utf-8",
                )

            n_structs = len(list(ds_jobs_dir.glob("struct_*")))
            logger.info(f"[DEBUG] Created {n_structs} stub job dirs for {ds}")

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

