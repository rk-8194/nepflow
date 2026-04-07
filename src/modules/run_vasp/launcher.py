"""Launcher sub-stage: submit, monitor, and self-resubmit VASP jobs."""

import csv
import json
import os
import re
import shutil
import subprocess
import time
from configparser import ConfigParser
from pathlib import Path

from ..base import SelfResubmitExit
from ._common import (
    RETRY_LEVELS,
    VASP_COMPLETION_MARKERS,
    format_walltime,
    logger,
    parse_walltime,
    read_retry_level,
    read_status,
    write_launcher_state,
    write_status,
)


# ==================================================================
# main launcher loop
# ==================================================================

def run_launcher(
    config: ConfigParser,
    jobs_dir: Path,
    vasp_dir: Path,
    datasets: list[str],
    project_name: str,
    project_dir: Path,
) -> None:
    """Submit, monitor, and self-resubmit VASP jobs.

    OOM retry escalation is handled by run_vasp.sh itself (self-requeue).
    The launcher detects self-requeues by comparing .vasp_retry_level
    against the last tracked retry_level in .vasp_status.
    """
    max_concurrent = config.getint("slurm", "max_concurrent", fallback=20)
    poll_interval = config.getint("slurm", "poll_interval", fallback=30)
    max_retry = config.getint("slurm", "max_retry_level", fallback=6)
    walltime_str = config.get("slurm", "walltime", fallback="03:00:00")

    walltime_seconds = parse_walltime(walltime_str)
    margin_seconds = 600  # 10 minutes
    start_time = time.time()
    deadline = start_time + walltime_seconds - margin_seconds

    shared_script = vasp_dir / "run_vasp.sh"

    # Cancel any stale launcher for this project
    _cancel_stale_launcher(vasp_dir)

    # Write our own launcher state
    launcher_job_id = os.environ.get("SLURM_JOB_ID", "local")
    write_launcher_state(vasp_dir, launcher_job_id, walltime_seconds)

    logger.info("")
    logger.info("Entering launcher loop")
    logger.info(f"  Max concurrent: {max_concurrent}")
    logger.info(f"  Walltime: {walltime_str} ({walltime_seconds}s), deadline in {walltime_seconds - margin_seconds}s")
    logger.info(f"  Poll interval: {poll_interval}s")

    while True:
        # Batch squeue query: all active job names for this project
        active_names = _get_project_job_names(project_name)
        all_terminal = True

        for ds in datasets:
            ds_jobs_dir = jobs_dir / ds
            if not ds_jobs_dir.exists():
                continue

            struct_dirs = sorted(
                [d for d in ds_jobs_dir.iterdir()
                 if d.is_dir() and d.name.startswith("struct_")],
                key=lambda p: p.name,
            )

            for struct_dir in struct_dirs:
                status_data = read_status(struct_dir)
                status = status_data.get("status", "pending")
                struct_num = int(struct_dir.name.split("_")[1])
                job_name = f"nf_{project_name}_{ds}_{struct_num:04d}"

                if status in ("completed", "failed"):
                    continue

                all_terminal = False

                if status == "submitted":
                    # Still running (or self-requeued with same name)?
                    if job_name in active_names:
                        continue

                    # Job left squeue — determine outcome
                    if _check_completed(struct_dir):
                        logger.info(f"  {ds}/{struct_dir.name}: completed")
                        _on_success(struct_dir, ds, vasp_dir)
                        write_status(
                            struct_dir, status="completed",
                            retry_level=read_retry_level(struct_dir),
                        )
                        continue

                    # Not completed — did the script self-requeue?
                    file_retry = read_retry_level(struct_dir)
                    tracked_retry = status_data.get("retry_level", 0)

                    if file_retry > tracked_retry:
                        # Script bumped .vasp_retry_level → it resubmitted
                        logger.info(f"  {ds}/{struct_dir.name}: OOM → self-requeued (level {file_retry})")
                        write_status(
                            struct_dir, status="submitted",
                            retry_level=file_retry,
                        )
                        continue

                    # Not completed, not self-requeued
                    if file_retry >= max_retry:
                        logger.warning(f"  {ds}/{struct_dir.name}: FAILED (max retries exhausted)")
                    elif (struct_dir / "vasp_output.log").exists():
                        logger.warning(f"  {ds}/{struct_dir.name}: FAILED (non-OOM error)")
                    else:
                        # Never ran or outputs cleaned — retry submission
                        write_status(struct_dir, status="pending",
                                     retry_level=file_retry)
                        continue

                    # Move to failed/
                    failed_dir = jobs_dir / ds / "failed"
                    failed_dir.mkdir(parents=True, exist_ok=True)
                    dest = failed_dir / struct_dir.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(struct_dir), str(dest))

                elif status == "pending":
                    if len(active_names) < max_concurrent:
                        retry_level = read_retry_level(struct_dir)
                        idx = min(retry_level, len(RETRY_LEVELS) - 1)
                        _, _, nodes, gpus = RETRY_LEVELS[idx]
                        job_id = _submit_job(
                            shared_script, struct_dir, job_name,
                            nodes, gpus,
                        )
                        if job_id:
                            write_status(
                                struct_dir, status="submitted",
                                retry_level=retry_level,
                                slurm_job_id=job_id,
                            )
                            active_names.add(job_name)
                            logger.info(f"  {ds}/{struct_dir.name}: submitted (job {job_id})")

        # Check if everything is terminal
        if all_terminal:
            logger.info("")
            logger.info("All VASP jobs completed or failed")
            _log_summary(jobs_dir, datasets, vasp_dir)
            return  # Normal return → workflow.py advances stage

        # Check walltime deadline
        if time.time() >= deadline:
            logger.info("")
            logger.info("Approaching walltime — resubmitting launcher")
            _resubmit_self(vasp_dir, walltime_seconds, project_name, project_dir)
            raise SelfResubmitExit("Launcher resubmitted before walltime expiry")

        time.sleep(poll_interval)


# ==================================================================
# SLURM interaction
# ==================================================================

def _get_project_job_names(project_name: str) -> set[str]:
    """Batch squeue query: return set of active job names for this project."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "--noheader",
             "-o", "%j", "--states=RUNNING,PENDING"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode == 0:
            prefix = f"nf_{project_name}_"
            return {
                name.strip()
                for name in result.stdout.strip().split("\n")
                if name.strip().startswith(prefix)
            }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return set()


def _submit_job(
    shared_script: Path,
    struct_dir: Path,
    job_name: str,
    nodes: int,
    gpus_per_node: int,
) -> str | None:
    """Submit run_vasp.sh for a structure via sbatch."""
    try:
        result = subprocess.run(
            [
                "sbatch",
                f"--job-name={job_name}",
                f"--nodes={nodes}",
                f"--ntasks-per-node={gpus_per_node}",
                f"--gres=gpu:{gpus_per_node}",
                str(shared_script),
                str(struct_dir.resolve()),
            ],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode == 0 and "Submitted batch job" in result.stdout:
            return result.stdout.strip().split()[-1]
        logger.warning(f"  sbatch failed for {struct_dir.name}: {result.stderr.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"  sbatch error for {struct_dir.name}: {e}")
    return None


def _job_in_squeue(job_id: str) -> bool:
    """Check if a specific job ID is still in squeue."""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _check_completed(struct_dir: Path) -> bool:
    """Check if VASP completed successfully by inspecting OUTCAR tail."""
    outcar = struct_dir / "OUTCAR"
    if not outcar.exists():
        return False
    try:
        with open(outcar, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 50_000))
            tail = f.read()
        return any(marker in tail for marker in VASP_COMPLETION_MARKERS)
    except OSError:
        return False


def _cancel_stale_launcher(vasp_dir: Path) -> None:
    """Cancel any previous launcher job for this project."""
    state_file = vasp_dir / ".launcher_state"
    if not state_file.exists():
        return
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        old_id = state.get("launcher_job_id", "")
        if old_id and old_id != "local" and _job_in_squeue(old_id):
            my_id = os.environ.get("SLURM_JOB_ID", "")
            if old_id != my_id:
                logger.info(f"  Cancelling stale launcher job {old_id}")
                subprocess.run(["scancel", old_id], capture_output=True, timeout=10, check=False)
    except (json.JSONDecodeError, OSError):
        pass


def _resubmit_self(
    vasp_dir: Path,
    walltime_seconds: int,
    project_name: str,
    project_dir: Path,
) -> None:
    """Resubmit nepflow as a new SLURM job and update launcher state."""
    slurm_header_path = project_dir / "config" / "slurm" / "header.slurm"
    nepflow_path = Path(__file__).resolve().parents[3] / "nepflow.py"

    header = slurm_header_path.read_text(encoding="utf-8").rstrip("\n")
    wt = format_walltime(walltime_seconds)

    script = re.sub(r"#SBATCH\s+--time=\S+", f"#SBATCH --time={wt}", header)
    script = re.sub(
        r"#SBATCH\s+--output=\S+",
        f"#SBATCH --output={project_dir}/logs/launcher_%j.out",
        script,
    )
    script = re.sub(
        r"#SBATCH\s+--error=\S+",
        f"#SBATCH --error={project_dir}/logs/launcher_%j.err",
        script,
    )
    if "--job-name" not in script:
        script = script.replace(
            "#!/bin/bash",
            f"#!/bin/bash\n#SBATCH --job-name=nf_{project_name}_launcher",
        )

    script += f"\n\npython3 {nepflow_path} --project {project_name} --stage run_vasp\n"

    resubmit_script = vasp_dir / ".launcher_resubmit.sh"
    resubmit_script.write_text(script, encoding="utf-8")

    try:
        result = subprocess.run(
            ["sbatch", str(resubmit_script)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode == 0 and "Submitted batch job" in result.stdout:
            new_id = result.stdout.strip().split()[-1]
            logger.info(f"  Resubmitted as job {new_id}")
            write_launcher_state(vasp_dir, new_id, walltime_seconds)
        else:
            logger.error(f"  Resubmission failed: {result.stderr.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"  Resubmission error: {e}")


# ==================================================================
# success / performance logging
# ==================================================================

def _on_success(struct_dir: Path, dataset: str, vasp_dir: Path) -> None:
    """Log performance on successful completion (cleanup done by run_vasp.sh)."""
    _log_performance(struct_dir, dataset, vasp_dir)


def _log_performance(struct_dir: Path, dataset: str, vasp_dir: Path) -> None:
    """Parse OUTCAR for performance data and append to .vasp_memory CSV."""
    outcar = struct_dir / "OUTCAR"
    poscar = struct_dir / "POSCAR"
    incar = struct_dir / "INCAR"
    csv_path = vasp_dir / ".vasp_memory"

    try:
        # Parse n_atoms from POSCAR
        poscar_lines = poscar.read_text(encoding="utf-8").splitlines()
        n_atoms = sum(int(x) for x in poscar_lines[6].split())

        # Parse NCORE/KPAR from INCAR (reflects actual parameters used)
        incar_text = incar.read_text(encoding="utf-8")
        ncore = kpar = 0
        for line in incar_text.splitlines():
            stripped = line.strip().upper()
            if stripped.startswith("NCORE"):
                ncore = int(line.split("=")[1].split("#")[0].strip())
            elif stripped.startswith("KPAR"):
                kpar = int(line.split("=")[1].split("#")[0].strip())

        # Infer nodes/gpus from NCORE/KPAR via RETRY_LEVELS table
        nodes, gpus_per_node = 1, 4
        for nc, kp, nd, gp in RETRY_LEVELS:
            if nc == ncore and kp == kpar:
                nodes, gpus_per_node = nd, gp
                break

        # Parse OUTCAR for LOOP times and k-points
        outcar_text = outcar.read_text(encoding="utf-8", errors="replace")

        loop_times = [
            float(m.group(1))
            for m in re.finditer(
                r"LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)", outcar_text
            )
        ]
        avg_loop = sum(loop_times) / len(loop_times) if loop_times else 0.0
        total_time = sum(loop_times)

        # Irreducible k-points
        n_kpoints_irr = 0
        m = re.search(r"Found\s+(\d+)\s+irreducible k-points", outcar_text)
        if m:
            n_kpoints_irr = int(m.group(1))

        # Number of electrons
        n_electrons = 0
        m = re.search(r"NELECT\s*=\s*([\d.]+)", outcar_text)
        if m:
            n_electrons = int(float(m.group(1)))

        # Write CSV header if file is new
        write_header = not csv_path.exists()
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "struct_name", "dataset", "n_atoms", "n_kpoints_irr",
                    "n_electrons", "nodes", "gpus", "ncore", "kpar",
                    "avg_loop_time", "total_time",
                ])
            writer.writerow([
                struct_dir.name, dataset, n_atoms, n_kpoints_irr,
                n_electrons, nodes, nodes * gpus_per_node, ncore, kpar,
                f"{avg_loop:.4f}", f"{total_time:.2f}",
            ])

    except (OSError, ValueError, IndexError) as e:
        logger.debug(f"  Could not log performance for {struct_dir.name}: {e}")


# ==================================================================
# summary
# ==================================================================

def _log_summary(jobs_dir: Path, datasets: list[str], vasp_dir: Path) -> None:
    """Log final summary of completed/failed jobs."""
    for ds in datasets:
        ds_dir = jobs_dir / ds
        failed_dir = ds_dir / "failed"
        completed = failed = 0
        for d in ds_dir.iterdir():
            if not d.is_dir() or d.name == "failed":
                continue
            status_file = d / ".vasp_status"
            if status_file.exists():
                try:
                    s = json.loads(status_file.read_text(encoding="utf-8"))
                    if s.get("status") == "completed":
                        completed += 1
                except (json.JSONDecodeError, OSError):
                    pass
        if failed_dir.exists():
            failed = sum(1 for d in failed_dir.iterdir() if d.is_dir())
        logger.info(f"  {ds}: {completed} completed, {failed} failed")

    csv_path = vasp_dir / ".vasp_memory"
    if csv_path.exists():
        logger.info(f"  Performance log: {csv_path}")
