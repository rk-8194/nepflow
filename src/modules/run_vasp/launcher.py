"""Launcher sub-stage: submit, monitor, and handle OOM escalation for VASP jobs."""

import csv
import json
import os
import re
import shutil
import subprocess
import time
from configparser import ConfigParser
from pathlib import Path

from ._common import (
    VASP_COMPLETION_MARKERS,
    build_retry_levels_for_gpu,
    estimate_kpoints_irr,
    estimate_n_electrons,
    logger,
    parse_zval,
    predict_vasp_params,
    read_status,
    write_launcher_state,
    write_status,
)
from ..base import SelfResubmitExit


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
    debug: bool = False,
    slurm_deadline: float | None = None,
) -> None:
    """Submit, monitor, and resubmit VASP jobs.

    On OOM, run_vasp.sh writes a .vasp_oom_detected marker and exits.
    The launcher detects this marker, builds a GPU-aware escalation
    table via build_retry_levels_for_gpu(), and resubmits with
    escalated parameters (NCORE/KPAR/GPU count).

    When debug=True, SLURM interactions are simulated.
    """
    max_concurrent = config.getint("slurm", "max_concurrent", fallback=20)
    poll_interval = 0 if debug else config.getint("slurm", "poll_interval", fallback=30)
    max_retry = config.getint("slurm", "max_retry_level", fallback=6)
    vasp_walltime = config.get("slurm", "vasp_walltime", fallback="00:30:00")

    cores_per_node = config.getint("hpc", "cores_per_node", fallback=64)
    gpus_per_node = config.getint("hpc", "gpus_per_node", fallback=4)

    # Default params when prediction is unavailable: 1 GPU, standard NCORE
    default_ncore = max(2, cores_per_node // gpus_per_node)
    default_kpar = 1
    default_gpus = 1

    shared_script = vasp_dir / "run_vasp.sh"

    # --- VASP memory: pre-run parameter prediction -----------------------
    nepflow_root = project_dir.parent.parent  # projects/project_X → nepflow root
    zval_cache: dict = {}
    vasp_config_dir = project_dir / "config" / "vasp"
    if vasp_config_dir.exists():
        for potcar_file in vasp_config_dir.glob("POTCAR_*"):
            elem = potcar_file.name.split("_", 1)[1]
            try:
                zval_cache[elem] = parse_zval(potcar_file)
            except (ValueError, OSError) as e:
                logger.debug(f"  Could not parse ZVAL from {potcar_file.name}: {e}")
    if zval_cache:
        logger.info(f"  ZVAL cache: {zval_cache}")
    else:
        logger.warning("  No ZVAL cache — parameter prediction disabled")
    # ---------------------------------------------------------------------

    # Cancel any stale launcher for this project
    if not debug:
        _cancel_stale_launcher(vasp_dir)

    # Write our own launcher state
    launcher_job_id = os.environ.get("SLURM_JOB_ID", "local")
    # Walltime is now managed at the nepflow level
    write_launcher_state(vasp_dir, launcher_job_id, 0)

    if debug:
        logger.info("")
        logger.info("[DEBUG] Entering launcher loop (simulated SLURM)")
    logger.info("")
    logger.info("═" * 60)
    logger.info("VASP Launcher")
    logger.info("═" * 60)
    logger.info(f"  Max concurrent : {max_concurrent}")
    logger.info(f"  Poll interval  : {poll_interval}s")
    logger.info(f"  Max OOM retries: {max_retry}")
    logger.info(f"  VASP walltime  : {vasp_walltime}")
    logger.info(f"  Prediction     : {'k-NN (n_atoms, n_electrons, kpts_irr)' if zval_cache else 'disabled'}")
    logger.info("")

    _debug_job_counter = 0
    _poll_count = 0

    while True:
        _poll_count += 1

        # Check SLURM deadline before each poll
        if slurm_deadline is not None and time.time() >= slurm_deadline:
            logger.warning("SLURM deadline reached inside launcher — triggering resubmit")
            raise SelfResubmitExit("SLURM walltime deadline reached in launcher poll loop")

        # Batch squeue query: get running and pending job names separately
        running_names, pending_names = _get_project_job_names(project_name) if not debug else (set(), set())
        if debug:
            running_names = set()
            pending_names = set()
        active_names = running_names | pending_names  # Union: all jobs in squeue
        all_terminal = True
        waiting_count = 0
        pending_count = 0
        running_count = 0
        completed_count = 0

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

                if status == "completed":
                    completed_count += 1
                    continue
                
                if status == "failed":
                    continue

                all_terminal = False

                if status == "submitted":
                    # Check if job is in SLURM queue
                    if job_name in running_names:
                        running_count += 1
                        continue
                    
                    if job_name in pending_names:
                        pending_count += 1
                        continue

                    # Job left squeue — determine outcome
                    if _check_completed(struct_dir):
                        logger.info(f"  ✓ {ds}/{struct_dir.name}")
                        _on_success(struct_dir, nepflow_root, gpus_per_node)
                        write_status(
                            struct_dir, status="completed",
                            retry_level=status_data.get("retry_level", 0),
                        )
                        completed_count += 1
                        continue

                    # Not completed — check for OOM marker from script
                    oom_marker = struct_dir / ".vasp_oom_detected"
                    if oom_marker.exists():
                        # Read retry level from .vasp_status (authoritative),
                        # NOT .vasp_retry_level (legacy, no longer written)
                        retry_count = status_data.get("retry_level", 0)
                        initial_gpu = status_data.get("initial_gpu", 1)
                        initial_ncore = status_data.get("initial_ncore")
                        initial_kpar = status_data.get("initial_kpar")
                        current_gpu = status_data.get("current_gpu", initial_gpu)
                        
                        # Read current INCAR values (for logging, and as
                        # fallback if initial params not stored yet)
                        current_ncore = 2
                        current_kpar = 1
                        incar_path = struct_dir / "INCAR"
                        if incar_path.exists():
                            try:
                                incar_content = incar_path.read_text(encoding="utf-8")
                                for line in incar_content.splitlines():
                                    if re.match(r"^\s*NCORE\s*=", line, re.IGNORECASE):
                                        current_ncore = int(line.split("=")[1].split()[0])
                                    elif re.match(r"^\s*KPAR\s*=", line, re.IGNORECASE):
                                        current_kpar = int(line.split("=")[1].split()[0])
                            except (OSError, ValueError):
                                pass
                        
                        # Use initial params for stable escalation table;
                        # fall back to current INCAR if not stored
                        if initial_ncore is None:
                            initial_ncore = current_ncore
                        if initial_kpar is None:
                            initial_kpar = current_kpar
                        
                        logger.info(
                            f"  ⚠ {ds}/{struct_dir.name}: OOM "
                            f"(retry {retry_count}/{max_retry}, "
                            f"gpu={current_gpu} NCORE={current_ncore} KPAR={current_kpar})"
                        )
                        
                        # Log OOM to .vasp_memory
                        _log_oom(struct_dir, nepflow_root, current_gpu, current_ncore, current_kpar, zval_cache, config)
                        
                        # Check max retries
                        if retry_count >= max_retry:
                            logger.warning(
                                f"  ✗ {ds}/{struct_dir.name}: max retries exhausted → failed/"
                            )
                            failed_dir = jobs_dir / ds / "failed"
                            failed_dir.mkdir(parents=True, exist_ok=True)
                            dest = failed_dir / struct_dir.name
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.move(str(struct_dir), str(dest))
                            continue
                        
                        # Build escalation table from ORIGINAL params
                        # (stable across retries — not affected by INCAR rewrites)
                        escalation_levels = build_retry_levels_for_gpu(
                            initial_gpu, initial_ncore, initial_kpar, config
                        )
                        
                        # retry_count is the number of OOM retries already
                        # attempted; it indexes directly into the table
                        esc_index = retry_count
                        next_retry = retry_count + 1
                        if esc_index < len(escalation_levels):
                            ncore, kpar, nodes, gpus = escalation_levels[esc_index]
                            logger.info(
                                f"    → escalate [{next_retry}]: "
                                f"NCORE={ncore} KPAR={kpar} gpus={nodes * gpus}"
                            )
                            
                            # Clean for fresh start
                            for f in ["CHG", "CHGCAR", "WAVECAR", "CONTCAR",
                                      "DOSCAR", "EIGENVAL", "PCDAT",
                                      "OUTCAR", "vasprun.xml", "OSZICAR",
                                      "vasp_output.log", ".vasp_oom_detected"]:
                                (struct_dir / f).unlink(missing_ok=True)
                            
                            # Update INCAR for escalated params
                            _write_incar_params(struct_dir, ncore, kpar)
                            
                            # Resubmit with escalated parameters
                            if debug:
                                _debug_job_counter += 1
                                job_id = _submit_job_debug(
                                    struct_dir, _debug_job_counter, fail=False,
                                )
                            else:
                                job_id = _submit_job(
                                    shared_script, struct_dir, job_name,
                                    nodes, gpus, vasp_walltime,
                                )
                            
                            if job_id:
                                logger.info(
                                    f"    → resubmitted job {job_id}"
                                )
                                write_status(
                                    struct_dir, status="submitted",
                                    retry_level=next_retry,
                                    slurm_job_id=job_id,
                                    initial_gpu=initial_gpu,
                                    initial_ncore=initial_ncore,
                                    initial_kpar=initial_kpar,
                                    current_gpu=nodes * gpus,
                                )
                                active_names.add(job_name)
                            continue
                        else:
                            logger.warning(
                                f"  ✗ {ds}/{struct_dir.name}: escalation exhausted → failed/"
                            )
                            failed_dir = jobs_dir / ds / "failed"
                            failed_dir.mkdir(parents=True, exist_ok=True)
                            dest = failed_dir / struct_dir.name
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.move(str(struct_dir), str(dest))
                            continue
                    
                    # OOM marker not found — check for other failures
                    file_retry = status_data.get("retry_level", 0)

                    if file_retry >= max_retry:
                        logger.warning(f"  ✗ {ds}/{struct_dir.name}: max retries exhausted → failed/")
                    elif (struct_dir / "vasp_output.log").exists():
                        logger.warning(f"  ✗ {ds}/{struct_dir.name}: non-OOM error → failed/")
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
                    waiting_count += 1
                    if len(active_names) < max_concurrent:
                        retry_level = status_data.get("retry_level", 0)

                        # Predict VASP params from memory for fresh jobs
                        predicted = None
                        n_kpoints_irr = None
                        if retry_level == 0 and zval_cache:
                            try:
                                n_atoms = sum(
                                    int(x) for x in
                                    (struct_dir / "POSCAR").read_text(
                                        encoding="utf-8"
                                    ).splitlines()[6].split()
                                )
                                n_electrons = estimate_n_electrons(
                                    struct_dir, zval_cache,
                                )
                                n_kpoints_irr = estimate_kpoints_irr(
                                    struct_dir, config,
                                )
                                predicted = predict_vasp_params(
                                    n_atoms, n_electrons,
                                    nepflow_root, config,
                                    n_kpoints_irr=n_kpoints_irr,
                                )
                            except (OSError, ValueError, IndexError) as e:
                                logger.debug(
                                    f"  Could not predict params for "
                                    f"{struct_dir.name}: {e}"
                                )

                        if predicted:
                            ncore, kpar, nodes, gpus = predicted
                            param_source = "predicted"
                        else:
                            ncore = default_ncore
                            kpar = default_kpar
                            nodes = 1
                            gpus = default_gpus
                            param_source = "default"
                        
                        kpt_info = f" kpts={n_kpoints_irr}" if n_kpoints_irr else ""
                        _write_incar_params(struct_dir, ncore, kpar)

                        if debug:
                            _debug_job_counter += 1
                            # Every 4th job is simulated as a failure
                            fail = (_debug_job_counter % 4 == 0)
                            job_id = _submit_job_debug(
                                struct_dir, _debug_job_counter, fail=fail,
                            )
                        else:
                            job_id = _submit_job(
                                shared_script, struct_dir, job_name,
                                nodes, gpus,
                                vasp_walltime,
                            )
                        if job_id:
                            logger.info(
                                f"  → {ds}/{struct_dir.name}: "
                                f"NCORE={ncore:<3d} KPAR={kpar} gpus={nodes * gpus}"
                                f"{kpt_info} ({param_source}) → job {job_id}"
                            )
                            write_status(
                                struct_dir, status="submitted",
                                retry_level=retry_level,
                                slurm_job_id=job_id,
                                initial_gpu=gpus,
                                initial_ncore=ncore,
                                initial_kpar=kpar,
                                current_gpu=nodes * gpus,
                            )
                            active_names.add(job_name)

        # Check if everything is terminal
        if all_terminal:
            logger.info("")
            logger.info("═" * 60)
            logger.info("All VASP jobs completed or failed")
            logger.info("═" * 60)
            _log_summary(jobs_dir, datasets, nepflow_root)
            return  # Normal return → workflow.py advances stage

        # Poll summary
        total = waiting_count + pending_count + running_count + completed_count
        logger.info(
            f"  [{completed_count}/{total} done] "
            f"running={running_count} pending={pending_count} "
            f"waiting={waiting_count}"
        )

        if poll_interval > 0:
            time.sleep(poll_interval)


# ==================================================================
# SLURM interaction
# ==================================================================

def _get_project_job_names(project_name: str) -> tuple[set[str], set[str]]:
    """Batch squeue query: return (running_names, pending_names) for this project."""
    running = set()
    pending = set()
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "--noheader",
             "-o", "%j %T", "--states=RUNNING,PENDING"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode == 0:
            prefix = f"nf_{project_name}_"
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().rsplit(None, 1)
                if len(parts) == 2:
                    name, state = parts
                    if name.startswith(prefix):
                        if state == "RUNNING":
                            running.add(name)
                        elif state == "PENDING":
                            pending.add(name)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return running, pending


def _write_incar_params(struct_dir: Path, ncore: int, kpar: int) -> None:
    """Update NCORE and KPAR in the structure's INCAR file."""
    incar_path = struct_dir / "INCAR"
    if not incar_path.exists():
        logger.warning(f"    INCAR not found: {incar_path}")
        return
    try:
        content = incar_path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        output = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                output.append(line)
            elif re.match(r"^NCORE\s*=", stripped, re.IGNORECASE):
                output.append(f"NCORE = {ncore}\n")
            elif re.match(r"^KPAR\s*=", stripped, re.IGNORECASE):
                output.append(f"KPAR = {kpar}\n")
            else:
                output.append(line)
        incar_path.write_text("".join(output), encoding="utf-8")
    except OSError as e:
        logger.warning(f"    Failed to write INCAR: {e}")


def _submit_job(
    shared_script: Path,
    struct_dir: Path,
    job_name: str,
    nodes: int,
    gpus: int,
    vasp_walltime: str,
) -> str | None:
    """Submit run_vasp.sh for a structure via sbatch."""
    try:
        sbatch_args = [
            "sbatch",
            f"--job-name={job_name}",
            f"--time={vasp_walltime}",
            f"--nodes={nodes}",
            f"--ntasks-per-node={gpus}",
            f"--gres=gpu:{gpus}",
            f"--output={struct_dir}/vasp_%j.out",
            f"--error={struct_dir}/vasp_%j.err",
            str(shared_script),
            str(struct_dir.resolve()),
        ]
        result = subprocess.run(
            sbatch_args,
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode == 0 and "Submitted batch job" in result.stdout:
            job_id = result.stdout.strip().split()[-1]
            return job_id
        logger.warning(f"  sbatch failed for {struct_dir.name}: {result.stderr.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"  sbatch error for {struct_dir.name}: {e}")
    return None


def _submit_job_debug(struct_dir: Path, counter: int, fail: bool = False) -> str:
    """Simulate VASP submission.

    When *fail* is True the stub OUTCAR omits the completion marker and a
    ``vasp_output.log`` is written, which triggers the "FAILED (non-OOM
    error)" path in the launcher state machine.
    """
    outcar = struct_dir / "OUTCAR"
    if fail:
        outcar.write_text(
            "  VASP simulated run — convergence NOT reached\n",
            encoding="utf-8",
        )
        (struct_dir / "vasp_output.log").write_text(
            "ERROR: debug-simulated failure\n", encoding="utf-8",
        )
    else:
        outcar.write_text(
            "  General timing and accounting informations for this job:\n"
            "  LOOP:  cpu time   10.00: real time   10.00\n"
            "  Found     1 irreducible k-points\n"
            "  NELECT =      100.0000\n"
            "  Voluntary context switches:        42\n",
            encoding="utf-8",
        )
    return f"DEBUG_{counter:05d}"


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


# ==================================================================
# success / performance logging
# ==================================================================

def _log_oom(
    struct_dir: Path,
    nepflow_root: Path,
    gpus: int,
    ncore: int,
    kpar: int,
    zval_cache: dict,
    config: "ConfigParser",
) -> None:
    """Log OOM event to .vasp_memory CSV with oom=1."""
    poscar = struct_dir / "POSCAR"
    csv_path = nepflow_root / ".vasp_memory"

    try:
        # Parse n_atoms from POSCAR
        poscar_lines = poscar.read_text(encoding="utf-8").splitlines()
        n_atoms = sum(int(x) for x in poscar_lines[6].split())

        # Try to get n_kpoints_irr and n_electrons from OUTCAR;
        # fall back to estimators if OUTCAR is absent (typical for OOM)
        outcar = struct_dir / "OUTCAR"
        n_kpoints_irr = 0
        n_electrons = 0
        if outcar.exists():
            try:
                outcar_text = outcar.read_text(encoding="utf-8", errors="replace")
                m = re.search(r"Found\s+(\d+)\s+irreducible k-points", outcar_text)
                if m:
                    n_kpoints_irr = int(m.group(1))
                m = re.search(r"NELECT\s*=\s*([\d.]+)", outcar_text)
                if m:
                    n_electrons = int(float(m.group(1)))
            except OSError:
                pass
        if n_kpoints_irr == 0:
            est = estimate_kpoints_irr(struct_dir, config)
            if est is not None:
                n_kpoints_irr = est
        if n_electrons == 0 and zval_cache:
            try:
                n_electrons = estimate_n_electrons(struct_dir, zval_cache)
            except (KeyError, OSError, ValueError, IndexError):
                pass

        # Nodes: 1 (single node, OOM'd before multi-node)
        nodes = 1

        # Write CSV header if file is new
        write_header = not csv_path.exists()
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "n_atoms", "n_kpoints_irr",
                    "n_electrons", "nodes", "gpus", "ncore", "kpar",
                    "avg_loop_time", "oom",
                ])
            writer.writerow([
                n_atoms, n_kpoints_irr,
                n_electrons, nodes, gpus, ncore, kpar,
                "0.0000", 1,
            ])

        logger.info(
            f"  Logged: OOM {n_atoms} atoms, GPU={gpus} "
            f"NCORE={ncore} KPAR={kpar}"
        )

    except (OSError, ValueError, IndexError) as e:
        logger.debug(f"  Could not log OOM for {struct_dir.name}: {e}")


def _on_success(struct_dir: Path, nepflow_root: Path, gpus_per_node: int) -> None:
    """Log performance on successful completion (cleanup done by run_vasp.sh)."""
    _log_performance(struct_dir, nepflow_root, gpus_per_node)


def _log_performance(struct_dir: Path, nepflow_root: Path, gpus_per_node: int) -> None:
    """Parse OUTCAR for performance data and append to .vasp_memory CSV."""
    outcar = struct_dir / "OUTCAR"
    poscar = struct_dir / "POSCAR"
    incar = struct_dir / "INCAR"
    csv_path = nepflow_root / ".vasp_memory"

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

        # Parse MPI ranks from OUTCAR to determine nodes/gpus
        outcar_text = outcar.read_text(encoding="utf-8", errors="replace")

        total_ranks = 0
        # GPU VASP: "running N mpi-ranks";  CPU VASP: "running on N total cores"
        m_ranks = re.search(r"running\s+(\d+)\s+mpi-ranks", outcar_text)
        if not m_ranks:
            m_ranks = re.search(r"running on\s+(\d+)\s+total cores", outcar_text)
        if m_ranks:
            total_ranks = int(m_ranks.group(1))
        nodes = max(1, total_ranks // gpus_per_node) if total_ranks > 0 else 1
        gpus = total_ranks if total_ranks > 0 else gpus_per_node

        loop_times = [
            float(m.group(1))
            for m in re.finditer(
                r"LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)", outcar_text
            )
        ]
        avg_loop = sum(loop_times) / len(loop_times) if loop_times else 0.0

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
                    "n_atoms", "n_kpoints_irr",
                    "n_electrons", "nodes", "gpus", "ncore", "kpar",
                    "avg_loop_time", "oom",
                ])
            writer.writerow([
                n_atoms, n_kpoints_irr,
                n_electrons, nodes, gpus, ncore, kpar,
                f"{avg_loop:.4f}", 0,
            ])
        
        logger.info(
            f"  Logged: {n_atoms} atoms, GPU={gpus} "
            f"NCORE={ncore} KPAR={kpar}, avg_loop={avg_loop:.2f}s"
        )

    except (OSError, ValueError, IndexError) as e:
        logger.debug(f"  Could not log performance for {struct_dir.name}: {e}")


# ==================================================================
# summary
# ==================================================================

def _log_summary(jobs_dir: Path, datasets: list[str], nepflow_root: Path) -> None:
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
        logger.info(f"  {ds:<8s} ✓ {completed:>4d} completed   ✗ {failed:>4d} failed")

    csv_path = nepflow_root / ".vasp_memory"
    if csv_path.exists():
        logger.info(f"  Memory: {csv_path}")
