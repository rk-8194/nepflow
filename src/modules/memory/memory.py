"""
VASP memory benchmarking stage.

Runs systematic VASP benchmarks on BCC W supercells across GPU counts
and NCORE/KPAR combinations. Records timings and OOM events to
.vasp_memory for use by the production parameter predictor.

Usage: python nepflow.py --project <name> --memory [--debug]
"""

import csv
import json
import os
import re
import subprocess
import time
from configparser import ConfigParser
from pathlib import Path
from typing import Optional

import numpy as np
from ase.build import bulk
from ase.io import write as ase_write

from ..base import Stage
from ..run_vasp._common import (
    VASP_COMPLETION_MARKERS,
    logger,
    parse_zval,
    read_status,
    write_status,
)
from ..run_vasp.prepare import inject_incar_defaults, write_poscar

CSV_HEADER = [
    "n_atoms", "n_kpoints_irr", "n_electrons",
    "nodes", "gpus", "ncore", "kpar", "avg_loop_time", "oom",
]


class MemoryStage(Stage):
    """Run VASP memory/performance benchmarks to populate .vasp_memory."""

    def run(self) -> None:
        config = self._load_config()

        vasp_dir = self.project_dir / "vasp"
        memory_dir = vasp_dir / "memory"
        vasp_config_dir = self.project_dir / "config" / "vasp"
        nepflow_root = self.project_dir.parent.parent

        # Phase 2: Generate benchmark structures
        supercells = self._generate_supercells(config)

        # Phase 3: Prepare job matrix
        combos = self._build_param_combos(config)
        self._prepare_jobs(
            config, memory_dir, vasp_config_dir, supercells, combos,
        )

        # Phase 4: Write runner script
        self._write_runner_script(config, memory_dir)

        # Phase 5: Submit & monitor
        self._run_launcher(config, memory_dir, debug=self.debug)

        # Phase 6: Collect results
        self._collect_results(memory_dir, nepflow_root)

    # ==================================================================
    # Phase 2: Generate BCC W supercells
    # ==================================================================

    def _generate_supercells(
        self, config: ConfigParser,
    ) -> list[tuple[int, "ase.Atoms"]]:
        """Generate BCC W supercells from unit cell up to target_n_atoms.

        Returns list of (n_atoms, atoms) tuples.
        """
        target = config.getint("generation", "target_n_atoms", fallback=128)
        unit = bulk("W", "bcc", cubic=True)  # 2 atoms in conventional cell
        n_unit = len(unit)

        supercells = []
        n = 1
        while True:
            sc = unit * (n, n, n)
            n_atoms = len(sc)
            if n_atoms > target:
                break
            supercells.append((n_atoms, sc))
            n += 1

        if not supercells:
            supercells.append((n_unit, unit))

        sizes = [s[0] for s in supercells]
        logger.info(f"  Benchmark supercells: {sizes} atoms (BCC W)")
        return supercells

    # ==================================================================
    # Phase 3: Build parameter combinations
    # ==================================================================

    def _build_param_combos(
        self, config: ConfigParser,
    ) -> list[tuple[int, int, int]]:
        """Build (gpus, ncore, kpar) combinations to benchmark.

        Returns sorted list of unique (gpus, ncore, kpar) tuples.
        """
        cores = config.getint("hpc", "cores_per_node", fallback=64)
        gpus_per_node = config.getint("hpc", "gpus_per_node", fallback=4)

        # Valid NCORE values: powers of 2 that divide cores_per_node
        valid_ncores = sorted(
            p for p in (2**i for i in range(1, 12))
            if p <= cores and cores % p == 0
        )
        if not valid_ncores:
            valid_ncores = [cores]

        # GPU counts to test: 1, 2, ... up to gpus_per_node
        gpu_counts = sorted(
            g for g in [1, 2, 4, 8]
            if g <= gpus_per_node
        )
        if not gpu_counts:
            gpu_counts = [gpus_per_node]

        # KPAR values: powers of 2 up to gpus_per_node (independent of
        # per-job GPU count — VASP handles KPAR > nranks gracefully).
        valid_kpars = sorted(
            k for k in (2**i for i in range(0, 8))
            if k <= gpus_per_node
        )
        if not valid_kpars:
            valid_kpars = [1]

        combos = []
        for gpus in gpu_counts:
            for ncore in valid_ncores:
                for kpar in valid_kpars:
                    combos.append((gpus, ncore, kpar))

        logger.info(
            f"  Parameter combos: {len(combos)} "
            f"(GPUs: {gpu_counts}, NCORE: {valid_ncores}, "
            f"KPAR: {valid_kpars})"
        )
        return combos

    # ==================================================================
    # Phase 3b: Prepare job directories
    # ==================================================================

    def _prepare_jobs(
        self,
        config: ConfigParser,
        memory_dir: Path,
        vasp_config_dir: Path,
        supercells: list[tuple[int, "ase.Atoms"]],
        combos: list[tuple[int, int, int]],
    ) -> None:
        """Create job directories with POSCAR/POTCAR/INCAR for each combo."""
        # Check if already prepared (idempotent)
        first_sc = supercells[0]
        first_combo = combos[0]
        first_dir = (
            memory_dir
            / f"atoms_{first_sc[0]:03d}"
            / f"gpu{first_combo[0]}_nc{first_combo[1]:02d}_kp{first_combo[2]}"
        )
        if first_dir.exists() and (first_dir / "POSCAR").exists():
            logger.info("  Memory benchmark jobs already prepared — skipping")
            return

        logger.info("  Preparing memory benchmark jobs")

        # Validate POTCAR_W
        potcar_w = vasp_config_dir / "POTCAR_W"
        if not potcar_w.exists():
            raise FileNotFoundError(
                f"POTCAR_W not found at {potcar_w}\n"
                f"Memory benchmarks use BCC W supercells. "
                f"Place the W pseudopotential file at: {potcar_w}"
            )
        potcar_bytes = potcar_w.read_bytes()

        # Read and modify INCAR template
        incar_template = vasp_config_dir / "INCAR"
        if not incar_template.exists():
            raise FileNotFoundError(
                f"INCAR template not found at {incar_template}\n"
                f"Place your VASP INCAR file in: {vasp_config_dir}/"
            )
        incar_text = incar_template.read_text(encoding="utf-8")
        incar_text = inject_incar_defaults(incar_text, config)
        incar_text = self._override_incar_for_benchmark(incar_text)

        count = 0
        for n_atoms, atoms in supercells:
            atoms_dir = memory_dir / f"atoms_{n_atoms:03d}"

            for gpus, ncore, kpar in combos:
                job_dir = atoms_dir / f"gpu{gpus}_nc{ncore:02d}_kp{kpar}"
                job_dir.mkdir(parents=True, exist_ok=True)

                write_poscar(atoms, job_dir / "POSCAR")
                (job_dir / "POTCAR").write_bytes(potcar_bytes)

                job_incar = self._set_incar_params(incar_text, ncore, kpar)
                (job_dir / "INCAR").write_text(job_incar, encoding="utf-8")

                write_status(job_dir, status="pending", retry_level=0)
                count += 1

        logger.info(
            f"  Prepared {count} benchmark jobs "
            f"({len(supercells)} sizes × {len(combos)} combos)"
        )

    @staticmethod
    def _override_incar_for_benchmark(incar_text: str) -> str:
        """Override INCAR settings for quick benchmarking (NSW=0, NELM=10)."""
        lines = incar_text.splitlines(keepends=True)
        output = []
        found_nsw = False
        found_nelm = False

        for line in lines:
            stripped = line.strip().upper()
            if re.match(r"^NSW\s*=", stripped):
                output.append("NSW = 0\n")
                found_nsw = True
            elif re.match(r"^NELM\s*=", stripped):
                output.append("NELM = 10\n")
                found_nelm = True
            else:
                output.append(line)

        additions = []
        if not found_nsw:
            additions.append("NSW = 0")
        if not found_nelm:
            additions.append("NELM = 10")

        if additions:
            result = "".join(output).rstrip("\n")
            result += "\n\n# --- Benchmark overrides (nepflow --memory) ---\n"
            result += "\n".join(additions) + "\n"
            return result
        return "".join(output)

    @staticmethod
    def _set_incar_params(incar_text: str, ncore: int, kpar: int) -> str:
        """Set NCORE and KPAR in INCAR text."""
        lines = incar_text.splitlines(keepends=True)
        output = []
        for line in lines:
            stripped = line.strip().upper()
            if re.match(r"^NCORE\s*=", stripped):
                output.append(f"NCORE = {ncore}\n")
            elif re.match(r"^KPAR\s*=", stripped):
                output.append(f"KPAR = {kpar}\n")
            else:
                output.append(line)
        return "".join(output)

    # ==================================================================
    # Phase 4: Write runner script
    # ==================================================================

    def _write_runner_script(
        self, config: ConfigParser, memory_dir: Path,
    ) -> None:
        """Write vasp/memory/run_memory.sh — simple single-run script."""
        slurm_header_path = (
            self.project_dir / "config" / "slurm" / "header.slurm"
        )
        if not slurm_header_path.exists():
            raise FileNotFoundError(
                f"SLURM header not found at {slurm_header_path}\n"
                f"Place your header.slurm in: {slurm_header_path.parent}/"
            )
        slurm_header = slurm_header_path.read_text(encoding="utf-8")

        # Strip resource directives from header — we set these per-job
        # via sbatch command-line args (--nodes, --ntasks-per-node, etc.)
        stripped_lines = []
        resource_flags = (
            "--nodes", "--ntasks-per-node", "--gres",
            "--time", "--output", "--error",
        )
        for line in slurm_header.splitlines(keepends=True):
            if line.strip().startswith("#SBATCH"):
                if any(flag in line for flag in resource_flags):
                    continue
            stripped_lines.append(line)
        slurm_header = "".join(stripped_lines)

        vasp_command = config.get(
            "hpc", "vasp_command", fallback="mpirun -np {ntasks} vasp_std",
        )
        bash_cmd = vasp_command.replace("{ntasks}", "$TOTAL_RANKS")
        # Prevent OpenMPI hwloc binding errors when requesting < max GPUs
        if "mpirun" in bash_cmd and "--bind-to" not in bash_cmd:
            bash_cmd = bash_cmd.replace("mpirun", "mpirun --bind-to none", 1)

        body = r"""
# ============================================================================
# VASP MEMORY BENCHMARK RUNNER (generated by nepflow --memory)
# ============================================================================
# Usage: sbatch --nodes=1 --ntasks-per-node=G --gres=gpu:G \
#              run_memory.sh <job_dir>
# Runs VASP once. Detects OOM. No self-requeue.

JOB_DIR="$1"
if [ -z "$JOB_DIR" ]; then
    echo "Usage: sbatch run_memory.sh <job_dir>"
    exit 1
fi

if [ ! -d "$JOB_DIR" ]; then
    echo "ERROR: Job directory not found: $JOB_DIR"
    exit 1
fi

echo "=========================================================================="
echo "nepflow VASP Memory Benchmark"
echo "=========================================================================="
echo "Job dir:   $JOB_DIR"
echo "Job name:  $SLURM_JOB_NAME"
echo "Timestamp: $(date)"
echo "=========================================================================="

cd "$JOB_DIR" || exit 1

NGPU=${SLURM_NTASKS_PER_NODE:-4}
NNODES=${SLURM_NNODES:-1}
TOTAL_RANKS=$((NNODES * NGPU))

NCORE_VAL=$(grep -i "^[[:space:]]*NCORE" INCAR | grep "=" | head -1 | sed 's/.*=//;s/#.*//' | xargs)
KPAR_VAL=$(grep -i "^[[:space:]]*KPAR" INCAR | grep "=" | head -1 | sed 's/.*=//;s/#.*//' | xargs)

echo "NCORE=$NCORE_VAL  KPAR=$KPAR_VAL  Nodes=$NNODES  GPUs/node=$NGPU  Ranks=$TOTAL_RANKS"
echo "Command: __VASP_CMD__"
echo "=========================================================================="

VASP_LOG="${JOB_DIR}/vasp_output.log"
__VASP_CMD__ > "$VASP_LOG" 2>&1
VASP_EXIT=$?

echo ""
echo "VASP exit code: $VASP_EXIT"

if [ -f "${JOB_DIR}/OUTCAR" ]; then
    echo "OUTCAR: $(wc -l < "${JOB_DIR}/OUTCAR") lines"
fi

# ============================================================================
# OOM DETECTION
# ============================================================================
if grep -q "oom_kill" "$VASP_LOG" 2>/dev/null || [ $VASP_EXIT -eq 137 ]; then
    echo ""
    echo "OOM detected!"
    echo "OOM" > "${JOB_DIR}/.vasp_oom_marker"
    exit 137
fi

# ============================================================================
# SUCCESS OR ERROR
# ============================================================================
if [ $VASP_EXIT -eq 0 ]; then
    echo "VASP completed successfully"
    rm -f CHG CHGCAR WAVECAR CONTCAR DOSCAR EIGENVAL PCDAT
    exit 0
else
    echo "VASP failed with exit code $VASP_EXIT (not OOM)"
    exit $VASP_EXIT
fi
"""
        body = body.replace("__VASP_CMD__", bash_cmd)
        script = slurm_header.rstrip("\n") + "\n" + body

        memory_dir.mkdir(parents=True, exist_ok=True)
        script_path = memory_dir / "run_memory.sh"
        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(script_path.stat().st_mode | 0o755)
        logger.info(f"  Memory benchmark script: {script_path}")

    # ==================================================================
    # Phase 5: Submit & monitor loop
    # ==================================================================

    def _run_launcher(
        self,
        config: ConfigParser,
        memory_dir: Path,
        debug: bool = False,
    ) -> None:
        """Submit and monitor memory benchmark jobs."""
        max_concurrent = config.getint("slurm", "max_concurrent", fallback=20)
        poll_interval = 0 if debug else config.getint(
            "slurm", "memory_poll_interval", fallback=10,
        )
        benchmark_walltime = config.get(
            "slurm", "memory_walltime", fallback="00:10:00",
        )
        shared_script = memory_dir / "run_memory.sh"

        # Discover all job directories
        job_dirs = self._discover_job_dirs(memory_dir)
        if not job_dirs:
            logger.warning("  No benchmark job directories found")
            return

        logger.info("")
        logger.info(f"  Memory benchmark launcher: {len(job_dirs)} jobs")
        logger.info(f"  Max concurrent: {max_concurrent}")
        if debug:
            logger.info("  [DEBUG] Simulated SLURM mode")

        _debug_counter = 0

        while True:
            running_names, pending_names = (
                self._get_job_names() if not debug else (set(), set())
            )
            active_names = running_names | pending_names
            all_terminal = True

            stats = {"pending": 0, "submitted": 0, "completed": 0, "oom": 0, "failed": 0}

            for job_dir in job_dirs:
                status_data = read_status(job_dir)
                status = status_data.get("status", "pending")

                if status in ("completed", "oom", "failed"):
                    stats[status] += 1
                    continue

                all_terminal = False
                job_name = self._job_name(job_dir)

                if status == "submitted":
                    if job_name in active_names:
                        stats["submitted"] += 1
                        continue

                    # Job left squeue — determine outcome
                    if self._check_completed(job_dir):
                        write_status(job_dir, status="completed")
                        logger.info(f"    {job_dir.parent.name}/{job_dir.name}: completed")
                        stats["completed"] += 1
                    elif self._check_oom(job_dir):
                        write_status(job_dir, status="oom")
                        logger.info(f"    {job_dir.parent.name}/{job_dir.name}: OOM")
                        stats["oom"] += 1
                    else:
                        write_status(job_dir, status="failed")
                        logger.warning(
                            f"    {job_dir.parent.name}/{job_dir.name}: failed (non-OOM)"
                        )
                        stats["failed"] += 1

                elif status == "pending":
                    stats["pending"] += 1
                    if len(active_names) < max_concurrent:
                        gpus = self._parse_gpus(job_dir)

                        if debug:
                            _debug_counter += 1
                            job_id = self._submit_debug(
                                job_dir, _debug_counter,
                            )
                        else:
                            job_id = self._submit_job(
                                shared_script, job_dir, job_name,
                                gpus, benchmark_walltime,
                            )

                        if job_id:
                            write_status(
                                job_dir, status="submitted",
                                slurm_job_id=job_id,
                            )
                            active_names.add(job_name)

            if all_terminal:
                logger.info("")
                logger.info(
                    f"  All benchmark jobs terminal: "
                    f"{stats['completed']} completed, "
                    f"{stats['oom']} OOM, "
                    f"{stats['failed']} failed"
                )
                return

            logger.info(
                f"  Status: {stats['pending']} pending, "
                f"{stats['submitted']} running, "
                f"{stats['completed']} completed, "
                f"{stats['oom']} OOM"
            )

            if poll_interval > 0:
                time.sleep(poll_interval)

    def _discover_job_dirs(self, memory_dir: Path) -> list[Path]:
        """Find all benchmark job directories (atoms_*/gpu*_nc*_kp*)."""
        dirs = []
        if not memory_dir.exists():
            return dirs
        for atoms_dir in sorted(memory_dir.iterdir()):
            if not atoms_dir.is_dir() or not atoms_dir.name.startswith("atoms_"):
                continue
            for job_dir in sorted(atoms_dir.iterdir()):
                if not job_dir.is_dir() or not job_dir.name.startswith("gpu"):
                    continue
                if (job_dir / "POSCAR").exists():
                    dirs.append(job_dir)
        return dirs

    def _job_name(self, job_dir: Path) -> str:
        """Generate SLURM job name from job directory."""
        return f"nf_{self.project_name}_mem_{job_dir.parent.name}_{job_dir.name}"

    @staticmethod
    def _parse_gpus(job_dir: Path) -> int:
        """Extract GPU count from directory name (gpu{N}_nc*_kp*)."""
        name = job_dir.name
        m = re.match(r"gpu(\d+)_", name)
        return int(m.group(1)) if m else 4

    def _get_job_names(self) -> tuple[set[str], set[str]]:
        """Batch squeue query for memory benchmark jobs."""
        running: set[str] = set()
        pending: set[str] = set()
        prefix = f"nf_{self.project_name}_mem_"
        try:
            result = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "--noheader",
                 "-o", "%j %T", "--states=RUNNING,PENDING"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            if result.returncode == 0:
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

    @staticmethod
    def _check_completed(job_dir: Path) -> bool:
        """Check if VASP completed successfully."""
        outcar = job_dir / "OUTCAR"
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

    @staticmethod
    def _check_oom(job_dir: Path) -> bool:
        """Check if job failed due to OOM."""
        if (job_dir / ".vasp_oom_marker").exists():
            return True
        vasp_log = job_dir / "vasp_output.log"
        if vasp_log.exists():
            try:
                text = vasp_log.read_text(encoding="utf-8", errors="replace")
                if "oom_kill" in text:
                    return True
            except OSError:
                pass
        return False

    def _submit_job(
        self,
        shared_script: Path,
        job_dir: Path,
        job_name: str,
        gpus: int,
        walltime: str,
    ) -> Optional[str]:
        """Submit a benchmark job via sbatch."""
        try:
            sbatch_args = [
                "sbatch",
                f"--job-name={job_name}",
                f"--time={walltime}",
                "--nodes=1",
                f"--ntasks-per-node={gpus}",
                f"--gres=gpu:{gpus}",
                f"--output={job_dir}/vasp_%j.out",
                f"--error={job_dir}/vasp_%j.err",
                "--mem=64G",
                str(shared_script),
                str(job_dir.resolve()),
            ]
            result = subprocess.run(
                sbatch_args,
                capture_output=True, text=True, timeout=30, check=False,
            )
            if result.returncode == 0 and "Submitted batch job" in result.stdout:
                job_id = result.stdout.strip().split()[-1]
                logger.info(
                    f"    {job_dir.parent.name}/{job_dir.name}: "
                    f"submitted (gpus={gpus}) job {job_id}"
                )
                return job_id
            logger.warning(
                f"    sbatch failed for {job_dir.name}: "
                f"{result.stderr.strip()}"
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"    sbatch error for {job_dir.name}: {e}")
        return None

    def _submit_debug(self, job_dir: Path, counter: int) -> str:
        """Simulate VASP submission for debug mode."""
        outcar = job_dir / "OUTCAR"

        # Every 5th job simulated as OOM
        if counter % 5 == 0:
            outcar.write_text(
                "  VASP simulated run — OOM\n", encoding="utf-8",
            )
            (job_dir / ".vasp_oom_marker").write_text("OOM\n", encoding="utf-8")
            (job_dir / "vasp_output.log").write_text(
                "oom_kill detected\n", encoding="utf-8",
            )
        else:
            # Parse n_atoms from POSCAR for realistic debug output
            try:
                poscar_lines = (job_dir / "POSCAR").read_text(
                    encoding="utf-8",
                ).splitlines()
                n_atoms = sum(int(x) for x in poscar_lines[6].split())
            except (OSError, ValueError, IndexError):
                n_atoms = 2

            # Compute simulated n_electrons (W has ZVAL=14 typically)
            n_electrons = n_atoms * 14

            outcar.write_text(
                "  General timing and accounting informations for this job:\n"
                "  LOOP:  cpu time   10.00: real time   10.00\n"
                f"  Found     1 irreducible k-points\n"
                f"  NELECT =      {n_electrons}.0000\n"
                "  running on       4 total cores\n"
                "  Voluntary context switches:        42\n",
                encoding="utf-8",
            )

        logger.info(
            f"    {job_dir.parent.name}/{job_dir.name}: "
            f"submitted (debug) job DEBUG_{counter:05d}"
        )
        return f"DEBUG_{counter:05d}"

    # ==================================================================
    # Phase 6: Collect results and update .vasp_memory
    # ==================================================================

    def _collect_results(self, memory_dir: Path, nepflow_root: Path) -> None:
        """Parse benchmark results and append to .vasp_memory."""
        job_dirs = self._discover_job_dirs(memory_dir)
        csv_path = nepflow_root / ".vasp_memory"

        rows: list[dict] = []
        for job_dir in job_dirs:
            status_data = read_status(job_dir)
            status = status_data.get("status", "pending")

            if status == "completed":
                row = self._parse_completed(job_dir)
                if row:
                    rows.append(row)
            elif status == "oom":
                row = self._parse_oom(job_dir)
                if row:
                    rows.append(row)

        if not rows:
            logger.info("  No benchmark results to record")
            return

        # Check if CSV exists and has the oom column already
        write_header = not csv_path.exists()
        if csv_path.exists():
            try:
                with open(csv_path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if "oom" not in first_line:
                    # Old format — need to migrate
                    self._migrate_csv(csv_path)
            except OSError:
                pass

        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(CSV_HEADER)
            for row in rows:
                writer.writerow([row[col] for col in CSV_HEADER])

        completed = sum(1 for r in rows if r["oom"] == 0)
        oom = sum(1 for r in rows if r["oom"] == 1)
        logger.info(
            f"  Wrote {len(rows)} entries to {csv_path} "
            f"({completed} timed, {oom} OOM)"
        )

        self._print_summary(rows)
        self._plot_results(rows, memory_dir)

    def _parse_completed(self, job_dir: Path) -> Optional[dict]:
        """Parse a completed benchmark job."""
        outcar = job_dir / "OUTCAR"
        poscar = job_dir / "POSCAR"
        incar = job_dir / "INCAR"

        try:
            poscar_lines = poscar.read_text(encoding="utf-8").splitlines()
            n_atoms = sum(int(x) for x in poscar_lines[6].split())

            incar_text = incar.read_text(encoding="utf-8")
            ncore = kpar = 0
            for line in incar_text.splitlines():
                stripped = line.strip().upper()
                if stripped.startswith("NCORE"):
                    ncore = int(line.split("=")[1].split("#")[0].strip())
                elif stripped.startswith("KPAR"):
                    kpar = int(line.split("=")[1].split("#")[0].strip())

            outcar_text = outcar.read_text(encoding="utf-8", errors="replace")

            # MPI ranks → gpus
            gpus = self._parse_gpus(job_dir)
            total_ranks = 0
            m_ranks = re.search(
                r"running on\s+(\d+)\s+total cores", outcar_text,
            )
            if m_ranks:
                total_ranks = int(m_ranks.group(1))
            if total_ranks > 0:
                gpus = total_ranks

            loop_times = [
                float(m.group(1))
                for m in re.finditer(
                    r"LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)",
                    outcar_text,
                )
            ]
            avg_loop = (
                sum(loop_times) / len(loop_times) if loop_times else 0.0
            )

            n_kpoints_irr = 0
            m = re.search(
                r"Found\s+(\d+)\s+irreducible k-points", outcar_text,
            )
            if m:
                n_kpoints_irr = int(m.group(1))

            n_electrons = 0
            m = re.search(r"NELECT\s*=\s*([\d.]+)", outcar_text)
            if m:
                n_electrons = int(float(m.group(1)))

            return {
                "n_atoms": n_atoms,
                "n_kpoints_irr": n_kpoints_irr,
                "n_electrons": n_electrons,
                "nodes": 1,
                "gpus": gpus,
                "ncore": ncore,
                "kpar": kpar,
                "avg_loop_time": f"{avg_loop:.4f}",
                "oom": 0,
            }
        except (OSError, ValueError, IndexError) as e:
            logger.debug(f"  Could not parse {job_dir.name}: {e}")
            return None

    def _parse_oom(self, job_dir: Path) -> Optional[dict]:
        """Parse an OOM benchmark job (record the failure)."""
        poscar = job_dir / "POSCAR"
        incar = job_dir / "INCAR"

        try:
            poscar_lines = poscar.read_text(encoding="utf-8").splitlines()
            n_atoms = sum(int(x) for x in poscar_lines[6].split())

            incar_text = incar.read_text(encoding="utf-8")
            ncore = kpar = 0
            for line in incar_text.splitlines():
                stripped = line.strip().upper()
                if stripped.startswith("NCORE"):
                    ncore = int(line.split("=")[1].split("#")[0].strip())
                elif stripped.startswith("KPAR"):
                    kpar = int(line.split("=")[1].split("#")[0].strip())

            gpus = self._parse_gpus(job_dir)

            # Try to get electron count from partial OUTCAR
            n_electrons = 0
            n_kpoints_irr = 0
            outcar = job_dir / "OUTCAR"
            if outcar.exists():
                try:
                    outcar_text = outcar.read_text(
                        encoding="utf-8", errors="replace",
                    )
                    m = re.search(r"NELECT\s*=\s*([\d.]+)", outcar_text)
                    if m:
                        n_electrons = int(float(m.group(1)))
                    m = re.search(
                        r"Found\s+(\d+)\s+irreducible k-points", outcar_text,
                    )
                    if m:
                        n_kpoints_irr = int(m.group(1))
                except OSError:
                    pass

            # Fallback: estimate electrons from POTCAR ZVAL if no OUTCAR
            if n_electrons == 0:
                vasp_config_dir = self.project_dir / "config" / "vasp"
                potcar_w = vasp_config_dir / "POTCAR_W"
                if potcar_w.exists():
                    try:
                        zval = parse_zval(potcar_w)
                        n_electrons = int(n_atoms * zval)
                    except (ValueError, OSError):
                        pass

            return {
                "n_atoms": n_atoms,
                "n_kpoints_irr": n_kpoints_irr,
                "n_electrons": n_electrons,
                "nodes": 1,
                "gpus": gpus,
                "ncore": ncore,
                "kpar": kpar,
                "avg_loop_time": "0.0000",
                "oom": 1,
            }
        except (OSError, ValueError, IndexError) as e:
            logger.debug(f"  Could not parse OOM job {job_dir.name}: {e}")
            return None

    @staticmethod
    def _migrate_csv(csv_path: Path) -> None:
        """Add oom=0 column to existing .vasp_memory without it."""
        try:
            text = csv_path.read_text(encoding="utf-8")
            lines = text.strip().split("\n")
            if not lines:
                return

            # Add oom to header
            new_lines = [lines[0] + ",oom"]
            # Add oom=0 to each data row
            for line in lines[1:]:
                if line.strip():
                    new_lines.append(line + ",0")

            csv_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            logger.info(f"  Migrated {csv_path} to include oom column")
        except OSError as e:
            logger.warning(f"  Could not migrate {csv_path}: {e}")

    @staticmethod
    def _print_summary(rows: list[dict]) -> None:
        """Print a summary of benchmark results."""
        logger.info("")
        logger.info("  ═══════════════════════════════════════════════════════")
        logger.info("  Memory Benchmark Summary")
        logger.info("  ═══════════════════════════════════════════════════════")

        # Group by n_atoms
        by_atoms: dict[int, list[dict]] = {}
        for row in rows:
            n = int(row["n_atoms"])
            by_atoms.setdefault(n, []).append(row)

        for n_atoms in sorted(by_atoms.keys()):
            group = by_atoms[n_atoms]
            completed = [r for r in group if r["oom"] == 0]
            oom_count = sum(1 for r in group if r["oom"] == 1)

            logger.info(f"")
            logger.info(
                f"  {n_atoms} atoms: "
                f"{len(completed)} timed, {oom_count} OOM"
            )

            if completed:
                # Find fastest per GPU count
                by_gpus: dict[int, list[dict]] = {}
                for r in completed:
                    g = int(r["gpus"])
                    by_gpus.setdefault(g, []).append(r)

                for gpus in sorted(by_gpus.keys()):
                    fastest = min(
                        by_gpus[gpus],
                        key=lambda r: float(r["avg_loop_time"]),
                    )
                    logger.info(
                        f"    GPU={gpus}: best NCORE={fastest['ncore']} "
                        f"KPAR={fastest['kpar']} "
                        f"avg_loop={fastest['avg_loop_time']}s"
                    )

        # ── Overall recommendation (prefer fewest GPUs) ──────────────
        logger.info("")
        logger.info("  ───────────────────────────────────────────────────────")
        logger.info("  Recommended parameters (prefer lowest GPU count):")
        logger.info("  ───────────────────────────────────────────────────────")

        for n_atoms in sorted(by_atoms.keys()):
            completed = [r for r in by_atoms[n_atoms] if r["oom"] == 0]
            if not completed:
                logger.info(f"  {n_atoms:>4d} atoms: ALL OOM — no viable config")
                continue

            # Sort by (gpus ASC, avg_loop ASC) → pick first = cheapest viable
            completed.sort(
                key=lambda r: (int(r["gpus"]), float(r["avg_loop_time"])),
            )
            pick = completed[0]
            abs_fastest = min(completed, key=lambda r: float(r["avg_loop_time"]))

            marker = ""
            if int(pick["gpus"]) != int(abs_fastest["gpus"]):
                speedup = float(pick["avg_loop_time"]) / float(abs_fastest["avg_loop_time"])
                marker = (
                    f"  ({speedup:.1f}× vs GPU={abs_fastest['gpus']} "
                    f"NC={abs_fastest['ncore']} KP={abs_fastest['kpar']})"
                )

            logger.info(
                f"  {n_atoms:>4d} atoms → GPU={pick['gpus']}  "
                f"NCORE={pick['ncore']}  KPAR={pick['kpar']}  "
                f"avg_loop={pick['avg_loop_time']}s{marker}"
            )

        logger.info("  ═══════════════════════════════════════════════════════")

    # ==================================================================
    # Phase 7: Plot results
    # ==================================================================

    @staticmethod
    def _plot_results(rows: list[dict], memory_dir: Path) -> None:
        """Generate benchmark summary plots and save to memory_dir/plots/."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
        except ImportError:
            logger.warning("  matplotlib not available — skipping plots")
            return

        completed = [r for r in rows if r["oom"] == 0]
        if not completed:
            logger.info("  No completed benchmarks to plot")
            return

        plot_dir = memory_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        all_atoms = sorted({int(r["n_atoms"]) for r in rows})
        all_gpus = sorted({int(r["gpus"]) for r in rows})
        all_ncores = sorted({int(r["ncore"]) for r in rows})
        all_kpars = sorted({int(r["kpar"]) for r in rows})

        # ── Plot 1: NCORE×KPAR heatmaps per (n_atoms, gpus) ──────────
        for n_atoms in all_atoms:
            atom_rows = [r for r in rows if int(r["n_atoms"]) == n_atoms]
            gpu_counts = sorted({int(r["gpus"]) for r in atom_rows})

            fig, axes = plt.subplots(
                1, len(gpu_counts),
                figsize=(5 * len(gpu_counts), 4),
                squeeze=False,
            )
            fig.suptitle(f"VASP Loop Time — {n_atoms} atoms", fontsize=14)

            for col, gpus in enumerate(gpu_counts):
                ax = axes[0, col]
                subset = [
                    r for r in atom_rows if int(r["gpus"]) == gpus
                ]

                # Build grid
                nc_idx = {v: i for i, v in enumerate(all_ncores)}
                kp_idx = {v: i for i, v in enumerate(all_kpars)}
                grid = np.full(
                    (len(all_ncores), len(all_kpars)), np.nan,
                )
                oom_mask = np.zeros_like(grid, dtype=bool)

                for r in subset:
                    ni = nc_idx.get(int(r["ncore"]))
                    ki = kp_idx.get(int(r["kpar"]))
                    if ni is None or ki is None:
                        continue
                    if r["oom"] == 1:
                        oom_mask[ni, ki] = True
                    else:
                        grid[ni, ki] = float(r["avg_loop_time"])

                # Plot heatmap
                valid = grid[~np.isnan(grid)]
                if len(valid) > 0:
                    im = ax.imshow(
                        grid, aspect="auto", origin="lower",
                        norm=LogNorm(
                            vmin=max(valid.min(), 1e-3),
                            vmax=valid.max(),
                        ),
                        cmap="viridis_r",
                    )
                    fig.colorbar(im, ax=ax, label="avg loop (s)")

                    # Annotate cells with values
                    for ni in range(len(all_ncores)):
                        for ki in range(len(all_kpars)):
                            if oom_mask[ni, ki]:
                                ax.text(
                                    ki, ni, "OOM", ha="center",
                                    va="center", color="red",
                                    fontweight="bold", fontsize=8,
                                )
                            elif not np.isnan(grid[ni, ki]):
                                ax.text(
                                    ki, ni, f"{grid[ni, ki]:.1f}",
                                    ha="center", va="center",
                                    color="white", fontsize=7,
                                )
                else:
                    # All OOM for this GPU count
                    ax.imshow(
                        np.zeros_like(grid), aspect="auto",
                        origin="lower", cmap="Greys", vmin=0, vmax=1,
                    )
                    for ni in range(len(all_ncores)):
                        for ki in range(len(all_kpars)):
                            if oom_mask[ni, ki]:
                                ax.text(
                                    ki, ni, "OOM", ha="center",
                                    va="center", color="red",
                                    fontweight="bold", fontsize=8,
                                )

                ax.set_xticks(range(len(all_kpars)))
                ax.set_xticklabels(all_kpars)
                ax.set_yticks(range(len(all_ncores)))
                ax.set_yticklabels(all_ncores)
                ax.set_xlabel("KPAR")
                ax.set_ylabel("NCORE")
                ax.set_title(f"GPU={gpus}")

            fig.tight_layout()
            path = plot_dir / f"heatmap_{n_atoms}atoms.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)

        # ── Plot 2: Scaling — best loop time vs n_atoms per GPU count ─
        fig, ax = plt.subplots(figsize=(7, 5))
        for gpus in all_gpus:
            x_vals, y_vals, labels = [], [], []
            for n_atoms in all_atoms:
                sub = [
                    r for r in completed
                    if int(r["gpus"]) == gpus
                    and int(r["n_atoms"]) == n_atoms
                ]
                if not sub:
                    continue
                best = min(sub, key=lambda r: float(r["avg_loop_time"]))
                x_vals.append(n_atoms)
                y_vals.append(float(best["avg_loop_time"]))
                labels.append(
                    f"NC={best['ncore']}\nKP={best['kpar']}"
                )
            if x_vals:
                ax.plot(
                    x_vals, y_vals, "o-",
                    label=f"GPU={gpus}", markersize=8,
                )
                for x, y, lbl in zip(x_vals, y_vals, labels):
                    ax.annotate(
                        lbl, (x, y), textcoords="offset points",
                        xytext=(8, 4), fontsize=7, color="gray",
                    )

        ax.set_xlabel("Number of atoms")
        ax.set_ylabel("Best avg loop time (s)")
        ax.set_title("VASP Scaling — Best Parameters per GPU Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if all_atoms:
            ax.set_xticks(all_atoms)
        fig.tight_layout()
        fig.savefig(plot_dir / "scaling.png", dpi=150)
        plt.close(fig)

        # ── Plot 3: OOM map — stacked bar per system size ─────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_width = 0.25
        x_pos = np.arange(len(all_atoms))

        for i, gpus in enumerate(all_gpus):
            ok_counts, oom_counts = [], []
            for n_atoms in all_atoms:
                sub = [
                    r for r in rows
                    if int(r["gpus"]) == gpus
                    and int(r["n_atoms"]) == n_atoms
                ]
                ok_counts.append(
                    sum(1 for r in sub if r["oom"] == 0)
                )
                oom_counts.append(
                    sum(1 for r in sub if r["oom"] == 1)
                )

            offset = (i - len(all_gpus) / 2 + 0.5) * bar_width
            ax.bar(
                x_pos + offset, ok_counts, bar_width,
                label=f"GPU={gpus} OK",
                color=f"C{i}", alpha=0.8,
            )
            ax.bar(
                x_pos + offset, oom_counts, bar_width,
                bottom=ok_counts, color=f"C{i}", alpha=0.3,
                hatch="//",
                label=f"GPU={gpus} OOM",
            )

        ax.set_xlabel("Number of atoms")
        ax.set_ylabel("Number of benchmarks")
        ax.set_title("Benchmark Outcomes — OK vs OOM")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_atoms)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(plot_dir / "oom_map.png", dpi=150)
        plt.close(fig)

        logger.info(f"  Plots saved to {plot_dir}/")
