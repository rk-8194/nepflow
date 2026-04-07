"""Prepare sub-stage: create VASP job folders and shared runner script."""

import re
import time
from configparser import ConfigParser
from pathlib import Path

import numpy as np
from ase.io import iread

from ._common import RETRY_LEVELS, logger, write_status


# ==================================================================
# job folder preparation
# ==================================================================

def prepare_jobs(
    config: ConfigParser,
    vasp_config_dir: Path,
    selected_dir: Path,
    jobs_dir: Path,
    datasets: list[str],
) -> None:
    """Create POSCAR/POTCAR/INCAR per structure (skips if already done)."""
    first_struct = jobs_dir / datasets[0] / "struct_0000"
    if first_struct.exists() and (first_struct / "POSCAR").exists():
        logger.info("Job folders already prepared — skipping preparation")
        return

    logger.info("Running VASP job preparation")

    # Validate INCAR template
    incar_template = vasp_config_dir / "INCAR"
    if not incar_template.exists():
        raise FileNotFoundError(
            f"INCAR template not found at {incar_template}\n"
            f"Place your VASP INCAR file in: {vasp_config_dir}/"
        )
    logger.info(f"  INCAR template: {incar_template}")

    incar_text = incar_template.read_text(encoding="utf-8")
    incar_text = inject_incar_defaults(incar_text, config)

    # Discover all unique elements across all datasets
    all_elements: set[str] = set()
    for ds in datasets:
        xyz_path = selected_dir / f"{ds}.xyz"
        if not xyz_path.exists():
            raise FileNotFoundError(
                f"Selected structures not found at {xyz_path}\n"
                f"Run the 'select' stage first."
            )
        for atoms in iread(str(xyz_path), format="extxyz"):
            all_elements.update(atoms.get_chemical_symbols())

    sorted_elements = sorted(all_elements)
    logger.info(f"  Elements: {', '.join(sorted_elements)}")

    # Validate pseudopotentials
    missing = [
        f"POTCAR_{e}" for e in sorted_elements
        if not (vasp_config_dir / f"POTCAR_{e}").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing pseudopotential files in {vasp_config_dir}/:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    # Pre-read POTCAR data
    potcar_data: dict[str, bytes] = {}
    for elem in sorted_elements:
        potcar_data[elem] = (vasp_config_dir / f"POTCAR_{elem}").read_bytes()

    # Create job folders for each dataset (POSCAR/POTCAR/INCAR only)
    for ds in datasets:
        logger.info(f"  Preparing {ds} jobs")
        xyz_path = selected_dir / f"{ds}.xyz"
        ds_jobs_dir = jobs_dir / ds
        ds_jobs_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        count = 0

        for i, atoms in enumerate(iread(str(xyz_path), format="extxyz")):
            struct_dir = ds_jobs_dir / f"struct_{i:04d}"
            struct_dir.mkdir(parents=True, exist_ok=True)

            write_poscar(atoms, struct_dir / "POSCAR")

            struct_elements = sorted(set(atoms.get_chemical_symbols()))
            with open(struct_dir / "POTCAR", "wb") as f:
                for elem in struct_elements:
                    f.write(potcar_data[elem])

            (struct_dir / "INCAR").write_text(incar_text, encoding="utf-8")
            write_status(struct_dir, status="pending", retry_level=0)

            count += 1
            if count % 100 == 0:
                logger.info(f"    {count} structures prepared...")

        elapsed = time.perf_counter() - t0
        logger.info(f"  {ds.capitalize()}: {count} jobs ({elapsed:.1f}s)")

    logger.info("Job preparation complete")


# ==================================================================
# shared VASP runner script (like adaptive_healing.sh)
# ==================================================================

def write_shared_vasp_script(
    vasp_dir: Path,
    slurm_header: str,
    vasp_command: str,
) -> None:
    """Write vasp/run_vasp.sh — self-healing script submitted per structure.

    Submitted for each structure via:
      sbatch --job-name=<name> --nodes=N --ntasks-per-node=G \\
             --gres=gpu:G run_vasp.sh <struct_dir>

    On OOM the script escalates NCORE/KPAR and resubmits itself,
    preserving the job name so the launcher can track it via squeue.
    """
    # Build case statements from RETRY_LEVELS table
    nk_cases = []
    for i, (ncore, kpar, _nodes, _gpus) in enumerate(RETRY_LEVELS):
        nk_cases.append(f"    {i}) NCORE_VAL={ncore}; KPAR_VAL={kpar} ;;")
    nk_cases.append('    *) echo "ERROR: Max retry level exceeded"; exit 1 ;;')

    rq_cases = []
    for i, (_ncore, _kpar, nodes, gpus) in enumerate(RETRY_LEVELS):
        rq_cases.append(
            f"        {i}) REQUEUE_NODES={nodes}; REQUEUE_GPU={gpus} ;;"
        )
    rq_cases.append('        *) echo "Max retries exhausted"; exit 1 ;;')

    bash_cmd = vasp_command.replace("{ntasks}", "$TOTAL_RANKS")

    body = r"""
# ============================================================================
# SELF-HEALING VASP RUNNER (generated by nepflow)
# ============================================================================
# Usage: sbatch --job-name=<name> --nodes=N --ntasks-per-node=G \
#              --gres=gpu:G run_vasp.sh <struct_dir>
# On OOM, escalates NCORE/KPAR and resubmits itself with the same job name.

STRUCT_DIR="$1"
if [ -z "$STRUCT_DIR" ]; then
    echo "Usage: sbatch run_vasp.sh <struct_dir>"
    exit 1
fi

if [ ! -d "$STRUCT_DIR" ]; then
    echo "ERROR: Structure directory not found: $STRUCT_DIR"
    exit 1
fi

STRUCT_NAME=$(basename "$STRUCT_DIR")
RETRY_LEVEL_FILE="${STRUCT_DIR}/.vasp_retry_level"

if [ -f "$RETRY_LEVEL_FILE" ]; then
    RETRY_LEVEL=$(cat "$RETRY_LEVEL_FILE")
else
    RETRY_LEVEL=0
fi

echo "=========================================================================="
echo "nepflow VASP Runner (Retry Level: $RETRY_LEVEL)"
echo "=========================================================================="
echo "Structure: $STRUCT_DIR"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Timestamp: $(date)"
echo "=========================================================================="

cd "$STRUCT_DIR" || exit 1

# Retry-level NCORE/KPAR
case $RETRY_LEVEL in
__NK_CASES__
esac

# Update INCAR
python3 << PYTHON_EOF
ncore = "$NCORE_VAL"
kpar  = "$KPAR_VAL"
with open("INCAR", "r") as f:
    lines = f.readlines()
output = []
for line in lines:
    s = line.strip().upper()
    if s.startswith("NCORE"):
        output.append(f"NCORE = {ncore}\n")
    elif s.startswith("KPAR"):
        output.append(f"KPAR = {kpar}\n")
    else:
        output.append(line)
with open("INCAR", "w") as f:
    f.writelines(output)
print(f"Updated INCAR: NCORE={ncore}, KPAR={kpar}")
PYTHON_EOF

NGPU=${SLURM_NTASKS_PER_NODE:-4}
NNODES=${SLURM_NNODES:-1}
TOTAL_RANKS=$((NNODES * NGPU))

echo ""
echo "NCORE=$NCORE_VAL  KPAR=$KPAR_VAL  Nodes=$NNODES  GPUs/node=$NGPU  Ranks=$TOTAL_RANKS"
echo "Command: __VASP_CMD__"
echo "=========================================================================="

VASP_LOG="${STRUCT_DIR}/vasp_output.log"
__VASP_CMD__ > "$VASP_LOG" 2>&1
VASP_EXIT=$?

echo ""
echo "VASP exit code: $VASP_EXIT"

if [ -f "${STRUCT_DIR}/OUTCAR" ]; then
    echo "OUTCAR: $(wc -l < "${STRUCT_DIR}/OUTCAR") lines"
fi

# ============================================================================
# OOM DETECTION AND SELF-REQUEUE
# ============================================================================
if grep -q "oom_kill" "$VASP_LOG" 2>/dev/null || [ $VASP_EXIT -eq 137 ]; then
    echo ""
    echo "OOM detected! Requeuing with escalated parameters..."

    NEXT_LEVEL=$((RETRY_LEVEL + 1))
    echo "$NEXT_LEVEL" > "$RETRY_LEVEL_FILE"

    case $NEXT_LEVEL in
__RQ_CASES__
    esac

    # Clean for fresh start
    rm -f CHG CHGCAR WAVECAR CONTCAR DOSCAR EIGENVAL PCDAT
    rm -f OUTCAR vasprun.xml OSZICAR vasp_output.log

    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    sbatch --job-name="$SLURM_JOB_NAME" \
        --nodes=$REQUEUE_NODES \
        --ntasks-per-node=$REQUEUE_GPU \
        --gres=gpu:$REQUEUE_GPU \
        "$SCRIPT_PATH" "$STRUCT_DIR"

    echo "Requeued: level $NEXT_LEVEL (nodes=$REQUEUE_NODES, gpus=$REQUEUE_GPU/node)"
    exit 0
fi

# ============================================================================
# SUCCESS OR ERROR
# ============================================================================
if [ $VASP_EXIT -eq 0 ]; then
    echo "VASP completed successfully"
    rm -f .vasp_retry_level CHG CHGCAR WAVECAR CONTCAR DOSCAR EIGENVAL PCDAT
    exit 0
else
    echo "VASP failed with exit code $VASP_EXIT (not OOM)"
    echo "Check $STRUCT_DIR/vasp_output.log for details"
    exit 1
fi
"""
    body = body.replace("__NK_CASES__", "\n".join(nk_cases))
    body = body.replace("__RQ_CASES__", "\n".join(rq_cases))
    body = body.replace("__VASP_CMD__", bash_cmd)

    script = slurm_header.rstrip("\n") + "\n" + body

    vasp_dir.mkdir(parents=True, exist_ok=True)
    script_path = vasp_dir / "run_vasp.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | 0o755)
    logger.info(f"  Shared VASP script: {script_path}")


# ==================================================================
# POSCAR writing
# ==================================================================

def write_poscar(atoms, path: Path) -> None:
    """Write an ASE Atoms object as a VASP5 POSCAR with atoms grouped by element."""
    symbols = np.array(atoms.get_chemical_symbols())
    unique_elements = sorted(set(symbols))

    sorted_indices = []
    counts = []
    for elem in unique_elements:
        mask = symbols == elem
        indices = np.where(mask)[0]
        sorted_indices.extend(indices.tolist())
        counts.append(int(mask.sum()))

    cell = atoms.get_cell()
    frac_positions = atoms.get_scaled_positions()

    with open(path, "w", encoding="utf-8") as f:
        info = atoms.info if hasattr(atoms, "info") else {}
        comment = info.get("config_type", " ".join(unique_elements))
        f.write(f"{comment}\n")
        f.write("1.0\n")
        for row in cell:
            f.write(f"  {row[0]:20.14f}  {row[1]:20.14f}  {row[2]:20.14f}\n")
        f.write("  " + "  ".join(unique_elements) + "\n")
        f.write("  " + "  ".join(str(c) for c in counts) + "\n")
        f.write("Direct\n")
        for idx in sorted_indices:
            p = frac_positions[idx]
            f.write(f"  {p[0]:20.14f}  {p[1]:20.14f}  {p[2]:20.14f}\n")


# ==================================================================
# INCAR injection
# ==================================================================

def inject_incar_defaults(incar_text: str, config: ConfigParser) -> str:
    """Append KSPACING and KGAMMA to INCAR text if not already present."""
    lines = incar_text.rstrip("\n")

    has_kspacing = bool(re.search(r"^\s*KSPACING\s*=", incar_text, re.MULTILINE | re.IGNORECASE))
    has_kgamma = bool(re.search(r"^\s*KGAMMA\s*=", incar_text, re.MULTILINE | re.IGNORECASE))

    additions = []
    if not has_kspacing:
        kspacing = config.get("vasp", "kspacing", fallback="0.22")
        additions.append(f"KSPACING = {kspacing}")
    if not has_kgamma:
        kgamma = config.get("vasp", "kgamma", fallback=".TRUE.")
        additions.append(f"KGAMMA = {kgamma}")

    if additions:
        lines += "\n\n# --- Injected by nepflow (not in user template) ---\n"
        lines += "\n".join(additions) + "\n"
    else:
        lines += "\n"

    return lines
