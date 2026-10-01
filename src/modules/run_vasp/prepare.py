"""Prepare sub-stage: create VASP job folders and shared runner script."""

import time
from configparser import ConfigParser
from pathlib import Path

from ase.io import iread

from nepflow.domain.identities import (
    normalise_dft_calculation_identity,
)
from nepflow.io.json import write_json
from nepflow.dft.vasp.inputs import (
    canonical_poscar_text,
    hash_incar_text,
    identity_for_structure,
    inject_incar_defaults as _canonical_inject_incar_defaults,
    read_identity,
)
from nepflow.dft.vasp.outputs import outcar_is_complete

from ._common import (
    get_nepflow_root,
    get_registry_entry,
    logger,
    read_completed_registry,
    read_status,
    write_status,
)


# ==================================================================
# job folder preparation
# ==================================================================

def prepare_jobs(
    config: ConfigParser,
    vasp_config_dir: Path,
    selected_dir: Path,
    jobs_dir: Path,
    datasets: list[str],
    project_dir: Path | None = None,
    project_name: str = "",
) -> None:
    """Create POSCAR/POTCAR/INCAR per structure and mark reusable jobs."""
    logger.info("Running VASP job preparation")

    incar_template = vasp_config_dir / "INCAR"
    if not incar_template.exists():
        raise FileNotFoundError(
            f"INCAR template not found at {incar_template}\n"
            f"Place your VASP INCAR file in: {vasp_config_dir}/"
        )
    logger.info(f"  INCAR template: {incar_template}")

    incar_text = inject_incar_defaults(
        incar_template.read_text(encoding="utf-8"),
        config,
    )
    incar_hash = hash_incar_text(incar_text)
    nepflow_root = get_nepflow_root(project_dir or jobs_dir.parent.parent)
    registry = read_completed_registry(nepflow_root)

    all_elements: set[str] = set()
    for ds in datasets:
        xyz_path = selected_dir / f"{ds}.xyz"
        if not xyz_path.exists():
            raise FileNotFoundError(
                f"Selected structures not found at {xyz_path}\n"
                "Run the 'select' stage first."
            )
        for atoms in iread(str(xyz_path), format="extxyz"):
            all_elements.update(atoms.get_chemical_symbols())

    sorted_elements = sorted(all_elements)
    logger.info(f"  Elements: {', '.join(sorted_elements)}")

    missing = [
        f"POTCAR_{e}" for e in sorted_elements
        if not (vasp_config_dir / f"POTCAR_{e}").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing pseudopotential files in {vasp_config_dir}/:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    potcar_data = {
        elem: (vasp_config_dir / f"POTCAR_{elem}").read_bytes()
        for elem in sorted_elements
    }

    for ds in datasets:
        logger.info(f"  Preparing {ds} jobs")
        xyz_path = selected_dir / f"{ds}.xyz"
        ds_jobs_dir = jobs_dir / ds
        ds_jobs_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        count = 0
        reused = 0

        for i, atoms in enumerate(iread(str(xyz_path), format="extxyz")):
            struct_dir = ds_jobs_dir / f"struct_{i:04d}"
            struct_dir.mkdir(parents=True, exist_ok=True)

            struct_elements = sorted(set(atoms.get_chemical_symbols()))
            potcar_bytes = b"".join(potcar_data[elem] for elem in struct_elements)
            input_identity = identity_for_structure(
                atoms,
                {"incar_hash": incar_hash, "potcar_data": potcar_data},
            )
            calculation = input_identity.calculation
            structure_id = input_identity.structure_id
            potcar_hash = input_identity.potcar_hash
            identity = {
                "project_name": project_name,
                "dataset": ds,
                "selected_index": i,
                "source_xyz": str(xyz_path.resolve()),
                "structure_id": structure_id,
                "incar_hash": incar_hash,
                "potcar_hash": potcar_hash,
                "calculation_id": calculation.calculation_id,
            }

            current_status = read_status(struct_dir).get("status", "pending")
            existing_identity = _read_identity(struct_dir)
            if current_status in {"submitted", "completed", "reused"}:
                if _identity_matches(existing_identity, identity):
                    count += 1
                    if current_status == "reused":
                        reused += 1
                    continue

            _clean_stale_outputs(struct_dir)
            write_poscar(atoms, struct_dir / "POSCAR")
            (struct_dir / "POTCAR").write_bytes(potcar_bytes)
            (struct_dir / "INCAR").write_text(incar_text, encoding="utf-8")
            _write_identity(struct_dir, identity)

            reusable_entry = _valid_registry_entry(
                registry,
                incar_hash,
                potcar_hash,
                structure_id,
            )
            if reusable_entry:
                reused_from = str(Path(reusable_entry["job_path"]).resolve())
                write_status(
                    struct_dir,
                    status="reused",
                    retry_level=0,
                    reused_from=reused_from,
                    structure_id=structure_id,
                    calculation_id=calculation.calculation_id,
                    incar_hash=incar_hash,
                    potcar_hash=potcar_hash,
                )
                reused += 1
            else:
                write_status(
                    struct_dir,
                    status="pending",
                    retry_level=0,
                    structure_id=structure_id,
                    calculation_id=calculation.calculation_id,
                    incar_hash=incar_hash,
                    potcar_hash=potcar_hash,
                )

            count += 1
            if count % 100 == 0:
                logger.info(f"    {count} structures prepared...")

        elapsed = time.perf_counter() - t0
        reuse_text = f", {reused} reused" if reused else ""
        logger.info(f"  {ds.capitalize()}: {count} jobs{reuse_text} ({elapsed:.1f}s)")

    logger.info("Job preparation complete")


# ==================================================================
# shared VASP runner script (like adaptive_healing.sh)
# ==================================================================

def write_shared_vasp_script(
    vasp_dir: Path,
    slurm_header: str,
    vasp_command: str,
) -> None:
    """Write vasp/run_vasp.sh - VASP runner submitted per structure."""
    bash_cmd = vasp_command.replace("{ntasks}", "$TOTAL_RANKS")
    if "mpirun" in bash_cmd and "--bind-to" not in bash_cmd:
        bash_cmd = bash_cmd.replace("mpirun", "mpirun --bind-to none", 1)

    body = r"""
# ============================================================================
# VASP RUNNER (generated by nepflow)
# ============================================================================
# Usage: sbatch --job-name=<name> --nodes=N --ntasks-per-node=G \
#              --gres=gpu:G run_vasp.sh <struct_dir>

STRUCT_DIR="$1"
if [ -z "$STRUCT_DIR" ]; then
    echo "Usage: sbatch run_vasp.sh <struct_dir>"
    exit 1
fi

if [ ! -d "$STRUCT_DIR" ]; then
    echo "ERROR: Structure directory not found: $STRUCT_DIR"
    exit 1
fi

echo "=========================================================================="
echo "nepflow VASP Runner"
echo "=========================================================================="
echo "Structure: $STRUCT_DIR"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Timestamp: $(date)"
echo "=========================================================================="

cd "$STRUCT_DIR" || exit 1

NCORE_VAL=$(grep -i "^[[:space:]]*NCORE" INCAR | grep "=" | head -1 | sed 's/.*=//;s/#.*//' | xargs)
KPAR_VAL=$(grep -i "^[[:space:]]*KPAR" INCAR | grep "=" | head -1 | sed 's/.*=//;s/#.*//' | xargs)
[ -z "$NCORE_VAL" ] && NCORE_VAL=2
[ -z "$KPAR_VAL" ] && KPAR_VAL=1

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

if grep -q "oom_kill" "$VASP_LOG" 2>/dev/null || [ $VASP_EXIT -eq 137 ]; then
    echo ""
    echo "OOM detected! Writing marker for launcher to handle retry..."
    echo "1" > "$STRUCT_DIR/.vasp_oom_detected"
    exit 1
fi

if [ $VASP_EXIT -eq 0 ]; then
    echo "VASP completed successfully"
    rm -f CHG CHGCAR WAVECAR CONTCAR DOSCAR EIGENVAL PCDAT
    exit 0
else
    echo "VASP failed with exit code $VASP_EXIT (not OOM)"
    echo "Check $STRUCT_DIR/vasp_output.log for details"
    exit 1
fi
"""
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
    path.write_text(canonical_poscar_text(atoms), encoding="utf-8")


def _read_identity(struct_dir: Path) -> dict:
    """Compatibility bridge for callers that still import this helper."""
    return read_identity(struct_dir)


def _write_identity(struct_dir: Path, identity: dict) -> None:
    write_json(struct_dir / ".vasp_identity", identity)


def _identity_matches(existing: dict, expected: dict) -> bool:
    existing = normalise_dft_calculation_identity(existing)
    expected = normalise_dft_calculation_identity(expected)
    keys = ("structure_id", "incar_hash", "potcar_hash", "calculation_id")
    return bool(existing) and all(existing.get(key) == expected[key] for key in keys)


def _clean_stale_outputs(struct_dir: Path) -> None:
    """Remove outputs from an old identity before preparing a new calculation."""
    for filename in (
        "OUTCAR",
        "vasprun.xml",
        "OSZICAR",
        "vasp_output.log",
        ".vasp_oom_detected",
        "CHG",
        "CHGCAR",
        "WAVECAR",
        "CONTCAR",
        "DOSCAR",
        "EIGENVAL",
        "PCDAT",
    ):
        (struct_dir / filename).unlink(missing_ok=True)


def _valid_registry_entry(
    registry: dict,
    incar_hash: str,
    potcar_hash: str,
    structure_id: str,
) -> dict | None:
    entry = get_registry_entry(registry, incar_hash, potcar_hash, structure_id)
    if entry is None:
        return None
    return entry if outcar_is_complete(Path(entry["job_path"]) / "OUTCAR") else None


# ==================================================================
# INCAR injection
# ==================================================================

def inject_incar_defaults(incar_text: str, config: ConfigParser) -> str:
    """Compatibility bridge to canonical INCAR default injection."""
    return _canonical_inject_incar_defaults(incar_text, config)
