"""Shared constants and utilities for the run_vasp sub-stages."""

import csv
import json
import logging
import re
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger("nepflow.run_vasp")

VASP_COMPLETION_MARKERS = ["General timing", "Voluntary context switches"]


# ==================================================================
# HPC-aware retry level generation
# ==================================================================

def build_retry_levels_for_gpu(
    starting_gpu: int,
    initial_ncore: int,
    initial_kpar: int,
    config: ConfigParser,
) -> list:
    """Build GPU-aware retry escalation from the job's initial parameters.

    Called with the ORIGINAL (first-submission) parameters so that the
    escalation table is stable across retries.  The launcher indexes
    into the returned list using the retry count.

    Escalation hierarchy:
      1. Fix KPAR = GPU count (optimal for GPU VASP), sweep all NCORE
      2. Try KPAR = 1 as aggressive memory-saving fallback, sweep NCORE
      3. Jump to next GPU tier (1→2→4) and repeat
      4. Multi-node escalation (2, 4, … nodes)

    Args:
      starting_gpu: GPU count from the initial submission
      initial_ncore: NCORE from the initial submission
      initial_kpar: KPAR from the initial submission
    """
    cores = config.getint("hpc", "cores_per_node", fallback=64)
    gpus_per_node = config.getint("hpc", "gpus_per_node", fallback=4)
    max_nodes = config.getint("hpc", "max_nodes", fallback=16)

    valid_ncores = sorted(p for p in (2**i for i in range(1, 12))
                          if p <= cores and cores % p == 0)
    if not valid_ncores:
        valid_ncores = [cores]

    levels: list[tuple[int, int, int, int]] = []
    seen = set()
    initial_key = (initial_ncore, initial_kpar, 1, starting_gpu)

    def _add(ncore, kpar, nodes, gpus):
        key = (ncore, kpar, nodes, gpus)
        if key not in seen and key != initial_key:
            seen.add(key)
            levels.append(key)

    valid_kpars = sorted(
        k for k in (2**i for i in range(0, 8))
        if k <= gpus_per_node
    ) or [1]

    def _sweep_gpu_tier(gpu_count):
        """Add NCORE/KPAR combinations for a given GPU count."""
        for kpar in valid_kpars:
            for nc in valid_ncores:
                _add(nc, kpar, 1, gpu_count)

    # Phase 1: Current GPU tier — fix KPAR to GPU count, try all NCORE
    _sweep_gpu_tier(starting_gpu)

    # Phase 2: Higher GPU tiers
    valid_gpus = sorted(g for g in [1, 2, 4, 8] if g <= gpus_per_node)
    for gpu in valid_gpus:
        if gpu > starting_gpu:
            _sweep_gpu_tier(gpu)

    # Phase 3: Multi-node escalation at highest GPU tier
    highest_gpu = valid_gpus[-1] if valid_gpus else starting_gpu
    nodes = 2
    while nodes <= max_nodes:
        kpar = nodes * highest_gpu
        for nc in valid_ncores:
            _add(nc, kpar, nodes, highest_gpu)
        nodes *= 2

    return levels


# ==================================================================
# per-structure status tracking
# ==================================================================

def write_status(
    struct_dir: Path,
    status: str,
    retry_level: int = 0,
    slurm_job_id: str = "",
    error: str = "",
    initial_gpu: int | None = None,
    initial_ncore: int | None = None,
    initial_kpar: int | None = None,
    current_gpu: int | None = None,
) -> None:
    """Write .vasp_status JSON to a structure directory."""
    data = {
        "status": status,
        "retry_level": retry_level,
        "slurm_job_id": slurm_job_id,
        "timestamp": datetime.now().isoformat(),
    }
    if error:
        data["error"] = error
    if initial_gpu is not None:
        data["initial_gpu"] = initial_gpu
    if initial_ncore is not None:
        data["initial_ncore"] = initial_ncore
    if initial_kpar is not None:
        data["initial_kpar"] = initial_kpar
    if current_gpu is not None:
        data["current_gpu"] = current_gpu
    (struct_dir / ".vasp_status").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def read_status(struct_dir: Path) -> dict:
    """Read .vasp_status JSON from a structure directory."""
    status_file = struct_dir / ".vasp_status"
    if status_file.exists():
        try:
            return json.loads(status_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"status": "pending", "retry_level": 0}


def write_launcher_state(vasp_dir: Path, job_id: str, walltime_seconds: int) -> None:
    """Write .launcher_state JSON."""
    vasp_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "launcher_job_id": job_id,
        "start_time": datetime.now().isoformat(),
        "walltime_seconds": walltime_seconds,
    }
    (vasp_dir / ".launcher_state").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )


# ==================================================================
# walltime helpers
# ==================================================================

def parse_walltime(wt: str) -> int:
    """Parse HH:MM:SS walltime string to seconds."""
    parts = wt.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 10800  # default 3h


def format_walltime(seconds: int) -> str:
    """Format seconds as HH:MM:SS."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==================================================================
# POTCAR helpers
# ==================================================================

def parse_zval(potcar_path: Path) -> float:
    """Extract ZVAL (valence electron count) from a single-element POTCAR."""
    text = potcar_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"ZVAL\s*=\s*([\d.]+)", text)
    if m:
        return float(m.group(1))
    raise ValueError(f"ZVAL not found in {potcar_path}")


def estimate_n_electrons(struct_dir: Path, zval_cache: dict) -> int:
    """Estimate total electrons for a structure from POSCAR + cached ZVALs.

    *zval_cache* maps element symbol → ZVAL (float).
    """
    poscar = struct_dir / "POSCAR"
    lines = poscar.read_text(encoding="utf-8").splitlines()
    # VASP5 POSCAR: line 6 = element symbols, line 7 = counts
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    return int(sum(zval_cache[e] * c for e, c in zip(elements, counts)))


def estimate_kpoints_irr(struct_dir: Path, config: ConfigParser) -> int | None:
    """Estimate the number of irreducible k-points from POSCAR + INCAR.

    Replicates VASP's KSPACING → k-mesh algorithm and uses spglib
    to reduce to the irreducible Brillouin zone wedge.
    Returns None if estimation fails.
    """
    poscar = struct_dir / "POSCAR"
    incar = struct_dir / "INCAR"
    if not poscar.exists():
        return None

    try:
        from ase.io import read as ase_read
        import spglib
    except ImportError:
        logger.debug("spglib or ase not available for k-point estimation")
        return None

    # Parse KSPACING and KGAMMA from INCAR (fall back to config defaults)
    kspacing = config.getfloat("vasp", "kspacing", fallback=0.30)
    kgamma = True
    if incar.exists():
        try:
            incar_text = incar.read_text(encoding="utf-8")
            m = re.search(
                r"^\s*KSPACING\s*=\s*([\d.]+)",
                incar_text, re.MULTILINE | re.IGNORECASE,
            )
            if m:
                kspacing = float(m.group(1))
            m = re.search(
                r"^\s*KGAMMA\s*=\s*\.?(TRUE|FALSE|T|F)\.?",
                incar_text, re.MULTILINE | re.IGNORECASE,
            )
            if m:
                kgamma = m.group(1).upper().startswith("T")
        except OSError:
            pass

    try:
        atoms = ase_read(str(poscar), format="vasp")
        cell = atoms.get_cell()

        # ASE reciprocal() returns b_i WITHOUT the 2π factor,
        # but VASP's KSPACING uses |b_i| WITH 2π:  N_i = ceil(2π|b_i| / KSPACING)
        reciprocal = cell.reciprocal()

        # K-mesh from KSPACING (replicates VASP's algorithm)
        mesh = [max(1, int(np.ceil(2 * np.pi * np.linalg.norm(b) / kspacing)))
                for b in reciprocal]

        # spglib cell: (lattice, scaled_positions, atomic_numbers)
        spg_cell = (
            cell.array,
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers(),
        )
        is_shift = [0, 0, 0] if kgamma else [1, 1, 1]
        mapping, _ = spglib.get_ir_reciprocal_mesh(
            mesh, spg_cell, is_shift=is_shift,
        )
        n_kpoints_irr = len(np.unique(mapping))

        logger.debug(
            f"    K-point estimate: mesh={mesh}, irred={n_kpoints_irr} "
            f"(KSPACING={kspacing}, KGAMMA={kgamma})"
        )
        return n_kpoints_irr
    except Exception as e:
        logger.debug(f"    Failed to estimate k-points: {e}")
        return None


# ==================================================================
# ML-based parameter prediction
# ==================================================================

def load_vasp_memory(nepflow_root: Path):
    """Load .vasp_memory CSV, filtering out OOM entries.

    Returns (features, params) where:
      features: ndarray shape (N, 3) — [n_atoms, n_electrons, n_kpoints_irr]
      params:   ndarray shape (N, 5) — [ncore, kpar, nodes, gpus, avg_loop_time]
    Returns (None, None) if the memory file is missing or empty.

    Rows with oom=1 are excluded from prediction data.
    Backward compatible: if ``oom`` or ``n_kpoints_irr`` columns are
    missing, they default to 0.
    """
    csv_path = nepflow_root / ".vasp_memory"
    if not csv_path.exists():
        return None, None

    features = []
    params = []
    try:
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip OOM rows (oom column may be absent in old CSVs)
                if int(row.get("oom", 0)) == 1:
                    continue
                features.append([
                    int(row["n_atoms"]),
                    int(row["n_electrons"]),
                    int(row.get("n_kpoints_irr", 0)),
                ])
                params.append([int(row["ncore"]), int(row["kpar"]),
                               int(row["nodes"]), int(row["gpus"]),
                               float(row.get("avg_loop_time", 0.0))])
    except (OSError, KeyError, ValueError) as e:
        logger.debug(f"Could not load .vasp_memory: {e}")
        return None, None

    if not features:
        return None, None

    return np.array(features, dtype=float), np.array(params, dtype=float)


def _snap_ncore(value: float, cores_per_node: int) -> int:
    """Snap to nearest power-of-2 that evenly divides cores_per_node."""
    valid = sorted(p for p in (2**i for i in range(1, 12))
                   if p <= cores_per_node and cores_per_node % p == 0)
    if not valid:
        return cores_per_node
    return min(valid, key=lambda v: abs(v - value))


def _snap_nodes(value: float, max_nodes: int) -> int:
    """Snap to nearest power-of-2 in [1, max_nodes]."""
    valid = [2**i for i in range(0, 20) if 2**i <= max_nodes]
    if not valid:
        return 1
    return min(valid, key=lambda v: abs(v - value))


def _snap_gpus(value: float, gpus_per_node: int) -> int:
    """Snap to nearest valid GPU count: 1, 2, …, gpus_per_node."""
    valid = sorted(g for g in [1, 2, 4, 8] if g <= gpus_per_node)
    if not valid:
        return gpus_per_node
    return min(valid, key=lambda v: abs(v - value))


def predict_vasp_params(
    n_atoms: int,
    n_electrons: int,
    nepflow_root: Path,
    config: ConfigParser,
    n_kpoints_irr: int | None = None,
):
    """Predict (ncore, kpar, nodes, gpus) for a structure from VASP memory.

    Uses k-NN (distance-weighted) on (n_atoms, n_electrons, n_kpoints_irr),
    averages the actual parameters from nearest neighbors, and snaps
    to HPC-valid values. Returns None when insufficient data exists.

    When *n_kpoints_irr* is provided and the memory contains k-point data,
    a 3D feature space is used so that structures with very different
    k-meshes (e.g. FCC primitive vs cubic supercells) are distinguished.
    Falls back to 2D (n_atoms, n_electrons) when k-point data is
    unavailable on either side.
    """
    features, params = load_vasp_memory(nepflow_root)
    if features is None or params is None or len(features) < 3:
        return None

    cores_per_node = config.getint("hpc", "cores_per_node", fallback=64)
    max_nodes = config.getint("hpc", "max_nodes", fallback=16)
    gpus_per_node = config.getint("hpc", "gpus_per_node", fallback=4)

    # Use 3D features if k-point data exists in both query and memory
    use_kpoints = (
        n_kpoints_irr is not None
        and n_kpoints_irr > 0
        and features[:, 2].max() > 0
    )
    if use_kpoints:
        query = np.array([[n_atoms, n_electrons, n_kpoints_irr]], dtype=float)
        feat = features
    else:
        query = np.array([[n_atoms, n_electrons]], dtype=float)
        feat = features[:, :2]

    # Normalise features (avoid division by zero)
    std = feat.std(axis=0)
    std[std == 0] = 1.0
    mean = feat.mean(axis=0)
    norm_features = (feat - mean) / std
    norm_query = (query - mean) / std

    # Euclidean distances
    dists = np.linalg.norm(norm_features - norm_query, axis=1)

    # Select ALL rows at (or within epsilon of) the minimum distance.
    # This handles the common case where many rows share the same feature
    # vector and are therefore all equidistant; k-NN argpartition would
    # return an arbitrary subset and could miss cheaper GPU options.
    min_dist = dists.min()
    close_mask = dists <= min_dist + 1e-6
    close_params = params[close_mask]  # (M, 5): ncore, kpar, nodes, gpus, avg_loop_time

    # GPU: prefer fewest GPUs present among close rows
    gpus = int(close_params[:, 3].min())
    gpus = _snap_gpus(gpus, gpus_per_node)

    # NCORE/KPAR/nodes: actual best row (lowest avg_loop_time) for that GPU count
    gpu_rows = close_params[close_params[:, 3] == gpus]
    best = gpu_rows[int(np.argmin(gpu_rows[:, 4]))]
    ncore = _snap_ncore(best[0], cores_per_node)
    kpar = max(1, int(best[1]))
    nodes = _snap_nodes(best[2], max_nodes)

    logger.debug(
        f"    predict: NCORE={ncore} KPAR={kpar} nodes={nodes} gpus={gpus} "
        f"({close_mask.sum()} neighbors, best_time={best[4]:.2f}s)"
    )
    return (ncore, kpar, nodes, gpus)
