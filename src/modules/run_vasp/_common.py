"""Shared constants and utilities for the run_vasp sub-stages."""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("nepflow.run_vasp")

# OOM retry escalation table (mirrors adaptive_healing.sh)
# (ncore, kpar, nodes, gpus_per_node)
RETRY_LEVELS = [
    (16, 2, 1, 4),   # Level 0: baseline
    (32, 2, 1, 4),   # Level 1: more CPU threading
    (64, 2, 1, 4),   # Level 2: max threading
    (64, 1, 1, 4),   # Level 3: single k-point group
    (16, 8, 4, 4),   # Level 4: 4 nodes / 16 GPUs
    (16, 16, 8, 4),  # Level 5: 8 nodes / 32 GPUs
    (16, 32, 16, 4), # Level 6: 16 nodes / 64 GPUs
]

VASP_COMPLETION_MARKERS = ["General timing", "Voluntary context switches"]


# ==================================================================
# per-structure status tracking
# ==================================================================

def read_retry_level(struct_dir: Path) -> int:
    """Read .vasp_retry_level written by run_vasp.sh on OOM self-requeue."""
    retry_file = struct_dir / ".vasp_retry_level"
    if retry_file.exists():
        try:
            return int(retry_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass
    return 0


def write_status(
    struct_dir: Path,
    status: str,
    retry_level: int = 0,
    slurm_job_id: str = "",
    error: str = "",
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
