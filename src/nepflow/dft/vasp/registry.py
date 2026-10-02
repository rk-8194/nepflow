"""Validated VASP job-status and completed-artifact registry records.

The launcher still consumes these records while DFT preparation moves into
the canonical stage package.  Keeping the file contract here prevents the
preparation service from depending on a legacy stage module.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from nepflow.errors import ArtifactError, StateError
from nepflow.io.json import read_json_object, write_json


VASP_REGISTRY_VERSION = 1
VALID_VASP_STATUSES = frozenset(
    {"pending", "submitted", "completed", "reused", "failed", "oom"}
)


def get_nepflow_root(project_dir: Path) -> Path:
    """Return the shared NEPFlow root for cross-project VASP artifacts."""
    project_dir = Path(project_dir)
    if project_dir.name.startswith("project_") and project_dir.parent.name == "projects":
        return project_dir.parent.parent
    return project_dir


def completed_jobs_registry_path(nepflow_root: Path) -> Path:
    """Return the shared completed-VASP registry path."""
    return Path(nepflow_root) / ".vasp_completed_jobs.json"


def validate_completed_registry(data: dict, path: Path | None = None) -> dict:
    """Validate the shared completed-VASP registry schema."""
    location = f" in {path}" if path is not None else ""
    version = data.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != VASP_REGISTRY_VERSION
    ):
        raise ArtifactError(f"Unsupported VASP registry version{location}: {version!r}")
    if "jobs" not in data or not isinstance(data["jobs"], dict):
        raise ArtifactError(f"VASP registry jobs must be an object{location}")
    return data


def read_completed_registry(nepflow_root: Path) -> dict:
    """Read the shared completed-VASP registry."""
    path = completed_jobs_registry_path(nepflow_root)
    data = read_json_object(
        path,
        default={"version": VASP_REGISTRY_VERSION, "jobs": {}},
        error_type=ArtifactError,
    )
    return validate_completed_registry(data, path)


def write_completed_registry(nepflow_root: Path, registry: dict) -> None:
    """Write the shared completed-VASP registry."""
    write_json(completed_jobs_registry_path(nepflow_root), registry)


def get_registry_entry(
    registry: dict,
    incar_hash: str,
    potcar_hash: str,
    structure_id: str,
) -> dict | None:
    """Return a registry entry for an exact scientific input identity."""
    jobs = registry.get("jobs")
    if not isinstance(jobs, dict):
        raise ArtifactError("VASP registry jobs must be an object")
    incar_entries = jobs.get(incar_hash)
    if incar_entries is None:
        return None
    if not isinstance(incar_entries, dict):
        raise ArtifactError(
            f"Registry entries for INCAR hash are malformed: {incar_hash}"
        )
    potcar_entries = incar_entries.get(potcar_hash)
    if potcar_entries is None:
        return None
    if not isinstance(potcar_entries, dict):
        raise ArtifactError(
            f"Registry entries for POTCAR hash are malformed: {potcar_hash}"
        )
    if structure_id not in potcar_entries:
        return None
    entry = potcar_entries[structure_id]
    if not isinstance(entry, dict):
        raise ArtifactError(
            f"Registry entry for structure ID is malformed: {structure_id}"
        )
    job_path = entry.get("job_path")
    if not isinstance(job_path, str) or not job_path.strip():
        raise ArtifactError(
            "Registry entry for structure ID lacks a valid job_path: "
            f"{structure_id}"
        )
    return entry


def upsert_registry_entry(
    nepflow_root: Path,
    incar_hash: str,
    potcar_hash: str,
    structure_id: str,
    entry: dict,
) -> None:
    """Insert or replace a completed-job registry entry."""
    registry = read_completed_registry(nepflow_root)
    registry["jobs"].setdefault(incar_hash, {}).setdefault(potcar_hash, {})[
        structure_id
    ] = entry
    write_completed_registry(nepflow_root, registry)


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
    **extra: object,
) -> None:
    """Write one validated-enough launcher status record."""
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
    data.update({key: value for key, value in extra.items() if value is not None})
    write_json(Path(struct_dir) / ".vasp_status", data)


def read_status(struct_dir: Path) -> dict:
    """Read and validate one launcher status record."""
    status_file = Path(struct_dir) / ".vasp_status"
    data = read_json_object(
        status_file,
        default={"status": "pending", "retry_level": 0},
        error_type=StateError,
    )
    status = data.get("status")
    if not isinstance(status, str) or status not in VALID_VASP_STATUSES:
        raise StateError(f"VASP status is missing or invalid: {status_file}")
    retry_level = data.get("retry_level", 0)
    if (
        isinstance(retry_level, bool)
        or not isinstance(retry_level, int)
        or retry_level < 0
    ):
        raise StateError(f"VASP status retry_level is invalid: {status_file}")
    return data


__all__ = [
    "VALID_VASP_STATUSES",
    "VASP_REGISTRY_VERSION",
    "completed_jobs_registry_path",
    "get_nepflow_root",
    "get_registry_entry",
    "read_completed_registry",
    "read_status",
    "upsert_registry_entry",
    "validate_completed_registry",
    "write_completed_registry",
    "write_status",
]
