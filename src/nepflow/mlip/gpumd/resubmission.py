"""Canonical GPUMD segment state and restart-file promotion services."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from nepflow.hpc.process import ProcessRunner

STATE_FILE_NAME = ".gpumd_self_resubmit_state.json"
DEFAULT_ARCHIVE_DIR = "final_xyz_history"


def load_segment_state(state_path: Path) -> dict[str, Any]:
    """Load one segment state record, or return a new state."""

    state_path = Path(state_path)
    if not state_path.exists():
        return {"segments_completed": 0, "history": [], "created": time.time()}
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read state file: {state_path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"GPUMD state file must contain an object: {state_path}")
    return value


def write_segment_state(state_path: Path, state: dict[str, Any]) -> None:
    """Persist the segment state with an update timestamp."""

    state["updated"] = time.time()
    Path(state_path).write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_gpumd_segment(
    command: str,
    workdir: Path,
    *,
    process_runner: ProcessRunner | None = None,
    dry_run: bool = False,
) -> None:
    """Run the operator-supplied GPUMD shell command."""

    if dry_run:
        return
    result = (process_runner or ProcessRunner()).run_shell(
        command,
        cwd=workdir,
        check=False,
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GPUMD command failed with exit code {result.returncode}.")


def archive_and_promote_final(
    workdir: Path,
    archive_dir_name: str,
    final_name: str,
    model_name: str,
    segment_index: int,
) -> dict[str, str]:
    """Archive a completed GPUMD restart and promote it to the model input."""

    workdir = Path(workdir)
    final_path = workdir / final_name
    if not final_path.exists():
        raise FileNotFoundError(f"GPUMD did not produce {final_path}.")
    if final_path.stat().st_size == 0:
        raise RuntimeError(f"GPUMD produced an empty restart file: {final_path}.")
    archive_dir = workdir / archive_dir_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_path = archive_dir / f"segment_{segment_index:05d}_{final_name}"
    shutil.copy2(final_path, archived_path)
    model_path = workdir / model_name
    final_path.replace(model_path)
    return {"archived_final": str(archived_path), "model_file": str(model_path)}


def should_stop(
    state: dict[str, Any],
    max_segments: int | None,
    stop_file: Path | None,
) -> tuple[bool, str]:
    """Apply the finite segment-stop policy."""

    if max_segments is not None and int(state.get("segments_completed", 0)) >= max_segments:
        return True, f"reached max segments ({max_segments})"
    if stop_file is not None and Path(stop_file).exists():
        return True, f"stop file present ({stop_file})"
    return False, ""


__all__ = [
    "DEFAULT_ARCHIVE_DIR",
    "STATE_FILE_NAME",
    "archive_and_promote_final",
    "load_segment_state",
    "run_gpumd_segment",
    "should_stop",
    "write_segment_state",
]
