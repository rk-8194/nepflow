#!/usr/bin/env python3
"""Run a GPUMD segment, archive ``final.xyz``, promote it to ``model.xyz``, and self-resubmit.

This utility is intentionally standalone so it can be copied into an MD working
directory and launched directly from a SLURM batch script. When NEPFlow is
available, it reuses NEPFlow's SLURM resubmission resolver. If not, it falls
back to the same resolution strategy locally.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


STATE_FILE_NAME = ".gpumd_self_resubmit_state.json"
DEFAULT_ARCHIVE_DIR = "final_xyz_history"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GPUMD in the current MD directory, archive each final.xyz, "
            "promote it to model.xyz, and resubmit the same SLURM script."
        )
    )
    parser.add_argument(
        "--gpumd-command",
        required=True,
        help=(
            "Shell command used to run one GPUMD segment, for example "
            "\"mpirun -np 1 --bind-to none $HOME/src/GPUMD/src/gpumd < run.in > gpumd.log 2>&1\"."
        ),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="MD simulation directory. Defaults to the current directory.",
    )
    parser.add_argument(
        "--final-file",
        default="final.xyz",
        help="Name of the GPUMD restart output file. Defaults to final.xyz.",
    )
    parser.add_argument(
        "--model-file",
        default="model.xyz",
        help="Name of the GPUMD restart input file. Defaults to model.xyz.",
    )
    parser.add_argument(
        "--archive-dir",
        default=DEFAULT_ARCHIVE_DIR,
        help=(
            "Directory that stores every completed final.xyz snapshot. "
            f"Defaults to {DEFAULT_ARCHIVE_DIR}."
        ),
    )
    parser.add_argument(
        "--state-file",
        default=STATE_FILE_NAME,
        help=f"State filename written inside the workdir. Defaults to {STATE_FILE_NAME}.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        help="Stop after this many completed GPUMD segments.",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="Optional sentinel file. If it exists after a segment, no resubmission occurs.",
    )
    parser.add_argument(
        "--submit-script",
        type=Path,
        help="Explicit SLURM submit script to use instead of auto-detection.",
    )
    parser.add_argument(
        "--nepflow-root",
        type=Path,
        help=(
            "Optional NEPFlow root directory. If supplied, the utility will "
            "reuse NEPFlow's SLURM resubmit resolver directly."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without launching GPUMD or sbatch.",
    )
    return parser.parse_args()


def ensure_stop_rule(args: argparse.Namespace) -> None:
    if args.max_segments is None and args.stop_file is None:
        raise ValueError("Specify at least one stop rule: --max-segments and/or --stop-file.")
    if args.max_segments is not None and args.max_segments < 1:
        raise ValueError("--max-segments must be at least 1.")


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {
            "segments_completed": 0,
            "history": [],
            "created": time.time(),
        }
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read state file: {state_path}") from exc


def write_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated"] = time.time()
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_gpumd(command: str, workdir: Path, dry_run: bool = False) -> None:
    print(f"[gpumd-self-resubmit] Running GPUMD command in {workdir}: {command}")
    if dry_run:
        return

    result = subprocess.run(command, shell=True, cwd=str(workdir), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GPUMD command failed with exit code {result.returncode}.")


def archive_and_promote_final(
    workdir: Path,
    archive_dir_name: str,
    final_name: str,
    model_name: str,
    segment_index: int,
) -> dict[str, str]:
    final_path = workdir / final_name
    if not final_path.exists():
        raise FileNotFoundError(f"GPUMD did not produce {final_path}.")
    if final_path.stat().st_size == 0:
        raise RuntimeError(f"GPUMD produced an empty restart file: {final_path}.")

    archive_dir = workdir / archive_dir_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived_name = f"segment_{segment_index:05d}_{final_name}"
    archived_path = archive_dir / archived_name
    shutil.copy2(final_path, archived_path)

    model_path = workdir / model_name
    final_path.replace(model_path)

    return {
        "archived_final": str(archived_path),
        "model_file": str(model_path),
    }


def _load_nepflow_module(nepflow_root: Path):
    nepflow_py = nepflow_root / "nepflow.py"
    if not nepflow_py.exists():
        raise FileNotFoundError(f"nepflow.py not found under {nepflow_root}")

    spec = importlib.util.spec_from_file_location("nepflow_cli_for_md_utility", nepflow_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load nepflow module from {nepflow_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_resubmit_from_scontrol(slurm_job_id: str) -> tuple[list[str], Path, str] | None:
    try:
        result = subprocess.run(
            ["scontrol", "show", "job", slurm_job_id],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    command_match = re.search(r"\bCommand=(\S+)", result.stdout)
    if not command_match:
        return None

    command_path = Path(command_match.group(1)).expanduser()
    if not command_path.is_absolute():
        workdir_match = re.search(r"\bWorkDir=(\S+)", result.stdout)
        if workdir_match:
            command_path = Path(workdir_match.group(1)) / command_path

    if not command_path.exists():
        return None

    return ["sbatch", str(command_path)], command_path.parent, f"scontrol job {slurm_job_id}"


def _fallback_resubmit_command(workdir: Path, submit_script: Path | None) -> tuple[list[str], Path, str]:
    if submit_script is not None:
        resolved = submit_script if submit_script.is_absolute() else (workdir / submit_script)
        resolved = resolved.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Submit script not found: {resolved}")
        return ["sbatch", str(resolved)], resolved.parent, f"explicit submit script {resolved}"

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        resolved = _resolve_resubmit_from_scontrol(slurm_job_id)
        if resolved is not None:
            return resolved

    candidate_dirs: list[Path] = []
    for directory in (
        os.environ.get("SLURM_SUBMIT_DIR"),
        str(workdir),
        os.getcwd(),
    ):
        if not directory:
            continue
        path = Path(directory).resolve()
        if path not in candidate_dirs:
            candidate_dirs.append(path)

    tried: list[str] = []
    for directory in candidate_dirs:
        candidate = directory / "submit.slurm"
        tried.append(str(candidate))
        if candidate.exists():
            return ["sbatch", str(candidate)], directory, f"submit.slurm in {directory}"

    tried_text = "\n".join(f"  - {candidate}" for candidate in tried)
    raise FileNotFoundError(f"Could not find a submit script to resubmit. Tried:\n{tried_text}")


def resolve_resubmit_command(
    workdir: Path,
    submit_script: Path | None,
    nepflow_root: Path | None,
) -> tuple[list[str], Path, str]:
    if nepflow_root is not None and submit_script is None:
        module = _load_nepflow_module(nepflow_root.resolve())
        return module._resolve_resubmit_command()
    return _fallback_resubmit_command(workdir=workdir, submit_script=submit_script)


def should_stop(
    state: dict[str, Any],
    max_segments: int | None,
    stop_file: Path | None,
) -> tuple[bool, str]:
    if max_segments is not None and int(state.get("segments_completed", 0)) >= max_segments:
        return True, f"reached max segments ({max_segments})"
    if stop_file is not None and stop_file.exists():
        return True, f"stop file present ({stop_file})"
    return False, ""


def resubmit_job(
    workdir: Path,
    submit_script: Path | None,
    nepflow_root: Path | None,
    dry_run: bool = False,
) -> None:
    command, submit_cwd, submit_source = resolve_resubmit_command(
        workdir=workdir,
        submit_script=submit_script,
        nepflow_root=nepflow_root,
    )
    print(
        "[gpumd-self-resubmit] Resubmitting via "
        f"{submit_source}: {' '.join(command)} (cwd={submit_cwd})"
    )
    if dry_run:
        return

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        cwd=str(submit_cwd),
    )
    stdout = result.stdout.strip()
    if stdout:
        print(f"[gpumd-self-resubmit] sbatch output: {stdout}")


def main() -> int:
    args = parse_args()
    ensure_stop_rule(args)

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    stop_file = args.stop_file.resolve() if args.stop_file is not None else None
    state_path = workdir / args.state_file
    state = load_state(state_path)

    if args.dry_run:
        next_segment = int(state.get("segments_completed", 0)) + 1
        print(f"[gpumd-self-resubmit] Dry run for workdir {workdir}")
        print(f"[gpumd-self-resubmit] Next segment would be {next_segment}")
        print(f"[gpumd-self-resubmit] GPUMD command: {args.gpumd_command}")
        stop, reason = should_stop(
            state={"segments_completed": next_segment},
            max_segments=args.max_segments,
            stop_file=stop_file,
        )
        if stop:
            print(f"[gpumd-self-resubmit] Would stop after the next segment because {reason}.")
        else:
            resubmit_job(
                workdir=workdir,
                submit_script=args.submit_script,
                nepflow_root=args.nepflow_root,
                dry_run=True,
            )
        return 0

    next_segment = int(state.get("segments_completed", 0)) + 1
    run_gpumd(args.gpumd_command, workdir=workdir, dry_run=False)

    archived = archive_and_promote_final(
        workdir=workdir,
        archive_dir_name=args.archive_dir,
        final_name=args.final_file,
        model_name=args.model_file,
        segment_index=next_segment,
    )

    state["segments_completed"] = next_segment
    state.setdefault("history", []).append(
        {
            "segment": next_segment,
            "time": time.time(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            **archived,
        }
    )
    write_state(state_path, state)

    stop, reason = should_stop(
        state=state,
        max_segments=args.max_segments,
        stop_file=stop_file,
    )
    if stop:
        print(f"[gpumd-self-resubmit] Completed segment {next_segment}; stopping because {reason}.")
        return 0

    resubmit_job(
        workdir=workdir,
        submit_script=args.submit_script,
        nepflow_root=args.nepflow_root,
        dry_run=False,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"[gpumd-self-resubmit] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
