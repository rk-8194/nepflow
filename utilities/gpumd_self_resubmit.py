#!/usr/bin/env python3
"""Thin operator CLI for canonical GPUMD segment/resubmission services."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from nepflow.hpc.slurm import SlurmScheduler
from nepflow.mlip.gpumd.resubmission import (
    DEFAULT_ARCHIVE_DIR,
    STATE_FILE_NAME,
    archive_and_promote_final,
    load_segment_state,
    run_gpumd_segment,
    should_stop,
    write_segment_state,
)
from nepflow.workflow.resubmission import (
    resolve_resubmit_command as canonical_resolve_resubmit_command,
)
from nepflow.workflow.resubmission import (
    submit_resubmission,
)

load_state = load_segment_state
write_state = write_segment_state


def resolve_resubmit_command(
    workdir: Path,
    submit_script: Path | None,
    nepflow_root: Path | None = None,
) -> tuple[list[str], Path, str]:
    """Compatibility-shaped thin wrapper over the workflow resolver."""

    _ = nepflow_root
    return canonical_resolve_resubmit_command(
        workdir,
        submit_script,
        scheduler=SlurmScheduler(),
    )


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
            '"mpirun -np 1 --bind-to none $HOME/src/GPUMD/src/gpumd < run.in > gpumd.log 2>&1".'
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


def run_gpumd(command: str, workdir: Path, dry_run: bool = False) -> None:
    """Thin CLI adapter over the canonical GPUMD segment runner."""

    print(f"[gpumd-self-resubmit] Running GPUMD command in {workdir}: {command}")
    run_gpumd_segment(command, workdir, dry_run=dry_run)


def resubmit_job(
    workdir: Path,
    submit_script: Path | None,
    nepflow_root: Path | None,
    dry_run: bool = False,
) -> None:
    """Thin CLI adapter over canonical workflow resubmission services."""

    _ = nepflow_root  # retained only for command-line compatibility
    scheduler = SlurmScheduler()
    command, submit_cwd, submit_source = canonical_resolve_resubmit_command(
        workdir,
        submit_script,
        scheduler=scheduler,
    )
    print(
        "[gpumd-self-resubmit] Resubmitting via "
        f"{submit_source}: {' '.join(command)} (cwd={submit_cwd})"
    )
    stdout = submit_resubmission(
        command,
        submit_cwd,
        scheduler=scheduler,
        dry_run=dry_run,
    )
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
