"""Typed SLURM resource values and shared directive rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math
from pathlib import Path
import re
from typing import Iterable


MemoryValue = int | str

_WALLTIME_PATTERN = re.compile(r"^(?:(?P<days>\d+)-)?(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)$")
_MEMORY_PATTERN = re.compile(r"^(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>[kmgtpe]?i?b?)?$", re.IGNORECASE)
_RESOURCE_DIRECTIVE_KEYS = (
    "--job-name",
    "--nodes",
    "--ntasks",
    "--ntasks-per-node",
    "--cpus-per-task",
    "--gpus",
    "--gpus-per-node",
    "--gres",
    "--mem",
    "--mem-per-cpu",
    "--mem-per-gpu",
    "--time",
    "--output",
    "--error",
)


@dataclass(frozen=True, slots=True)
class JobResources:
    """Unambiguous resources requested for one scheduler job.

    ``gpus_per_node`` and ``total_gpus`` are deliberately not interchangeable:
    the latter is derived from the requested node count.  Memory must be
    supplied either per node or as a total, never both.  Memory values may be
    integer megabytes or scheduler-compatible strings such as ``"64G"``.
    """

    nodes: int = 1
    gpus_per_node: int = 0
    total_gpus: int | None = None
    mpi_ranks: int = 1
    cpus_per_task: int | None = None
    memory_per_node: MemoryValue | None = None
    memory_total: MemoryValue | None = None
    walltime: str | timedelta | None = None
    partition: str | None = None
    account: str | None = None
    constraint: str | None = None
    extra_directives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError("nodes must be at least 1")
        if self.gpus_per_node < 0:
            raise ValueError("gpus_per_node must not be negative")
        expected_total_gpus = self.nodes * self.gpus_per_node
        if self.total_gpus is not None and self.total_gpus != expected_total_gpus:
            raise ValueError("total_gpus must equal nodes * gpus_per_node")
        object.__setattr__(self, "total_gpus", expected_total_gpus)
        if self.mpi_ranks < 1:
            raise ValueError("mpi_ranks must be at least 1")
        if self.cpus_per_task is not None and self.cpus_per_task < 1:
            raise ValueError("cpus_per_task must be at least 1 when supplied")
        if self.memory_per_node is not None and self.memory_total is not None:
            raise ValueError("memory_per_node and memory_total are mutually exclusive")
        _validate_memory(self.memory_per_node, "memory_per_node")
        _validate_memory(self.memory_total, "memory_total")
        _validate_walltime(self.walltime)
        for directive in self.extra_directives:
            if not directive.strip().startswith("#SBATCH"):
                raise ValueError("extra_directives must contain #SBATCH directives")

    @property
    def tasks(self) -> int:
        """Alias for MPI ranks at the scheduler boundary."""

        return self.mpi_ranks


def _validate_memory(value: MemoryValue | None, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{field_name} must be positive")
        return
    if not isinstance(value, str) or not value.strip() or not _MEMORY_PATTERN.fullmatch(value.strip()):
        raise ValueError(f"{field_name} must be a positive megabyte count or SLURM memory value")
    if float(_MEMORY_PATTERN.fullmatch(value.strip()).group("amount")) <= 0:  # type: ignore[union-attr]
        raise ValueError(f"{field_name} must be positive")


def _validate_walltime(value: str | timedelta | None) -> None:
    if value is None:
        return
    if isinstance(value, timedelta):
        if value.total_seconds() <= 0:
            raise ValueError("walltime must be positive")
        return
    if not isinstance(value, str) or not _WALLTIME_PATTERN.fullmatch(value.strip()):
        raise ValueError("walltime must use HH:MM:SS or D-HH:MM:SS format")


def _format_walltime(value: str | timedelta) -> str:
    if isinstance(value, str):
        return value.strip()
    total_seconds = int(value.total_seconds())
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}" if days else f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _memory_to_megabytes(value: MemoryValue) -> int:
    if isinstance(value, int):
        return value
    match = _MEMORY_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Invalid memory value: {value}")
    amount = float(match.group("amount"))
    unit = (match.group("unit") or "m").lower().rstrip("b")
    multiplier = {"k": 1 / 1024, "ki": 1 / 1024, "m": 1, "mi": 1, "g": 1024, "gi": 1024, "t": 1024 * 1024, "ti": 1024 * 1024}.get(unit)
    if multiplier is None:
        raise ValueError(f"Unsupported memory unit: {value}")
    return max(1, math.ceil(amount * multiplier))


def _render_memory(resources: JobResources) -> str | None:
    if resources.memory_per_node is not None:
        value = resources.memory_per_node
        return str(value) if isinstance(value, str) else f"{value}M"
    if resources.memory_total is None:
        return None
    total_mb = _memory_to_megabytes(resources.memory_total)
    per_node_mb = math.ceil(total_mb / resources.nodes)
    return f"{per_node_mb}M"


def render_sbatch_directives(
    resources: JobResources,
    *,
    job_name: str | None = None,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Render common resource directives without rendering a backend body."""

    directives: list[str] = []
    if job_name:
        directives.append(f"#SBATCH --job-name={job_name}")
    directives.append(f"#SBATCH --nodes={resources.nodes}")
    if resources.mpi_ranks:
        directives.append(f"#SBATCH --ntasks={resources.mpi_ranks}")
    if resources.cpus_per_task is not None:
        directives.append(f"#SBATCH --cpus-per-task={resources.cpus_per_task}")
    if resources.gpus_per_node:
        directives.append(f"#SBATCH --gpus-per-node={resources.gpus_per_node}")
    memory = _render_memory(resources)
    if memory is not None:
        directives.append(f"#SBATCH --mem={memory}")
    if resources.walltime is not None:
        directives.append(f"#SBATCH --time={_format_walltime(resources.walltime)}")
    if resources.partition:
        directives.append(f"#SBATCH --partition={resources.partition}")
    if resources.account:
        directives.append(f"#SBATCH --account={resources.account}")
    if resources.constraint:
        directives.append(f"#SBATCH --constraint={resources.constraint}")
    if stdout_path is not None:
        directives.append(f"#SBATCH --output={stdout_path}")
    if stderr_path is not None:
        directives.append(f"#SBATCH --error={stderr_path}")
    directives.extend(resources.extra_directives)
    return tuple(directives)


def _directive_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("#SBATCH"):
        return None
    return next(
        (
            key
            for key in _RESOURCE_DIRECTIVE_KEYS
            if re.search(rf"{re.escape(key)}(?:[=\s]|$)", stripped)
        ),
        None,
    )


def render_slurm_header(
    resources: JobResources,
    *,
    base_header: str = "",
    job_name: str | None = None,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    site_prologue: Iterable[str] = (),
) -> str:
    """Render common directives while preserving site header/prologue content.

    Resource directives in ``base_header`` are replaced by the typed values;
    unrelated ``#SBATCH`` directives, module loads, exports, and other site
    prologue lines remain in their original order.
    """

    base_lines = base_header.splitlines()
    shebang = next((line for line in base_lines if line.startswith("#!")), "#!/bin/bash")
    replacement_keys = {"--nodes", "--ntasks", "--ntasks-per-node"}
    if resources.cpus_per_task is not None:
        replacement_keys.add("--cpus-per-task")
    if resources.gpus_per_node:
        replacement_keys.update({"--gpus-per-node", "--gres"})
    if resources.memory_per_node is not None or resources.memory_total is not None:
        replacement_keys.update({"--mem", "--mem-per-cpu", "--mem-per-gpu"})
    if resources.walltime is not None:
        replacement_keys.add("--time")
    if job_name is not None:
        replacement_keys.add("--job-name")
    if stdout_path is not None:
        replacement_keys.add("--output")
    if stderr_path is not None:
        replacement_keys.add("--error")
    preserved = [
        line
        for line in base_lines
        if not line.startswith("#!") and _directive_key(line) not in replacement_keys
    ]
    prologue = [line.rstrip("\n") for line in site_prologue]
    lines = [shebang, *render_sbatch_directives(
        resources,
        job_name=job_name,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )]
    lines.extend(preserved)
    lines.extend(prologue)
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["JobResources", "render_sbatch_directives", "render_slurm_header"]
