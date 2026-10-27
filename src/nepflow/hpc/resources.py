"""Typed SLURM resources and shared HPC script/prologue rendering."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Sequence

from nepflow.errors import ConfigurationError

MemoryValue = int | str

SITE_ENVIRONMENT_PROLOGUE = Path("config/slurm/header.slurm")
DEFAULT_SITE_ENVIRONMENT_PROLOGUE = """#!/usr/bin/env bash
# Site environment setup for child HPC jobs.
# Scheduler resources belong in project.config; #SBATCH directives are not allowed here.
# Example:
# module purge
# module load <vasp-module>
# source <site-environment-script>
# export <runtime-variable>=<value>
"""

_WALLTIME_PATTERN = re.compile(
    r"^(?:(?P<days>\d+)-)?(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)$"
)
_MEMORY_PATTERN = re.compile(
    r"^(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>[kmgtpe]?i?b?)?$", re.IGNORECASE
)
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
        """Compatibility alias for the total MPI-rank/task count."""

        return self.mpi_ranks

    @property
    def total_mpi_ranks(self) -> int:
        """Return the total MPI ranks requested across all nodes."""

        return self.mpi_ranks

    @property
    def total_cpus(self) -> int | None:
        """Return total requested CPU slots when ``cpus_per_task`` is known."""

        if self.cpus_per_task is None:
            return None
        return self.mpi_ranks * self.cpus_per_task

    @property
    def walltime_seconds(self) -> int | None:
        """Return scheduler walltime as whole seconds, when configured."""

        if self.walltime is None:
            return None
        if isinstance(self.walltime, timedelta):
            return int(self.walltime.total_seconds())
        match = _WALLTIME_PATTERN.fullmatch(self.walltime.strip())
        if match is None:
            raise ValueError("walltime must use HH:MM:SS or D-HH:MM:SS format")
        days = int(match.group("days") or 0)
        return (
            days * 24 * 60 * 60
            + int(match.group("hours")) * 60 * 60
            + int(match.group("minutes")) * 60
            + int(match.group("seconds"))
        )


def _validate_memory(value: MemoryValue | None, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{field_name} must be positive")
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a positive megabyte count or SLURM memory value")
    match = _MEMORY_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"{field_name} must be a positive megabyte count or SLURM memory value")
    if float(match.group("amount")) <= 0:
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
    return (
        f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
        if days
        else f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    )


def _memory_to_megabytes(value: MemoryValue) -> int:
    """Convert a scheduler memory value to whole megabytes.

    Integer inputs are already MB.  Unit-bearing strings use binary
    scheduler units (1024 MB per GiB) and are rounded up so a request never
    falls below the configured amount.
    """

    if isinstance(value, int):
        return value
    match = _MEMORY_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Invalid memory value: {value}")
    amount = float(match.group("amount"))
    unit = (match.group("unit") or "m").lower().rstrip("b")
    multiplier = {
        "k": 1 / 1024,
        "ki": 1 / 1024,
        "m": 1,
        "mi": 1,
        "g": 1024,
        "gi": 1024,
        "t": 1024 * 1024,
        "ti": 1024 * 1024,
    }.get(unit)
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
    memory_directive = _render_memory(resources)
    if memory_directive is not None:
        directives.append(f"#SBATCH --mem={memory_directive}")
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


def normalize_site_environment_prologue(
    content: str,
    *,
    source: str | Path = SITE_ENVIRONMENT_PROLOGUE,
) -> tuple[str, ...]:
    """Validate and normalize a shell-only site environment prologue.

    Scheduler directives are intentionally forbidden here.  Typed project
    configuration and ``JobResources`` are the sole owners of scheduler and
    resource policy.  Shebangs are removed because final scripts render one
    canonical interpreter line.
    """

    if not isinstance(content, str):
        raise ConfigurationError(f"Site environment prologue is not text: {source}")
    normalized: list[str] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#!"):
            continue
        if stripped.startswith("#SBATCH"):
            directive = _directive_key(stripped)
            detail = directive or stripped.removeprefix("#SBATCH").strip() or "directive"
            raise ConfigurationError(
                f"Site environment prologue {source} contains forbidden scheduler "
                f"directive {detail!r} on line {line_number}; configure scheduler "
                "resources in project.config"
            )
        normalized.append(line.rstrip())
    while normalized and not normalized[0].strip():
        normalized.pop(0)
    while normalized and not normalized[-1].strip():
        normalized.pop()
    return tuple(normalized)


def load_site_environment_prologue(project_dir: str | Path) -> tuple[str, ...]:
    """Load the project's optional shell bootstrap from its canonical path.

    A missing file is an explicit empty-prologue policy.  Callers that require
    an executable unavailable in the current environment should pair this
    with :func:`require_resolvable_command_without_prologue` before submission.
    """

    path = Path(project_dir) / SITE_ENVIRONMENT_PROLOGUE
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except IsADirectoryError as exc:
        raise ConfigurationError(f"Site environment prologue is not a file: {path}") from exc
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"Could not read site environment prologue: {path}") from exc
    prologue = normalize_site_environment_prologue(content, source=path)
    bash = shutil.which("bash")
    if bash is not None:
        try:
            syntax = subprocess.run(
                (bash, "-n"),
                input=render_shell_script("", site_prologue=prologue),
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ConfigurationError(
                f"Could not validate site environment prologue syntax: {path}"
            ) from exc
        if syntax.returncode != 0:
            detail = (syntax.stderr or syntax.stdout or "invalid bash syntax").strip()
            raise ConfigurationError(
                f"Site environment prologue has invalid bash syntax: {path}: {detail}"
            )
    return prologue


def render_shell_script(
    body: str,
    *,
    site_prologue: Iterable[str] = (),
) -> str:
    """Render one canonical bash script around a validated site prologue."""

    prologue = normalize_site_environment_prologue(
        "\n".join(str(line).rstrip("\n") for line in site_prologue),
        source="site environment prologue",
    )
    body_lines = [line for line in str(body).splitlines() if not line.strip().startswith("#!")]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    lines = ["#!/usr/bin/env bash"]
    if prologue:
        lines.extend(("", *prologue))
    if body_lines:
        lines.extend(("", *body_lines))
    return "\n".join(lines).rstrip() + "\n"


def require_resolvable_command_without_prologue(
    command: Sequence[str | Path],
    site_prologue: Iterable[str],
    *,
    source: str | Path,
) -> None:
    """Fail before submission when an empty prologue cannot expose a command."""

    if any(line.strip() and not line.lstrip().startswith("#") for line in site_prologue):
        return
    values = [os.path.expanduser(os.path.expandvars(os.fspath(part))) for part in command]
    if not values:
        raise ConfigurationError("Configured HPC execution command is empty")
    launchers = {"mpirun", "mpiexec", "prterun", "srun"}
    first = Path(values[0]).name.lower()
    candidate = values[0]
    if first in launchers:
        candidates = [
            value
            for value in values[1:]
            if value
            and not value.startswith("-")
            and not value.isdigit()
            and "{ntasks}" not in value
        ]
        if candidates:
            candidate = candidates[-1]
    candidate_path = Path(candidate)
    available = (
        candidate_path.is_file() and os.access(candidate_path, os.X_OK)
        if any(separator in candidate for separator in ("/", "\\"))
        else shutil.which(candidate) is not None
    )
    if not available:
        raise ConfigurationError(
            f"Site environment prologue {source} is missing or contains no bootstrap "
            f"commands, and configured executable {candidate!r} is not resolvable; "
            f"configure {SITE_ENVIRONMENT_PROLOGUE} before submitting child jobs"
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
    prologue = normalize_site_environment_prologue(
        "\n".join(str(line).rstrip("\n") for line in site_prologue),
        source="site environment prologue",
    )
    lines = [
        *render_sbatch_directives(
            resources,
            job_name=job_name,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        ),
    ]
    lines.extend(preserved)
    lines.extend(prologue)
    rendered = render_shell_script("\n".join(lines), site_prologue=())
    if shebang != "#!/usr/bin/env bash":
        rendered = shebang + rendered[rendered.index("\n") :]
    return rendered


__all__ = [
    "DEFAULT_SITE_ENVIRONMENT_PROLOGUE",
    "JobResources",
    "SITE_ENVIRONMENT_PROLOGUE",
    "load_site_environment_prologue",
    "normalize_site_environment_prologue",
    "render_sbatch_directives",
    "render_shell_script",
    "render_slurm_header",
    "require_resolvable_command_without_prologue",
]
