"""Process-local CPU allocation discovery.

The worker pools in NEPFlow must follow the CPUs available to the current
process, not the number of CPUs in the machine or on a SLURM node. This
module keeps that policy in one place for generation and selection.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_CGROUP = Path("/proc/self/cgroup")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _parse_positive_integer(value: str) -> int | None:
    token = value.strip().split(",", 1)[0].split("(", 1)[0].strip()
    try:
        parsed = int(token)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_cpu_set(value: str) -> int | None:
    """Return the number of CPUs represented by a Linux cpuset string."""

    text = value.strip()
    if not text:
        return None
    total = 0
    for item in text.split(","):
        token = item.strip()
        if not token:
            return None
        if "-" in token:
            first, last = token.split("-", 1)
            try:
                start = int(first)
                stop = int(last)
            except ValueError:
                return None
            if start < 0 or stop < start:
                return None
            total += stop - start + 1
        else:
            try:
                cpu = int(token)
            except ValueError:
                return None
            if cpu < 0:
                return None
            total += 1
    return total if total > 0 else None


def _cgroup_memberships() -> list[tuple[str, str, frozenset[str]]]:
    """Return cgroup version, relative path, and controllers for this process."""

    memberships: list[tuple[str, str, frozenset[str]]] = []
    text = _read_text(_PROC_CGROUP)
    if text is None:
        return memberships
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers_text, relative_path = fields
        if hierarchy == "0" and not controllers_text:
            memberships.append(("v2", relative_path or "/", frozenset()))
        else:
            memberships.append(
                ("v1", relative_path or "/", frozenset(controllers_text.split(",")))
            )
    return memberships


def _cgroup_mounts() -> list[tuple[str, str, Path, frozenset[str]]]:
    """Return mounted cgroup roots visible from this process."""

    mounts: list[tuple[str, str, Path, frozenset[str]]] = []
    mountinfo = _read_text(_PROC_CGROUP.parent / "mountinfo")
    if mountinfo is None:
        return mounts
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            mount_root = fields[3]
            mount_point = Path(fields[4])
            filesystem = fields[separator + 1]
            super_options = frozenset(fields[separator + 3].split(","))
        except (IndexError, ValueError):
            continue
        if filesystem == "cgroup2":
            mounts.append(("v2", mount_root, mount_point, frozenset()))
        elif filesystem == "cgroup":
            mounts.append(("v1", mount_root, mount_point, super_options))
    return mounts


def _relative_cgroup_path(relative_path: str, mount_root: str) -> str | None:
    relative = "/" + relative_path.strip("/")
    root = "/" + mount_root.strip("/")
    if root == "/":
        return relative
    if relative == root:
        return "/"
    if relative.startswith(root + "/"):
        return relative[len(root) :]
    return None


def _ancestor_paths(root: Path, relative_path: str) -> list[Path]:
    parts = tuple(part for part in relative_path.strip("/").split("/") if part)
    return [root.joinpath(*parts[:index]) for index in range(len(parts), -1, -1)]


def _cgroup_directories(controller: str) -> list[Path]:
    """Return this process's cgroup directory and its constrained ancestors."""

    directories: list[Path] = []
    memberships = _cgroup_memberships()
    mounts = _cgroup_mounts()
    for version, relative_path, controllers in memberships:
        if version == "v2":
            if controller in {"cpu", "cpuset"}:
                mounted = False
                for mount_version, mount_root, mount_point, _mount_controllers in mounts:
                    if mount_version != "v2":
                        continue
                    mapped = _relative_cgroup_path(relative_path, mount_root)
                    if mapped is not None:
                        directories.extend(_ancestor_paths(mount_point, mapped))
                        mounted = True
                if not mounted:
                    directories.extend(_ancestor_paths(_CGROUP_ROOT, relative_path))
            continue
        if controller == "cpu" and not ({"cpu", "cpuacct"} & controllers):
            continue
        if controller == "cpuset" and "cpuset" not in controllers:
            continue
        mounted = False
        for mount_version, mount_root, mount_point, mount_controllers in mounts:
            if mount_version != "v1" or controller not in mount_controllers:
                continue
            mapped = _relative_cgroup_path(relative_path, mount_root)
            if mapped is not None:
                directories.extend(_ancestor_paths(mount_point, mapped))
                mounted = True
        if not mounted:
            roots = (_CGROUP_ROOT / controller,)
            if controller == "cpu":
                roots += (_CGROUP_ROOT / "cpu,cpuacct",)
            for root in roots:
                directories.extend(_ancestor_paths(root, relative_path))

    if not memberships:
        directories.append(_CGROUP_ROOT)
        directories.append(_CGROUP_ROOT / controller)
        if controller == "cpu":
            directories.append(_CGROUP_ROOT / "cpu,cpuacct")

    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        if directory not in seen:
            unique.append(directory)
            seen.add(directory)
    return unique


def _cgroup_cpu_quota() -> int | None:
    limits: list[int] = []
    for directory in _cgroup_directories("cpu"):
        v2 = _read_text(directory / "cpu.max")
        if v2 is not None:
            fields = v2.split()
            if len(fields) >= 2 and fields[0] != "max":
                quota = _parse_positive_integer(fields[0])
                period = _parse_positive_integer(fields[1])
                if quota is not None and period is not None:
                    limits.append(max(1, math.floor(quota / period)))

        quota = _parse_positive_integer(_read_text(directory / "cpu.cfs_quota_us") or "")
        period = _parse_positive_integer(_read_text(directory / "cpu.cfs_period_us") or "")
        if quota is not None and period is not None:
            limits.append(max(1, math.floor(quota / period)))
    return min(limits) if limits else None


def _cgroup_cpuset_limit() -> int | None:
    limits: list[int] = []
    for directory in _cgroup_directories("cpuset"):
        for filename in ("cpuset.cpus.effective", "cpuset.cpus"):
            value = _read_text(directory / filename)
            if value is None:
                continue
            parsed = _parse_cpu_set(value)
            if parsed is not None:
                limits.append(parsed)
    return min(limits) if limits else None


def effective_cpu_allocation() -> int:
    """Return the smallest CPU allocation visible to the current process.

    The result combines host CPU count, process affinity, cgroup CPU quota and
    cpuset limits, and the per-task SLURM allocation. Node-wide SLURM values
    such as ``SLURM_CPUS_ON_NODE`` are intentionally ignored: a GPU request or
    a node allocation does not grant all of those CPUs to this process.
    """

    try:
        host_count = max(1, int(os.cpu_count() or 1))
    except (TypeError, ValueError):
        host_count = 1
    limits = [host_count]

    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            affinity_count = len(affinity(0))
        except (OSError, TypeError):
            affinity_count = 0
        if affinity_count > 0:
            limits.append(affinity_count)

    slurm_task_cpus = _parse_positive_integer(os.environ.get("SLURM_CPUS_PER_TASK", ""))
    if slurm_task_cpus is not None:
        limits.append(slurm_task_cpus)

    for cgroup_limit in (_cgroup_cpu_quota(), _cgroup_cpuset_limit()):
        if cgroup_limit is not None and cgroup_limit > 0:
            limits.append(cgroup_limit)
    return max(1, min(limits))


__all__ = ["effective_cpu_allocation"]
