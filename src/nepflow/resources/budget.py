"""Portable runtime memory and scratch budgets.

The values in this module are operational observations.  They must never be
included in scientific fingerprints or used to change a numerical algorithm.
The service deliberately fails closed when automatic discovery cannot establish
an allocation: an unknown memory limit is not treated as unlimited memory.
"""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

try:
    import resource as _resource
except ImportError:  # pragma: no cover - exercised on Windows
    _resource = None  # type: ignore[assignment]

DEFAULT_SAFETY_MARGIN_FRACTION = 0.10
_UNLIMITED = (1 << 60) - 1


def _optional_nonnegative(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer when provided")
    return int(value)


def _optional_positive(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer when provided")
    return int(value)


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Point-in-time resource observations for the current process."""

    host_available_memory_bytes: int | None = None
    effective_cgroup_memory_limit_bytes: int | None = None
    effective_cgroup_current_bytes: int | None = None
    rlimit_address_space_bytes: int | None = None
    rlimit_data_bytes: int | None = None
    scheduler_allocation_bytes: int | None = None
    effective_available_memory_bytes: int | None = None
    scratch_available_bytes: int | None = None
    scratch_path: str | None = None
    scratch_is_memory_backed: bool = False
    process_rss_bytes: int | None = None
    detection_sources: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = (
            "host_available_memory_bytes",
            "effective_cgroup_memory_limit_bytes",
            "effective_cgroup_current_bytes",
            "rlimit_address_space_bytes",
            "rlimit_data_bytes",
            "scheduler_allocation_bytes",
            "effective_available_memory_bytes",
            "scratch_available_bytes",
            "process_rss_bytes",
        )
        for name in names:
            object.__setattr__(self, name, _optional_nonnegative(getattr(self, name), name))
        if self.effective_cgroup_current_bytes is not None and (
            self.effective_cgroup_memory_limit_bytes is not None
            and self.effective_cgroup_current_bytes
            > self.effective_cgroup_memory_limit_bytes
        ):
            raise ValueError("cgroup current usage cannot exceed its effective limit")
        candidates: list[int] = []
        if self.host_available_memory_bytes is not None:
            candidates.append(self.host_available_memory_bytes)
        if self.cgroup_headroom_bytes is not None:
            candidates.append(self.cgroup_headroom_bytes)
        if self.scheduler_allocation_bytes is not None:
            candidates.append(self.scheduler_allocation_bytes)
        if self.rlimit_address_space_bytes is not None:
            candidates.append(
                max(0, self.rlimit_address_space_bytes - (self.process_rss_bytes or 0))
            )
        if self.rlimit_data_bytes is not None:
            candidates.append(max(0, self.rlimit_data_bytes - (self.process_rss_bytes or 0)))
        if candidates:
            detected = min(candidates)
            if self.effective_available_memory_bytes is None:
                object.__setattr__(self, "effective_available_memory_bytes", detected)
            else:
                object.__setattr__(
                    self,
                    "effective_available_memory_bytes",
                    min(self.effective_available_memory_bytes, detected),
                )
        if not isinstance(self.scratch_is_memory_backed, bool):
            raise ValueError("scratch_is_memory_backed must be boolean")
        object.__setattr__(self, "detection_sources", tuple(map(str, self.detection_sources)))
        object.__setattr__(self, "uncertainty", tuple(map(str, self.uncertainty)))

    @property
    def cgroup_headroom_bytes(self) -> int | None:
        if self.effective_cgroup_memory_limit_bytes is None:
            return None
        current = self.effective_cgroup_current_bytes or 0
        return max(0, self.effective_cgroup_memory_limit_bytes - current)


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Derived operational ceilings for one execution context."""

    execution_mode: str = "auto"
    memory_budget_bytes: int | None = None
    reserved_headroom_bytes: int = 0
    scratch_budget_bytes: int | None = None
    provenance_of_budget: str = "unknown"
    snapshot: ResourceSnapshot = field(default_factory=ResourceSnapshot)
    scratch_is_memory_backed: bool = False
    worker_cap: int | None = None

    def __post_init__(self) -> None:
        mode = str(self.execution_mode).strip().lower()
        if mode not in {"auto", "explicit"}:
            raise ValueError("execution_mode must be 'auto' or 'explicit'")
        object.__setattr__(self, "execution_mode", mode)
        object.__setattr__(
            self,
            "memory_budget_bytes",
            _optional_nonnegative(self.memory_budget_bytes, "memory_budget_bytes"),
        )
        reserve = _optional_nonnegative(
            self.reserved_headroom_bytes, "reserved_headroom_bytes"
        )
        assert reserve is not None
        object.__setattr__(self, "reserved_headroom_bytes", reserve)
        object.__setattr__(
            self,
            "scratch_budget_bytes",
            _optional_nonnegative(self.scratch_budget_bytes, "scratch_budget_bytes"),
        )
        if not isinstance(self.snapshot, ResourceSnapshot):
            raise TypeError("snapshot must be a ResourceSnapshot")
        object.__setattr__(self, "worker_cap", _optional_positive(self.worker_cap, "worker_cap"))

    @property
    def available_memory_bytes(self) -> int | None:
        """Maximum additional managed memory after reserved headroom."""

        if self.memory_budget_bytes is None:
            return None
        return max(0, self.memory_budget_bytes - self.reserved_headroom_bytes)


class ResourceCapacityError(MemoryError):
    """Typed fail-fast error for an operation that cannot fit its budget."""

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        requested_bytes: int | None = None,
        available_bytes: int | None = None,
        reserved_headroom_bytes: int | None = None,
        scratch_needed_bytes: int | None = None,
        scratch_available_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.requested_bytes = requested_bytes
        self.available_bytes = available_bytes
        self.reserved_headroom_bytes = reserved_headroom_bytes
        self.scratch_needed_bytes = scratch_needed_bytes
        self.scratch_available_bytes = scratch_available_bytes


class WorkspaceLease:
    """Idempotent reservation released by context-manager exit or ``close``."""

    def __init__(
        self,
        service: "ResourceBudgetService",
        operation: str,
        requested_peak_bytes: int,
        scratch_bytes: int,
        memory_charge_bytes: int | None = None,
    ) -> None:
        self._service = service
        self.operation = operation
        self.requested_peak_bytes = requested_peak_bytes
        self.granted_bytes = requested_peak_bytes
        self.scratch_bytes = scratch_bytes
        self._memory_charge_bytes = (
            requested_peak_bytes
            + (scratch_bytes if service.budget.scratch_is_memory_backed else 0)
            if memory_charge_bytes is None
            else memory_charge_bytes
        )
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._service._release(self)

    def resize(self, requested_peak_bytes: int, *, scratch_bytes: int | None = None) -> None:
        """Grow or shrink this operation's reservation at a work boundary."""

        self._service._resize(
            self,
            requested_peak_bytes,
            self.scratch_bytes if scratch_bytes is None else scratch_bytes,
        )

    def __enter__(self) -> "WorkspaceLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


ResourceReservation = WorkspaceLease


class ResourceBudgetService:
    """Thread-safe authority for bounded operation reservations."""

    def __init__(self, budget: ResourceBudget) -> None:
        self.budget = budget
        self._reserved_memory = 0
        self._reserved_scratch = 0
        self._leases: set[WorkspaceLease] = set()
        self._lock = threading.RLock()

    @property
    def reserved_memory_bytes(self) -> int:
        with self._lock:
            return self._reserved_memory

    @property
    def reserved_scratch_bytes(self) -> int:
        with self._lock:
            return self._reserved_scratch

    @property
    def remaining_scratch_bytes(self) -> int | None:
        with self._lock:
            if self.budget.scratch_budget_bytes is None:
                return None
            return max(0, self.budget.scratch_budget_bytes - self._reserved_scratch)

    @property
    def remaining_managed_budget(self) -> int | None:
        with self._lock:
            available = self.budget.available_memory_bytes
            if available is None:
                return None
            return max(0, available - self._reserved_memory)

    @property
    def remaining_memory_bytes(self) -> int | None:
        """Compatibility alias for the managed additional-memory headroom."""

        return self.remaining_managed_budget

    @property
    def active_operation_reservations(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(lease.operation for lease in self._leases if not lease.closed))

    def acquire(
        self,
        operation: str,
        requested_peak_bytes: int,
        *,
        scratch_bytes: int = 0,
    ) -> WorkspaceLease:
        requested = _optional_positive(requested_peak_bytes, "requested_peak_bytes")
        scratch = _optional_nonnegative(scratch_bytes, "scratch_bytes")
        assert requested is not None and scratch is not None
        with self._lock:
            memory_request = requested + (scratch if self.budget.scratch_is_memory_backed else 0)
            available = self.budget.available_memory_bytes
            if available is None:
                raise ResourceCapacityError(
                    "runtime memory availability is unknown; provide an explicit expert "
                    f"memory budget before starting {operation!r}",
                    operation=operation,
                    requested_bytes=memory_request,
                    available_bytes=None,
                    reserved_headroom_bytes=self.budget.reserved_headroom_bytes,
                    scratch_needed_bytes=scratch,
                    scratch_available_bytes=self.budget.scratch_budget_bytes,
                )
            if self._reserved_memory + memory_request > available:
                raise ResourceCapacityError(
                    f"{operation} exceeds the managed runtime memory budget: "
                    f"requested_bytes={memory_request}, "
                    f"remaining_bytes={max(0, available - self._reserved_memory)}, "
                    f"reserved_headroom_bytes={self.budget.reserved_headroom_bytes}",
                    operation=operation,
                    requested_bytes=memory_request,
                    available_bytes=max(0, available - self._reserved_memory),
                    reserved_headroom_bytes=self.budget.reserved_headroom_bytes,
                    scratch_needed_bytes=scratch,
                    scratch_available_bytes=self.budget.scratch_budget_bytes,
                )
            if (
                scratch > 0 and self.budget.scratch_budget_bytes is None
            ):
                raise ResourceCapacityError(
                    f"{operation} requires scratch capacity, but scratch availability is unknown",
                    operation=operation,
                    requested_bytes=memory_request,
                    available_bytes=max(0, available - self._reserved_memory),
                    reserved_headroom_bytes=self.budget.reserved_headroom_bytes,
                    scratch_needed_bytes=scratch,
                    scratch_available_bytes=None,
                )
            if (
                self.budget.scratch_budget_bytes is not None
                and self._reserved_scratch + scratch > self.budget.scratch_budget_bytes
            ):
                raise ResourceCapacityError(
                    f"{operation} exceeds the managed scratch budget: "
                    f"requested_bytes={scratch}, "
                    "remaining_bytes="
                    f"{max(0, self.budget.scratch_budget_bytes - self._reserved_scratch)}",
                    operation=operation,
                    requested_bytes=memory_request,
                    available_bytes=max(0, available - self._reserved_memory),
                    reserved_headroom_bytes=self.budget.reserved_headroom_bytes,
                    scratch_needed_bytes=scratch,
                    scratch_available_bytes=self.budget.scratch_budget_bytes,
                )
            lease = WorkspaceLease(
                self,
                str(operation),
                requested,
                scratch,
                memory_charge_bytes=memory_request,
            )
            self._reserved_memory += memory_request
            self._reserved_scratch += scratch
            self._leases.add(lease)
            return lease

    reserve = acquire

    def _release(self, lease: WorkspaceLease) -> None:
        with self._lock:
            if lease in self._leases:
                self._leases.remove(lease)
                self._reserved_memory -= lease._memory_charge_bytes
                self._reserved_scratch -= lease.scratch_bytes

    def _resize(self, lease: WorkspaceLease, requested_peak_bytes: int, scratch_bytes: int) -> None:
        requested = _optional_positive(requested_peak_bytes, "requested_peak_bytes")
        scratch = _optional_nonnegative(scratch_bytes, "scratch_bytes")
        assert requested is not None and scratch is not None
        with self._lock:
            if lease.closed or lease not in self._leases:
                raise RuntimeError("cannot resize a closed workspace lease")
            old_memory = lease._memory_charge_bytes
            old_scratch = lease.scratch_bytes
            new_memory = requested + (scratch if self.budget.scratch_is_memory_backed else 0)
            available = self.budget.available_memory_bytes
            if available is None:
                raise ResourceCapacityError(
                    "runtime memory availability is unknown while resizing "
                    f"{lease.operation!r}",
                    operation=lease.operation,
                    requested_bytes=new_memory,
                    reserved_headroom_bytes=self.budget.reserved_headroom_bytes,
                )
            if self._reserved_memory - old_memory + new_memory > available:
                raise ResourceCapacityError(
                    f"{lease.operation} exceeds the managed runtime memory budget while "
                    f"resizing: requested_bytes={new_memory}, "
                    f"remaining_bytes={max(0, available - (self._reserved_memory - old_memory))}",
                    operation=lease.operation,
                    requested_bytes=new_memory,
                    available_bytes=max(0, available - (self._reserved_memory - old_memory)),
                    reserved_headroom_bytes=self.budget.reserved_headroom_bytes,
                    scratch_needed_bytes=scratch,
                    scratch_available_bytes=self.budget.scratch_budget_bytes,
                )
            if (
                scratch > 0 and self.budget.scratch_budget_bytes is None
            ):
                raise ResourceCapacityError(
                    f"{lease.operation} requires scratch capacity, but scratch "
                    "availability is unknown",
                    operation=lease.operation,
                    requested_bytes=new_memory,
                    available_bytes=max(0, available - (self._reserved_memory - old_memory)),
                    reserved_headroom_bytes=self.budget.reserved_headroom_bytes,
                    scratch_needed_bytes=scratch,
                    scratch_available_bytes=None,
                )
            if (
                self.budget.scratch_budget_bytes is not None
                and self._reserved_scratch - old_scratch + scratch
                > self.budget.scratch_budget_bytes
            ):
                raise ResourceCapacityError(
                    f"{lease.operation} exceeds the managed scratch budget while resizing",
                    operation=lease.operation,
                    requested_bytes=new_memory,
                    available_bytes=max(0, available - (self._reserved_memory - old_memory)),
                    reserved_headroom_bytes=self.budget.reserved_headroom_bytes,
                    scratch_needed_bytes=scratch,
                    scratch_available_bytes=self.budget.scratch_budget_bytes,
                )
            self._reserved_memory += new_memory - old_memory
            self._reserved_scratch += scratch - old_scratch
            lease.requested_peak_bytes = requested
            lease.granted_bytes = requested
            lease.scratch_bytes = scratch
            lease._memory_charge_bytes = new_memory


def _parse_meminfo(text: str) -> int | None:
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() != "MemAvailable":
            continue
        fields = value.split()
        if not fields:
            return None
        try:
            amount = int(fields[0])
        except (TypeError, ValueError, OverflowError):
            return None
        unit = fields[1].lower() if len(fields) > 1 else "b"
        multiplier = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}.get(unit)
        if multiplier is None or amount < 0:
            return None
        return amount * multiplier
    return None


def _parse_size(value: str, *, default_unit: str = "b") -> int | None:
    token = value.strip().lower()
    suffixes = (
        ("kib", 1024),
        ("kb", 1024),
        ("k", 1024),
        ("mib", 1024**2),
        ("mb", 1024**2),
        ("m", 1024**2),
        ("gib", 1024**3),
        ("gb", 1024**3),
        ("g", 1024**3),
        ("tib", 1024**4),
        ("tb", 1024**4),
        ("t", 1024**4),
    )
    multiplier = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}.get(
        default_unit.lower(), 1
    )
    for suffix, candidate_multiplier in suffixes:
        if token.endswith(suffix):
            token = token[: -len(suffix)].strip()
            multiplier = candidate_multiplier
            break
    try:
        number = float(token)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return int(number * multiplier)


def _safe_read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _unescape_mountinfo(value: str) -> str:
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")


def _linux_cgroup_paths(
    proc_root: Path, cgroup_root: Path
) -> list[tuple[Path, Path, str]]:
    membership = _safe_read(proc_root / "self" / "cgroup") or ""
    mountinfo = _safe_read(proc_root / "self" / "mountinfo") or ""
    paths: list[tuple[Path, Path, str]] = []
    for line in membership.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controllers, relative = fields[1], fields[2].lstrip("/")
        wanted = "v2" if not controllers else "memory"
        for mount_line in mountinfo.splitlines():
            left, separator, right = mount_line.partition(" - ")
            if not separator:
                continue
            left_fields = left.split()
            right_fields = right.split()
            if len(left_fields) < 5 or len(right_fields) < 3:
                continue
            filesystem = right_fields[0]
            mount_options = set(left_fields[5].split(",")) if len(left_fields) > 5 else set()
            mount_options.update(right_fields[2].split(","))
            if filesystem != ("cgroup2" if wanted == "v2" else "cgroup"):
                continue
            if wanted == "memory" and "memory" not in mount_options:
                continue
            mount_point = Path(_unescape_mountinfo(left_fields[4]))
            path = mount_point / relative
            paths.append((path, mount_point, filesystem))
            break
    if not paths and membership and not mountinfo:
        for line in membership.splitlines():
            fields = line.split(":", 2)
            if len(fields) == 3 and (fields[1] == "" or "memory" in fields[1].split(",")):
                paths.append(
                    (
                        cgroup_root / fields[2].lstrip("/"),
                        cgroup_root,
                        "cgroup2" if not fields[1] else "cgroup",
                    )
                )
    return paths


def _ancestor_values(path: Path, mount_root: Path, names: tuple[str, ...]) -> list[int]:
    values: list[int] = []
    current = path
    try:
        while True:
            for name in names:
                text = _safe_read(current / name)
                if text is None:
                    continue
                token = text.strip()
                if token == "max":
                    continue
                try:
                    value = int(token)
                except ValueError:
                    continue
                if 0 <= value < _UNLIMITED:
                    values.append(value)
                    break
            if current == mount_root or current.parent == current:
                break
            current = current.parent
    except OSError:
        pass
    return values


def _ancestor_memory_metrics(
    path: Path,
    mount_root: Path,
    limit_name: str,
    current_name: str,
) -> list[tuple[int, int | None]]:
    metrics: list[tuple[int, int | None]] = []
    current = path
    try:
        while True:
            limit_text = _safe_read(current / limit_name)
            if limit_text is not None:
                token = limit_text.strip()
                if token != "max":
                    try:
                        limit = int(token)
                    except ValueError:
                        limit = -1
                    if 0 <= limit < _UNLIMITED:
                        current_text = _safe_read(current / current_name)
                        current_value: int | None = None
                        if current_text is not None:
                            try:
                                parsed = int(current_text.strip())
                            except ValueError:
                                parsed = -1
                            if parsed >= 0:
                                current_value = parsed
                        metrics.append((limit, current_value))
            if current == mount_root or current.parent == current:
                break
            current = current.parent
    except OSError:
        pass
    return metrics


def _detect_linux_cgroup(
    proc_root: Path, cgroup_root: Path
) -> tuple[int | None, int | None, list[str], list[str]]:
    metrics: list[tuple[int, int | None]] = []
    sources: list[str] = []
    uncertainty: list[str] = []
    paths = _linux_cgroup_paths(proc_root, cgroup_root)
    for path, mount_point, filesystem in paths:
        if filesystem == "cgroup2":
            limit_names = ("memory.max",)
            current_names = ("memory.current",)
        else:
            limit_names = ("memory.limit_in_bytes",)
            current_names = ("memory.usage_in_bytes",)
        path_metrics = _ancestor_memory_metrics(
            path,
            mount_point,
            limit_names[0],
            current_names[0],
        )
        metrics.extend(path_metrics)
        if path_metrics:
            sources.append(f"{filesystem}:{path}")
    if not metrics:
        if paths:
            uncertainty.append("cgroup memory metrics are unavailable")
        return None, None, sources, uncertainty
    effective_limit = min(limit for limit, _ in metrics)
    headrooms: list[int] = []
    for limit, current in metrics:
        if current is None:
            headrooms.append(limit)
        elif current > limit:
            uncertainty.append("cgroup usage exceeded a discovered limit")
            headrooms.append(0)
        else:
            headrooms.append(limit - current)
    effective_headroom_index = min(range(len(headrooms)), key=headrooms.__getitem__)
    effective_current = metrics[effective_headroom_index][1]
    if effective_current is not None and effective_current > effective_limit:
        uncertainty.append("nested cgroup metrics were inconsistent")
        effective_current = effective_limit
    return effective_limit, effective_current, sources, uncertainty


def _detect_rlimits(proc_root: Path) -> tuple[int | None, int | None, int | None, list[str]]:
    sources: list[str] = []
    uncertainty: list[str] = []
    rss: int | None = None
    status = _safe_read(proc_root / "self" / "status") or ""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            try:
                rss = int(line.split()[1]) * 1024
            except (IndexError, ValueError):
                pass
    as_limit: int | None = None
    data_limit: int | None = None
    if os.name != "nt" and _resource is not None:
        try:
            as_raw, _ = _resource.getrlimit(_resource.RLIMIT_AS)
            if as_raw not in (-1, _resource.RLIM_INFINITY):
                as_limit = int(as_raw)
                sources.append("rlimit_as")
        except (AttributeError, OSError, ValueError):
            uncertainty.append("RLIMIT_AS unavailable")
        try:
            data_raw, _ = _resource.getrlimit(_resource.RLIMIT_DATA)
            if data_raw not in (-1, _resource.RLIM_INFINITY):
                data_limit = int(data_raw)
                sources.append("rlimit_data")
        except (AttributeError, OSError, ValueError):
            uncertainty.append("RLIMIT_DATA unavailable")
    return as_limit, data_limit, rss, sources + uncertainty


def _windows_available_memory() -> int | None:
    class _MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    try:
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        windll = getattr(ctypes, "windll", None)
        kernel32 = getattr(windll, "kernel32", None)
        if kernel32 is None or kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) == 0:
            return None
        return int(status.ullAvailPhys)
    except (AttributeError, OSError, TypeError):
        return None


def _scratch_details(path: Path, proc_root: Path) -> tuple[int | None, bool, list[str]]:
    try:
        available = int(shutil.disk_usage(path).free)
    except (OSError, ValueError):
        return None, False, [f"scratch unavailable: {path}"]
    memory_backed = False
    mounts = _safe_read(proc_root / "self" / "mountinfo") or ""
    best_length = -1
    for line in mounts.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        right_fields = right.split()
        if len(fields) < 5 or not right_fields:
            continue
        mount_point = Path(_unescape_mountinfo(fields[4]))
        try:
            path.relative_to(mount_point)
        except ValueError:
            continue
        if len(str(mount_point)) > best_length:
            best_length = len(str(mount_point))
            memory_backed = right_fields[0] in {"tmpfs", "ramfs"}
    return available, memory_backed, [f"scratch:{path}"]


def detect_resource_snapshot(
    *,
    scratch_path: str | Path | None = None,
    proc_root: str | Path = "/proc",
    cgroup_root: str | Path = "/sys/fs/cgroup",
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ResourceSnapshot:
    """Detect process-local memory headroom and scratch capacity."""

    platform_value = platform_name or sys.platform
    proc = Path(proc_root)
    cgroup = Path(cgroup_root)
    sources: list[str] = []
    uncertainty: list[str] = []
    host_available: int | None = None
    cgroup_limit: int | None = None
    cgroup_current: int | None = None
    rlimit_as: int | None = None
    rlimit_data: int | None = None
    rss: int | None = None

    if platform_value.startswith("win"):
        host_available = _windows_available_memory()
        if host_available is not None:
            sources.append("GlobalMemoryStatusEx")
        else:
            uncertainty.append("Windows available memory is unavailable")
    elif platform_value.startswith("linux"):
        meminfo = _safe_read(proc / "meminfo")
        if meminfo is not None:
            try:
                host_available = _parse_meminfo(meminfo)
            except (TypeError, ValueError, OverflowError):
                uncertainty.append("MemAvailable is invalid")
            if host_available is not None:
                sources.append("/proc/meminfo:MemAvailable")
        else:
            uncertainty.append("/proc/meminfo is unavailable")
        cgroup_limit, cgroup_current, cg_sources, cg_uncertainty = _detect_linux_cgroup(
            proc, cgroup
        )
        sources.extend(cg_sources)
        uncertainty.extend(cg_uncertainty)
    elif platform_value == "darwin":
        uncertainty.append("macOS available-memory probe is unavailable")
    else:
        uncertainty.append(f"unsupported platform: {platform_value}")

    if platform_value.startswith("linux") or platform_value == "darwin":
        rlimit_as, rlimit_data, rss, rlimit_notes = _detect_rlimits(proc)
        for note in rlimit_notes:
            (sources if note.startswith("rlimit_") else uncertainty).append(note)

    env = os.environ if environ is None else environ
    scheduler_bytes: int | None = None
    if env.get("SLURM_JOB_ID"):
        scheduler_name = "SLURM_MEM_PER_NODE"
        raw = env.get(scheduler_name)
        if raw is None:
            scheduler_name = "SLURM_MEM_PER_CPU"
            raw = env.get(scheduler_name)
        if raw:
            try:
                scheduler_bytes = _parse_size(raw, default_unit="mb")
                if scheduler_bytes is None:
                    raise ValueError(raw)
                if scheduler_bytes == 0:
                    scheduler_bytes = None
                    uncertainty.append(f"{scheduler_name} reports no finite limit")
                else:
                    sources.append(scheduler_name)
            except (ValueError, OverflowError):
                uncertainty.append("SLURM memory allocation is invalid")

    scratch = Path(scratch_path or env.get("TMPDIR") or tempfile.gettempdir())
    scratch_available, scratch_memory_backed, scratch_sources = _scratch_details(scratch, proc)
    sources.extend(scratch_sources)

    candidates: list[int] = []
    if host_available is not None:
        candidates.append(host_available)
    cgroup_headroom = None
    if cgroup_limit is not None:
        cgroup_headroom = max(0, cgroup_limit - (cgroup_current or 0))
        candidates.append(cgroup_headroom)
    if scheduler_bytes is not None:
        candidates.append(scheduler_bytes)
    if rlimit_as is not None:
        candidates.append(max(0, rlimit_as - (rss or 0)))
    if rlimit_data is not None:
        candidates.append(max(0, rlimit_data - (rss or 0)))
    effective = min(candidates) if candidates else None
    return ResourceSnapshot(
        host_available_memory_bytes=host_available,
        effective_cgroup_memory_limit_bytes=cgroup_limit,
        effective_cgroup_current_bytes=cgroup_current,
        rlimit_address_space_bytes=rlimit_as,
        rlimit_data_bytes=rlimit_data,
        scheduler_allocation_bytes=scheduler_bytes,
        effective_available_memory_bytes=effective,
        scratch_available_bytes=scratch_available,
        scratch_path=str(scratch),
        scratch_is_memory_backed=scratch_memory_backed,
        process_rss_bytes=rss,
        detection_sources=tuple(sources),
        uncertainty=tuple(uncertainty),
    )


def build_resource_budget(
    snapshot: ResourceSnapshot | None = None,
    *,
    execution_mode: str = "auto",
    memory_budget_bytes: int | None = None,
    scratch_budget_bytes: int | None = None,
    reserved_headroom_bytes: int | None = None,
    safety_margin_fraction: float = DEFAULT_SAFETY_MARGIN_FRACTION,
    scratch_path: str | Path | None = None,
    worker_cap: int | None = None,
) -> ResourceBudgetService:
    """Create one shared service from a snapshot and optional expert ceilings."""

    current = snapshot or detect_resource_snapshot(scratch_path=scratch_path)
    normalized_mode = str(execution_mode).strip().lower()
    if normalized_mode == "explicit" and memory_budget_bytes is None:
        raise ValueError("explicit execution mode requires memory_budget_bytes")
    if not math.isfinite(float(safety_margin_fraction)) or not 0.0 <= safety_margin_fraction < 1.0:
        raise ValueError("safety_margin_fraction must be finite and in [0, 1)")
    detected = current.effective_available_memory_bytes
    if normalized_mode == "auto" and any(
        note == "cgroup memory metrics are unavailable" for note in current.uncertainty
    ):
        detected = None
    if memory_budget_bytes is not None:
        memory_budget_bytes = _optional_positive(memory_budget_bytes, "memory_budget_bytes")
        detected = memory_budget_bytes if detected is None else min(detected, memory_budget_bytes)
    if reserved_headroom_bytes is None:
        reserve = (
            math.ceil(detected * safety_margin_fraction) if detected is not None else 0
        )
    else:
        reserve = _optional_nonnegative(reserved_headroom_bytes, "reserved_headroom_bytes")
        assert reserve is not None
    scratch = current.scratch_available_bytes
    if scratch_budget_bytes is not None:
        scratch_budget_bytes = _optional_positive(scratch_budget_bytes, "scratch_budget_bytes")
        scratch = scratch_budget_bytes if scratch is None else min(scratch, scratch_budget_bytes)
    budget = ResourceBudget(
        execution_mode=execution_mode,
        memory_budget_bytes=detected,
        reserved_headroom_bytes=reserve,
        scratch_budget_bytes=scratch,
        provenance_of_budget=";".join(current.detection_sources) or "unknown",
        snapshot=current,
        scratch_is_memory_backed=current.scratch_is_memory_backed,
        worker_cap=worker_cap,
    )
    return ResourceBudgetService(budget)


__all__ = [
    "DEFAULT_SAFETY_MARGIN_FRACTION",
    "ResourceBudget",
    "ResourceBudgetService",
    "ResourceCapacityError",
    "ResourceReservation",
    "ResourceSnapshot",
    "WorkspaceLease",
    "build_resource_budget",
    "detect_resource_snapshot",
]
