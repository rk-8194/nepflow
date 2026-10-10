"""Deterministic resource-probe regressions for Linux container runtimes."""

from pathlib import Path

import pytest

from nepflow.resources.budget import (
    ResourceCapacityError,
    ResourceSnapshot,
    build_resource_budget,
    detect_resource_snapshot,
)


GIB = 1024**3


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    membership: str = "0::/",
    mount_root: str = "/",
    limit: str | None = "max",
    current: str | None = "0",
    host_available_gib: int = 8,
) -> tuple[Path, Path, Path]:
    proc = tmp_path / "proc"
    mount_point = tmp_path / "cgroup"
    relative = membership.split(":", 2)[2].lstrip("/")
    if mount_root == "/":
        mapped = relative
    else:
        prefix = mount_root.strip("/")
        assert relative == prefix or relative.startswith(prefix + "/")
        mapped = relative[len(prefix) :].lstrip("/")
    cgroup_path = mount_point / mapped
    _write(proc / "meminfo", f"MemAvailable: {host_available_gib * 1024 * 1024} kB\n")
    _write(proc / "self" / "cgroup", membership + "\n")
    _write(
        proc / "self" / "mountinfo",
        f"29 23 0:26 {mount_root} {mount_point} rw,nosuid,nodev - cgroup2 cgroup rw\n",
    )
    _write(proc / "self" / "status", "Name:\tfixture\n")
    if limit is not None:
        _write(cgroup_path / "memory.max", limit + "\n")
    if current is not None:
        _write(cgroup_path / "memory.current", current + "\n")
    return proc, tmp_path / "scratch", mount_point


def test_wsl_style_unlimited_cgroup_uses_host_memavailable(tmp_path: Path) -> None:
    proc, scratch, _ = _fixture(tmp_path, limit="max", current="1048576")

    snapshot = detect_resource_snapshot(
        proc_root=proc,
        cgroup_root=tmp_path / "unused-cgroup-root",
        scratch_path=scratch,
        platform_name="linux",
        environ={},
    )
    service = build_resource_budget(snapshot, safety_margin_fraction=0.10)

    assert snapshot.cgroup_status == "unlimited"
    assert snapshot.host_available_memory_bytes == 8 * GIB
    assert service.budget.memory_budget_bytes == 8 * GIB
    assert service.remaining_managed_budget == int(8 * GIB * 0.90)


def test_finite_nested_cgroup_uses_tightest_headroom(tmp_path: Path) -> None:
    proc, scratch, mount_point = _fixture(
        tmp_path,
        membership="0::/slice/job",
        mount_root="/slice",
        limit=str(4 * GIB),
        current=str(3 * GIB),
        host_available_gib=16,
    )
    _write(mount_point / "memory.max", str(16 * GIB) + "\n")
    _write(mount_point / "memory.current", str(5 * GIB) + "\n")

    snapshot = detect_resource_snapshot(
        proc_root=proc,
        scratch_path=scratch,
        platform_name="linux",
        environ={},
    )
    service = build_resource_budget(snapshot, safety_margin_fraction=0.0)

    assert snapshot.cgroup_status == "finite"
    assert snapshot.cgroup_headroom_bytes == GIB
    assert service.budget.memory_budget_bytes == GIB


def test_unlimited_child_still_honours_finite_parent(tmp_path: Path) -> None:
    proc, scratch, mount_point = _fixture(
        tmp_path,
        membership="0::/slice/job",
        mount_root="/slice",
        limit="max",
        current="0",
        host_available_gib=16,
    )
    _write(mount_point / "memory.max", str(6 * GIB) + "\n")
    _write(mount_point / "memory.current", str(2 * GIB) + "\n")

    snapshot = detect_resource_snapshot(
        proc_root=proc,
        scratch_path=scratch,
        platform_name="linux",
        environ={},
    )

    assert snapshot.cgroup_status == "finite"
    assert snapshot.cgroup_headroom_bytes == 4 * GIB


def test_absent_unlimited_and_unreadable_cgroups_are_distinct(tmp_path: Path) -> None:
    absent_proc = tmp_path / "absent-proc"
    _write(absent_proc / "meminfo", "MemAvailable: 8388608 kB\n")
    _write(absent_proc / "self" / "cgroup", "\n")
    _write(absent_proc / "self" / "mountinfo", "\n")
    absent = detect_resource_snapshot(
        proc_root=absent_proc,
        scratch_path=tmp_path / "scratch-absent",
        platform_name="linux",
        environ={},
    )

    unlimited_proc, unlimited_scratch, _ = _fixture(tmp_path / "unlimited")
    unlimited = detect_resource_snapshot(
        proc_root=unlimited_proc,
        scratch_path=unlimited_scratch,
        platform_name="linux",
        environ={},
    )

    unreadable_proc, unreadable_scratch, _ = _fixture(
        tmp_path / "unreadable",
        limit=str(4 * GIB),
        current=None,
    )
    unreadable = detect_resource_snapshot(
        proc_root=unreadable_proc,
        scratch_path=unreadable_scratch,
        platform_name="linux",
        environ={},
    )

    assert absent.cgroup_status == "absent"
    assert unlimited.cgroup_status == "unlimited"
    assert unreadable.cgroup_status == "unknown"
    assert unreadable.effective_cgroup_current_bytes is None
    assert "unreadable" in " ".join(unreadable.uncertainty)


def test_independent_bounds_survive_known_unlimited_or_unknown_cgroup() -> None:
    unlimited = ResourceSnapshot(
        host_available_memory_bytes=8 * GIB,
        cgroup_status="unlimited",
        scheduler_allocation_bytes=2 * GIB,
    )
    unknown = ResourceSnapshot(
        host_available_memory_bytes=8 * GIB,
        cgroup_status="unknown",
        cgroup_paths=("/sys/fs/cgroup/job",),
        scheduler_allocation_bytes=2 * GIB,
    )
    no_independent_bound = ResourceSnapshot(
        host_available_memory_bytes=8 * GIB,
        cgroup_status="unknown",
        cgroup_paths=("/sys/fs/cgroup/job",),
    )

    assert (
        build_resource_budget(unlimited, safety_margin_fraction=0.0).budget.memory_budget_bytes
        == 2 * GIB
    )
    assert (
        build_resource_budget(unknown, safety_margin_fraction=0.0).budget.memory_budget_bytes
        == 2 * GIB
    )
    assert build_resource_budget(no_independent_bound).budget.memory_budget_bytes is None


def test_windows_portable_mode_requires_explicit_memory_when_unavailable() -> None:
    snapshot = ResourceSnapshot(cgroup_status="absent")

    assert build_resource_budget(snapshot).budget.memory_budget_bytes is None
    explicit = build_resource_budget(
        snapshot,
        execution_mode="explicit",
        memory_budget_bytes=64 * 1024**2,
        safety_margin_fraction=0.0,
    )
    assert explicit.budget.memory_budget_bytes == 64 * 1024**2


def test_unknown_capacity_error_contains_probe_diagnostics() -> None:
    snapshot = ResourceSnapshot(
        host_available_memory_bytes=8 * GIB,
        cgroup_status="unknown",
        cgroup_paths=("/sys/fs/cgroup/job",),
        uncertainty=("cgroup memory limit is unreadable",),
        detection_sources=("/proc/meminfo:MemAvailable",),
    )
    service = build_resource_budget(snapshot)

    with pytest.raises(ResourceCapacityError, match="cgroup_status=unknown") as error:
        service.acquire("diagnostic test", 1)

    message = str(error.value)
    assert "host_available=8589934592" in message
    assert "cgroup_paths=/sys/fs/cgroup/job" in message
    assert "sources=/proc/meminfo:MemAvailable" in message
    assert "memory_budget_override=False" in message
