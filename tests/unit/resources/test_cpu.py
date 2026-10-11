"""Process-local CPU allocation detection regressions."""

from pathlib import Path

import nepflow.resources.cpu as cpu_module


def _stable_cpu_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cpu_module.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(cpu_module.os, "sched_getaffinity", lambda _pid: set(range(128)))
    monkeypatch.setattr(cpu_module, "_PROC_CGROUP", tmp_path / "missing-cgroup-membership")
    monkeypatch.setattr(cpu_module, "_CGROUP_ROOT", tmp_path / "missing-cgroup-root")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "128")


def test_slurm_node_cpu_count_does_not_become_a_process_allocation(
    monkeypatch, tmp_path: Path
) -> None:
    _stable_cpu_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "128")

    assert cpu_module.effective_cpu_allocation() == 128

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert cpu_module.effective_cpu_allocation() == 4


def test_affinity_and_cgroup_quota_and_cpuset_are_all_limits(monkeypatch, tmp_path: Path) -> None:
    _stable_cpu_environment(monkeypatch, tmp_path)
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cpu.max").write_text("400000 100000\n", encoding="utf-8")
    (root / "cpuset.cpus.effective").write_text("0-7\n", encoding="utf-8")
    monkeypatch.setattr(cpu_module, "_CGROUP_ROOT", root)
    monkeypatch.setattr(cpu_module.os, "sched_getaffinity", lambda _pid: set(range(16)))

    assert cpu_module.effective_cpu_allocation() == 4

    monkeypatch.setattr(cpu_module.os, "sched_getaffinity", lambda _pid: {0, 1})
    assert cpu_module.effective_cpu_allocation() == 2
