"""Explicit VASP recovery decisions and resource-only retry changes."""

from __future__ import annotations

import re
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def build_retry_levels_for_gpu(
    starting_gpu: int,
    initial_ncore: int,
    initial_kpar: int,
    config: ConfigParser,
) -> list[tuple[int, int, int, int]]:
    """Build the accepted Phase 2 GPU-aware retry escalation table."""
    cores = config.getint("hpc", "cores_per_node", fallback=64)
    gpus_per_node = config.getint("hpc", "gpus_per_node", fallback=4)
    max_nodes = config.getint("hpc", "max_nodes", fallback=16)

    valid_ncores = sorted(
        p
        for p in (2**i for i in range(1, 12))
        if p <= cores and cores % p == 0
    )
    if not valid_ncores:
        valid_ncores = [cores]

    levels: list[tuple[int, int, int, int]] = []
    seen = set()
    initial_key = (initial_ncore, initial_kpar, 1, starting_gpu)

    def add(ncore: int, kpar: int, nodes: int, gpus: int) -> None:
        key = (ncore, kpar, nodes, gpus)
        if key not in seen and key != initial_key:
            seen.add(key)
            levels.append(key)

    valid_kpars = sorted(
        k for k in (2**i for i in range(0, 8)) if k <= gpus_per_node
    ) or [1]

    def sweep_gpu_tier(gpu_count: int) -> None:
        for kpar in valid_kpars:
            for ncore in valid_ncores:
                add(ncore, kpar, 1, gpu_count)

    sweep_gpu_tier(starting_gpu)

    valid_gpus = sorted(g for g in [1, 2, 4, 8] if g <= gpus_per_node)
    for gpu in valid_gpus:
        if gpu > starting_gpu:
            sweep_gpu_tier(gpu)

    highest_gpu = valid_gpus[-1] if valid_gpus else starting_gpu
    nodes = 2
    while nodes <= max_nodes:
        kpar = nodes * highest_gpu
        for ncore in valid_ncores:
            add(ncore, kpar, nodes, highest_gpu)
        nodes *= 2
    return levels


def write_incar_resource_parameters(
    struct_dir: Path,
    ncore: int,
    kpar: int,
    *,
    warn: Callable[[str], object] | None = None,
) -> None:
    """Update only launcher-controlled NCORE/KPAR values in an INCAR."""
    incar_path = Path(struct_dir) / "INCAR"
    if not incar_path.exists():
        if warn is not None:
            warn(f"    INCAR not found: {incar_path}")
        return
    try:
        content = incar_path.read_text(encoding="utf-8")
        output = []
        for line in content.splitlines(keepends=True):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                output.append(line)
            elif re.match(r"^NCORE\s*=", stripped, re.IGNORECASE):
                output.append(f"NCORE = {ncore}\n")
            elif re.match(r"^KPAR\s*=", stripped, re.IGNORECASE):
                output.append(f"KPAR = {kpar}\n")
            else:
                output.append(line)
        incar_path.write_text("".join(output), encoding="utf-8")
    except OSError as exc:
        if warn is not None:
            warn(f"    Failed to write INCAR: {exc}")


@dataclass(frozen=True, slots=True)
class VaspRecoveryDecision:
    """A recorded recovery choice, separate from applying the choice."""

    retry: bool
    retry_level: int
    ncore: int | None = None
    kpar: int | None = None
    nodes: int | None = None
    gpus: int | None = None
    reason: str = ""


def decide_retry(
    levels: list[tuple[int, int, int, int]],
    retry_level: int,
    *,
    max_retry_level: int,
) -> VaspRecoveryDecision:
    """Choose the next explicit retry level without applying side effects."""
    next_level = retry_level + 1
    if next_level > max_retry_level or next_level > len(levels):
        return VaspRecoveryDecision(
            retry=False,
            retry_level=retry_level,
            reason="retry_limit_exhausted",
        )
    ncore, kpar, nodes, gpus = levels[next_level - 1]
    return VaspRecoveryDecision(
        retry=True,
        retry_level=next_level,
        ncore=ncore,
        kpar=kpar,
        nodes=nodes,
        gpus=gpus,
        reason="oom_escalation",
    )


__all__ = [
    "VaspRecoveryDecision",
    "build_retry_levels_for_gpu",
    "decide_retry",
    "write_incar_resource_parameters",
]
