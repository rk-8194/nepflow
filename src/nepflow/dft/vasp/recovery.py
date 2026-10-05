"""Explicit VASP recovery decisions and resource-only retry changes."""

from __future__ import annotations

import re
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from nepflow.config.models import NepflowConfig
from nepflow.errors import VaspError
from nepflow.io.atomic import atomic_write_text


def build_retry_levels_for_gpu(
    starting_gpu: int,
    initial_ncore: int,
    initial_kpar: int,
    config: ConfigParser,
) -> list[tuple[int, int, int, int]]:
    """Build the accepted Phase 2 GPU-aware retry escalation table.

    The parser form remains for compatibility with existing recovery callers.
    New DFT stage code should use :func:`build_retry_levels_for_config`.
    """
    starting_gpus_per_node = starting_gpu
    return _build_retry_levels(
        starting_gpus_per_node,
        initial_ncore,
        initial_kpar,
        cores=config.getint("hpc", "cores_per_node", fallback=64),
        gpus_per_node=config.getint("hpc", "gpus_per_node", fallback=4),
        max_nodes=config.getint("hpc", "max_nodes", fallback=16),
    )


def build_retry_levels_for_config(
    starting_gpu: int,
    initial_ncore: int,
    initial_kpar: int,
    config: NepflowConfig,
) -> list[tuple[int, int, int, int]]:
    """Build retry levels from the canonical typed configuration."""

    starting_gpus_per_node = starting_gpu
    return _build_retry_levels(
        starting_gpus_per_node,
        initial_ncore,
        initial_kpar,
        cores=config.hpc.cores_per_node,
        gpus_per_node=config.hpc.gpus_per_node,
        max_nodes=config.hpc.max_nodes,
    )


def _build_retry_levels(
    starting_gpus_per_node: int,
    initial_ncore: int,
    initial_kpar: int,
    *,
    cores: int,
    gpus_per_node: int,
    max_nodes: int,
) -> list[tuple[int, int, int, int]]:

    valid_ncores = sorted(p for p in (2**i for i in range(1, 12)) if p <= cores and cores % p == 0)
    if not valid_ncores:
        valid_ncores = [cores]

    levels: list[tuple[int, int, int, int]] = []
    seen = set()
    initial_key = (initial_ncore, initial_kpar, 1, starting_gpus_per_node)

    def add(ncore: int, kpar: int, nodes: int, gpus_per_node: int) -> None:
        key = (ncore, kpar, nodes, gpus_per_node)
        if key not in seen and key != initial_key:
            seen.add(key)
            levels.append(key)

    valid_kpars = sorted(k for k in (2**i for i in range(0, 8)) if k <= gpus_per_node) or [1]

    def sweep_gpu_tier(gpu_count: int) -> None:
        for kpar in valid_kpars:
            for ncore in valid_ncores:
                add(ncore, kpar, 1, gpu_count)

    sweep_gpu_tier(starting_gpus_per_node)

    valid_gpus = sorted(g for g in [1, 2, 4, 8] if g <= gpus_per_node)
    for gpu in valid_gpus:
        if gpu > starting_gpus_per_node:
            sweep_gpu_tier(gpu)

    highest_gpus_per_node = valid_gpus[-1] if valid_gpus else starting_gpus_per_node
    nodes = 2
    while nodes <= max_nodes:
        kpar = nodes * highest_gpus_per_node
        for ncore in valid_ncores:
            add(ncore, kpar, nodes, highest_gpus_per_node)
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
        raise VaspError(f"Required VASP recovery artifact is missing: {incar_path}")
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
        atomic_write_text(incar_path, "".join(output), encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        if warn is not None:
            warn(f"    Failed to write INCAR: {exc}")
        raise VaspError(f"Could not apply VASP recovery to {incar_path}: {exc}") from exc


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

    @property
    def gpus_per_node(self) -> int | None:
        """Return the legacy ``gpus`` value with its node scope explicit."""

        return self.gpus


class VaspRecoveryPolicy:
    """Typed VASP retry policy used by the canonical DFT reconciler."""

    def __init__(self, config: NepflowConfig) -> None:
        self.config = config

    def decide(
        self,
        *,
        starting_gpu: int,
        initial_ncore: int,
        initial_kpar: int,
        retry_level: int,
    ) -> VaspRecoveryDecision:
        levels = build_retry_levels_for_config(
            starting_gpu,
            initial_ncore,
            initial_kpar,
            self.config,
        )
        return decide_retry(
            levels,
            retry_level,
            max_retry_level=self.config.dft_recovery.max_retry_level,
        )

    def apply(self, decision: VaspRecoveryDecision, working_directory: Path) -> None:
        """Apply one recorded resource decision before a retry submission."""
        if not decision.retry:
            return
        if decision.ncore is None or decision.kpar is None:
            raise VaspError("VASP retry decision is missing INCAR resources")
        write_incar_resource_parameters(
            working_directory,
            decision.ncore,
            decision.kpar,
        )
        for filename in (
            "CHG",
            "CHGCAR",
            "WAVECAR",
            "CONTCAR",
            "DOSCAR",
            "EIGENVAL",
            "PCDAT",
            "OUTCAR",
            "vasprun.xml",
            "OSZICAR",
            "vasp_output.log",
            ".vasp_oom_detected",
        ):
            (Path(working_directory) / filename).unlink(missing_ok=True)


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
    ncore, kpar, nodes, gpus_per_node = levels[next_level - 1]
    return VaspRecoveryDecision(
        retry=True,
        retry_level=next_level,
        ncore=ncore,
        kpar=kpar,
        nodes=nodes,
        gpus=gpus_per_node,
        reason="oom_escalation",
    )


__all__ = [
    "VaspRecoveryDecision",
    "VaspRecoveryPolicy",
    "build_retry_levels_for_gpu",
    "build_retry_levels_for_config",
    "decide_retry",
    "write_incar_resource_parameters",
]
