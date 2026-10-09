"""Rattle/displacement perturbations backed by HipHive."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nepflow.domain.identities import calculate_structure_id

from .models import PerturbationSettings, derive_child_seed

Annotate = Callable[..., Any]


def sample_rattle_stds(settings: PerturbationSettings, n: int) -> list[float]:
    """Return the inclusive linear rattle schedule in Angstroms.

    The endpoints come from ``settings.rattle_std_min/max``; a single value
    uses the configured standard deviation.  No random state is consumed.
    """

    if n <= 0:
        return []
    minimum = settings.rattle_std_min
    maximum = settings.rattle_std_max
    if minimum is None or maximum is None:
        minimum = maximum = settings.rattle_std
    if abs(maximum - minimum) < 1e-12:
        return [float(settings.rattle_std)] * n
    if n == 1:
        return [minimum]
    step = (maximum - minimum) / float(n - 1)
    return [float(minimum + step * index) for index in range(n)]


def rattled(
    supercell: Any,
    base: Any,
    n: int,
    settings: PerturbationSettings,
    seed: int,
    annotate: Annotate,
    *,
    slot_start: int = 0,
    total_count: int | None = None,
) -> list[Any]:
    """Generate HipHive rattled structures with a minimum distance in Angstroms.

    ``supercell`` is treated as a periodic ASE-like cell.  Each rattle slot
    receives a deterministic child seed derived from the base, root seed, and
    slot before it is passed to HipHive.  HipHive failure raises
    ``RuntimeError``; Gaussian substitution would not be scientifically
    equivalent.
    """

    from hiphive.structure_generation import generate_mc_rattled_structures

    total = n if total_count is None else total_count
    output: list[Any] = []
    base_structure_id = calculate_structure_id(base)
    for index, rattle_std in enumerate(
        sample_rattle_stds(settings, total)[slot_start : slot_start + n],
        start=slot_start,
    ):
        child_seed = derive_child_seed(base_structure_id, seed, "rattled", index)
        try:
            rattled_structures = generate_mc_rattled_structures(
                supercell,
                n_structures=1,
                rattle_std=rattle_std,
                d_min=settings.rattle_d_min,
                seed=child_seed,
            )
        except Exception as exc:
            # HipHive is an external generator with version-dependent
            # exception classes.  Its failure is terminal for this requested
            # perturbation family; a Gaussian substitute is non-equivalent.
            raise RuntimeError(
                "HipHive rattling failed; Gaussian substitution is disabled"
            ) from exc
        for structure in rattled_structures:
            annotate(
                structure,
                base,
                "rattled",
                random_seed=child_seed,
                parameters={"rattle_std": float(rattle_std), "rattle_index": index},
                operation_id=f"rattled:{index}",
            )
            output.append(structure)
    return output


__all__ = ["rattled", "sample_rattle_stds"]
