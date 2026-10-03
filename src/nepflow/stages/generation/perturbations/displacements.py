"""Rattle/displacement perturbations backed by HipHive."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import PerturbationSettings


Annotate = Callable[..., Any]


def sample_rattle_stds(settings: PerturbationSettings, n: int) -> list[float]:
    """Return the accepted inclusive linear rattle-amplitude schedule."""

    if n <= 0:
        return []
    minimum = float(settings.rattle_std_min)
    maximum = float(settings.rattle_std_max)
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
) -> list[Any]:
    """Generate HipHive rattled structures without a scientific fallback."""

    from hiphive.structure_generation import generate_mc_rattled_structures

    output: list[Any] = []
    for index, rattle_std in enumerate(sample_rattle_stds(settings, n)):
        try:
            rattled_structures = generate_mc_rattled_structures(
                supercell,
                n_structures=1,
                rattle_std=rattle_std,
                d_min=settings.rattle_d_min,
                seed=int(seed),
            )
        except Exception as exc:
            raise RuntimeError(
                "HipHive rattling failed; Gaussian substitution is disabled"
            ) from exc
        for structure in rattled_structures:
            annotate(
                structure,
                base,
                "rattled",
                random_seed=seed,
                parameters={"rattle_std": float(rattle_std), "rattle_index": index},
                operation_id=f"rattled:{index}",
            )
            output.append(structure)
    return output


__all__ = ["rattled", "sample_rattle_stds"]
