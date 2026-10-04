"""Isotropic volume-profile perturbations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .models import PerturbationSettings

Annotate = Callable[..., Any]


def volume_profile(
    supercell: Any,
    base: Any,
    settings: PerturbationSettings,
    annotate: Annotate,
) -> list[Any]:
    """Generate the accepted isotropic E-V profile in configured order."""

    scale_factors = np.linspace(
        settings.volume_scale_range[0],
        settings.volume_scale_range[1],
        settings.n_volume_points,
    )
    output: list[Any] = []
    for index, scale_factor in enumerate(scale_factors):
        scaled = supercell.copy()
        linear_scale = float(scale_factor) ** (1.0 / 3.0)
        scaled.set_cell(supercell.cell * linear_scale, scale_atoms=True)
        annotate(
            scaled,
            base,
            "volume_profile",
            parameters={
                "volume_scale": float(scale_factor),
                "volume_index": index,
            },
            operation_id=f"volume:{index}",
        )
        output.append(scaled)
    return output


__all__ = ["volume_profile"]
