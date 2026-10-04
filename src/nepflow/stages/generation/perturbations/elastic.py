"""Normal, coupled-normal, and tensor-shear strain perturbations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .models import PerturbationSettings

Annotate = Callable[..., Any]


def normal_strain_matrix(amplitude: float, axis: int) -> np.ndarray:
    """Return a diagonal normal strain tensor, with amplitude in fractional units."""

    matrix = np.eye(3)
    matrix[axis, axis] += amplitude
    return matrix


def coupled_strain_matrix(amplitude: float, axis_a: int, axis_b: int) -> np.ndarray:
    """Return a volume-preserving coupled normal strain tensor."""

    matrix = np.eye(3)
    axis_c = ({0, 1, 2} - {axis_a, axis_b}).pop()
    matrix[axis_a, axis_a] = 1.0 + amplitude
    matrix[axis_b, axis_b] = 1.0 - amplitude
    matrix[axis_c, axis_c] = 1.0 / (1.0 - amplitude * amplitude)
    return matrix


def shear_strain_matrix(amplitude: float, axis_a: int, axis_b: int) -> np.ndarray:
    """Return tensor shear (not engineering shear) for one axis pair."""

    matrix = np.eye(3)
    matrix[axis_a, axis_b] = amplitude
    matrix[axis_b, axis_a] = amplitude
    return matrix


def elastic_stress_set(
    supercell: Any,
    base: Any,
    settings: PerturbationSettings,
    annotate: Annotate,
) -> list[Any]:
    """Generate every enabled strain mode for every non-zero amplitude."""

    if not settings.elastic_stress_enabled:
        return []
    modes = (
        ("normal_xx", normal_strain_matrix, (0,)),
        ("normal_yy", normal_strain_matrix, (1,)),
        ("normal_zz", normal_strain_matrix, (2,)),
        ("coupled_xy", coupled_strain_matrix, (0, 1)),
        ("coupled_xz", coupled_strain_matrix, (0, 2)),
        ("coupled_yz", coupled_strain_matrix, (1, 2)),
        ("shear_xy", shear_strain_matrix, (0, 1)),
        ("shear_xz", shear_strain_matrix, (0, 2)),
        ("shear_yz", shear_strain_matrix, (1, 2)),
    )

    output: list[Any] = []
    for amplitude in settings.elastic_strain_amplitudes:
        value = float(amplitude)
        if value == 0.0:
            continue
        for mode, matrix_factory, axes in modes:
            matrix = matrix_factory(value, *axes)
            strained = supercell.copy()
            strained.set_cell(matrix @ strained.cell[:], scale_atoms=True)
            annotate(
                strained,
                base,
                "elastic_stress",
                parameters={
                    "elastic_mode": mode,
                    "strain_amplitude": value,
                    "strain_matrix": (matrix - np.eye(3)).reshape(-1).tolist(),
                },
                operation_id=f"elastic:{mode}:{value}",
            )
            output.append(strained)
    return output


__all__ = [
    "coupled_strain_matrix",
    "elastic_stress_set",
    "normal_strain_matrix",
    "shear_strain_matrix",
]
